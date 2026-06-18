# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.vl.constants import Modality

from .patch import add_prefix
from .qwen3_vl import Qwen3VLTextModel
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixinV1


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute output lengths of the Qwen3-ASR convolutional audio front-end."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class Qwen3ASRAudioAttention(nn.Module):
    """Audio encoder self-attention."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}.')

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True, dtype=dtype, device=device)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Forward variable-length bidirectional attention."""
        seq_length = hidden_states.size(0)
        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)

        outputs = []
        boundaries = cu_seqlens.tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            query = query_states[start:end].transpose(0, 1).unsqueeze(0)
            key = key_states[start:end].transpose(0, 1).unsqueeze(0)
            value = value_states[start:end].transpose(0, 1).unsqueeze(0)
            attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            outputs.append(attn_output.squeeze(0).transpose(0, 1).reshape(end - start, self.embed_dim))

        attn_output = torch.cat(outputs, dim=0) if outputs else hidden_states.new_empty(0, self.embed_dim)
        return self.out_proj(attn_output)


class Qwen3ASRAudioEncoderLayer(nn.Module):
    """Qwen3-ASR audio encoder layer."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(config, dtype=dtype, device=device)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(embed_dim, config.encoder_ffn_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, embed_dim, dtype=dtype, device=device)
        self.final_layer_norm = nn.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states=hidden_states, cu_seqlens=cu_seqlens)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class SinusoidsPositionEmbedding(nn.Module):
    """Sinusoidal position embedding used by Qwen3-ASR audio tower."""

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError('SinusoidsPositionEmbedding needs even channels input.')
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer('positional_embedding',
                             torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
                             persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]


class Qwen3ASRAudioEncoder(nn.Module):
    """Qwen3-ASR audio encoder."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize
        self.positional_embedding = SinusoidsPositionEmbedding(config.max_source_positions, embed_dim)
        self.layers = nn.ModuleList(
            [Qwen3ASRAudioEncoderLayer(config, dtype=dtype, device=device) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model, dtype=dtype, device=device)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1, dtype=dtype, device=device)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1, dtype=dtype, device=device)
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1, dtype=dtype, device=device)
        conv_out_dim = config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = nn.Linear(conv_out_dim, config.d_model, bias=False, dtype=dtype, device=device)
        self.proj1 = nn.Linear(config.d_model, config.d_model, dtype=dtype, device=device)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim, dtype=dtype, device=device)

    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        """Encode one audio feature tensor."""
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_count = int(chunk_num.sum().item())
        chunk_lengths = torch.full((chunk_count,), self.n_window * 2, dtype=torch.long, device=feature_lens.device)
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(int(length.item()), dtype=torch.bool, device=padded_feature.device)
             for length in feature_lens_after_cnn],
            batch_first=True,
        )

        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

        bsz, channels, freq, time = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(bsz, time, channels * freq))
        positional_embedding = self.positional_embedding(padded_embed.shape[1]).unsqueeze(0).to(padded_embed)
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]

        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        window_aftercnn = max(int(window_aftercnn), 1)
        for cnn_len in aftercnn_lens.tolist():
            cnn_len = int(cnn_len)
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj2(self.act(self.proj1(hidden_states)))
        return hidden_states


class Qwen3ASRForConditionalGeneration(nn.Module, DeployModelMixinV1, CudaGraphMixin):
    """Qwen3-ASR model for PyTorch backend inference."""

    packed_modules_mapping = {
        'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        'gate_up_proj': ['gate_proj', 'up_proj'],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.input_processor = Qwen3ASRInputProcessor(config, dtype)

        thinker_config = config.thinker_config
        self.audio_tower = Qwen3ASRAudioEncoder(thinker_config.audio_config, dtype=dtype, device=device)
        self.model = Qwen3VLTextModel(thinker_config.text_config,
                                      dtype=dtype,
                                      device=device,
                                      prefix=add_prefix('model', prefix))
        self.lm_head = self.build_lm_head(thinker_config.text_config.hidden_size,
                                          thinker_config.text_config.vocab_size,
                                          bias=False,
                                          dtype=dtype,
                                          device=device)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def _get_audio_features(self, audio_inputs: list[MultiModalData], dtype: torch.dtype) -> torch.Tensor:
        audio_features = []
        for audio_input in audio_inputs:
            input_feature = audio_input.data.to(dtype=dtype)
            feature_len = audio_input.meta['feature_len']
            audio_feature = self.audio_tower(input_feature[:, :int(feature_len.item())],
                                             feature_lens=feature_len.unsqueeze(0))
            audio_features.append(audio_feature)
        return torch.cat(audio_features, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        mrope_position_ids: torch.Tensor = None,
        audio_inputs: list[MultiModalData] | None = None,
        **kwargs,
    ):
        """Model forward."""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if audio_inputs:
                audio_features = self._get_audio_features(audio_inputs, inputs_embeds.dtype).to(inputs_embeds)
                audio_mask = input_ids == self.config.thinker_config.audio_token_id
                if audio_mask.sum().item() != audio_features.size(0):
                    raise ValueError('Audio token count does not match Qwen3-ASR audio feature count: '
                                     f'{audio_mask.sum().item()} vs {audio_features.size(0)}.')
                audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
        )
        return hidden_states

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        audio_inputs = None
        if context.input_multimodals is not None:
            mm_inputs = [input_mm.get('mm_data', []) for input_mm in context.input_multimodals]
            mm_inputs = [item for sublist in mm_inputs for item in sublist]
            audio_inputs = [item for item in mm_inputs if item.modality == Modality.AUDIO]

        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=getattr(context, 'mrope_position_ids', None),
            audio_inputs=audio_inputs,
        )

    @classmethod
    def rename_weight(cls, name: str) -> str:
        """Rename Qwen3-ASR checkpoint weights to LMDeploy module names."""
        if name.startswith('thinker.audio_tower.'):
            return 'audio_tower.' + name[len('thinker.audio_tower.'):]
        if name.startswith('thinker.model.'):
            return 'model.' + name[len('thinker.model.'):]
        if name.startswith('thinker.lm_head.'):
            return 'lm_head.' + name[len('thinker.lm_head.'):]
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights."""
        stacked_params_mapping = [
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue
            name = self.rename_weight(name)
            is_text_decoder_weight = name.startswith('model.')
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if not is_text_decoder_weight or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class Qwen3ASRInputProcessor(BaseModelInputProcessor):
    """Qwen3-ASR multimodal input processor."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype) -> None:
        self.config = config
        self.dtype = dtype

    def _make_audio_mm_data(self, input_mm: dict[str, Any]) -> MultiModalData:
        input_features = input_mm['input_features']
        if self.dtype is not None:
            input_features = input_features.to(self.dtype)
        feature_attention_mask = input_mm['feature_attention_mask']
        feature_len = feature_attention_mask.sum(-1).to(dtype=torch.long)
        offset = input_mm['offset']
        audio_token_id = input_mm['audio_token_id']
        return MultiModalData(
            modality=Modality.AUDIO,
            data=input_features,
            start=offset[0],
            end=offset[1],
            meta=dict(feature_len=feature_len, audio_token_id=audio_token_id),
        )

    def preprocess_input(self,
                         input_ids: list[int],
                         input_multimodals: list[dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """Prepare audio input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_mm_data = []
        for input_mm in input_multimodals:
            modality = input_mm.get('modality')
            if modality == Modality.AUDIO or modality == Modality.AUDIO.value:
                input_mm_data.append(self._make_audio_mm_data(input_mm))

        result = PreprocessInputResult(input_ids=input_ids, input_multimodals=dict(mm_data=input_mm_data))
        return result
