# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VISION_MODELS, MultimodalSpecialTokens, VisionModel
from lmdeploy.vl.model.preprocess_utils import get_mm_items_offset

logger = get_logger('lmdeploy')


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute Qwen3-ASR audio token counts from feature mask lengths."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


@VISION_MODELS.register_module()
class Qwen3ASRModel(VisionModel):
    """Qwen3-ASR audio preprocessor."""

    _arch = ['Qwen3ASRForConditionalGeneration']
    executor_max_workers = 2

    def build_preprocessor(self, trust_remote_code: bool = False):
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        chat_template_path = Path(self.model_path) / 'chat_template.json'
        if getattr(self.tokenizer, 'chat_template', None) is None and chat_template_path.exists():
            with open(chat_template_path, encoding='utf-8') as f:
                self.tokenizer.chat_template = json.load(f).get('chat_template')
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_path)

        self.audio_token = getattr(self.tokenizer, 'audio_token', '<|audio_pad|>')
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_start_token = getattr(self.tokenizer, 'audio_bos_token', '<|audio_start|>')
        self.audio_end_token = getattr(self.tokenizer, 'audio_eos_token', '<|audio_end|>')
        self.mm_tokens = MultimodalSpecialTokens(audio_token=self.audio_token, audio_token_id=self.audio_token_id)
        self.use_native_audio_tower = os.getenv('LMDEPLOY_QWEN3_ASR_NATIVE_AUDIO', '1') != '0'

    @staticmethod
    def _normalize_audio_item(data):
        if isinstance(data, tuple):
            data = data[0]
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _as_config(value) -> PretrainedConfig:
        if isinstance(value, PretrainedConfig):
            return value
        return PretrainedConfig(**value)

    @staticmethod
    def _get_nested_config(config, name: str):
        if isinstance(config, dict):
            return config[name]
        return getattr(config, name)

    @staticmethod
    def _is_audio_modality(modality) -> bool:
        return modality == Modality.AUDIO or modality == Modality.AUDIO.value

    def _expand_audio_tokens(self, prompt: str, audio_lengths: list[int]) -> str:
        for length in audio_lengths:
            prompt = prompt.replace(self.audio_token, self.audio_token * int(length), 1)
        return prompt

    def get_input_prompt(self,
                         messages: list[dict],
                         chat_template,
                         sequence_start: bool,
                         chat_template_kwargs: dict | None = None) -> str:
        """Render Qwen3-ASR prompts with the checkpoint chat template."""
        kwargs = chat_template_kwargs or {}
        add_generation_prompt = messages[-1]['role'] != 'assistant'
        return self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=add_generation_prompt,
                                                  **kwargs)

    def preprocess(self,
                   messages: list[dict],
                   input_prompt: str | list[int],
                   mm_processor_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Preprocess audio and return input ids with audio feature tensors."""
        if not isinstance(input_prompt, str):
            raise TypeError('Qwen3-ASR audio preprocessing expects a rendered prompt string.')

        mm_items = self.collect_multimodal_items(messages)
        raw_audios = [
            self._normalize_audio_item(data) for modality, data, _ in mm_items if self._is_audio_modality(modality)
        ]
        if not raw_audios:
            input_ids = self.tokenizer.encode(input_prompt, add_special_tokens=False)
            return dict(prompt=input_prompt, input_ids=input_ids, multimodal=[])

        audio_kwargs = dict(sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors='pt')
        if mm_processor_kwargs:
            audio_kwargs.update(mm_processor_kwargs.get('audio', {}))

        audio_inputs = self.feature_extractor(raw_audios, **audio_kwargs)
        input_features = audio_inputs['input_features']
        feature_attention_mask = audio_inputs['attention_mask']
        audio_lengths = _get_feat_extract_output_lengths(feature_attention_mask.sum(-1)).tolist()

        expanded_prompt = self._expand_audio_tokens(input_prompt, audio_lengths)
        input_ids = torch.tensor(self.tokenizer.encode(expanded_prompt, add_special_tokens=False), dtype=torch.long)
        offsets = get_mm_items_offset(input_ids, self.audio_token_id)
        assert len(offsets) == len(raw_audios), (
            f'the number of {self.audio_token} ranges is not equal to input audios, '
            f'{len(offsets)} vs {len(raw_audios)}')

        multimodal = []
        for idx, offset in enumerate(offsets):
            multimodal.append(
                dict(
                    modality=Modality.AUDIO,
                    input_features=input_features[idx],
                    feature_attention_mask=feature_attention_mask[idx],
                    offset=offset,
                    audio_token_id=self.audio_token_id,
                ))

        return dict(prompt=expanded_prompt, input_ids=input_ids.tolist(), multimodal=multimodal)

    def build_model(self, trust_remote_code: bool = False):
        """Build the Qwen3-ASR audio tower used by TurboMind hybrid mode."""
        if getattr(self, 'use_native_audio_tower', False):
            logger.info('Qwen3-ASR audio tower will run natively in TurboMind.')
            return

        from lmdeploy.pytorch.models.qwen3_asr import Qwen3ASRAudioEncoder
        from lmdeploy.turbomind.deploy.loader import create_loader

        thinker_config = self._get_nested_config(self.hf_config, 'thinker_config')
        audio_config = self._as_config(self._get_nested_config(thinker_config, 'audio_config'))
        has_cuda = torch.cuda.is_available()
        dtype = torch.float16 if has_cuda else torch.float32
        device = torch.device('cuda:0' if has_cuda else 'cpu')
        audio_tower = Qwen3ASRAudioEncoder(audio_config, dtype=dtype, device='cpu')

        prefix = 'thinker.audio_tower.'
        loader = create_loader(self.model_path, r'thinker\.audio_tower\.layers\.([0-9]+).', [])
        state_dict = {}
        for _, params in loader.items():
            for name, tensor in params.items():
                if name.startswith(prefix):
                    state_dict[name[len(prefix):]] = tensor
        missing, unexpected = audio_tower.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError('Failed to load Qwen3-ASR audio tower weights: '
                               f'missing={missing}, unexpected={unexpected}.')

        self.audio_tower = audio_tower.to(device=device, dtype=dtype).eval()
        self.audio_tower_device = device
        self.audio_tower_dtype = dtype
        logger.info('Qwen3-ASR audio tower loaded for TurboMind hybrid inference.')

    @torch.no_grad()
    def forward(self, messages: dict[str, Any], max_batch_size: int = 1) -> dict[str, Any]:
        """Encode audio features for TurboMind input embedding injection."""
        if not isinstance(messages, dict):
            raise TypeError('Qwen3-ASR TurboMind forward expects the dict returned by preprocess().')
        if getattr(self, 'use_native_audio_tower', False):
            return messages
        if not hasattr(self, 'audio_tower'):
            raise RuntimeError('Qwen3-ASR audio tower is not built. Call build_model() before forward().')

        embeddings = []
        for item in messages.get('multimodal', []):
            input_features = item['input_features'].to(device=self.audio_tower_device, dtype=self.audio_tower_dtype)
            feature_attention_mask = item['feature_attention_mask'].to(device=self.audio_tower_device)
            feature_len = feature_attention_mask.sum(-1).to(dtype=torch.long)
            audio_feature = self.audio_tower(input_features[:, :int(feature_len.item())],
                                             feature_lens=feature_len.unsqueeze(0))
            embeddings.append(audio_feature.cpu())

        messages['input_embeddings'] = embeddings
        return messages

    def to_turbomind(self,
                     messages: dict[str, Any],
                     chat_template,
                     tokenizer,
                     sequence_start,
                     chat_template_kwargs=None,
                     **kwargs) -> dict[str, Any]:
        """Pack Qwen3-ASR audio embeddings for TurboMind decoder inference."""
        if not isinstance(messages, dict):
            raise TypeError('Qwen3-ASR TurboMind wrap expects the dict returned by forward().')

        input_ids = messages['input_ids']
        ranges = [item['offset'] for item in messages.get('multimodal', [])]
        embeddings = messages.get('input_embeddings', [])
        if getattr(self, 'use_native_audio_tower', False) and not embeddings:
            feature_lens = []
            features = []
            for item, embedding_range in zip(messages.get('multimodal', []), ranges):
                feature_attention_mask = item['feature_attention_mask']
                feature_len = feature_attention_mask.sum(-1).to(dtype=torch.long)
                expected_tokens = embedding_range[1] - embedding_range[0]
                actual_tokens = int(_get_feat_extract_output_lengths(feature_len).item())
                if actual_tokens != expected_tokens:
                    raise ValueError('Audio feature length does not match token range: '
                                     f'{actual_tokens} vs {expected_tokens}.')
                features.append(item['input_features'])
                feature_lens.append(int(feature_len.item()))

            audio_features = torch.stack(features, dim=0).contiguous() if features else None
            audio_feature_lens = torch.tensor(feature_lens, dtype=torch.int32) if feature_lens else None
            audio_embedding_ranges = torch.tensor(ranges, dtype=torch.int32) if ranges else None
            return dict(
                prompt=messages.get('prompt'),
                input_ids=input_ids,
                audio_features=audio_features,
                audio_feature_lens=audio_feature_lens,
                audio_embedding_ranges=audio_embedding_ranges,
            )

        for embedding, embedding_range in zip(embeddings, ranges):
            expected_tokens = embedding_range[1] - embedding_range[0]
            if embedding.shape[0] != expected_tokens:
                raise ValueError('Audio embedding length does not match token range: '
                                 f'{embedding.shape[0]} vs {expected_tokens}.')

        return dict(
            prompt=messages.get('prompt'),
            input_ids=input_ids,
            input_embeddings=embeddings,
            input_embedding_ranges=ranges,
        )
