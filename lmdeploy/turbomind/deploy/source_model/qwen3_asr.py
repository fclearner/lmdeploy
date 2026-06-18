# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.archs import get_model_arch

from ..loader import create_loader
from .base import INPUT_MODELS, BaseInputModel
from .qwen import Qwen3Model, Qwen3Reader


def _get_text_config(config):
    """Return Qwen3-ASR thinker text config as a plain dict."""
    thinker_config = getattr(config, 'thinker_config', None)
    if thinker_config is None and isinstance(config, dict):
        thinker_config = config.get('thinker_config')
    if thinker_config is None:
        raise RuntimeError('Qwen3-ASR config does not contain thinker_config.')

    text_config = getattr(thinker_config, 'text_config', None)
    if text_config is None and isinstance(thinker_config, dict):
        text_config = thinker_config.get('text_config')
    if text_config is None:
        raise RuntimeError('Qwen3-ASR thinker_config does not contain text_config.')

    return text_config.to_dict() if hasattr(text_config, 'to_dict') else dict(text_config)


def _get_audio_config(config):
    """Return Qwen3-ASR thinker audio config as a plain dict."""
    thinker_config = getattr(config, 'thinker_config', None)
    if thinker_config is None and isinstance(config, dict):
        thinker_config = config.get('thinker_config')
    if thinker_config is None:
        raise RuntimeError('Qwen3-ASR config does not contain thinker_config.')

    audio_config = getattr(thinker_config, 'audio_config', None)
    if audio_config is None and isinstance(thinker_config, dict):
        audio_config = thinker_config.get('audio_config')
    if audio_config is None:
        raise RuntimeError('Qwen3-ASR thinker_config does not contain audio_config.')

    return audio_config.to_dict() if hasattr(audio_config, 'to_dict') else dict(audio_config)


class Qwen3ASRReader(Qwen3Reader):
    """Reader for the Qwen3-ASR text decoder inside ``thinker.model``."""

    attn_layer_prefix = 'thinker.model.layers'
    attn_layer_patten = r'thinker\.model\.layers\.([0-9]+).'
    tok_embeddings_key = 'thinker.model.embed_tokens.weight'
    norm_weight_key = 'thinker.model.norm.weight'
    output_weight_key = 'thinker.lm_head.weight'


@INPUT_MODELS.register_module(name='qwen3_asr')
class Qwen3ASRModel(Qwen3Model):
    """Qwen3-ASR text decoder in HF checkpoint format."""

    Reader = Qwen3ASRReader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs: dict):
        BaseInputModel.__init__(self, model_path, tokenizer_path)
        self.policy = kwargs.get('input_policy')
        _, root_config = get_model_arch(model_path)
        self.root_config = root_config
        self.model_config = _get_text_config(root_config)
        self.audio_config = _get_audio_config(root_config)
        self.model_config['tie_word_embeddings'] = False
        self.fp8_quant = kwargs.get('fp8_quant', False)

    def audio_info(self):
        """Return TurboMind config values for the Qwen3-ASR audio tower."""
        keys = [
            'd_model',
            'output_dim',
            'num_mel_bins',
            'downsample_hidden_size',
            'encoder_layers',
            'encoder_attention_heads',
            'encoder_ffn_dim',
            'max_source_positions',
            'n_window',
            'n_window_infer',
            'conv_chunksize',
            'activation_function',
        ]
        return {'enabled': True, **{key: self.audio_config[key] for key in keys}}

    def export_extra(self, output_model):
        """Export Qwen3-ASR audio tower weights for native TurboMind execution."""
        prefix = 'thinker.audio_tower.'
        loader = create_loader(self.model_path, r'thinker\.audio_tower\.layers\.([0-9]+).', [])
        exported = set()
        for _, params in loader.items():
            for name, tensor in params.items():
                if not name.startswith(prefix) or name in exported:
                    continue
                if tensor.dim() == 2 and name.endswith('.weight'):
                    tensor = tensor.t()
                output_model.export_weight(tensor, f'audio_tower.{name[len(prefix):]}')
                exported.add(name)
