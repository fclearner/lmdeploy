# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Qwen3ASRModelConfigBuilder(AutoModelConfigBuilder):
    """Model config builder for Qwen3-ASR."""

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return getattr(hf_config, 'model_type', None) == 'qwen3_asr' and hasattr(hf_config, 'thinker_config')

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        text_config = hf_config.thinker_config.text_config
        if hasattr(hf_config, 'quantization_config') and not hasattr(text_config, 'quantization_config'):
            setattr(text_config, 'quantization_config', hf_config.quantization_config)

        cfg = DefaultModelConfigBuilder.build(text_config, model_path, **kwargs)
        cfg.hf_config = hf_config
        cfg.llm_config = text_config
        return cfg
