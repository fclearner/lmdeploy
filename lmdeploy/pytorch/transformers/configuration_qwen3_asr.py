# Copyright (c) OpenMMLab. All rights reserved.
from transformers.configuration_utils import PretrainedConfig


def _as_config(value: dict | PretrainedConfig | None) -> PretrainedConfig:
    """Convert nested config dictionaries into attribute configs."""
    if isinstance(value, PretrainedConfig):
        return value
    return PretrainedConfig(**(value or {}))


class Qwen3ASRConfig(PretrainedConfig):
    """Minimal Qwen3-ASR config for LMDeploy model loading.

    Recent transformers releases include this config natively. This fallback is
    used when AutoConfig does not yet know ``model_type=qwen3_asr``.
    """

    model_type = 'qwen3_asr'

    def __init__(self, thinker_config: dict | PretrainedConfig | None = None, support_languages=None, **kwargs):
        super().__init__(**kwargs)
        thinker_config = thinker_config or {}
        thinker_dict = thinker_config.to_dict() if isinstance(thinker_config, PretrainedConfig) else thinker_config

        self.support_languages = support_languages or []
        self.thinker_config = _as_config(thinker_config)
        self.thinker_config.audio_config = _as_config(thinker_dict.get('audio_config'))
        self.thinker_config.text_config = _as_config(thinker_dict.get('text_config'))

        for name in ('audio_token_id', 'audio_start_token_id', 'audio_end_token_id', 'dtype'):
            if name in thinker_dict:
                setattr(self.thinker_config, name, thinker_dict[name])

        self.audio_token_id = getattr(self.thinker_config, 'audio_token_id', None)
        self.audio_start_token_id = getattr(self.thinker_config, 'audio_start_token_id', None)
        self.audio_end_token_id = getattr(self.thinker_config, 'audio_end_token_id', None)
        # Qwen3-ASR checkpoints include thinker.lm_head.weight, so the top-level
        # LMDeploy loader must not treat lm_head as tied and skip it.
        self.tie_word_embeddings = False
