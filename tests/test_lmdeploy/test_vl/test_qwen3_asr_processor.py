import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.messages import VisionConfig
from lmdeploy.pytorch.transformers.configuration_qwen3_asr import Qwen3ASRConfig
from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.utils import _get_and_verify_max_len
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.engine import ImageEncoder
from lmdeploy.vl.media.audio import AudioMediaIO
from lmdeploy.vl.model.qwen3_asr import Qwen3ASRModel, _get_feat_extract_output_lengths


def test_qwen3_asr_config_exposes_nested_thinker_config():
    cfg = Qwen3ASRConfig(
        thinker_config=dict(
            audio_token_id=151676,
            audio_start_token_id=151669,
            audio_end_token_id=151670,
            audio_config=dict(d_model=1024),
            text_config=dict(hidden_size=1024),
        ),
        support_languages=['en', 'zh'],
    )

    assert cfg.model_type == 'qwen3_asr'
    assert cfg.audio_token_id == 151676
    assert cfg.thinker_config.audio_config.d_model == 1024
    assert cfg.thinker_config.text_config.hidden_size == 1024
    assert cfg.tie_word_embeddings is False


def test_qwen3_asr_audio_output_lengths():
    input_lengths = torch.tensor([100, 101, 200])
    output_lengths = _get_feat_extract_output_lengths(input_lengths)

    assert output_lengths.tolist() == [13, 14, 26]


def test_qwen3_asr_expands_audio_placeholders():
    model = object.__new__(Qwen3ASRModel)
    model.audio_token = '<|audio_pad|>'

    prompt = 'a<|audio_pad|>b<|audio_pad|>c'
    assert model._expand_audio_tokens(prompt, [2, 1]) == 'a<|audio_pad|><|audio_pad|>b<|audio_pad|>c'
    assert model._is_audio_modality(Modality.AUDIO)
    assert model._is_audio_modality('audio')


def test_qwen3_asr_uses_checkpoint_chat_template_for_audio_prompt():
    class FakeTokenizer:

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages[0]['content'][0]['type'] == Modality.AUDIO.value
            assert kwargs == {'foo': 'bar'}
            return '<|audio_start|><|audio_pad|><|audio_end|>'

    model = object.__new__(Qwen3ASRModel)
    model.tokenizer = FakeTokenizer()
    messages = [{'role': 'user', 'content': [{'type': Modality.AUDIO.value, 'data': object()}]}]

    prompt = model.get_input_prompt(messages, chat_template=None, sequence_start=True, chat_template_kwargs={'foo': 'bar'})

    assert prompt == '<|audio_start|><|audio_pad|><|audio_end|>'


def test_audio_media_io_normalizes_multichannel_integer_audio():
    audio = np.array([[0, 32767], [-32768, 0]], dtype=np.int16)
    result = AudioMediaIO._to_float32(audio, mono=True)

    assert result.dtype == np.float32
    assert result.shape == (2,)
    np.testing.assert_allclose(result, np.array([32767 / 65536, -0.5], dtype=np.float32), rtol=1e-5)


def test_multimodal_processor_accepts_audio_data():
    raw_audio = np.zeros(1600, dtype=np.float32)
    messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'text',
                'text': 'transcribe',
            },
            {
                'type': 'audio_data',
                'audio_data': {
                    'data': raw_audio,
                },
            },
        ],
    }]

    parsed = asyncio.run(MultimodalProcessor.async_parse_multimodal_item(messages))

    assert parsed[0]['content'][0] == {'type': 'text', 'text': 'transcribe'}
    assert parsed[0]['content'][1]['type'] == Modality.AUDIO.value
    assert parsed[0]['content'][1]['data'] is raw_audio
    processor = object.__new__(MultimodalProcessor)
    assert processor._has_multimodal_input(messages)


def test_qwen3_asr_turbomind_reader_uses_thinker_text_decoder():
    try:
        from lmdeploy.turbomind.deploy.source_model.qwen3_asr import Qwen3ASRReader
        from lmdeploy.turbomind.supported_models import SUPPORTED_ARCHS
    except ImportError as exc:
        pytest.skip(f'TurboMind extension is unavailable in this environment: {exc}')

    assert SUPPORTED_ARCHS['Qwen3ASRForConditionalGeneration'] == 'qwen3_asr'
    assert Qwen3ASRReader.attn_layer_prefix == 'thinker.model.layers'
    assert Qwen3ASRReader.tok_embeddings_key == 'thinker.model.embed_tokens.weight'
    assert Qwen3ASRReader.output_weight_key == 'thinker.lm_head.weight'


def test_qwen3_asr_max_len_uses_nested_text_config():
    cfg = {'thinker_config': {'text_config': {'max_position_embeddings': 4096}}}

    assert _get_and_verify_max_len(cfg, None) == 4096


def test_qwen3_asr_to_turbomind_wraps_audio_embeddings():
    model = object.__new__(Qwen3ASRModel)
    embedding = torch.zeros(2, 4)
    messages = dict(
        prompt='prompt',
        input_ids=[10, 151676, 151676, 11],
        input_embeddings=[embedding],
        multimodal=[dict(offset=(1, 3))],
    )

    result = model.to_turbomind(messages, chat_template=None, tokenizer=None, sequence_start=True)

    assert result['input_embeddings'][0] is embedding
    assert result['input_embedding_ranges'] == [(1, 3)]
    assert 'input_meta' not in result


def test_qwen3_asr_to_turbomind_wraps_native_audio_inputs():
    model = object.__new__(Qwen3ASRModel)
    model.use_native_audio_tower = True
    input_features = torch.zeros(128, 100)
    messages = dict(
        prompt='prompt',
        input_ids=list(range(16)),
        multimodal=[
            dict(
                input_features=input_features,
                feature_attention_mask=torch.ones(100),
                offset=(1, 14),
            )
        ],
    )

    result = model.to_turbomind(messages, chat_template=None, tokenizer=None, sequence_start=True)

    assert 'input_embeddings' not in result
    assert result['audio_features'].shape == (1, 128, 100)
    assert torch.equal(result['audio_features'][0], input_features)
    assert result['audio_feature_lens'].tolist() == [100]
    assert result['audio_feature_lens'].dtype == torch.int32
    assert result['audio_embedding_ranges'].tolist() == [[1, 14]]
    assert result['audio_embedding_ranges'].dtype == torch.int32


def test_image_encoder_uses_model_executor_workers(monkeypatch):
    class DummyModel:

        executor_max_workers = 3

        def preprocess(self, messages):
            return messages

    monkeypatch.setattr('lmdeploy.vl.engine.load_vl_model', lambda *args, **kwargs: DummyModel())

    encoder = ImageEncoder('dummy',
                           'turbomind',
                           VisionConfig(),
                           backend_config=SimpleNamespace(max_batch_size=2))

    try:
        assert encoder.executor._max_workers == 2
    finally:
        encoder.executor.shutdown(wait=True)


def test_image_encoder_uses_explicit_thread_safe_vision_config(monkeypatch):
    class DummyModel:

        executor_max_workers = 2

        def preprocess(self, messages):
            return messages

    monkeypatch.setattr('lmdeploy.vl.engine.load_vl_model', lambda *args, **kwargs: DummyModel())

    encoder = ImageEncoder('dummy',
                           'turbomind',
                           VisionConfig(max_batch_size=4, thread_safe=True),
                           backend_config=SimpleNamespace(max_batch_size=8))

    try:
        assert encoder.executor._max_workers == 4
    finally:
        encoder.executor.shutdown(wait=True)
