import asyncio

from duplex.lmdeploy_client import DuplexLmdeployClient, DuplexLmdeployClientConfig, TokenDecisionLabels
from turbomind_service_core import GenerateResult


def run(coro):
    return asyncio.run(coro)


class FakeGrpcClient:

    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    async def health_async(self):
        return {"status": "ok", "target": "fake"}

    async def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.results.pop(0)


def make_client(results):
    labels = TokenDecisionLabels(valid_id=11, invalid_id=22, end_id=33)
    config = DuplexLmdeployClientConfig(
        target="fake",
        timeout_s=1,
        health_interval_s=60,
        max_new_tokens=1,
        certainty_threshold=0.05,
        completion_threshold=0.1,
        invalid_bias=0.1,
    )
    fake = FakeGrpcClient(results)
    return DuplexLmdeployClient(config, labels=labels, grpc_client=fake), fake


def result(token_id=None, text=""):
    return GenerateResult(
        ok=True,
        request_id="req-1",
        status="finished",
        status_code=200,
        tokens=1,
        text=text,
        token_ids=[] if token_id is None else [token_id],
    )


def test_validity_decision_maps_token_id_and_config():
    client, fake = make_client([result(11)])

    async def scenario():
        assert await client.infer("req-1", "prompt", decoding_type=0, timeout=1) == "<valid>"
        await client.close()

    run(scenario())

    call = fake.calls[0]
    assert call["infer_type"] == 0
    assert call["timeout_s"] == 1
    assert call["max_new_tokens"] == 1
    assert call["include_token_ids"] is True
    assert call["generation_config"]["token_decision_valid_id"] == 11
    assert call["generation_config"]["token_decision_invalid_id"] == 22
    assert call["generation_config"]["token_decision_end_id"] == 33
    assert call["generation_config"]["token_decision_certainty_threshold"] == 0.05
    assert call["generation_config"]["token_decision_invalid_bias"] == 0.1


def test_completion_decision_maps_end_token_id():
    client, fake = make_client([result(33)])

    async def scenario():
        assert await client.infer("req-1", "prompt<valid>", decoding_type=1, timeout=1) == "<|im_end|>"
        await client.close()

    run(scenario())

    call = fake.calls[0]
    assert call["infer_type"] == 1
    assert call["max_new_tokens"] == 1
    assert call["generation_config"]["token_decision_completion_threshold"] == 0.1
    assert "token_decision_certainty_threshold" not in call["generation_config"]


def test_invalidity_decision_falls_back_to_text_prefix():
    client, _fake = make_client([result(text="<invalid> extra")])

    async def scenario():
        assert await client.infer("req-1", "prompt", decoding_type=0, timeout=1) == "<invalid>"
        await client.close()

    run(scenario())


def test_failed_grpc_result_raises_runtime_error():
    client, _fake = make_client([
        GenerateResult(
            ok=False,
            request_id="req-1",
            status="bad_request",
            status_code=400,
            error="missing ids",
        )
    ])

    async def scenario():
        try:
            await client.infer("req-1", "prompt", decoding_type=0, timeout=1)
        except RuntimeError as exc:
            assert "missing ids" in str(exc)
        else:
            raise AssertionError("RuntimeError was not raised")
        await client.close()

    run(scenario())


def test_config_uses_normalized_grpc_client_channels(monkeypatch):
    monkeypatch.delenv("DUPLEX_GRPC_CLIENT_CHANNELS", raising=False)
    monkeypatch.delenv("TM_GRPC_CLIENT_CHANNELS", raising=False)

    config = DuplexLmdeployClientConfig.from_env(default_grpc_client_channels=7)
    assert config.channels == 7

    monkeypatch.setenv("DUPLEX_GRPC_CLIENT_CHANNELS", "13")
    config = DuplexLmdeployClientConfig.from_env(default_grpc_client_channels=7)
    assert config.channels == 13
