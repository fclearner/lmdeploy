import asyncio

from duplex.full_duplex import get_duplex_response, turn_end_logging
from duplex.server import desensitize
from duplex.schemas import EndData, InferData, InputData


def run(coro):
    return asyncio.run(coro)


class FakeDuplexClient:

    def __init__(self, healthy=True):
        self.healthy = healthy
        self.calls = []

    async def health_check(self):
        return self.healthy

    async def infer(self, request_id, text_input, decoding_type=0, timeout=None):
        self.calls.append(
            {
                "request_id": request_id,
                "text_input": text_input,
                "decoding_type": decoding_type,
                "timeout": timeout,
            }
        )
        if decoding_type == 0:
            return "<valid>"
        if decoding_type == 1:
            return "<|im_end|>"
        return ""


def make_data(asr_text="hello", vad_final=True):
    return InferData(
        callId="call-1",
        sessionId="session-1",
        requestId="req-1",
        roundId="round-1",
        input=InputData(
            asrText=asr_text,
            startTime=0.0,
            endTime=1.0,
            vadFinal=vad_final,
            dualVad=True,
        ),
    )


def test_get_duplex_response_calls_validity_then_completion():
    client = FakeDuplexClient()

    async def scenario():
        return await get_duplex_response(client, make_data(), max_context_len=6000, timeout=0.25)

    ret = run(scenario())

    assert [call["decoding_type"] for call in client.calls] == [0, 1]
    assert all(call["request_id"] == "req-1" for call in client.calls)
    assert ret["state"] == 1
    assert ret["finalText"] == ["hello"]
    assert ret["history"]["lastOutput"] == "<valid>"
    assert ret["history"]["isSemComplete"] is True


def test_get_duplex_response_uses_fallback_when_backend_is_unhealthy():
    client = FakeDuplexClient(healthy=False)

    async def scenario():
        return await get_duplex_response(client, make_data(asr_text="h", vad_final=False), 6000, 0.25)

    ret = run(scenario())

    assert client.calls == []
    assert ret["fallbackMsg"] == "LMDeploy gRPC server is not healthy"
    assert ret["history"]["lastOutput"] == "<uncertain>"


def test_turn_end_logging_flushes_buffer_to_final_text():
    client = FakeDuplexClient()

    async def scenario():
        infer_ret = await get_duplex_response(client, make_data(), 6000, 0.25)
        end_data = EndData(
            callId="call-1",
            sessionId="session-1",
            requestId="req-2",
            roundId="round-1",
            input=InputData(vadFinal=True, dualVad=True),
            history=infer_ret["history"],
        )
        return await turn_end_logging(end_data)

    ret = run(scenario())

    assert ret["state"] == 2
    assert ret["finalText"] == ["hello"]
    assert ret["history"]["buffer"] == []


def test_desensitize_masks_digit_runs_and_honors_business_whitelist():
    payload = {
        "callId": "call-123456",
        "requestId": "req-123456",
        "input": {
            "asrText": "phone 13800138000 id 42",
            "startTime": 123.45,
            "score": 12345,
        },
        "history": {
            "context": "acct 6222020202020202",
            "countBatch": [123456],
        },
    }

    masked = desensitize(payload)

    assert masked["callId"] == "call-123456"
    assert masked["requestId"] == "req-123456"
    assert masked["input"]["asrText"] == "phone *********** id 42"
    assert masked["input"]["startTime"] == 123.45
    assert masked["input"]["score"] == "*****"
    assert masked["history"]["context"] == "acct ****************"
    assert masked["history"]["countBatch"] == [123456]
