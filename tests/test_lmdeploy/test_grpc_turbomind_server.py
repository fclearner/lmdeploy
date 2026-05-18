import asyncio
import threading
from types import SimpleNamespace

from grpc_turbomind_server import (
    DedicatedModelLoopService,
    GrpcServerConfig,
    TurboMindGrpcHandlers,
    _run_startup_warmup,
    dict_to_struct,
    struct_to_dict,
)
from turbomind_grpc_client import result_from_dict
from turbomind_service_core import ServerConfig, TurboMindGenerationService, build_logits_processor


def run(coro):
    return asyncio.run(coro)


class FakeContext:

    def __init__(self, cancelled=False):
        self._cancelled = cancelled

    def cancelled(self):
        return self._cancelled


class FakeService:

    def __init__(self):
        self.generate_kwargs = None
        self.aborted = []

    async def generate(self, prompt, **kwargs):
        self.generate_kwargs = {"prompt": prompt, **kwargs}
        return SimpleNamespace(
            to_dict=lambda: {
                "request_id": kwargs["request_id"],
                "status": "finished",
                "tokens": 1,
                "text": "A",
                "performance": {
                    "queue_time_s": 0.001,
                    "generation_time_s": 0.002,
                    "total_time_s": 0.003,
                    "logits_enabled": False,
                    "logits_processor_time_s": 0.0,
                },
            }
        )

    def health(self):
        return {"status": "ok", "idle_instances": 1}

    def metrics(self):
        return {"accepted": 1, "completed": 1}

    async def abort(self, request_id):
        self.aborted.append(request_id)
        return request_id == "req-1"

    async def abort_all(self):
        self.aborted.append("*")
        return 3


class LoopBoundFakeService:

    def __init__(self):
        self.thread_names = []
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True
        self.thread_names.append(threading.current_thread().name)

    async def close(self):
        self.closed = True
        self.thread_names.append(threading.current_thread().name)

    async def generate(self, prompt, **kwargs):
        self.thread_names.append(threading.current_thread().name)
        return f"{prompt}:{kwargs['request_id']}"

    def health(self):
        self.thread_names.append(threading.current_thread().name)
        return {"status": "ok"}

    def metrics(self):
        self.thread_names.append(threading.current_thread().name)
        return {"completed": 1}

    async def abort(self, request_id):
        self.thread_names.append(threading.current_thread().name)
        return request_id == "req-1"

    async def abort_all(self):
        self.thread_names.append(threading.current_thread().name)
        return 1


class WarmupFakeService:

    def __init__(self):
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return SimpleNamespace(ok=True)


class FakeTokenizer:

    def encode(self, prompt, add_bos=True):
        return [1, 2, 3]

    def decode(self, token_ids):
        return "A"


class CppProcessorFakeModel:

    def __init__(self, instance):
        self.instance = instance
        self.tokenizer = FakeTokenizer()

    def create_instance(self, cuda_stream_id=0):
        return self.instance


class CppProcessorFakeInstance:

    def __init__(self):
        self.gen_config = None

    def async_stream_infer(self, **kwargs):
        self.gen_config = kwargs["gen_config"]

        async def gen():
            yield SimpleNamespace(token_ids=[123], logits=None, status=SimpleNamespace(name="FINISH"))

        return gen()


def test_struct_roundtrip_preserves_nested_payload():
    payload = {
        "request_id": "req-1",
        "prompt": "hello",
        "generation_config": {"temperature": 0.0, "stop_words": ["\n"]},
        "include_token_ids": True,
    }
    assert struct_to_dict(dict_to_struct(payload)) == payload


def test_generate_handler_maps_struct_to_service_and_back():
    service = FakeService()
    handlers = TurboMindGrpcHandlers(service)

    async def scenario():
        response = await handlers.Generate(
            dict_to_struct({
                "request_id": "req-1",
                "prompt": "hello",
                "max_new_tokens": 1,
                "infer_type": 0,
                "include_text": True,
                "include_token_ids": True,
                "generation_config": {"temperature": 0.0},
            }),
            FakeContext(cancelled=False),
        )
        body = struct_to_dict(response)
        assert body["status"] == "finished"
        assert body["text"] == "A"
        assert service.generate_kwargs["prompt"] == "hello"
        assert service.generate_kwargs["max_new_tokens"] == 1
        assert service.generate_kwargs["infer_type"] == 0
        assert not service.generate_kwargs["should_abort"]()

    run(scenario())


def test_health_metrics_and_abort_handlers():
    service = FakeService()
    handlers = TurboMindGrpcHandlers(service)

    async def scenario():
        assert struct_to_dict(await handlers.Health(dict_to_struct({}), FakeContext()))["status"] == "ok"
        assert struct_to_dict(await handlers.Metrics(dict_to_struct({}), FakeContext()))["completed"] == 1
        ok = struct_to_dict(await handlers.AbortRequest(dict_to_struct({"request_id": "req-1"}), FakeContext()))
        missing = struct_to_dict(await handlers.AbortRequest(dict_to_struct({}), FakeContext()))
        all_aborted = struct_to_dict(await handlers.AbortRequest(dict_to_struct({"abort_all": True}), FakeContext()))
        assert ok == {"request_id": "req-1", "aborted": True, "status_code": 200}
        assert missing["status_code"] == 400
        assert all_aborted["aborted"] == 3

    run(scenario())


def test_dedicated_model_loop_runs_service_off_grpc_loop():
    service = LoopBoundFakeService()
    proxy = DedicatedModelLoopService(service)

    async def scenario():
        await proxy.start()
        assert await proxy.generate("hello", request_id="req-1") == "hello:req-1"
        assert await proxy.health_async() == {"status": "ok"}
        assert await proxy.metrics_async() == {"completed": 1}
        assert await proxy.abort("req-1")
        assert await proxy.abort_all() == 1
        await proxy.close()
        assert service.started
        assert service.closed
        assert service.thread_names
        assert all(name == "tm-model-loop" for name in service.thread_names)

    run(scenario())


def test_startup_warmup_runs_configured_synthetic_requests():
    service = WarmupFakeService()
    config = GrpcServerConfig(
        startup_warmup_requests=5,
        startup_warmup_concurrency=2,
        startup_warmup_min_chars=10,
        startup_warmup_max_chars=20,
        startup_warmup_infer_types="-1,1",
        max_new_tokens=1,
    )

    run(_run_startup_warmup(service, config))

    assert len(service.calls) == 5
    assert {call["infer_type"] for call in service.calls} == {-1, 1}
    assert all(10 <= len(call["prompt"]) <= 20 for call in service.calls)
    assert all(call["include_text"] is False for call in service.calls)
    assert all(call["include_logits"] is False for call in service.calls)


def test_grpc_client_result_from_dict_preserves_status_and_perf():
    result = result_from_dict({
        "request_id": "req-1",
        "status": "finished",
        "status_code": 200,
        "tokens": 1,
        "text": "A",
        "token_ids": [65],
        "performance": {
            "input_tokens": 12,
            "queue_time_s": 0.001,
            "first_token_time_s": 0.002,
            "generation_time_s": 0.003,
            "total_time_s": 0.004,
            "tokens_per_s": 333.0,
            "logits_enabled": True,
            "logits_processor_time_s": 0.0005,
            "grpc_handler_time_s": 0.006,
        },
    })
    assert result.ok
    assert result.status_code == 200
    assert result.request_id == "req-1"
    assert result.token_ids == [65]
    assert result.input_tokens == 12
    assert result.logits_enabled
    assert result.logits_processor_time_s == 0.0005
    assert result.to_dict()["performance"]["grpc_handler_time_s"] == 0.006


def test_cpp_logits_processor_sets_generation_config_without_output_logits():
    instance = CppProcessorFakeInstance()
    config = ServerConfig(
        max_instances=1,
        enable_cpp_logits_processor=True,
        enable_builtin_logits_processor=True,
        valid_id=123,
        invalid_id=456,
        end_id=789,
        certainty_threshold=0.7,
        completion_threshold=0.8,
        invalid_bias=0.05,
    )
    assert build_logits_processor(config) is None
    service = TurboMindGenerationService(CppProcessorFakeModel(instance), config)

    async def scenario():
        await service.start()
        result = await service.generate("hello", infer_type=1, max_new_tokens=1)
        assert result.ok
        assert not result.logits_enabled
        gen_config = instance.gen_config
        assert gen_config.output_logits is None
        assert gen_config.token_decision_infer_type == 1
        assert gen_config.token_decision_valid_id == 123
        assert gen_config.token_decision_invalid_id == 456
        assert gen_config.token_decision_end_id == 789
        assert gen_config.token_decision_certainty_threshold == 0.7
        assert gen_config.token_decision_completion_threshold == 0.8
        assert gen_config.token_decision_invalid_bias == 0.05

    run(scenario())


def test_cpp_logits_processor_infer_type_negative_one_is_noop():
    instance = CppProcessorFakeInstance()
    config = ServerConfig(
        max_instances=1,
        enable_cpp_logits_processor=True,
        enable_builtin_logits_processor=True,
        valid_id=123,
        invalid_id=456,
        end_id=789,
    )
    service = TurboMindGenerationService(CppProcessorFakeModel(instance), config)

    async def scenario():
        await service.start()
        result = await service.generate("hello", infer_type=-1, max_new_tokens=1)
        assert result.ok
        assert not result.logits_enabled
        gen_config = instance.gen_config
        assert gen_config.output_logits is None
        assert gen_config.token_decision_infer_type == -1
        assert gen_config.token_decision_valid_id == -1
        assert gen_config.token_decision_invalid_id == -1
        assert gen_config.token_decision_end_id == -1

    run(scenario())
