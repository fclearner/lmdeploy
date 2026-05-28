import asyncio
import inspect
import os
import stat
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

# Keep production pressure tests from inheriting TurboMind debug logging. This
# must run before TurboMind's C++ extension is imported/initialized. Set
# TM_ALLOW_DEBUG_LOGS=1 when the noisy C++ DEBUG/TRACE logs are intentional.
if os.getenv("TM_ALLOW_DEBUG_LOGS", "0").lower() not in {"1", "true", "yes", "on"}:
    if os.getenv("LMDEPLOY_LOG_LEVEL", "").upper() in {"TRACE", "DEBUG"}:
        os.environ["LMDEPLOY_LOG_LEVEL"] = "WARNING"
    if os.getenv("TM_LOG_LEVEL", "").upper() in {"TRACE", "DEBUG"}:
        os.environ["TM_LOG_LEVEL"] = "WARNING"
    os.environ.setdefault("LMDEPLOY_LOG_LEVEL", "WARNING")
    os.environ.setdefault("TM_LOG_LEVEL", "WARNING")
    os.environ.setdefault("TM_DEBUG_LEVEL", "")

from google.protobuf import struct_pb2

from turbomind_service_core import (
    ServerConfig,
    TurboMindGenerationService,
    build_logits_processor,
    build_tm_model,
    ignored_env_vars,
)
from turbomind_grpc_protocol import GRPC_SERVICE_NAME, dict_to_struct, struct_to_dict


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class GrpcServerConfig(ServerConfig):
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    grpc_unix_socket: str | None = None
    grpc_max_receive_message_length: int = 16 * 1024 * 1024
    grpc_max_send_message_length: int = 16 * 1024 * 1024
    grpc_max_concurrent_rpcs: int | None = None
    grpc_max_concurrent_streams: int = 1024
    grpc_loop_lag_interval_s: float = 0.01
    grpc_loop_lag_window_s: float = 1.0
    grpc_dedicated_model_loop: bool = False
    grpc_serde_thread_workers: int = 0
    startup_warmup_requests: int = 0
    startup_warmup_concurrency: int = 8
    startup_warmup_min_chars: int = 128
    startup_warmup_max_chars: int = 1024
    startup_warmup_infer_types: str = "-1"

    @classmethod
    def from_env(cls) -> "GrpcServerConfig":
        base = ServerConfig.from_env()
        max_concurrent_rpcs = os.getenv("TM_GRPC_MAX_CONCURRENT_RPCS")
        return cls(
            **base.__dict__,
            grpc_host=os.getenv("TM_GRPC_HOST", os.getenv("TM_HOST", "0.0.0.0")),
            grpc_port=int(os.getenv("TM_GRPC_PORT", "50051")),
            grpc_unix_socket=os.getenv("TM_GRPC_UNIX_SOCKET") or None,
            grpc_max_receive_message_length=int(
                os.getenv("TM_GRPC_MAX_RECEIVE_MESSAGE_LENGTH", str(16 * 1024 * 1024))
            ),
            grpc_max_send_message_length=int(
                os.getenv("TM_GRPC_MAX_SEND_MESSAGE_LENGTH", str(16 * 1024 * 1024))
            ),
            grpc_max_concurrent_rpcs=(
                int(max_concurrent_rpcs) if max_concurrent_rpcs else base.max_instances + base.max_queue_size
            ),
            grpc_max_concurrent_streams=int(os.getenv("TM_GRPC_MAX_CONCURRENT_STREAMS", "1024")),
            grpc_loop_lag_interval_s=float(os.getenv("TM_GRPC_LOOP_LAG_INTERVAL_S", "0.01")),
            grpc_loop_lag_window_s=float(os.getenv("TM_GRPC_LOOP_LAG_WINDOW_S", "1.0")),
            grpc_dedicated_model_loop=_env_bool("TM_GRPC_DEDICATED_MODEL_LOOP", False),
            grpc_serde_thread_workers=int(os.getenv("TM_GRPC_SERDE_THREAD_WORKERS", "0")),
            startup_warmup_requests=int(os.getenv("TM_STARTUP_WARMUP_REQUESTS", "0")),
            startup_warmup_concurrency=int(os.getenv("TM_STARTUP_WARMUP_CONCURRENCY", "8")),
            startup_warmup_min_chars=int(os.getenv("TM_STARTUP_WARMUP_MIN_CHARS", "128")),
            startup_warmup_max_chars=int(os.getenv("TM_STARTUP_WARMUP_MAX_CHARS", "1024")),
            startup_warmup_infer_types=os.getenv("TM_STARTUP_WARMUP_INFER_TYPES", "-1"),
        )


def _context_cancelled(context: Any) -> bool:
    cancelled = getattr(context, "cancelled", None)
    if callable(cancelled):
        return bool(cancelled())
    return False


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class DedicatedModelLoopService:
    """Run a TurboMindGenerationService on a private asyncio loop/thread."""

    def __init__(
        self,
        service: Any | None = None,
        *,
        service_factory: Callable[[], Any] | None = None,
        thread_name: str = "tm-model-loop",
    ):
        if service is None and service_factory is None:
            raise ValueError("service or service_factory is required")
        self._service = service
        self._service_factory = service_factory
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name=thread_name, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    @classmethod
    def from_factory(cls, service_factory: Callable[[], Any]) -> "DedicatedModelLoopService":
        return cls(service_factory=service_factory)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    async def _submit(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)

    async def _ensure_service(self) -> Any:
        if self._service is None:
            assert self._service_factory is not None
            self._service = self._service_factory()
        return self._service

    async def start(self) -> None:
        async def do_start() -> None:
            service = await self._ensure_service()
            start = getattr(service, "start", None)
            if start is not None:
                await _maybe_await(start())

        await self._submit(do_start())

    async def close(self) -> None:
        async def do_close() -> None:
            if self._service is not None:
                close = getattr(self._service, "close", None)
                if close is not None:
                    await _maybe_await(close())

        try:
            await self._submit(do_close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        submitted = time.monotonic()

        async def do_generate() -> Any:
            model_loop_started = time.monotonic()
            service = await self._ensure_service()
            result = await service.generate(*args, **kwargs)
            model_loop_time_s = time.monotonic() - model_loop_started
            submit_delay_s = model_loop_started - submitted
            extra = getattr(result, "extra_performance", None)
            if isinstance(extra, dict):
                extra["dedicated_model_submit_delay_s"] = round(submit_delay_s, 6)
                extra["dedicated_model_loop_time_s"] = round(model_loop_time_s, 6)
            return result

        return await self._submit(do_generate())

    async def health_async(self) -> dict[str, Any]:
        async def do_health() -> dict[str, Any]:
            service = await self._ensure_service()
            return await _maybe_await(service.health())

        return await self._submit(do_health())

    async def metrics_async(self) -> dict[str, Any]:
        async def do_metrics() -> dict[str, Any]:
            service = await self._ensure_service()
            return await _maybe_await(service.metrics())

        return await self._submit(do_metrics())

    async def abort(self, request_id: str) -> bool:
        async def do_abort() -> bool:
            service = await self._ensure_service()
            abort = getattr(service, "abort", None)
            if abort is None:
                return False
            return bool(await _maybe_await(abort(request_id)))

        return await self._submit(do_abort())

    async def abort_all(self) -> int:
        async def do_abort_all() -> int:
            service = await self._ensure_service()
            abort_all = getattr(service, "abort_all", None)
            if abort_all is None:
                return 0
            return int(await _maybe_await(abort_all()))

        return await self._submit(do_abort_all())


class EventLoopLagMonitor:
    """Periodically measure recent scheduling lag on the gRPC event loop."""

    def __init__(self, interval_s: float = 0.01, window_s: float = 1.0):
        self._interval_s = max(0.001, float(interval_s))
        self._window_s = max(self._interval_s, float(window_s))
        self._last_lag = 0.0
        self._lifetime_max_lag = 0.0
        self._samples: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._monitor())

    async def _monitor(self) -> None:
        while True:
            started = time.monotonic()
            await asyncio.sleep(self._interval_s)
            now = time.monotonic()
            lag = max(0.0, now - started - self._interval_s)
            with self._lock:
                self._last_lag = lag
                self._lifetime_max_lag = max(self._lifetime_max_lag, lag)
                self._samples.append((now, lag))
                self._prune_locked(now)

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def snapshot(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            window_max = max((lag for _, lag in self._samples), default=self._last_lag)
            return {
                "last": self._last_lag,
                "max": window_max,
                "lifetime_max": self._lifetime_max_lag,
            }

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()


class TurboMindGrpcHandlers:
    """gRPC handlers backed by TurboMindGenerationService."""

    def __init__(
        self,
        service: Any,
        loop_lag_monitor: EventLoopLagMonitor | None = None,
        serde_executor: ThreadPoolExecutor | None = None,
        health_overrides: dict[str, Any] | None = None,
    ):
        self.service = service
        self.loop_lag_monitor = loop_lag_monitor
        self._serde_executor = serde_executor
        self._health_overrides = health_overrides or {}

    async def _struct_to_dict_async(self, request: struct_pb2.Struct) -> dict[str, Any]:
        if self._serde_executor is None:
            return struct_to_dict(request)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._serde_executor, struct_to_dict, request)

    async def _dict_to_struct_async(self, payload: dict[str, Any]) -> struct_pb2.Struct:
        if self._serde_executor is None:
            return dict_to_struct(payload)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._serde_executor, dict_to_struct, payload)

    async def Generate(self, request: struct_pb2.Struct, context: Any) -> struct_pb2.Struct:
        started = time.monotonic()
        handler_entered_wall = time.time()

        parse_started = time.monotonic()
        data = await self._struct_to_dict_async(request)
        request_struct_time_s = time.monotonic() - parse_started
        client_call_started = data.pop("_grpc_client_call_started_wall_time_s", None)

        result = await self.service.generate(
            data.get("prompt", ""),
            request_id=data.get("request_id"),
            max_new_tokens=data.get("max_new_tokens"),
            generation_config=data.get("generation_config"),
            include_text=bool(data.get("include_text", True)),
            include_token_ids=bool(data.get("include_token_ids", False)),
            include_logits=bool(data.get("include_logits", False)),
            logits_format=str(data.get("logits_format", "topk")),
            logits_top_k=int(data.get("logits_top_k", 20)),
            logits_token_ids=data.get("logits_token_ids"),
            infer_type=int(data.get("infer_type", -1)),
            should_abort=lambda: _context_cancelled(context),
        )

        build_started = time.monotonic()
        payload = result.to_dict()
        perf = payload.setdefault("performance", {})
        perf["grpc_request_struct_time_s"] = round(request_struct_time_s, 6)
        perf["grpc_response_build_time_s"] = round(time.monotonic() - build_started, 6)
        perf["grpc_handler_time_s"] = round(time.monotonic() - started, 6)
        if client_call_started is not None:
            perf["grpc_pre_handler_delay_s"] = round(max(0.0, handler_entered_wall - float(client_call_started)), 6)
        if self.loop_lag_monitor is not None:
            lag = self.loop_lag_monitor.snapshot()
            perf["grpc_loop_lag_last_s"] = round(lag["last"], 6)
            perf["grpc_loop_lag_max_s"] = round(lag["max"], 6)
            perf["grpc_loop_lag_lifetime_max_s"] = round(lag["lifetime_max"], 6)

        return await self._dict_to_struct_async(payload)

    async def Health(self, _request: struct_pb2.Struct, _context: Any) -> struct_pb2.Struct:
        health_async = getattr(self.service, "health_async", None)
        if health_async is not None:
            data = await health_async()
        else:
            data = await _maybe_await(self.service.health())
        data.update(self._health_overrides)
        return await self._dict_to_struct_async(data)

    async def Metrics(self, _request: struct_pb2.Struct, _context: Any) -> struct_pb2.Struct:
        metrics_async = getattr(self.service, "metrics_async", None)
        if metrics_async is not None:
            data = await metrics_async()
        else:
            data = await _maybe_await(self.service.metrics())
        return await self._dict_to_struct_async(data)

    async def AbortRequest(self, request: struct_pb2.Struct, _context: Any) -> struct_pb2.Struct:
        data = await self._struct_to_dict_async(request)
        if data.get("abort_all"):
            abort_all = getattr(self.service, "abort_all", None)
            aborted = int(await _maybe_await(abort_all())) if abort_all is not None else 0
            return await self._dict_to_struct_async({"aborted": aborted, "status_code": 200})

        request_id = str(data.get("request_id") or "")
        if not request_id:
            return await self._dict_to_struct_async({
                "request_id": request_id,
                "aborted": False,
                "status_code": 400,
                "error": "request_id is required",
            })
        abort_fn = getattr(self.service, "abort", None)
        aborted = bool(await _maybe_await(abort_fn(request_id))) if abort_fn is not None else False
        return await self._dict_to_struct_async({"request_id": request_id, "aborted": aborted, "status_code": 200})


class _GenericServiceHandler:
    """Fallback generic handler compatible with older grpcio versions."""

    def __init__(self, service_name: str, rpc_method_handlers: dict[str, Any]):
        self._name = service_name
        self._handlers = rpc_method_handlers

    def service_name(self) -> str:
        return self._name

    def service(self, handler_call_details: Any) -> Any:
        method = handler_call_details.method.split("/")[-1]
        return self._handlers.get(method)


def _make_generic_handler(handlers: TurboMindGrpcHandlers, service_name: str) -> _GenericServiceHandler:
    try:
        import grpc
    except ImportError as exc:
        raise RuntimeError("grpcio is required: pip install grpcio protobuf") from exc

    return _GenericServiceHandler(
        service_name,
        {
            "Generate": grpc.unary_unary_rpc_method_handler(
                handlers.Generate,
                request_deserializer=struct_pb2.Struct.FromString,
                response_serializer=struct_pb2.Struct.SerializeToString,
            ),
            "Health": grpc.unary_unary_rpc_method_handler(
                handlers.Health,
                request_deserializer=struct_pb2.Struct.FromString,
                response_serializer=struct_pb2.Struct.SerializeToString,
            ),
            "Metrics": grpc.unary_unary_rpc_method_handler(
                handlers.Metrics,
                request_deserializer=struct_pb2.Struct.FromString,
                response_serializer=struct_pb2.Struct.SerializeToString,
            ),
            "AbortRequest": grpc.unary_unary_rpc_method_handler(
                handlers.AbortRequest,
                request_deserializer=struct_pb2.Struct.FromString,
                response_serializer=struct_pb2.Struct.SerializeToString,
            ),
        },
    )


def _build_service(config: GrpcServerConfig) -> TurboMindGenerationService:
    tm_model = build_tm_model(config)
    logits_processor = build_logits_processor(config)
    return TurboMindGenerationService(
        tm_model,
        config,
        logits_processor=logits_processor,
    )


def _warmup_prompt(target_chars: int, index: int) -> str:
    seeds = [
        "Please decide whether the current utterance has finished; return one token only.",
        "ASR text says hello and then pauses for a short time.",
        "request_id=warmup status=finished queue_ms=0 generation_ms=0",
        '{"callId":"warmup","sessionId":"s","input":{"vadFinal":true}}',
        "mixed English Chinese 123 latency queue token end valid invalid",
    ]
    chunks: list[str] = []
    while len(" ".join(chunks)) < target_chars:
        chunks.append(seeds[(index + len(chunks)) % len(seeds)])
    return " ".join(chunks)[:target_chars]


def _warmup_infer_types(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.replace(",", " ").split():
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values or [-1]


async def _run_startup_warmup(service: Any, config: GrpcServerConfig) -> None:
    total = max(0, int(config.startup_warmup_requests))
    if total <= 0:
        return

    concurrency = max(1, min(total, int(config.startup_warmup_concurrency)))
    min_chars = max(1, int(config.startup_warmup_min_chars))
    max_chars = max(min_chars, int(config.startup_warmup_max_chars))
    infer_types = _warmup_infer_types(config.startup_warmup_infer_types)
    started = time.monotonic()
    semaphore = asyncio.Semaphore(concurrency)
    ok = 0
    errors = 0

    async def one(index: int) -> None:
        nonlocal ok, errors
        span = max_chars - min_chars
        target_chars = min_chars + (index * 9973 % (span + 1) if span else 0)
        infer_type = infer_types[index % len(infer_types)]
        async with semaphore:
            result = await service.generate(
                _warmup_prompt(target_chars, index),
                request_id=f"startup-warmup-{index}",
                max_new_tokens=config.max_new_tokens,
                include_text=False,
                include_token_ids=False,
                include_logits=False,
                infer_type=infer_type,
            )
            if getattr(result, "ok", False):
                ok += 1
            else:
                errors += 1

    print(
        "[warmup] startup "
        f"requests={total} concurrency={concurrency} chars={min_chars}-{max_chars} "
        f"infer_types={infer_types}",
        flush=True,
    )
    await asyncio.gather(*(one(index) for index in range(total)))
    print(
        "[warmup] startup finished "
        f"ok={ok} error={errors} elapsed={time.monotonic() - started:.3f}s",
        flush=True,
    )


async def serve(
    config: GrpcServerConfig | None = None,
    service: Any | None = None,
) -> None:
    try:
        import grpc
        from grpc import aio as grpc_aio
    except ImportError as exc:
        raise RuntimeError("grpcio is required: pip install grpcio protobuf") from exc

    config = config or GrpcServerConfig.from_env()
    ignored = ignored_env_vars()
    if ignored:
        print(f"[wan] Ignored legacy env vars: {ignored}", flush=True)
    print(
        "[effective-config] "
        f"model={config.model_path} "
        f"max_instances={config.max_instances} "
        f"max_batch_size={config.engine_max_batch_size} "
        f"admission_concurrency={config.effective_admission_concurrency} "
        f"cuda_streams={config.cuda_streams} "
        f"session_len={config.session_len} "
        f"cache_max_entry_count={config.cache_max_entry_count} "
        f"enable_prefix_caching={config.enable_prefix_caching} "
        f"dtype={config.dtype} "
        f"max_new_tokens={config.max_new_tokens} "
        f"builtin_logits={config.enable_builtin_logits_processor} "
        f"cpp_logits={config.enable_cpp_logits_processor} "
        f"logits_processor={config.logits_processor} "
        f"grpc_unix_socket={config.grpc_unix_socket} "
        f"grpc_port={config.grpc_port} "
        f"grpc_dedicated_model_loop={config.grpc_dedicated_model_loop} "
        f"startup_warmup_requests={config.startup_warmup_requests}",
        flush=True,
    )
    owns_service = service is None
    if service is None:
        if config.grpc_dedicated_model_loop:
            service = DedicatedModelLoopService.from_factory(lambda: _build_service(config))
        else:
            service = _build_service(config)
    await service.start()
    print("[init] TurboMind service started.", flush=True)
    await _run_startup_warmup(service, config)

    serde_executor = (
        ThreadPoolExecutor(max_workers=config.grpc_serde_thread_workers, thread_name_prefix="grpc-serde")
        if config.grpc_serde_thread_workers > 0
        else None
    )

    loop_lag_monitor = EventLoopLagMonitor(
        interval_s=config.grpc_loop_lag_interval_s,
        window_s=config.grpc_loop_lag_window_s,
    )
    await loop_lag_monitor.start()

    handlers = TurboMindGrpcHandlers(
        service=service,
        loop_lag_monitor=loop_lag_monitor,
        serde_executor=serde_executor,
        health_overrides={
            "grpc_dedicated_model_loop": config.grpc_dedicated_model_loop,
            "grpc_serde_thread_workers": config.grpc_serde_thread_workers,
        },
    )

    options = [
        ("grpc.max_receive_message_length", config.grpc_max_receive_message_length),
        ("grpc.max_send_message_length", config.grpc_max_send_message_length),
        ("grpc.max_concurrent_streams", config.grpc_max_concurrent_streams),
    ]
    server = grpc_aio.server(
        options=options,
        maximum_concurrent_rpcs=config.grpc_max_concurrent_rpcs,
    )
    server.add_generic_rpc_handlers([_make_generic_handler(handlers, GRPC_SERVICE_NAME)])

    if config.grpc_unix_socket:
        try:
            os.unlink(config.grpc_unix_socket)
        except FileNotFoundError:
            pass
        address = f"unix:{config.grpc_unix_socket}"
    else:
        address = f"{config.grpc_host}:{config.grpc_port}"

    server.add_insecure_port(address)
    await server.start()
    print(f"[init] gRPC server listening on {address}", flush=True)

    if config.grpc_unix_socket:
        try:
            os.chmod(config.grpc_unix_socket, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except OSError:
            pass

    try:
        await server.wait_for_termination()
    finally:
        print("[shutdown] Stopping gRPC server...", flush=True)
        await server.stop(grace=5)
        loop_lag_monitor.stop()
        if owns_service and service is not None:
            await service.close()
        elif service is not None:
            close = getattr(service, "close", None)
            if close is not None:
                await _maybe_await(close())
        if serde_executor is not None:
            serde_executor.shutdown(wait=False)
        print("[shutdown] Done.", flush=True)


def main() -> None:
    config = GrpcServerConfig.from_env()
    print(f"[config] {config}", flush=True)
    asyncio.run(serve(config=config))


if __name__ == "__main__":
    main()
