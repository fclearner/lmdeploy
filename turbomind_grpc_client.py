import asyncio
import os
import time
from typing import Any, Callable

from google.protobuf import struct_pb2

from turbomind_grpc_protocol import GRPC_SERVICE_NAME, dict_to_struct, struct_to_dict
from turbomind_service_core import GenerateResult


def _import_grpc():
    try:
        import grpc
    except ImportError as exc:
        raise RuntimeError("grpcio is required for the Sanic gRPC gateway: pip install grpcio protobuf") from exc
    return grpc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class TurboMindGrpcClient:
    """Async client used by Sanic to call the gRPC model service."""

    def __init__(
        self,
        target: str,
        *,
        timeout_s: float = 120.0,
        max_receive_message_length: int = 16 * 1024 * 1024,
        max_send_message_length: int = 16 * 1024 * 1024,
        channels: int = 4,
        poll_interval_s: float = 0.005,
        max_inflight: int | None = None,
        connect_timeout_s: float = 10.0,
        ready_on_start: bool = True,
    ):
        self.target = target
        self.timeout_s = timeout_s
        self.max_receive_message_length = max_receive_message_length
        self.max_send_message_length = max_send_message_length
        self.channels = max(1, int(channels))
        self.poll_interval_s = max(0.001, float(poll_interval_s))
        self.max_inflight = max_inflight
        if self.max_inflight is not None and self.max_inflight <= 0:
            self.max_inflight = None
        self.connect_timeout_s = float(connect_timeout_s)
        self.ready_on_start = bool(ready_on_start)
        self._grpc = None
        self._channels = []
        self._generate = []
        self._health = []
        self._metrics = []
        self._abort_request = []
        self._next_channel = 0
        self._channel_lock = asyncio.Lock()
        self._inflight_semaphore = asyncio.Semaphore(self.max_inflight) if self.max_inflight else None

    @classmethod
    def from_env(cls) -> "TurboMindGrpcClient":
        unix_socket = os.getenv("TM_GRPC_UNIX_SOCKET")
        target = f"unix:{unix_socket}" if unix_socket else os.getenv("TM_GRPC_TARGET", "127.0.0.1:50051")
        return cls(
            target,
            timeout_s=float(os.getenv("TM_GRPC_TIMEOUT_S", "120")),
            max_receive_message_length=int(os.getenv("TM_GRPC_MAX_RECEIVE_MESSAGE_LENGTH", str(16 * 1024 * 1024))),
            max_send_message_length=int(os.getenv("TM_GRPC_MAX_SEND_MESSAGE_LENGTH", str(16 * 1024 * 1024))),
            channels=int(os.getenv("TM_GRPC_CLIENT_CHANNELS", "4")),
            poll_interval_s=float(os.getenv("TM_GRPC_CLIENT_POLL_INTERVAL_S", "0.005")),
            max_inflight=int(os.getenv("TM_GRPC_CLIENT_MAX_INFLIGHT", "0")) or None,
            connect_timeout_s=float(os.getenv("TM_GRPC_CLIENT_CONNECT_TIMEOUT_S", "10")),
            ready_on_start=_env_bool("TM_GRPC_CLIENT_READY_ON_START", True),
        )

    async def start(self):
        if self._channels:
            return
        self._grpc = _import_grpc()
        options = [
            ("grpc.max_receive_message_length", self.max_receive_message_length),
            ("grpc.max_send_message_length", self.max_send_message_length),
            ("grpc.enable_retries", 0),
            ("grpc.use_local_subchannel_pool", 1),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
        ]
        for _ in range(self.channels):
            channel = self._grpc.aio.insecure_channel(self.target, options=options)
            self._channels.append(channel)
            self._generate.append(
                channel.unary_unary(
                    f"/{GRPC_SERVICE_NAME}/Generate",
                    request_serializer=struct_pb2.Struct.SerializeToString,
                    response_deserializer=struct_pb2.Struct.FromString,
                )
            )
            self._health.append(
                channel.unary_unary(
                    f"/{GRPC_SERVICE_NAME}/Health",
                    request_serializer=struct_pb2.Struct.SerializeToString,
                    response_deserializer=struct_pb2.Struct.FromString,
                )
            )
            self._metrics.append(
                channel.unary_unary(
                    f"/{GRPC_SERVICE_NAME}/Metrics",
                    request_serializer=struct_pb2.Struct.SerializeToString,
                    response_deserializer=struct_pb2.Struct.FromString,
                )
            )
            self._abort_request.append(
                channel.unary_unary(
                    f"/{GRPC_SERVICE_NAME}/AbortRequest",
                    request_serializer=struct_pb2.Struct.SerializeToString,
                    response_deserializer=struct_pb2.Struct.FromString,
                )
            )
        if self.ready_on_start:
            await asyncio.gather(
                *(
                    asyncio.wait_for(channel.channel_ready(), timeout=self.connect_timeout_s)
                    for channel in self._channels
                )
            )

    async def close(self):
        if self._channels:
            await asyncio.gather(*(channel.close() for channel in self._channels))
            self._channels = []
            self._generate = []
            self._health = []
            self._metrics = []
            self._abort_request = []

    async def _pick_generate(self):
        async with self._channel_lock:
            index = self._next_channel
            self._next_channel = (self._next_channel + 1) % len(self._generate)
        return self._generate[index]

    async def _acquire_inflight_slot(self, should_abort: Callable[[], bool] | None) -> bool:
        if self._inflight_semaphore is None:
            return False
        while True:
            if should_abort is not None and should_abort():
                raise asyncio.CancelledError
            try:
                await asyncio.wait_for(self._inflight_semaphore.acquire(), timeout=self.poll_interval_s)
                return True
            except asyncio.TimeoutError:
                continue

    async def generate(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        max_new_tokens: int | None = None,
        generation_config: dict[str, Any] | None = None,
        include_text: bool = True,
        include_token_ids: bool = False,
        include_logits: bool = False,
        logits_format: str = "topk",
        logits_top_k: int = 20,
        logits_token_ids: list[int] | None = None,
        infer_type: int = -1,
        should_abort: Callable[[], bool] | None = None,
    ) -> GenerateResult:
        await self.start()
        payload = {
            "prompt": prompt,
            "request_id": request_id,
            "max_new_tokens": max_new_tokens,
            "generation_config": generation_config,
            "include_text": include_text,
            "include_token_ids": include_token_ids,
            "include_logits": include_logits,
            "logits_format": logits_format,
            "logits_top_k": logits_top_k,
            "logits_token_ids": logits_token_ids,
            "infer_type": infer_type,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        queue_started = asyncio.get_running_loop().time()
        acquired_slot = False
        call_started = None
        result = None
        try:
            acquired_slot = await self._acquire_inflight_slot(should_abort)
            grpc_client_queue_time_s = asyncio.get_running_loop().time() - queue_started
        except asyncio.CancelledError:
            return _error_result(request_id or "", "cancelled", 499, "request was cancelled")

        generate = await self._pick_generate()
        request_struct_started = asyncio.get_running_loop().time()
        request_msg = dict_to_struct(payload)
        grpc_client_request_struct_time_s = asyncio.get_running_loop().time() - request_struct_started
        call_started = asyncio.get_running_loop().time()
        payload_start_wall_time_s = time.time()
        request_msg.fields["_grpc_client_call_started_wall_time_s"].number_value = payload_start_wall_time_s
        call = generate(request_msg, timeout=self.timeout_s)
        call_task = None
        abort_watch_task = None
        try:
            if should_abort is None:
                response = await call
                result = result_from_dict(struct_to_dict(response))
                return _with_grpc_client_perf(
                    result,
                    grpc_client_queue_time_s,
                    call_started,
                    grpc_client_request_struct_time_s,
                )

            async def abort_watch() -> None:
                while True:
                    await asyncio.sleep(self.poll_interval_s)
                    if call_task is not None and call_task.done():
                        return
                    if should_abort():
                        call.cancel()
                        return

            call_task = asyncio.ensure_future(call)
            abort_watch_task = asyncio.create_task(abort_watch())
            response = await call_task
            result = result_from_dict(struct_to_dict(response))
            return _with_grpc_client_perf(
                result,
                grpc_client_queue_time_s,
                call_started,
                grpc_client_request_struct_time_s,
            )
        except asyncio.CancelledError:
            call.cancel()
            raise
        except Exception as exc:
            result = _error_result(request_id or "", "grpc_error", 502, str(exc))
            return _with_grpc_client_perf(
                result,
                grpc_client_queue_time_s,
                call_started,
                grpc_client_request_struct_time_s,
            )
        finally:
            if abort_watch_task is not None:
                abort_watch_task.cancel()
            if acquired_slot and self._inflight_semaphore is not None:
                self._inflight_semaphore.release()

    def health(self) -> dict[str, Any]:
        return {"status": "unknown", "target": self.target}

    async def health_async(self) -> dict[str, Any]:
        await self.start()
        response = await self._health[0](dict_to_struct({}), timeout=5)
        data = struct_to_dict(response)
        data["target"] = self.target
        data["grpc_client_channels"] = self.channels
        data["grpc_client_max_inflight"] = self.max_inflight
        data["grpc_client_ready_on_start"] = self.ready_on_start
        data["grpc_client_connect_timeout_s"] = self.connect_timeout_s
        data["grpc_client_available_inflight"] = (
            self._inflight_semaphore._value if self._inflight_semaphore is not None else None
        )
        return data

    def metrics(self) -> dict[str, Any]:
        return {"target": self.target}

    async def metrics_async(self) -> dict[str, Any]:
        await self.start()
        response = await self._metrics[0](dict_to_struct({}), timeout=5)
        data = struct_to_dict(response)
        data["target"] = self.target
        data["grpc_client_channels"] = self.channels
        data["grpc_client_max_inflight"] = self.max_inflight
        data["grpc_client_ready_on_start"] = self.ready_on_start
        data["grpc_client_connect_timeout_s"] = self.connect_timeout_s
        data["grpc_client_available_inflight"] = (
            self._inflight_semaphore._value if self._inflight_semaphore is not None else None
        )
        return data

    async def abort(self, request_id: str) -> bool:
        await self.start()
        response = await self._abort_request[0](dict_to_struct({"request_id": request_id}), timeout=5)
        return bool(struct_to_dict(response).get("aborted"))

    async def abort_all(self) -> int:
        await self.start()
        response = await self._abort_request[0](dict_to_struct({"abort_all": True}), timeout=5)
        return int(struct_to_dict(response).get("aborted", 0))


def result_from_dict(data: dict[str, Any]) -> GenerateResult:
    perf = data.get("performance") if isinstance(data.get("performance"), dict) else {}
    known_perf_keys = {
        "input_tokens",
        "output_tokens",
        "queue_time_s",
        "first_token_time_s",
        "generation_time_s",
        "total_time_s",
        "tokens_per_s",
        "logits_enabled",
        "logits_processor_time_s",
    }
    extra_performance = {key: value for key, value in perf.items() if key not in known_perf_keys}
    return GenerateResult(
        ok=data.get("status") == "finished",
        request_id=str(data.get("request_id", "")),
        status=str(data.get("status", "unknown")),
        status_code=int(data.get("status_code", 200 if data.get("status") == "finished" else 500)),
        tokens=int(data.get("tokens", 0)),
        text=str(data.get("text", "")),
        token_ids=[int(token_id) for token_id in data.get("token_ids", [])],
        queue_time_s=float(perf.get("queue_time_s", 0.0)),
        generation_time_s=float(perf.get("generation_time_s", 0.0)),
        total_time_s=float(perf.get("total_time_s", 0.0)),
        input_tokens=int(perf.get("input_tokens", 0)),
        first_token_time_s=perf.get("first_token_time_s"),
        tokens_per_s=float(perf.get("tokens_per_s", 0.0)),
        logits_enabled=bool(perf.get("logits_enabled", False)),
        logits_processor_time_s=float(perf.get("logits_processor_time_s", 0.0)),
        extra_performance=extra_performance,
        logits=data.get("logits"),
        error=data.get("error"),
    )


def _with_grpc_client_perf(
    result: GenerateResult,
    queue_time_s: float,
    call_started: float | None,
    request_struct_time_s: float | None = None,
) -> GenerateResult:
    result.extra_performance["grpc_client_queue_time_s"] = round(queue_time_s, 6)
    if request_struct_time_s is not None:
        result.extra_performance["grpc_client_request_struct_time_s"] = round(request_struct_time_s, 6)
    if call_started is not None:
        result.extra_performance["grpc_client_call_time_s"] = round(
            asyncio.get_running_loop().time() - call_started,
            6,
        )
    return result


def _error_result(request_id: str, status: str, status_code: int, error: str) -> GenerateResult:
    return GenerateResult(
        ok=False,
        request_id=request_id,
        status=status,
        status_code=status_code,
        error=error,
    )
