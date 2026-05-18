import asyncio
import importlib
import inspect
import math
import os
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-0.5B"
IGNORED_ENV_VARS = (
    "TM_INFER_INIT_THREAD_WORKERS",
    "TM_GENERATION_THREAD_WORKERS",
)


@dataclass
class ServerConfig:
    model_path: str = DEFAULT_MODEL_PATH
    host: str = "0.0.0.0"
    port: int = 8000
    max_instances: int = 8
    max_batch_size: int | None = None
    admission_concurrency: int | None = None
    cuda_streams: int = 8
    max_queue_size: int = 128
    queue_timeout_s: float = 5.0
    generation_timeout_s: float = 60.0
    session_len: int = 4096
    cache_max_entry_count: float = 0.85
    enable_prefix_caching: bool = False
    dtype: str = "auto"
    max_new_tokens: int = 512
    acquire_poll_interval_s: float = 0.05
    keep_alive: bool = False
    stream_output: bool = False
    logits_processor: str | None = None
    enable_builtin_logits_processor: bool = False
    enable_cpp_logits_processor: bool = False
    valid_id: int | None = None
    invalid_id: int | None = None
    end_id: int | None = None
    certainty_threshold: float = 0.0
    completion_threshold: float = 0.0
    invalid_bias: float = 0.0
    hold_instance_for_logits_processor: bool = False

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            model_path=os.getenv("TM_MODEL_PATH", DEFAULT_MODEL_PATH),
            host=os.getenv("TM_HOST", "0.0.0.0"),
            port=int(os.getenv("TM_PORT", "8000")),
            max_instances=int(os.getenv("TM_MAX_INSTANCES", "8")),
            max_batch_size=_env_optional_int("TM_MAX_BATCH_SIZE"),
            admission_concurrency=_env_optional_int("TM_ADMISSION_CONCURRENCY"),
            cuda_streams=int(os.getenv("TM_CUDA_STREAMS", "8")),
            max_queue_size=int(os.getenv("TM_MAX_QUEUE_SIZE", "128")),
            queue_timeout_s=float(os.getenv("TM_QUEUE_TIMEOUT_S", "5")),
            generation_timeout_s=float(os.getenv("TM_GENERATION_TIMEOUT_S", "60")),
            session_len=int(os.getenv("TM_SESSION_LEN", "4096")),
            cache_max_entry_count=float(os.getenv("TM_CACHE_MAX_ENTRY_COUNT", "0.85")),
            enable_prefix_caching=_env_bool("TM_ENABLE_PREFIX_CACHING", True),
            dtype=os.getenv("TM_DTYPE", "auto"),
            max_new_tokens=int(os.getenv("TM_MAX_NEW_TOKENS", "512")),
            acquire_poll_interval_s=float(os.getenv("TM_ACQUIRE_POLL_INTERVAL_S", "0.05")),
            keep_alive=_env_bool("TM_KEEP_ALIVE", False),
            stream_output=_env_bool("TM_STREAM_OUTPUT", False),
            logits_processor=os.getenv("TM_LOGITS_PROCESSOR"),
            enable_builtin_logits_processor=_env_bool("TM_ENABLE_BUILTIN_LOGITS_PROCESSOR", False),
            enable_cpp_logits_processor=_env_bool("TM_ENABLE_CPP_LOGITS_PROCESSOR", False),
            valid_id=_env_optional_int("TM_VALID_ID"),
            invalid_id=_env_optional_int("TM_INVALID_ID"),
            end_id=_env_optional_int("TM_END_ID"),
            certainty_threshold=float(os.getenv("TM_CERTAINTY_THRESHOLD", "0")),
            completion_threshold=float(os.getenv("TM_COMPLETION_THRESHOLD", "0")),
            invalid_bias=float(os.getenv("TM_INVALID_BIAS", "0")),
            hold_instance_for_logits_processor=_env_bool("TM_HOLD_INSTANCE_FOR_LOGITS_PROCESSOR", False),
        )

    @property
    def engine_max_batch_size(self) -> int:
        return int(self.max_batch_size or self.max_instances)

    @property
    def effective_admission_concurrency(self) -> int:
        if self.admission_concurrency is not None:
            return max(1, min(self.max_instances, int(self.admission_concurrency)))
        return max(1, min(self.max_instances, self.engine_max_batch_size))


@dataclass
class ServiceStats:
    accepted: int = 0
    completed: int = 0
    queue_full: int = 0
    queue_timeout: int = 0
    generation_timeout: int = 0
    cancelled: int = 0
    engine_errors: int = 0
    validation_errors: int = 0
    total_queue_time_s: float = 0.0
    total_generation_time_s: float = 0.0
    total_first_token_time_s: float = 0.0
    total_generated_tokens: int = 0


@dataclass
class RequestState:
    request_id: str
    session_id: int
    created_at: float
    task: asyncio.Task | None = None
    instance: Any = None
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    status: str = "queued"


@dataclass
class GenerateResult:
    ok: bool
    request_id: str
    status: str
    status_code: int
    tokens: int = 0
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    queue_time_s: float = 0.0
    generation_time_s: float = 0.0
    total_time_s: float = 0.0
    input_tokens: int = 0
    first_token_time_s: float | None = None
    tokens_per_s: float = 0.0
    logits_enabled: bool = False
    logits_processor_time_s: float = 0.0
    extra_performance: dict[str, Any] = field(default_factory=dict)
    logits: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        perf = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.tokens,
            "queue_time_s": round(self.queue_time_s, 6),
            "first_token_time_s": None
            if self.first_token_time_s is None
            else round(self.first_token_time_s, 6),
            "generation_time_s": round(self.generation_time_s, 6),
            "total_time_s": round(self.total_time_s, 6),
            "tokens_per_s": round(self.tokens_per_s, 6),
            "logits_enabled": self.logits_enabled,
            "logits_processor_time_s": round(self.logits_processor_time_s, 6),
        }
        for key, value in self.extra_performance.items():
            perf[key] = value
        data = {
            "request_id": self.request_id,
            "status": self.status,
            "status_code": self.status_code,
            "tokens": self.tokens,
            "performance": perf,
        }
        if self.text:
            data["text"] = self.text
        if self.token_ids:
            data["token_ids"] = self.token_ids
        if self.logits is not None:
            data["logits"] = self.logits
        if self.error:
            data["error"] = self.error
        return data


@dataclass
class SimpleGenerationConfig:
    n: int = 1
    max_new_tokens: int = 512
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int = 50
    min_p: float = 0.0
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    random_seed: int | None = None
    stop_words: list[str] | None = None
    bad_words: list[str] | None = None
    stop_token_ids: list[int] | None = None
    bad_token_ids: list[int] | None = None
    min_new_tokens: int | None = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    logprobs: int | None = None
    response_format: dict | None = None
    logits_processors: list[Any] | None = None
    output_logits: str | None = None
    output_last_hidden_state: str | None = None
    token_decision_infer_type: int = -1
    token_decision_valid_id: int = -1
    token_decision_invalid_id: int = -1
    token_decision_end_id: int = -1
    token_decision_certainty_threshold: float = 0.0
    token_decision_completion_threshold: float = 0.0
    token_decision_invalid_bias: float = 0.0
    include_stop_str_in_output: bool = False


@dataclass
class GenerationPerf:
    input_tokens: int = 0
    first_token_time_s: float | None = None
    generation_time_s: float = 0.0
    logits_enabled: bool = False
    logits_processor_time_s: float = 0.0
    tokenize_time_s: float = 0.0
    prepare_time_s: float = 0.0
    infer_init_time_s: float = 0.0
    engine_queue_time_s: float | None = None
    engine_first_token_time_s: float | None = None
    input_ids: list[int] = field(default_factory=list)
    generation_config: Any = None

    def tokens_per_s(self, output_tokens: int) -> float:
        if output_tokens <= 0 or self.generation_time_s <= 0:
            return 0.0
        return output_tokens / self.generation_time_s


class TurboMindGenerationService:
    """Small production wrapper around TurboMind instances.

    TurboMind already has its own scheduler, but this wrapper adds the HTTP-facing
    controls a long-running service needs: bounded admission, queue timeout,
    per-request timeout, explicit abort, disconnect cleanup and observable stats.
    """

    def __init__(
        self,
        tm_model: Any,
        config: ServerConfig,
        *,
        instance_factory: Callable[[int], Any] | None = None,
        generation_config_factory: Callable[..., Any] | None = None,
        logits_processor: Callable[..., Any] | None = None,
    ):
        self.tm_model = tm_model
        self.config = config
        self.instance_factory = instance_factory or self._create_instance
        self.generation_config_factory = generation_config_factory or _build_generation_config
        self.logits_processor = logits_processor
        self.admission_concurrency = config.effective_admission_concurrency
        self.pool: asyncio.Queue[Any] = asyncio.Queue(maxsize=self.admission_concurrency)
        self.stats = ServiceStats()
        self._active: dict[str, RequestState] = {}
        self._active_lock = asyncio.Lock()
        self._waiting = 0
        self._started = False

    async def start(self):
        if self._started:
            return
        for index in range(self.admission_concurrency):
            stream_id = index % max(1, self.config.cuda_streams)
            self.pool.put_nowait(self.instance_factory(stream_id))
        self._started = True

    async def close(self):
        for request_id in list(self._active):
            await self.abort(request_id)
        close = getattr(self.tm_model, "close", None)
        if close:
            close()

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
        now = time.monotonic()
        request_id = request_id or uuid.uuid4().hex
        state = RequestState(request_id=request_id, session_id=_new_session_id(), created_at=now)
        state.task = asyncio.current_task()

        if not isinstance(prompt, str) or not prompt:
            self.stats.validation_errors += 1
            return _error_result(request_id, "bad_request", 400, "prompt must be a non-empty string")

        should_abort = should_abort or (lambda: False)
        async with self._active_lock:
            if request_id in self._active:
                self.stats.validation_errors += 1
                return _error_result(request_id, "duplicate_request_id", 409, "request_id is already active")
            if self.pool.empty() and self._waiting >= self.config.max_queue_size:
                self.stats.queue_full += 1
                return _error_result(request_id, "queue_full", 429, "server queue is full")
            self._active[request_id] = state
            self.stats.accepted += 1

        instance = None
        queue_started = time.monotonic()
        try:
            instance = await self._acquire_instance(state, should_abort)
            queue_time_s = time.monotonic() - queue_started
            self.stats.total_queue_time_s += queue_time_s

            if state.abort_event.is_set() or should_abort():
                self.stats.cancelled += 1
                return _error_result(request_id, "cancelled", 499, "request was cancelled", queue_time_s=queue_time_s)

            state.instance = instance
            state.status = "running"
            if self.config.enable_cpp_logits_processor and infer_type >= 0:
                generation_config = dict(generation_config or {})
                generation_config.setdefault("token_decision_infer_type", int(infer_type))
                generation_config.setdefault(
                    "token_decision_valid_id",
                    int(self.config.valid_id if self.config.valid_id is not None else -1),
                )
                generation_config.setdefault(
                    "token_decision_invalid_id",
                    int(self.config.invalid_id if self.config.invalid_id is not None else -1),
                )
                generation_config.setdefault(
                    "token_decision_end_id",
                    int(self.config.end_id if self.config.end_id is not None else -1),
                )
                generation_config.setdefault(
                    "token_decision_certainty_threshold",
                    float(self.config.certainty_threshold),
                )
                generation_config.setdefault(
                    "token_decision_completion_threshold",
                    float(self.config.completion_threshold),
                )
                generation_config.setdefault("token_decision_invalid_bias", float(self.config.invalid_bias))

            need_logits = (
                include_logits
                or (not self.config.enable_cpp_logits_processor and self.logits_processor is not None and infer_type >= 0)
            )
            if need_logits:
                generation_config = dict(generation_config or {})
                generation_config.setdefault("output_logits", "generation")

            generation_kwargs = dict(
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
                collect_logits=need_logits,
                return_logits=include_logits,
                logits_format=logits_format,
                logits_top_k=logits_top_k,
                logits_token_ids=logits_token_ids,
                infer_type=infer_type,
                should_abort=should_abort,
            )
            token_ids, perf, logits = await self._run_generation(instance, state, prompt, **generation_kwargs)
            if not self.config.hold_instance_for_logits_processor:
                state.instance = None
                self.pool.put_nowait(instance)
                instance = None
            if self.logits_processor is not None and logits is not None:
                processor_started = time.monotonic()
                token_ids = await _apply_logits_processor(
                    self.logits_processor,
                    request_id=state.request_id,
                    prompt=prompt,
                    input_ids=perf.input_ids,
                    token_ids=token_ids,
                    logits=logits,
                    tokenizer=self.tm_model.tokenizer,
                    generation_config=perf.generation_config,
                    infer_type=infer_type,
                )
                perf.logits_processor_time_s = time.monotonic() - processor_started
            logits_payload = _serialize_logits(
                logits,
                fmt=logits_format,
                top_k=logits_top_k,
                token_ids=logits_token_ids,
            ) if include_logits else None
            if instance is not None:
                state.instance = None
                self.pool.put_nowait(instance)
                instance = None
            self.stats.completed += 1
            self.stats.total_generation_time_s += perf.generation_time_s
            self.stats.total_generated_tokens += len(token_ids)
            if perf.first_token_time_s is not None:
                self.stats.total_first_token_time_s += perf.first_token_time_s
            text = self._decode(token_ids) if include_text else ""
            return GenerateResult(
                ok=True,
                request_id=request_id,
                status="finished",
                status_code=200,
                tokens=len(token_ids),
                text=text,
                token_ids=token_ids if include_token_ids else [],
                queue_time_s=queue_time_s,
                generation_time_s=perf.generation_time_s,
                total_time_s=time.monotonic() - now,
                input_tokens=perf.input_tokens,
                first_token_time_s=perf.first_token_time_s,
                tokens_per_s=perf.tokens_per_s(len(token_ids)),
                logits_enabled=perf.logits_enabled,
                logits_processor_time_s=perf.logits_processor_time_s,
                extra_performance={
                    "tokenize_time_s": round(perf.tokenize_time_s, 6),
                    "prepare_time_s": round(perf.prepare_time_s, 6),
                    "infer_init_time_s": round(perf.infer_init_time_s, 6),
                    "engine_queue_time_s": None
                    if perf.engine_queue_time_s is None
                    else round(perf.engine_queue_time_s, 6),
                    "engine_first_token_time_s": None
                    if perf.engine_first_token_time_s is None
                    else round(perf.engine_first_token_time_s, 6),
                },
                logits=logits_payload,
            )
        except asyncio.TimeoutError:
            self.stats.queue_timeout += int(instance is None)
            self.stats.generation_timeout += int(instance is not None)
            if instance is not None:
                await self._cancel_instance(instance, state.session_id)
            status = "queue_timeout" if instance is None else "generation_timeout"
            return _error_result(
                request_id,
                status,
                503 if instance is None else 504,
                f"{status} exceeded",
                queue_time_s=time.monotonic() - queue_started if instance is None else 0.0,
                total_time_s=time.monotonic() - now,
            )
        except asyncio.CancelledError:
            self.stats.cancelled += 1
            if instance is not None:
                await self._cancel_instance(instance, state.session_id)
            return _error_result(request_id, "cancelled", 499, "request was cancelled")
        except Exception as exc:
            self.stats.engine_errors += 1
            if instance is not None:
                await self._cancel_instance(instance, state.session_id)
            return _error_result(request_id, "engine_error", 500, str(exc))
        finally:
            if instance is not None:
                state.instance = None
                with suppress(asyncio.QueueFull):
                    self.pool.put_nowait(instance)
            async with self._active_lock:
                self._active.pop(request_id, None)

    async def abort(self, request_id: str) -> bool:
        async with self._active_lock:
            state = self._active.get(request_id)
        if state is None:
            return False
        state.abort_event.set()
        if state.instance is not None:
            await self._cancel_instance(state.instance, state.session_id)
        if state.task is not None:
            state.task.cancel()
        return True

    async def abort_all(self) -> int:
        request_ids = list(self._active)
        results = await asyncio.gather(*(self.abort(request_id) for request_id in request_ids), return_exceptions=True)
        return sum(result is True for result in results)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self._started else "starting",
            "model": self.config.model_path,
            "idle_instances": self.pool.qsize(),
            "max_instances": self.config.max_instances,
            "max_batch_size": self.config.engine_max_batch_size,
            "admission_concurrency": self.admission_concurrency,
            "active_requests": len(self._active),
            "waiting_requests": self._waiting,
            "session_len": self.config.session_len,
            "cache_max_entry_count": self.config.cache_max_entry_count,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "dtype": self.config.dtype,
            "hold_instance_for_logits_processor": self.config.hold_instance_for_logits_processor,
            "enable_cpp_logits_processor": self.config.enable_cpp_logits_processor,
            "ignored_env_vars": ignored_env_vars(),
        }

    def metrics(self) -> dict[str, Any]:
        accepted = max(1, self.stats.accepted)
        completed = max(1, self.stats.completed)
        avg_generation_time_s = self.stats.total_generation_time_s / completed
        total_time_for_tokens = max(self.stats.total_generation_time_s, 1e-9)
        return {
            **self.stats.__dict__,
            "active_requests": len(self._active),
            "idle_instances": self.pool.qsize(),
            "admission_concurrency": self.admission_concurrency,
            "avg_queue_time_s": self.stats.total_queue_time_s / accepted,
            "avg_generation_time_s": avg_generation_time_s,
            "avg_first_token_time_s": self.stats.total_first_token_time_s / completed,
            "avg_tokens_per_s": self.stats.total_generated_tokens / total_time_for_tokens,
        }

    def _create_instance(self, cuda_stream_id: int):
        return self.tm_model.create_instance(cuda_stream_id=cuda_stream_id)

    async def _acquire_instance(self, state: RequestState, should_abort: Callable[[], bool]):
        deadline = time.monotonic() + self.config.queue_timeout_s
        self._waiting += 1
        try:
            while True:
                if state.abort_event.is_set() or should_abort():
                    raise asyncio.CancelledError
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                try:
                    return await asyncio.wait_for(
                        self.pool.get(),
                        timeout=min(remaining, self.config.acquire_poll_interval_s),
                    )
                except asyncio.TimeoutError:
                    continue
        finally:
            self._waiting -= 1

    async def _run_generation(
        self,
        instance: Any,
        state: RequestState,
        prompt: str,
        *,
        max_new_tokens: int | None,
        generation_config: dict[str, Any] | None,
        collect_logits: bool,
        return_logits: bool,
        logits_format: str,
        logits_top_k: int,
        logits_token_ids: list[int] | None,
        infer_type: int,
        should_abort: Callable[[], bool],
    ) -> tuple[list[int], GenerationPerf, Any]:
        prepare_started = time.monotonic()
        tokenize_started = prepare_started
        input_ids = self.tm_model.tokenizer.encode(prompt, add_bos=True)
        tokenize_time_s = time.monotonic() - tokenize_started
        perf = GenerationPerf(input_tokens=len(input_ids))
        perf.tokenize_time_s = tokenize_time_s
        perf.input_ids = input_ids
        gen_config = self.generation_config_factory(
            generation_config or {},
            max_new_tokens=max_new_tokens,
            default_max_new_tokens=self.config.max_new_tokens,
        )
        perf.generation_config = gen_config
        infer_kwargs = dict(
            session_id=state.session_id,
            input_ids=input_ids,
            gen_config=gen_config,
            sequence_start=True,
            sequence_end=True,
            stream_output=self.config.stream_output,
        )
        infer_init_started = time.monotonic()
        generator = instance.async_stream_infer(**infer_kwargs)
        perf.infer_init_time_s = time.monotonic() - infer_init_started
        perf.prepare_time_s = time.monotonic() - prepare_started
        token_ids: list[int] = []
        logits = None
        deadline = time.monotonic() + self.config.generation_timeout_s
        started = time.monotonic()
        try:
            while True:
                if state.abort_event.is_set() or should_abort():
                    raise asyncio.CancelledError
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                next_output = asyncio.create_task(generator.__anext__())
                done, _ = await asyncio.wait({next_output}, timeout=remaining)
                if not done:
                    next_output.cancel()
                    with suppress(Exception, asyncio.CancelledError):
                        await asyncio.wait_for(next_output, timeout=1.0)
                    raise asyncio.TimeoutError
                try:
                    output = next_output.result()
                except StopAsyncIteration:
                    break
                if output.token_ids and perf.first_token_time_s is None:
                    perf.first_token_time_s = time.monotonic() - started
                    _fill_engine_metrics(perf, output)
                token_ids.extend(output.token_ids)
                if collect_logits and getattr(output, "logits", None) is not None:
                    logits = output.logits
                status_name = _status_name(output.status)
                if status_name == "FINISH":
                    break
                if status_name == "CANCEL":
                    raise asyncio.CancelledError
                if status_name != "SUCCESS":
                    raise RuntimeError(f"engine error: {status_name}")
        finally:
            await generator.aclose()
        perf.generation_time_s = time.monotonic() - started
        perf.logits_enabled = collect_logits
        return token_ids, perf, logits

    async def _cancel_instance(self, instance: Any, session_id: int):
        cancel = getattr(instance, "async_cancel", None)
        if cancel is not None:
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(cancel(session_id))
            return
        model_inst = getattr(instance, "model_inst", None)
        cancel = getattr(model_inst, "cancel", None)
        if cancel is not None:
            with suppress(Exception):
                cancel()

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        decode = getattr(self.tm_model.tokenizer, "decode", None)
        if decode is None:
            return ""
        with suppress(Exception):
            return decode(token_ids)
        return ""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value)


def ignored_env_vars() -> dict[str, str]:
    return {name: os.getenv(name, "") for name in IGNORED_ENV_VARS if name in os.environ}


def _new_session_id() -> int:
    return uuid.uuid4().int % (2**31 - 1)


def _build_generation_config(
    payload: dict[str, Any],
    *,
    max_new_tokens: int | None,
    default_max_new_tokens: int,
) -> Any:
    allowed = {field_name for field_name in SimpleGenerationConfig.__dataclass_fields__}
    data = {key: value for key, value in payload.items() if key in allowed}
    data["max_new_tokens"] = int(max_new_tokens or data.get("max_new_tokens") or default_max_new_tokens)
    return SimpleGenerationConfig(**data)


def load_logits_processor(path: str | None) -> Callable[..., Any] | None:
    """Load a server-side logits processor from ``module:function``.

    The callable is invoked after TurboMind returns generation logits and before
    the final token ids are decoded for the HTTP response. It may return None to
    keep the original output, a list[int] to replace token_ids, or a dict with a
    ``token_ids`` field.
    """
    if not path:
        return None
    module_name, sep, func_name = path.partition(":")
    if not sep or not module_name or not func_name:
        raise ValueError("TM_LOGITS_PROCESSOR must use module:function format")
    module = importlib.import_module(module_name)
    processor = getattr(module, func_name)
    if not callable(processor):
        raise TypeError(f"{path} is not callable")
    return processor


def build_logits_processor(config: ServerConfig) -> Callable[..., Any] | None:
    if config.enable_cpp_logits_processor:
        return None
    if config.logits_processor:
        if config.logits_processor == "builtin_token_decision":
            return BuiltinTokenDecisionProcessor(config)
        return load_logits_processor(config.logits_processor)
    if config.enable_builtin_logits_processor:
        return BuiltinTokenDecisionProcessor(config)
    return None


class BuiltinTokenDecisionProcessor:
    """TRT-LLM-style server-side token decision from generation logits."""

    prefer_inline = True

    def __init__(self, config: ServerConfig):
        missing = []
        for name in ("valid_id", "invalid_id", "end_id"):
            if getattr(config, name) is None:
                missing.append(name)
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"builtin token decision processor requires: {names}")
        self.valid_id = int(config.valid_id)
        self.invalid_id = int(config.invalid_id)
        self.end_id = int(config.end_id)
        self.certainty_threshold = float(config.certainty_threshold)
        self.completion_threshold = float(config.completion_threshold)
        self.invalid_bias = float(config.invalid_bias)

    def __call__(self, *, token_ids: list[int], logits: Any, infer_type: int = -1, **_kwargs) -> list[int] | None:
        if infer_type < 0:
            return None

        if infer_type == 0:
            probs = _logit_probs(logits, [self.valid_id, self.invalid_id])
            valid_prob = probs[self.valid_id]
            invalid_prob = probs[self.invalid_id]
            if max(valid_prob, invalid_prob) <= self.certainty_threshold:
                return None
            ret_id = self.valid_id if valid_prob > invalid_prob + self.invalid_bias else self.invalid_id
            return [ret_id]

        end_prob = _logit_probs(logits, [self.end_id])[self.end_id]
        if end_prob > self.completion_threshold:
            return [self.end_id]
        return None


async def _apply_logits_processor(
    processor: Callable[..., Any],
    *,
    request_id: str,
    prompt: str,
    input_ids: list[int],
    token_ids: list[int],
    logits: Any,
    tokenizer: Any,
    generation_config: Any,
    infer_type: int = -1,
) -> list[int]:
    kwargs = dict(
        request_id=request_id,
        prompt=prompt,
        input_ids=input_ids,
        token_ids=list(token_ids),
        logits=logits,
        tokenizer=tokenizer,
        generation_config=generation_config,
        infer_type=infer_type,
    )
    if getattr(processor, "prefer_inline", False):
        result = processor(**kwargs)
    elif _is_async_callable(processor):
        result = await processor(**kwargs)
    else:
        result = await asyncio.to_thread(processor, **kwargs)
    if result is None:
        return token_ids
    if isinstance(result, dict):
        result = result.get("token_ids", token_ids)
    return [int(token_id) for token_id in result]


def _is_async_callable(processor: Callable[..., Any]) -> bool:
    if inspect.iscoroutinefunction(processor):
        return True
    call = getattr(processor, "__call__", None)
    return bool(call is not None and inspect.iscoroutinefunction(call))


def _logit_probs(logits: Any, token_ids: list[int]) -> dict[int, float]:
    if hasattr(logits, "detach"):
        tensor = _select_logits_row(logits).detach()
        ids = tensor.new_tensor([int(token_id) for token_id in token_ids]).long()
        selected = tensor.index_select(0, ids)
        log_denom = tensor.logsumexp(dim=-1)
        probs = (selected.float() - log_denom.float()).exp()
        return {
            int(token_id): float(prob)
            for token_id, prob in zip(token_ids, probs.detach().cpu().tolist())
        }

    row = _select_logits_row(logits)
    if not isinstance(row, list):
        row = [float(row)]
    max_logit = max(float(value) for value in row)
    denom = sum(math.exp(float(value) - max_logit) for value in row)
    return {
        int(token_id): math.exp(float(row[int(token_id)]) - max_logit) / denom
        for token_id in token_ids
    }


def _select_logits_row(logits: Any) -> Any:
    """Select one vocab row from logits with vocab on the last dimension.

    TurboMind generation logits may be returned as ``[vocab]``,
    ``[step, vocab]`` or with extra leading batch/beam dimensions. The builtin
    token decision is a one-token classifier path, so it should operate on a
    single vocab row rather than accidentally indexing a leading dimension by
    token id.
    """
    if hasattr(logits, "detach"):
        tensor = logits.detach()
        if tensor.dim() == 0:
            return tensor.reshape(1)
        if tensor.dim() == 1:
            return tensor
        return tensor.reshape(-1, tensor.shape[-1])[0]

    row = logits
    while isinstance(row, list) and row and isinstance(row[0], list):
        row = row[0]
    return row


def _status_name(status: Any) -> str:
    return getattr(status, "name", str(status))


def _fill_engine_metrics(perf: GenerationPerf, output: Any) -> None:
    metrics = getattr(output, "req_metrics", None)
    if metrics is None:
        return
    events = getattr(metrics, "engine_events", None) or []
    event_times: dict[str, float] = {}
    for event in events:
        event_type = getattr(event, "type", None)
        name = getattr(event_type, "name", str(event_type))
        timestamp = getattr(event, "timestamp", None)
        if timestamp is not None:
            event_times[name] = float(timestamp)

    queued = event_times.get("QUEUED")
    scheduled = event_times.get("SCHEDULED")
    token_timestamp = getattr(metrics, "token_timestamp", None)
    if queued is not None and scheduled is not None:
        perf.engine_queue_time_s = max(0.0, scheduled - queued)
    if queued is not None and token_timestamp is not None:
        perf.engine_first_token_time_s = max(0.0, float(token_timestamp) - queued)


def _shape_of(values: Any) -> list[int]:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    shape = []
    item = values
    while isinstance(item, list):
        shape.append(len(item))
        if not item:
            break
        item = item[0]
    return shape


def _as_2d_list(values: Any) -> list[list[float]]:
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        return [[float(values)]]
    if not values:
        return []
    if isinstance(values[0], list):
        return [[float(item) for item in row] for row in values]
    return [[float(item) for item in values]]


def _serialize_logits(
    logits: Any,
    *,
    fmt: str,
    top_k: int,
    token_ids: list[int] | None,
) -> dict[str, Any] | None:
    if logits is None:
        return None

    fmt = (fmt or "topk").lower()
    if fmt not in {"topk", "selected", "full"}:
        raise ValueError("logits_format must be one of: topk, selected, full")

    if fmt == "topk" and int(top_k or 0) < 0:
        fmt = "full"

    shape = _shape_of(logits)
    payload: dict[str, Any] = {
        "format": fmt,
        "shape": shape,
    }

    if hasattr(logits, "detach"):
        tensor = logits.detach().float().cpu()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        if fmt == "full":
            payload["values"] = tensor.tolist()
            return payload

        if fmt == "selected":
            selected = [int(token_id) for token_id in (token_ids or [])]
            vocab_size = tensor.shape[-1]
            payload["token_ids"] = selected
            payload["values"] = [
                [row[token_id].item() if 0 <= token_id < vocab_size else None for token_id in selected]
                for row in tensor
            ]
            return payload

        k = min(max(1, int(top_k or 1)), tensor.shape[-1])
        values, indexes = tensor.topk(k, dim=-1)
        payload["top_k"] = k
        payload["values"] = [
            {
                "token_ids": [int(token_id) for token_id in row_indexes.tolist()],
                "values": [float(value) for value in row_values.tolist()],
            }
            for row_indexes, row_values in zip(indexes, values)
        ]
        return payload

    rows = _as_2d_list(logits)

    if fmt == "full":
        payload["values"] = rows
        return payload

    if fmt == "selected":
        selected = [int(token_id) for token_id in (token_ids or [])]
        payload["token_ids"] = selected
        payload["values"] = [
            [row[token_id] if 0 <= token_id < len(row) else None for token_id in selected]
            for row in rows
        ]
        return payload

    k = max(1, int(top_k or 1))
    top_rows = []
    for row in rows:
        indexed = sorted(enumerate(row), key=lambda item: item[1], reverse=True)[:k]
        top_rows.append({
            "token_ids": [int(token_id) for token_id, _ in indexed],
            "values": [float(value) for _, value in indexed],
        })
    payload["top_k"] = k
    payload["values"] = top_rows
    return payload


def _error_result(
    request_id: str,
    status: str,
    status_code: int,
    error: str,
    *,
    queue_time_s: float = 0.0,
    total_time_s: float | None = None,
) -> GenerateResult:
    return GenerateResult(
        ok=False,
        request_id=request_id,
        status=status,
        status_code=status_code,
        error=error,
        queue_time_s=queue_time_s,
        total_time_s=queue_time_s if total_time_s is None else total_time_s,
    )


def build_tm_model(config: ServerConfig):
    from lmdeploy.messages import TurbomindEngineConfig
    from lmdeploy.turbomind import TurboMind

    dtype = _resolve_turbomind_dtype(config.dtype)
    engine_config = TurbomindEngineConfig(
        dtype=dtype,
        session_len=config.session_len,
        cache_max_entry_count=config.cache_max_entry_count,
        enable_prefix_caching=config.enable_prefix_caching,
        max_batch_size=config.engine_max_batch_size,
    )
    return TurboMind.from_pretrained(config.model_path, engine_config=engine_config)


def _resolve_turbomind_dtype(dtype: str) -> str:
    dtype = (dtype or "auto").lower()
    if dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError("TM_DTYPE must be one of: auto, float16, bfloat16")
    if dtype != "auto":
        return dtype
    try:
        import torch

        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            if major < 8:
                return "float16"
    except Exception:
        pass
    return "auto"
