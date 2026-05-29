from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from turbomind_grpc_client import TurboMindGrpcClient
from turbomind_service_core import GenerateResult

from .service_config import CERTAIN_PROB_THRESHOLD, END_PROB_THRESHOLD, INVALID_BIAS


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_optional_int(*names: str) -> int | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return int(value)
    return None


def _env_float(names: tuple[str, ...], default: float) -> float:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return float(value)
    return float(default)


@dataclass(frozen=True)
class TokenDecisionLabels:
    valid_id: int | None = None
    invalid_id: int | None = None
    end_id: int | None = None
    valid_text: str = "<valid>"
    invalid_text: str = "<invalid>"
    end_text: str = "<|im_end|>"

    @classmethod
    def from_env(cls) -> "TokenDecisionLabels":
        return cls(
            valid_id=_env_optional_int("DUPLEX_VALID_ID", "TM_VALID_ID"),
            invalid_id=_env_optional_int("DUPLEX_INVALID_ID", "TM_INVALID_ID"),
            end_id=_env_optional_int("DUPLEX_END_ID", "TM_END_ID"),
            valid_text=os.getenv("DUPLEX_VALID_TEXT", "<valid>"),
            invalid_text=os.getenv("DUPLEX_INVALID_TEXT", "<invalid>"),
            end_text=os.getenv("DUPLEX_END_TEXT", "<|im_end|>"),
        )


@dataclass(frozen=True)
class DuplexLmdeployClientConfig:
    target: str = "127.0.0.1:50051"
    timeout_s: float = 0.25
    channels: int = 4
    max_inflight: int | None = None
    connect_timeout_s: float = 10.0
    ready_on_start: bool = True
    health_interval_s: float = 2.0
    max_new_tokens: int = 1
    raw_max_new_tokens: int = 512
    certainty_threshold: float = CERTAIN_PROB_THRESHOLD
    completion_threshold: float = END_PROB_THRESHOLD
    invalid_bias: float = INVALID_BIAS

    @classmethod
    def from_env(cls, *, default_channels: int = 4) -> "DuplexLmdeployClientConfig":
        target = os.getenv("DUPLEX_GRPC_TARGET") or os.getenv("TM_GRPC_TARGET") or "127.0.0.1:50051"
        return cls(
            target=target,
            timeout_s=_env_float(("DUPLEX_REQUEST_TIMEOUT", "DUPLEX_GRPC_TIMEOUT_S", "TM_GRPC_TIMEOUT_S"), 0.25),
            channels=int(
                os.getenv(
                    "DUPLEX_GRPC_CLIENT_CHANNELS",
                    os.getenv("TM_GRPC_CLIENT_CHANNELS", default_channels),
                )
            ),
            max_inflight=_env_optional_int("DUPLEX_GRPC_CLIENT_MAX_INFLIGHT", "TM_GRPC_CLIENT_MAX_INFLIGHT"),
            connect_timeout_s=_env_float(
                ("DUPLEX_GRPC_CLIENT_CONNECT_TIMEOUT_S", "TM_GRPC_CLIENT_CONNECT_TIMEOUT_S"),
                10.0,
            ),
            ready_on_start=_env_bool(
                "DUPLEX_GRPC_CLIENT_READY_ON_START",
                _env_bool("TM_GRPC_CLIENT_READY_ON_START", True),
            ),
            health_interval_s=_env_float(("DUPLEX_HEALTH_INTERVAL_S",), 2.0),
            max_new_tokens=int(os.getenv("DUPLEX_MAX_NEW_TOKENS", "1")),
            raw_max_new_tokens=int(os.getenv("DUPLEX_RAW_MAX_NEW_TOKENS", "512")),
            certainty_threshold=_env_float(
                ("DUPLEX_CERTAINTY_THRESHOLD", "TM_CERTAINTY_THRESHOLD"),
                CERTAIN_PROB_THRESHOLD,
            ),
            completion_threshold=_env_float(
                ("DUPLEX_COMPLETION_THRESHOLD", "TM_COMPLETION_THRESHOLD"),
                END_PROB_THRESHOLD,
            ),
            invalid_bias=_env_float(("DUPLEX_INVALID_BIAS", "TM_INVALID_BIAS"), INVALID_BIAS),
        )


class DuplexLmdeployClient:
    """Adapter that keeps the old duplex business interface over LMDeploy gRPC."""

    def __init__(
        self,
        config: DuplexLmdeployClientConfig,
        *,
        labels: TokenDecisionLabels | None = None,
        grpc_client: Any | None = None,
    ):
        self.config = config
        self.labels = labels or TokenDecisionLabels.from_env()
        self.grpc_client = grpc_client or TurboMindGrpcClient(
            config.target,
            timeout_s=config.timeout_s,
            channels=config.channels,
            max_inflight=config.max_inflight,
            connect_timeout_s=config.connect_timeout_s,
            ready_on_start=config.ready_on_start,
        )
        self._healthy = False
        self._last_health: dict[str, Any] = {"status": "unknown", "target": config.target}
        self._health_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._started = False

    @classmethod
    def from_env(cls, *, default_channels: int = 4) -> "DuplexLmdeployClient":
        return cls(DuplexLmdeployClientConfig.from_env(default_channels=default_channels))

    async def start(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            start = getattr(self.grpc_client, "start", None)
            if start is not None:
                await start()
            await self._refresh_health()
            self._health_task = asyncio.create_task(self._health_loop())
            self._started = True

    async def close(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
        close = getattr(self.grpc_client, "close", None)
        if close is not None:
            await close()
        self._started = False

    async def health_check(self) -> bool:
        if not self._started:
            try:
                await self.start()
            except Exception:
                self._healthy = False
        return self._healthy

    async def health(self) -> dict[str, Any]:
        if not self._started:
            await self.health_check()
        return dict(self._last_health)

    async def infer(
        self,
        request_id: str,
        text_input: str,
        decoding_type: int = 0,
        timeout: float | None = None,
    ) -> str:
        if not await self.health_check():
            raise RuntimeError(f"LMDeploy gRPC target is not healthy: {self._last_health}")

        coro = self._generate_once(request_id, text_input, decoding_type)
        if timeout is not None and timeout > 0:
            result = await asyncio.wait_for(coro, timeout=timeout)
        else:
            result = await coro
        return self._normalize_output(result, decoding_type)

    async def _generate_once(self, request_id: str, text_input: str, decoding_type: int) -> GenerateResult:
        generation_config = self._generation_config(decoding_type)
        max_new_tokens = self.config.raw_max_new_tokens if decoding_type < 0 else self.config.max_new_tokens
        return await self.grpc_client.generate(
            text_input,
            request_id=request_id,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
            include_text=True,
            include_token_ids=True,
            infer_type=decoding_type,
        )

    def _generation_config(self, decoding_type: int) -> dict[str, Any]:
        config: dict[str, Any] = {
            "token_decision_valid_id": self.labels.valid_id,
            "token_decision_invalid_id": self.labels.invalid_id,
            "token_decision_end_id": self.labels.end_id,
        }
        if decoding_type == 0:
            config["token_decision_certainty_threshold"] = self.config.certainty_threshold
            config["token_decision_invalid_bias"] = self.config.invalid_bias
        elif decoding_type > 0:
            config["token_decision_completion_threshold"] = self.config.completion_threshold
        return {key: value for key, value in config.items() if value is not None}

    def _normalize_output(self, result: GenerateResult, decoding_type: int) -> str:
        if not result.ok:
            error = result.error or result.status
            raise RuntimeError(f"LMDeploy gRPC request failed: {error}")

        token_id = result.token_ids[0] if result.token_ids else None
        if token_id == self.labels.valid_id:
            return self.labels.valid_text
        if token_id == self.labels.invalid_id:
            return self.labels.invalid_text
        if decoding_type > 0 and token_id == self.labels.end_id:
            return self.labels.end_text

        text = (result.text or "").strip()
        if text.startswith(self.labels.valid_text):
            return self.labels.valid_text
        if text.startswith(self.labels.invalid_text):
            return self.labels.invalid_text
        if decoding_type > 0 and self.labels.end_text in text:
            return self.labels.end_text
        return "" if decoding_type == 0 else text

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(max(0.1, self.config.health_interval_s))
            await self._refresh_health()

    async def _refresh_health(self) -> None:
        try:
            data = await self.grpc_client.health_async()
            status = str(data.get("status", "")).lower()
            self._last_health = data
            self._healthy = status in {"ok", "healthy"}
        except Exception as exc:
            self._healthy = False
            self._last_health = {
                "status": "unhealthy",
                "target": self.config.target,
                "error": str(exc),
            }
