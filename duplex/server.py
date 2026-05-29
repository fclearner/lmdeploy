from __future__ import annotations

import argparse
import asyncio
import itertools
import inspect
import json
import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from types import SimpleNamespace
from typing import Any

from pydantic import ValidationError

from .full_duplex import get_duplex_response, turn_end_logging
from .lmdeploy_client import DuplexLmdeployClient
from .schemas import EndData, InferData, model_to_dict
from .service_config import warmup_prompt


LOGGER = logging.getLogger("duplex.server")
SENSITIVE_KEYS = {
    "asrText",
    "ttsText",
    "context",
    "user",
    "finalText",
    "textToConcat",
    "textToTrim",
    "lastReply",
    "labelText",
    "bufferText",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def setup_logging(logfile: str | None = None, max_bytes: int = 100 * 1024 * 1024, backup_count: int = 5) -> None:
    log_queue: queue.Queue = queue.Queue(-1)
    queue_handler = QueueHandler(log_queue)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(queue_handler)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile:
        handlers.append(RotatingFileHandler(logfile, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)

    listener = QueueListener(log_queue, *handlers)
    listener.start()
    root._duplex_queue_listener = listener  # keep listener alive for the process lifetime


def create_app(
    *,
    max_concurrency: int | None = None,
    max_context_len: int | None = None,
    request_timeout: float | None = None,
    client_pool_size: int | None = None,
    warmup_enabled: bool | None = None,
):
    try:
        from sanic import Sanic, text
        from sanic.response import raw
    except ImportError as exc:
        raise RuntimeError("sanic is required for duplex.server: pip install sanic") from exc

    try:
        from sanic.worker.manager import WorkerManager

        WorkerManager.THRESHOLD = _env_int("DUPLEX_WORKER_THRESHOLD", 600)
    except Exception:
        pass

    max_concurrency = max_concurrency or _env_int("DUPLEX_MAX_CONCURRENCY", _env_int("MAX_CONCURRENCY", 50))
    max_context_len = max_context_len or _env_int("DUPLEX_MAX_CONTEXT_LEN", _env_int("MAX_CONTEXT_LEN", 6000))
    request_timeout = request_timeout or _env_float(
        "DUPLEX_REQUEST_TIMEOUT",
        _env_float("TRITON_REQUEST_TIMEOUT", 0.25),
    )
    client_pool_size = client_pool_size or _env_int("DUPLEX_CLIENT_POOL_SIZE", 1)
    warmup_enabled = _env_bool("DUPLEX_WARMUP_ENABLED", True) if warmup_enabled is None else warmup_enabled
    default_channels = _env_int("DUPLEX_DEFAULT_GRPC_CHANNELS", max(1, min(64, max_concurrency)))

    app = Sanic("duplex-lmdeploy-gateway")
    app.ctx.config = SimpleNamespace(
        max_concurrency=max_concurrency,
        max_context_len=max_context_len,
        request_timeout=request_timeout,
        client_pool_size=client_pool_size,
        warmup_enabled=warmup_enabled,
        default_channels=default_channels,
    )

    @app.before_server_start
    async def _startup(app, _loop) -> None:
        clients = [
            DuplexLmdeployClient.from_env(default_channels=default_channels)
            for _ in range(client_pool_size)
        ]
        await asyncio.gather(*(client.start() for client in clients))
        if warmup_enabled:
            await asyncio.gather(*(model_warmup(client) for client in clients))
        app.ctx.clients = clients
        app.ctx.client_cycle = itertools.cycle(clients)
        LOGGER.info(
            "duplex gateway started target=%s clients=%s default_channels=%s timeout=%.3f",
            clients[0].config.target if clients else "n/a",
            client_pool_size,
            default_channels,
            request_timeout,
        )

    @app.after_server_stop
    async def _shutdown(app, _loop) -> None:
        clients = getattr(app.ctx, "clients", [])
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)

    @app.get("/health/check")
    async def _health_check(_request):
        return text("success")

    @app.get("/health/ready")
    async def _health_ready(request):
        clients = getattr(request.app.ctx, "clients", [])
        raw_statuses = await asyncio.gather(*(client.health() for client in clients), return_exceptions=True)
        statuses = [
            {"status": "unhealthy", "error": str(item)}
            if isinstance(item, Exception)
            else item
            for item in raw_statuses
        ]
        ready = all(isinstance(item, dict) and item.get("status") == "ok" for item in statuses)
        body = {
            "status": "ok" if ready else "unhealthy",
            "clients": statuses,
        }
        return _json_response(raw, body, status=200 if body["status"] == "ok" else 503)

    @app.post("/infer/end_turn")
    async def _infer_end_turn(request):
        try:
            data = EndData(**(request.json or {}))
        except ValidationError as exc:
            return _json_response(raw, {"error": exc.errors()}, status=400)

        LOGGER.info("end_turn request=%s", desensitize(model_to_dict(data)))
        ret = await turn_end_logging(data)
        LOGGER.info("end_turn response=%s", desensitize(ret))
        return _json_response(raw, ret)

    @app.post("/infer")
    async def _infer(request):
        try:
            data = InferData(**(request.json or {}))
        except ValidationError as exc:
            return _json_response(raw, {"error": exc.errors()}, status=400)

        client = next(request.app.ctx.client_cycle)
        LOGGER.info("infer request=%s", desensitize(model_to_dict(data)))
        ret = await get_duplex_response(
            client,
            data,
            request.app.ctx.config.max_context_len,
            request.app.ctx.config.request_timeout,
        )
        LOGGER.info("infer response=%s", desensitize(ret))
        return _json_response(raw, ret)

    return app


async def model_warmup(client: DuplexLmdeployClient) -> None:
    for index, prompt in enumerate(warmup_prompt):
        await client.infer(f"duplex-warmup-{index}", prompt, decoding_type=1)


def desensitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _mask_value(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [desensitize(item) for item in value]
    return value


def _mask_value(key: str, value: Any) -> Any:
    if key in SENSITIVE_KEYS:
        if isinstance(value, str):
            return f"<len:{len(value)}>"
        if isinstance(value, list):
            return f"<list:{len(value)}>"
    return desensitize(value)


def _json_response(raw, payload: dict[str, Any], *, status: int = 200):
    return raw(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        status=status,
        content_type="application/json; charset=utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanic duplex gateway for LMDeploy gRPC")
    parser.add_argument("--host", default=os.getenv("DUPLEX_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_env_int("DUPLEX_PORT", 18080))
    parser.add_argument("--workers", type=int, default=_env_int("DUPLEX_WORKERS", 1))
    parser.add_argument(
        "--single-process",
        action="store_true",
        default=_env_bool("DUPLEX_SINGLE_PROCESS", True),
        help="Run Sanic in a single process. This is the default for module-based deployment.",
    )
    parser.add_argument("--log-file", default=os.getenv("DUPLEX_LOG_FILE"))
    parser.add_argument("--access-log", action="store_true", default=_env_bool("DUPLEX_ACCESS_LOG", False))
    parser.add_argument("--debug", action="store_true", default=_env_bool("DUPLEX_DEBUG", False))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    app = create_app()
    run_kwargs = {
        "host": args.host,
        "port": args.port,
        "workers": args.workers,
        "access_log": args.access_log,
        "debug": args.debug,
    }
    if "single_process" in inspect.signature(app.run).parameters:
        run_kwargs["single_process"] = args.single_process
    app.run(**run_kwargs)


if __name__ == "__main__":
    main()
