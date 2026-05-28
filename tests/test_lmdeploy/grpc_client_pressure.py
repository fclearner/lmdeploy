#!/usr/bin/env python3
"""Pure gRPC client pressure tester for grpc_turbomind_server.py.

The server uses google.protobuf.Struct as the request/response envelope, so this
client does not need generated pb2 files. It calls:

  /lmdeploy.turbomind.TurboMindService/Generate

This script is for normal latency/throughput pressure. It does not intentionally
send invalid prompts, cancel requests, or abruptly close streams.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from google.protobuf import json_format, struct_pb2


GRPC_SERVICE = "lmdeploy.turbomind.TurboMindService"
TEXT_CORPORA = {
    "zh": [
        "\u4f60\u597d\uff0c\u8bf7\u5224\u65ad\u7528\u6237\u8fd9\u53e5\u8bdd\u662f\u5426\u5df2\u7ecf\u8bf4\u5b8c\u3002",
        "\u7ebf\u4e0a\u65e5\u5fd7\u663e\u793a\u961f\u5217\u65f6\u95f4\u8fc7\u9ad8\uff0c\u9700\u8981\u533a\u5206\u662f\u6392\u961f\u8fd8\u662f\u751f\u6210\u6162\u3002",
        "\u8bf7\u6839\u636e\u4e0a\u4e0b\u6587\u8f93\u51fa\u4e00\u4e2a\u5b57\uff0c\u4e0d\u8981\u89e3\u91ca\u3002",
        "\u5ba2\u6237\u60f3\u8981\u53d6\u6d88\u8bf7\u6c42\uff0c\u670d\u52a1\u7aef\u9700\u8981\u5c3d\u5feb\u91ca\u653e\u5b9e\u4f8b\u3002",
    ],
    "mixed": [
        "\u4e2dEnglish\u6df7\u5408 input 123, latency_ms=42, queue_ms=18.",
        "ASR\u7ed3\u679c\uff1a\u7528\u6237\u8bf4 hello\uff0c\u7136\u540e\u505c\u987f 300ms\u3002",
        "session_id=s-001 round=3 \u8bf7\u5224\u65ad vadFinal=true \u662f\u5426\u7ed3\u675f\u3002",
        "\u8fd4\u56de valid/invalid/end \u4e09\u7c7b\u4e4b\u4e00\uff0cmax_new_tokens=1\u3002",
    ],
    "log": [
        "2026-05-06T20:00:01.123Z level=INFO request_id=req-001 queue_ms=17 gen_ms=24 status=finished",
        "WARN request_id=req-002 upstream=llm queue_ms=502 active=8 waiting=16",
        "INFO status=finished input_tokens=64 output_tokens=1 generation_ms=24",
        "TRACE callId=call-abc sessionId=s-42 roundId=7 vadFinal=false",
    ],
    "json": [
        '{"callId":"call-001","sessionId":"s-001","input":{"asrText":"\u4f60\u597d","vadFinal":true}}',
        '{"request_id":"req-001","infer_type":0,"valid_id":1,"invalid_id":2,"end_id":3}',
        '{"metrics":{"queue_p95":148.45,"generation_p95":26.71,"tokens":1}}',
    ],
    "en": [
        "Classify the request and return exactly one token.",
        "The server should report queue time, generation time, and total time.",
        "Please decide whether the current utterance is complete.",
        "A timeout happened after the request waited in the queue.",
    ],
}


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    kind: str
    prompt: str


@dataclass(frozen=True)
class RequestTrace:
    request_id: str
    status: str
    kind: str
    chars: int
    latency_ms: float
    server_total_ms: float | None
    queue_ms: float | None
    engine_queue_ms: float | None
    ttft_ms: float | None
    generation_ms: float | None
    grpc_pre_handler_ms: float | None
    grpc_handler_ms: float | None
    dedicated_model_submit_delay_ms: float | None
    dedicated_model_loop_ms: float | None
    client_minus_server_ms: float | None
    error: str | None = None


@dataclass
class Counters:
    sent: int = 0
    ok: int = 0
    error: int = 0
    status_codes: dict[str, int] = field(default_factory=dict)
    app_statuses: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    queue_ms: list[float] = field(default_factory=list)
    engine_queue_ms: list[float] = field(default_factory=list)
    engine_first_token_ms: list[float] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    generation_ms: list[float] = field(default_factory=list)
    server_total_ms: list[float] = field(default_factory=list)
    grpc_handler_ms: list[float] = field(default_factory=list)
    grpc_pre_handler_ms: list[float] = field(default_factory=list)
    dedicated_model_submit_delay_ms: list[float] = field(default_factory=list)
    dedicated_model_loop_ms: list[float] = field(default_factory=list)
    client_minus_server_ms: list[float] = field(default_factory=list)
    logits_processor_ms: list[float] = field(default_factory=list)
    logits_enabled: int = 0
    input_chars: list[int] = field(default_factory=list)
    first_errors: list[str] = field(default_factory=list)
    traces: list[RequestTrace] = field(default_factory=list)

    def record_error(self, msg: str) -> None:
        self.error += 1
        if len(self.first_errors) < 20:
            self.first_errors.append(msg[:1200])


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
    return xs[idx]


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def dict_to_struct(data: dict[str, Any]) -> struct_pb2.Struct:
    msg = struct_pb2.Struct()
    json_format.ParseDict(data, msg)
    return msg


def struct_to_dict(msg: struct_pb2.Struct) -> dict[str, Any]:
    return json_format.MessageToDict(msg, preserving_proto_field_name=True)


def synthesize_text(kind: str, target_chars: int, rng: random.Random) -> str:
    corpus = TEXT_CORPORA[kind]
    chunks: list[str] = []
    while len(" ".join(chunks)) < target_chars:
        chunks.append(rng.choice(corpus))
    text = " ".join(chunks)[:target_chars]
    return text if text.strip("\x00 \t\r\n") else "hello"


def build_workload(args: argparse.Namespace) -> list[WorkItem]:
    rng = random.Random(args.seed)
    kinds = [item.strip() for item in args.kinds.split(",") if item.strip()]
    unknown = sorted(set(kinds) - set(TEXT_CORPORA))
    if unknown:
        raise ValueError(f"unknown kinds: {unknown}; valid={sorted(TEXT_CORPORA)}")

    items: list[WorkItem] = []
    for _ in range(args.requests):
        kind = rng.choice(kinds)
        target_chars = rng.randint(args.min_chars, args.max_chars)
        items.append(
            WorkItem(
                request_id=f"grpc-pressure-{uuid.uuid4()}",
                kind=kind,
                prompt=synthesize_text(kind, target_chars, rng),
            )
        )
    return items


def parse_infer_types(args: argparse.Namespace) -> list[int]:
    raw = args.infer_types
    if raw is None or str(raw).strip() == "":
        return [int(args.infer_type)]
    values: list[int] = []
    for item in str(raw).replace(",", " ").split():
        values.append(int(item))
    return values or [int(args.infer_type)]


def build_payload(args: argparse.Namespace, item: WorkItem) -> dict[str, Any]:
    return {
        "request_id": item.request_id,
        "prompt": item.prompt,
        "max_new_tokens": args.max_tokens,
        "infer_type": args.infer_type,
        "include_text": args.include_text,
        "include_token_ids": args.include_token_ids,
    }


def perf_ms(perf: dict[str, Any], key: str) -> float | None:
    value = perf.get(key)
    if value is None:
        return None
    return float(value) * 1000


def record_response(counters: Counters, item: WorkItem, body: dict[str, Any], latency_ms: float) -> None:
    status = str(body.get("status", "unknown"))
    perf = body.get("performance") if isinstance(body.get("performance"), dict) else {}
    counters.latencies_ms.append(latency_ms)
    counters.input_chars.append(len(item.prompt))
    counters.app_statuses[status] = counters.app_statuses.get(status, 0) + 1
    if status == "finished":
        counters.ok += 1
    else:
        counters.record_error(f"{status}: {body.get('error', body)}")
    for key, target in [
        ("queue_time_s", counters.queue_ms),
        ("engine_queue_time_s", counters.engine_queue_ms),
        ("engine_first_token_time_s", counters.engine_first_token_ms),
        ("first_token_time_s", counters.ttft_ms),
        ("generation_time_s", counters.generation_ms),
        ("total_time_s", counters.server_total_ms),
        ("grpc_pre_handler_delay_s", counters.grpc_pre_handler_ms),
        ("grpc_handler_time_s", counters.grpc_handler_ms),
        ("dedicated_model_submit_delay_s", counters.dedicated_model_submit_delay_ms),
        ("dedicated_model_loop_time_s", counters.dedicated_model_loop_ms),
        ("logits_processor_time_s", counters.logits_processor_ms),
    ]:
        value = perf_ms(perf, key)
        if value is not None:
            target.append(value)
    server_total_ms = perf_ms(perf, "total_time_s")
    client_minus_server_ms = None
    if server_total_ms is not None:
        client_minus_server_ms = max(0.0, latency_ms - server_total_ms)
        counters.client_minus_server_ms.append(client_minus_server_ms)
    if perf.get("logits_enabled"):
        counters.logits_enabled += 1
    counters.traces.append(
        RequestTrace(
            request_id=item.request_id,
            status=status,
            kind=item.kind,
            chars=len(item.prompt),
            latency_ms=latency_ms,
            server_total_ms=server_total_ms,
            queue_ms=perf_ms(perf, "queue_time_s"),
            engine_queue_ms=perf_ms(perf, "engine_queue_time_s"),
            ttft_ms=perf_ms(perf, "first_token_time_s"),
            generation_ms=perf_ms(perf, "generation_time_s"),
            grpc_pre_handler_ms=perf_ms(perf, "grpc_pre_handler_delay_s"),
            grpc_handler_ms=perf_ms(perf, "grpc_handler_time_s"),
            dedicated_model_submit_delay_ms=perf_ms(perf, "dedicated_model_submit_delay_s"),
            dedicated_model_loop_ms=perf_ms(perf, "dedicated_model_loop_time_s"),
            client_minus_server_ms=client_minus_server_ms,
            error=None if status == "finished" else str(body.get("error", body)),
        )
    )


async def request_once(stub: Any, args: argparse.Namespace, item: WorkItem, counters: Counters) -> None:
    counters.sent += 1
    started = time.perf_counter()
    try:
        payload = build_payload(args, item)
        payload["_grpc_client_call_started_wall_time_s"] = time.time()
        response = await stub(dict_to_struct(payload), timeout=args.timeout_sec)
        latency_ms = (time.perf_counter() - started) * 1000
        counters.status_codes["OK"] = counters.status_codes.get("OK", 0) + 1
        record_response(counters, item, struct_to_dict(response), latency_ms)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        counters.latencies_ms.append(latency_ms)
        code = getattr(exc, "code", lambda: type(exc).__name__)()
        code_name = getattr(code, "name", str(code))
        counters.status_codes[code_name] = counters.status_codes.get(code_name, 0) + 1
        counters.record_error(f"{type(exc).__name__}: {exc}")
        counters.traces.append(
            RequestTrace(
                request_id=item.request_id,
                status=code_name,
                kind=item.kind,
                chars=len(item.prompt),
                latency_ms=latency_ms,
                server_total_ms=None,
                queue_ms=None,
                engine_queue_ms=None,
                ttft_ms=None,
                generation_ms=None,
                grpc_pre_handler_ms=None,
                grpc_handler_ms=None,
                dedicated_model_submit_delay_ms=None,
                dedicated_model_loop_ms=None,
                client_minus_server_ms=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        )


async def run_phase(name: str, stub: Any, args: argparse.Namespace, items: list[WorkItem]) -> Counters:
    counters = Counters()
    queue: asyncio.Queue[WorkItem] = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    async def worker(worker_id: int) -> None:
        worker_stub = stub[worker_id % len(stub)] if isinstance(stub, list) else stub
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await request_once(worker_stub, args, item, counters)
            finally:
                queue.task_done()

    print(f"\n[phase] {name}: total={len(items)} concurrency={args.concurrency}")
    started = time.perf_counter()
    await asyncio.gather(*(worker(worker_id) for worker_id in range(args.concurrency)))
    elapsed = time.perf_counter() - started
    print_summary(name, counters, elapsed, args.top_slow)
    return counters


async def run_rate_phase(name: str, stub: Any, args: argparse.Namespace, items: list[WorkItem]) -> Counters:
    counters = Counters()
    max_inflight = max(1, args.concurrency)
    rate_qps = float(args.rate_qps)
    semaphore = asyncio.Semaphore(max_inflight)
    tasks: list[asyncio.Task[None]] = []
    started = time.perf_counter()

    async def one(index: int, item: WorkItem) -> None:
        async with semaphore:
            worker_stub = stub[index % len(stub)] if isinstance(stub, list) else stub
            await request_once(worker_stub, args, item, counters)

    print(f"\n[phase] {name}: total={len(items)} rate_qps={rate_qps:.2f} max_inflight={max_inflight}")
    for index, item in enumerate(items):
        target_time = started + index / rate_qps
        delay = target_time - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        task = asyncio.create_task(one(index, item))
        tasks.append(task)

    if tasks:
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    print_summary(name, counters, elapsed, args.top_slow)
    return counters


def print_summary(name: str, counters: Counters, elapsed_sec: float, top_slow: int = 0) -> None:
    print(
        f"[summary] {name}: elapsed={elapsed_sec:.2f}s sent={counters.sent} ok={counters.ok} "
        f"error={counters.error} qps={counters.sent / elapsed_sec if elapsed_sec > 0 else 0:.2f}"
    )
    print(f"  grpc_statuses={dict(sorted(counters.status_codes.items()))}")
    print(f"  app_statuses={dict(sorted(counters.app_statuses.items()))}")
    print(
        "  latency_ms "
        f"p50={fmt(percentile(counters.latencies_ms, 0.50))} "
        f"p95={fmt(percentile(counters.latencies_ms, 0.95))} "
        f"max={fmt(max_or_none(counters.latencies_ms))}"
    )
    print(
        "  server_ms  "
        f"queue_p95={fmt(percentile(counters.queue_ms, 0.95))} "
        f"queue_max={fmt(max_or_none(counters.queue_ms))} "
        f"engine_queue_p95={fmt(percentile(counters.engine_queue_ms, 0.95))} "
        f"engine_queue_max={fmt(max_or_none(counters.engine_queue_ms))} "
        f"engine_ttft_p95={fmt(percentile(counters.engine_first_token_ms, 0.95))} "
        f"engine_ttft_max={fmt(max_or_none(counters.engine_first_token_ms))} "
        f"ttft_p95={fmt(percentile(counters.ttft_ms, 0.95))} "
        f"ttft_max={fmt(max_or_none(counters.ttft_ms))} "
        f"gen_p95={fmt(percentile(counters.generation_ms, 0.95))} "
        f"gen_max={fmt(max_or_none(counters.generation_ms))} "
        f"total_p95={fmt(percentile(counters.server_total_ms, 0.95))} "
        f"total_max={fmt(max_or_none(counters.server_total_ms))}"
    )
    print(
        "  transport  "
        f"pre_handler_p95={fmt(percentile(counters.grpc_pre_handler_ms, 0.95))} "
        f"pre_handler_max={fmt(max_or_none(counters.grpc_pre_handler_ms))} "
        f"handler_p95={fmt(percentile(counters.grpc_handler_ms, 0.95))} "
        f"handler_max={fmt(max_or_none(counters.grpc_handler_ms))} "
        f"dedicated_submit_p95={fmt(percentile(counters.dedicated_model_submit_delay_ms, 0.95))} "
        f"dedicated_submit_max={fmt(max_or_none(counters.dedicated_model_submit_delay_ms))} "
        f"dedicated_loop_p95={fmt(percentile(counters.dedicated_model_loop_ms, 0.95))} "
        f"dedicated_loop_max={fmt(max_or_none(counters.dedicated_model_loop_ms))} "
        f"client_minus_server_p50={fmt(percentile(counters.client_minus_server_ms, 0.50))} "
        f"client_minus_server_p95={fmt(percentile(counters.client_minus_server_ms, 0.95))} "
        f"client_minus_server_max={fmt(max_or_none(counters.client_minus_server_ms))}"
    )
    print(
        "  input      "
        f"chars_p50={fmt(percentile(counters.input_chars, 0.50))} "
        f"chars_p95={fmt(percentile(counters.input_chars, 0.95))} "
        f"chars_max={fmt(max(counters.input_chars) if counters.input_chars else None)}"
    )
    print(
        "  logits     "
        f"enabled={counters.logits_enabled}/{counters.ok} "
        f"processor_p95={fmt(percentile(counters.logits_processor_ms, 0.95))}"
    )
    if counters.first_errors:
        print("  first_errors:")
        for msg in counters.first_errors:
            print(f"  - {msg}")
    if top_slow > 0 and counters.traces:
        print(f"  top_slow latency_ms top={top_slow}:")
        for trace in sorted(counters.traces, key=lambda item: item.latency_ms, reverse=True)[:top_slow]:
            print(
                "  - "
                f"lat={fmt(trace.latency_ms)} "
                f"status={trace.status} "
                f"kind={trace.kind} "
                f"chars={trace.chars} "
                f"server_total={fmt(trace.server_total_ms)} "
                f"queue={fmt(trace.queue_ms)} "
                f"engine_queue={fmt(trace.engine_queue_ms)} "
                f"ttft={fmt(trace.ttft_ms)} "
                f"gen={fmt(trace.generation_ms)} "
                f"pre_handler={fmt(trace.grpc_pre_handler_ms)} "
                f"handler={fmt(trace.grpc_handler_ms)} "
                f"dedicated_submit={fmt(trace.dedicated_model_submit_delay_ms)} "
                f"dedicated_loop={fmt(trace.dedicated_model_loop_ms)} "
                f"client_minus_server={fmt(trace.client_minus_server_ms)} "
                f"request_id={trace.request_id}"
            )
            if trace.error:
                print(f"    error={trace.error[:300]}")


def import_grpc():
    try:
        import grpc
    except ImportError:
        print("Missing dependency: grpcio. Install with: pip install grpcio protobuf", file=sys.stderr)
        raise
    return grpc


async def amain(args: argparse.Namespace) -> None:
    infer_types = parse_infer_types(args)
    preview_args = copy.copy(args)
    preview_args.infer_type = infer_types[0]
    items = build_workload(preview_args)
    if args.dry_run:
        print(
            json.dumps(
                [build_payload(preview_args, item) for item in items[: args.dry_run]],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    grpc = import_grpc()
    options = [
        ("grpc.max_receive_message_length", args.max_receive_message_length),
        ("grpc.max_send_message_length", args.max_send_message_length),
    ]
    channels = [grpc.aio.insecure_channel(args.target, options=options) for _ in range(args.channels)]
    try:
        if args.channel_ready_timeout_sec > 0:
            await asyncio.wait_for(
                asyncio.gather(*(channel.channel_ready() for channel in channels)),
                timeout=args.channel_ready_timeout_sec,
            )
        generate = [
            channel.unary_unary(
                f"/{GRPC_SERVICE}/Generate",
                request_serializer=struct_pb2.Struct.SerializeToString,
                response_deserializer=struct_pb2.Struct.FromString,
            )
            for channel in channels
        ]
        health = channels[0].unary_unary(
            f"/{GRPC_SERVICE}/Health",
            request_serializer=struct_pb2.Struct.SerializeToString,
            response_deserializer=struct_pb2.Struct.FromString,
        )
        if args.health:
            response = await health(dict_to_struct({}), timeout=5)
            print(f"health={json.dumps(struct_to_dict(response), ensure_ascii=False)}")

        for repeat_index in range(max(1, args.repeat)):
            for infer_type in infer_types:
                phase_args = copy.copy(args)
                phase_args.infer_type = infer_type
                if not args.same_workload_repeats:
                    phase_args.seed = int(args.seed) + repeat_index * 100000
                items = build_workload(phase_args)
                prefix = f"repeat={repeat_index + 1} infer={infer_type}"
                if args.warmup > 0:
                    await run_phase(f"{prefix} warmup", generate, phase_args, items[: args.warmup])
                    if args.phase_gap_sec > 0:
                        await asyncio.sleep(args.phase_gap_sec)
                if args.rate_qps > 0:
                    await run_rate_phase(f"{prefix} pressure", generate, phase_args, items)
                else:
                    await run_phase(f"{prefix} pressure", generate, phase_args, items)
                if args.phase_gap_sec > 0:
                    await asyncio.sleep(args.phase_gap_sec)
    finally:
        await asyncio.gather(*(channel.close() for channel in channels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--rate-qps",
        type=float,
        default=0.0,
        help="Open-loop arrival rate for pressure phase. 0 keeps fixed-concurrency closed-loop mode.",
    )
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--infer-type", type=int, default=-1)
    parser.add_argument(
        "--infer-types",
        default=None,
        help="Comma/space separated infer_type list, for example '-1,0,1'. Overrides --infer-type.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each infer_type this many times.")
    parser.add_argument(
        "--same-workload-repeats",
        action="store_true",
        help="Reuse the same generated request texts across repeats for fair A/B latency comparisons.",
    )
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--include-token-ids", action="store_true")
    parser.add_argument("--kinds", default="zh,mixed,log,json,en")
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--top-slow", type=int, default=0, help="Print the slowest N requests with timing breakdown.")
    parser.add_argument(
        "--phase-gap-sec",
        type=float,
        default=0.0,
        help="Sleep between measured phases to let client/server transport queues drain.",
    )
    parser.add_argument(
        "--channel-ready-timeout-sec",
        type=float,
        default=10.0,
        help="Wait for all gRPC channels to connect before measuring. Set <=0 to skip.",
    )
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--max-receive-message-length", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-send-message-length", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
