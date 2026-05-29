#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import random
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any

import aiohttp


WORDS = (
    "hello",
    "please",
    "check",
    "policy",
    "today",
    "account",
    "service",
    "transfer",
    "balance",
    "question",
)


@dataclass
class Sample:
    index: int
    latency_ms: float
    ok: bool
    status: int
    state: Any = None
    fallback: bool = False
    error: str | None = None
    chars: int = 0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def make_text(index: int, min_chars: int, max_chars: int) -> str:
    rng = random.Random(index)
    target = rng.randint(min_chars, max_chars)
    pieces: list[str] = []
    while len(" ".join(pieces)) < target:
        pieces.append(rng.choice(WORDS))
    return " ".join(pieces)[:target]


def make_payload(index: int, args: argparse.Namespace, *, warmup: bool = False) -> dict[str, Any]:
    request_id = f"duplex-{'warmup' if warmup else 'pressure'}-{uuid.uuid4()}"
    text = make_text(index, args.min_chars, args.max_chars)
    return {
        "callId": f"call-{index}",
        "sessionId": f"session-{index}",
        "requestId": request_id,
        "roundId": f"round-{index}",
        "input": {
            "ttsText": "",
            "asrText": text,
            "startTime": float(index),
            "endTime": float(index) + 1.0,
            "vadFinal": args.vad_final,
            "dualVad": True,
            "timeOut": 0,
        },
        "history": None,
        "whiteList": [],
        "blackList": [],
    }


async def post_once(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    index: int,
    timeout: float,
) -> Sample:
    started = time.perf_counter()
    chars = len(payload["input"]["asrText"])
    try:
        async with session.post(url, json=payload, timeout=timeout) as response:
            body = await response.json(content_type=None)
            latency_ms = (time.perf_counter() - started) * 1000
            ok = response.status == 200 and "error" not in body
            return Sample(
                index=index,
                latency_ms=latency_ms,
                ok=ok,
                status=response.status,
                state=body.get("state"),
                fallback="fallbackMsg" in body,
                error=body.get("error") if isinstance(body, dict) else None,
                chars=chars,
            )
    except Exception as exc:
        return Sample(
            index=index,
            latency_ms=(time.perf_counter() - started) * 1000,
            ok=False,
            status=0,
            error=str(exc),
            chars=chars,
        )


async def run_phase(args: argparse.Namespace, *, total: int, warmup: bool) -> list[Sample]:
    connector = aiohttp.TCPConnector(
        limit=max(args.concurrency, args.channels),
        limit_per_host=max(args.concurrency, args.channels),
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_sec)
    sem = asyncio.Semaphore(args.concurrency)
    results: list[Sample] = []
    pending: set[asyncio.Task] = set()
    phase_started = time.perf_counter()
    next_at = phase_started

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def submit(index: int) -> Sample:
            async with sem:
                return await post_once(
                    session,
                    args.url,
                    make_payload(index, args, warmup=warmup),
                    index,
                    args.timeout_sec,
                )

        for index in range(total):
            if args.rate_qps > 0 and not warmup:
                next_at += 1.0 / args.rate_qps
                delay = next_at - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            pending.add(asyncio.create_task(submit(index)))
            done = {task for task in pending if task.done()}
            for task in done:
                pending.remove(task)
                results.append(await task)

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                results.append(await task)

    return results


def print_summary(name: str, samples: list[Sample], elapsed_s: float, top_slow: int) -> None:
    latencies = [sample.latency_ms for sample in samples]
    ok = [sample for sample in samples if sample.ok]
    status_counts = Counter(sample.status for sample in samples)
    state_counts = Counter(sample.state for sample in samples if sample.state is not None)
    fallback_count = sum(1 for sample in samples if sample.fallback)
    qps = len(samples) / elapsed_s if elapsed_s > 0 else 0.0
    print(
        f"[summary] {name}: elapsed={elapsed_s:.2f}s sent={len(samples)} "
        f"ok={len(ok)} error={len(samples)-len(ok)} qps={qps:.2f}"
    )
    print(f"  http_statuses={dict(status_counts)} app_states={dict(state_counts)} fallback={fallback_count}")
    print(
        "  latency_ms "
        f"p50={percentile(latencies, 0.50):.2f} "
        f"p95={percentile(latencies, 0.95):.2f} "
        f"p99={percentile(latencies, 0.99):.2f} "
        f"max={max(latencies) if latencies else 0.0:.2f}"
    )
    if top_slow > 0:
        print(f"  top_slow latency_ms top={top_slow}:")
        for sample in sorted(samples, key=lambda item: item.latency_ms, reverse=True)[:top_slow]:
            print(
                f"    - lat={sample.latency_ms:.2f} status={sample.status} "
                f"state={sample.state} fallback={sample.fallback} chars={sample.chars} error={sample.error}"
            )


async def main_async(args: argparse.Namespace) -> None:
    if args.warmup > 0:
        started = time.perf_counter()
        samples = await run_phase(args, total=args.warmup, warmup=True)
        print_summary("warmup", samples, time.perf_counter() - started, args.top_slow)
    for repeat in range(1, args.repeat + 1):
        started = time.perf_counter()
        samples = await run_phase(args, total=args.requests, warmup=False)
        print_summary(f"repeat={repeat} pressure", samples, time.perf_counter() - started, args.top_slow)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pressure test the duplex Sanic /infer endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:18080/infer")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--channels", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--rate-qps", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=256)
    parser.add_argument("--top-slow", type=int, default=5)
    parser.add_argument("--vad-final", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
