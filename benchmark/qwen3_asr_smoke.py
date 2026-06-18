# Copyright (c) OpenMMLab. All rights reserved.
"""Smoke test Qwen3-ASR through LMDeploy pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path


def _default_model_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / '.cache_models' / 'modelscope' / 'Qwen' / 'Qwen3-ASR-0.6B')


def _default_audio_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / 'artifacts' / 'qwen3_asr_smoke' / 'hello_world.wav')


def _default_manifest_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / 'artifacts' / 'qwen3_asr_smoke' / 'manifest.json')


def _build_messages(audio_path: str, system: str | None) -> list[dict]:
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({
        'role':
        'user',
        'content': [{
            'type': 'audio_url',
            'audio_url': {
                'url': audio_path,
            },
        }],
    })
    return messages


def _make_backend_config(backend: str, args):
    from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig

    kwargs = dict(
        dtype=args.dtype,
        session_len=args.session_len,
        max_batch_size=args.batch_size,
        cache_max_entry_count=args.cache_max_entry_count,
        max_prefill_token_num=args.max_prefill_token_num,
    )
    if backend == 'pytorch':
        return PytorchEngineConfig(**kwargs)
    if backend == 'turbomind':
        return TurbomindEngineConfig(**kwargs)
    raise ValueError(f'Unsupported backend: {backend}')


def _audio_duration(path: str) -> float | None:
    try:
        with wave.open(path) as f:
            return round(f.getnframes() / float(f.getframerate()), 3)
    except Exception:
        return None


def _extract_asr_text(text: str | None) -> str:
    if not text:
        return ''
    marker = '<asr_text>'
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def _normalize_text(text: str | None) -> str:
    text = _extract_asr_text(text).lower()
    text = re.sub(r'[^a-z0-9 ]+', ' ', text)
    return ' '.join(text.split())


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    dp = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, 1):
        prev = dp[0]
        dp[0] = i
        for j, hyp_token in enumerate(hyp, 1):
            old = dp[j]
            cost = 0 if ref_token == hyp_token else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old
    return dp[-1]


def _wer(expected: str | None, actual: str | None) -> float | None:
    if expected is None:
        return None
    ref = _normalize_text(expected).split()
    hyp = _normalize_text(actual).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(_edit_distance(ref, hyp) / len(ref), 4)


def _normalize_chars(text: str | None) -> str:
    text = _extract_asr_text(text).lower()
    return ''.join(char for char in text if char.isalnum())


def _cer(expected: str | None, actual: str | None) -> float | None:
    if expected is None:
        return None
    ref = list(_normalize_chars(expected))
    hyp = list(_normalize_chars(actual))
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(_edit_distance(ref, hyp) / len(ref), 4)


def _response_to_dict(response) -> dict:
    return {
        'text': getattr(response, 'text', None),
        'input_token_len': getattr(response, 'input_token_len', None),
        'generate_token_len': getattr(response, 'generate_token_len', None),
        'finish_reason': getattr(response, 'finish_reason', None),
        'token_ids': getattr(response, 'token_ids', None),
    }


def _nvidia_smi_memory() -> dict:
    try:
        completed = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        line = completed.stdout.strip().splitlines()[0]
        used_mib, total_mib = [int(value.strip()) for value in line.split(',')[:2]]
        return {
            'nvidia_smi_memory_used_mib': used_mib,
            'nvidia_smi_memory_total_mib': total_mib,
        }
    except Exception as exc:  # noqa: BLE001
        return {'nvidia_smi_error': repr(exc)}


def _cuda_stats() -> dict:
    stats = _nvidia_smi_memory()
    try:
        import torch

        if not torch.cuda.is_available():
            stats['cuda_available'] = False
            return stats
        stats.update({
            'cuda_available': True,
            'device': torch.cuda.get_device_name(0),
            'memory_allocated': int(torch.cuda.memory_allocated()),
            'memory_reserved': int(torch.cuda.memory_reserved()),
            'max_memory_allocated': int(torch.cuda.max_memory_allocated()),
            'max_memory_reserved': int(torch.cuda.max_memory_reserved()),
        })
        return stats
    except Exception as exc:  # noqa: BLE001
        stats['cuda_error'] = repr(exc)
        return stats


def _load_samples(args) -> list[dict]:
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        with open(manifest_path, encoding='utf-8-sig') as f:
            samples = json.load(f)
        base_dir = manifest_path.parent
    else:
        samples = [{'id': Path(args.audio).stem, 'audio': args.audio, 'expected': args.expected}]
        base_dir = Path.cwd()

    resolved = []
    for i, sample in enumerate(samples):
        audio_path = Path(sample['audio'])
        if not audio_path.is_absolute():
            audio_path = base_dir / audio_path
        audio_path = audio_path.expanduser().resolve()
        resolved.append({
            'id': sample.get('id', audio_path.stem),
            'audio': str(audio_path),
            'expected': sample.get('expected'),
            'duration_sec': _audio_duration(str(audio_path)),
            'index': i,
        })
    if args.limit is not None:
        resolved = resolved[:args.limit]
    return resolved


def _iter_chunks(items: list[dict], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield start, items[start:start + chunk_size]


def _summarize_items(items: list[dict], batches: list[dict] | None = None) -> dict:
    infer_times = [item['infer_elapsed_sec'] for item in items if item.get('ok')]
    wers = [item['wer'] for item in items if item.get('wer') is not None]
    cers = [item['cer'] for item in items if item.get('cer') is not None]
    if not infer_times:
        return {'ok_count': 0, 'total_count': len(items)}
    summary = {
        'ok_count': len(infer_times),
        'total_count': len(items),
        'infer_mean_sec': round(sum(infer_times) / len(infer_times), 3),
        'infer_min_sec': round(min(infer_times), 3),
        'infer_max_sec': round(max(infer_times), 3),
        'wer_mean': round(sum(wers) / len(wers), 4) if wers else None,
        'cer_mean': round(sum(cers) / len(cers), 4) if cers else None,
    }
    if batches:
        wall_sec = sum(batch['elapsed_sec'] for batch in batches)
        audio_sec = sum(item['duration_sec'] for item in items if item.get('duration_sec'))
        summary.update({
            'batch_count': len(batches),
            'infer_wall_sec': round(wall_sec, 3),
            'requests_per_sec': round(len(items) / wall_sec, 3) if wall_sec > 0 else None,
            'audio_rtf_wall': round(wall_sec / audio_sec, 4) if audio_sec > 0 else None,
        })
    return summary


def _make_item(sample: dict, repeat: int, response, elapsed: float, batch_index: int | None = None) -> dict:
    response_dict = _response_to_dict(response)
    return {
        'id': sample['id'],
        'repeat': repeat,
        'audio': sample['audio'],
        'duration_sec': sample['duration_sec'],
        'expected': sample['expected'],
        'infer_elapsed_sec': round(elapsed, 3),
        'rtf': round(elapsed / sample['duration_sec'], 3) if sample['duration_sec'] else None,
        'batch_index': batch_index,
        'response': response_dict,
        'asr_text': _extract_asr_text(response_dict.get('text')),
        'wer': _wer(sample['expected'], response_dict.get('text')),
        'cer': _cer(sample['expected'], response_dict.get('text')),
        'ok': response_dict.get('finish_reason') != 'error',
    }


def run_backend(backend: str, args) -> dict:
    if backend == 'turbomind' and args.tm_lib_path:
        sys.path.insert(0, str(Path(args.tm_lib_path).expanduser().resolve()))

    import torch
    from lmdeploy import GenerationConfig, VisionConfig, pipeline

    model_path = str(Path(args.model_path).expanduser().resolve())
    samples = _load_samples(args)
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    result = {
        'backend': backend,
        'model_path': model_path,
        'samples': samples,
        'tm_lib_path': args.tm_lib_path if backend == 'turbomind' else None,
        'ok': False,
    }
    load_start = time.perf_counter()
    pipe = None
    try:
        backend_config = _make_backend_config(backend, args)
        vision_config = VisionConfig(max_batch_size=args.vision_max_batch_size, thread_safe=args.vision_thread_safe)
        pipe = pipeline(model_path, backend_config=backend_config, log_level=args.log_level, vision_config=vision_config)
        result['load_elapsed_sec'] = round(time.perf_counter() - load_start, 3)
        result['cuda_after_load'] = _cuda_stats()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        items = []
        batches = []
        for repeat in range(args.repeat):
            for batch_index, batch_samples in _iter_chunks(samples, args.batch_size):
                batch_messages = [_build_messages(sample['audio'], args.system) for sample in batch_samples]
                infer_start = time.perf_counter()
                responses = pipe(batch_messages, gen_config=gen_config, do_preprocess=True)
                batch_elapsed = time.perf_counter() - infer_start
                responses = responses if isinstance(responses, list) else [responses]
                batch_elapsed_per_item = batch_elapsed / len(batch_samples)
                batches.append({
                    'repeat': repeat,
                    'batch_index': batch_index,
                    'size': len(batch_samples),
                    'elapsed_sec': round(batch_elapsed, 3),
                    'elapsed_per_item_sec': round(batch_elapsed_per_item, 3),
                })
                for sample, response in zip(batch_samples, responses):
                    item = _make_item(sample, repeat, response, batch_elapsed_per_item, batch_index=batch_index)
                    if not item['ok']:
                        item['error'] = f"pipeline returned error response: {item['response'].get('text')}"
                    items.append(item)

        summary = _summarize_items(items, batches)
        result.update({
            'ok': summary['ok_count'] == summary['total_count'],
            'items': items,
            'batches': batches,
            'summary': summary,
            'cuda_after_infer': _cuda_stats(),
        })
    except Exception as exc:  # noqa: BLE001
        result.update({
            'elapsed_sec': round(time.perf_counter() - load_start, 3),
            'error': repr(exc),
            'traceback': traceback.format_exc(),
            'cuda': _cuda_stats(),
        })
    finally:
        if pipe is not None:
            pipe.close()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Smoke test Qwen3-ASR with LMDeploy.')
    parser.add_argument('--model-path', default=_default_model_path())
    parser.add_argument('--audio', default=_default_audio_path())
    parser.add_argument('--expected')
    parser.add_argument('--manifest')
    parser.add_argument('--backend', choices=['pytorch', 'turbomind', 'both'], default='pytorch')
    parser.add_argument('--dtype', choices=['auto', 'float16', 'bfloat16'], default='float16')
    parser.add_argument('--session-len', type=int, default=2048)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.2)
    parser.add_argument('--max-prefill-token-num', type=int, default=1024)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--vision-max-batch-size', type=int, default=1)
    parser.add_argument('--vision-thread-safe', action='store_true')
    parser.add_argument('--system', default=None)
    parser.add_argument('--log-level', default=os.environ.get('LMDEPLOY_LOG_LEVEL', 'WARNING'))
    parser.add_argument('--output')
    parser.add_argument('--tm-lib-path', default=os.environ.get('LMDEPLOY_TURBOMIND_LIB_PATH'))
    parser.add_argument('--stdout-summary', action='store_true')
    if Path(_default_manifest_path()).exists():
        parser.set_defaults(manifest=_default_manifest_path())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backends = ['pytorch', 'turbomind'] if args.backend == 'both' else [args.backend]
    results = [run_backend(backend, args) for backend in backends]
    output = json.dumps(results, ensure_ascii=False, indent=2)
    if args.stdout_summary:
        print(
            json.dumps(
                [{
                    'backend': result.get('backend'),
                    'ok': result.get('ok'),
                    'load_elapsed_sec': result.get('load_elapsed_sec'),
                    'summary': result.get('summary'),
                    'error': result.get('error'),
                } for result in results],
                ensure_ascii=False,
                indent=2,
            ))
    else:
        print(output)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + '\n', encoding='utf-8')
    return 0 if all(result.get('ok') for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
