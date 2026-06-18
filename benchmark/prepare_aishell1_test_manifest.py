# Copyright (c) OpenMMLab. All rights reserved.
"""Prepare an AISHELL-1 test manifest for Qwen3-ASR benchmarking."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_cache_dir() -> str:
    return str(_repo_root() / '.cache_models' / 'datasets' / 'aishell_1_zh_test')


def _default_out_dir() -> str:
    return str(_repo_root() / 'artifacts' / 'aishell1_test')


def _audio_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path)) as f:
            return round(f.getnframes() / float(f.getframerate()), 3)
    except Exception:
        return None


def _iter_test_shards(repo_id: str, revision: str | None) -> list[str]:
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    shards = sorted(
        sibling.rfilename for sibling in info.siblings
        if sibling.rfilename.startswith('data/test-') and sibling.rfilename.endswith('.parquet'))
    if not shards:
        raise RuntimeError(f'No test parquet shards found in dataset {repo_id!r}')
    return shards


def _write_audio(audio_bytes: bytes, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)


def prepare_manifest(args) -> dict:
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    wav_dir = out_dir / 'wav'
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else out_dir / 'manifest.json'
    metadata_path = out_dir / 'metadata.json'

    out_dir.mkdir(parents=True, exist_ok=True)
    shards = _iter_test_shards(args.repo_id, args.revision)
    rows = []
    seen = 0
    take_limit = None if args.limit <= 0 else args.limit

    for shard in shards:
        parquet_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type='dataset',
            filename=shard,
            revision=args.revision,
            local_dir=str(cache_dir),
        )
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=args.batch_size):
            for record in batch.to_pylist():
                if seen < args.start:
                    seen += 1
                    continue
                if take_limit is not None and len(rows) >= take_limit:
                    break

                audio = record.get('context') or {}
                audio_bytes = audio.get('bytes')
                if not audio_bytes:
                    raise RuntimeError(f'Missing audio bytes at source row {seen}')

                sample_id = f'aishell1_test_{seen:06d}'
                audio_path = wav_dir / f'{sample_id}.wav'
                _write_audio(audio_bytes, audio_path, args.force)

                rows.append({
                    'id': sample_id,
                    'audio': str(audio_path.relative_to(manifest_path.parent)),
                    'expected': record.get('answer') or '',
                    'duration_sec': _audio_duration(audio_path),
                    'source_index': seen,
                })
                seen += 1
            if take_limit is not None and len(rows) >= take_limit:
                break
        if take_limit is not None and len(rows) >= take_limit:
            break

    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    metadata = {
        'repo_id': args.repo_id,
        'revision': args.revision,
        'cache_dir': str(cache_dir),
        'manifest': str(manifest_path),
        'out_dir': str(out_dir),
        'start': args.start,
        'limit': args.limit,
        'num_samples': len(rows),
        'shards': shards,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='AudioLLMs/aishell_1_zh_test')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--cache-dir', default=_default_cache_dir())
    parser.add_argument('--out-dir', default=_default_out_dir())
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=100, help='Use <= 0 to prepare the full test split.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = prepare_manifest(args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
