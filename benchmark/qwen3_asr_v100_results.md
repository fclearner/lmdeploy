# Qwen3-ASR V100 Validation Results

This note records the validation evidence for Qwen3-ASR-0.6B support in LMDeploy.
Raw benchmark JSON files, cached models, and AISHELL-1 audio samples are intentionally
left out of git to avoid committing local artifacts and dataset files.

## Setup

- Model: Qwen3-ASR-0.6B
- Dataset: AISHELL-1 test subset, 100 utterances
- GPU: Tesla V100-PCIE-32GB
- Batch size: 8
- Max new tokens: 64

Audio throughput is reported as `1 / audio_rtf_wall`, equivalent to processed
audio seconds per wall-clock second.

## Final Results

| Backend | Repeats | Samples | OK | CER | WER | Requests/s | Audio RTF | Audio throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboMind native audio | 8 | 800 | 800/800 | 0.0175 | 0.0 | 38.905 | 0.0055 | 181.8x realtime |
| PyTorch | 4 | 400 | 400/400 | 0.0175 | 0.0 | 27.209 | 0.0078 | 128.2x realtime |

The final TurboMind native audio run matched the PyTorch CER and output text on
the repeated AISHELL-1 subset. Each of the 8 TurboMind repeats had `diffs=0`
against PyTorch repeat 0 output text and CER.

## Regression Controls

- Before greedy decoding mapping, native TurboMind `100 x 4`, `bs=8` produced
  `CER=0.0931`.
- With greedy decoding mapping but a shared `LlamaLinear` workspace, native
  TurboMind `bs=8` still drifted, for example `CER=0.0207` and `CER=0.0197`.
- `CUDA_LAUNCH_BLOCKING=1` restored `CER=0.0175`, which isolated the issue to
  asynchronous workspace overlap rather than model conversion or data handling.
- True hybrid mode with `LMDEPLOY_QWEN3_ASR_NATIVE_AUDIO=0` was stable at
  `CER=0.0175`, further isolating the remaining regression to the native audio
  tower.

## Key Fixes

- `lmdeploy/turbomind/turbomind.py` maps `GenerationConfig(do_sample=False)` to
  TurboMind greedy settings: `top_k=1`, `top_p=0.0`, `min_p=0.0`, and
  `temperature=1.0`.
- `src/turbomind/models/llama/Qwen3AsrAudioTower.{h,cu}` owns a dedicated
  `LlamaLinear` instance and workspace instead of reusing the main LLM
  `LlamaLinear&`, preventing scheduler-thread audio tower GEMM workspace overlap
  with model execution.
- `src/turbomind/kernels/core/sub_byte_ptr.h` uses `TM_HOST_DEVICE` for CUDA
  12.1 and V100 build compatibility.

## Reproduction Commands

Build TurboMind for V100:

```bash
cmake --build build/v100_sm70 --config Release -j 4
cmake --install build/v100_sm70 --config Release
```

Run the final TurboMind native audio benchmark:

```bash
PYTHONPATH=/path/to/lmdeploy \
LMDEPLOY_QWEN3_ASR_NATIVE_AUDIO=1 \
python benchmark/qwen3_asr_smoke.py \
  --backend turbomind \
  --manifest artifacts/aishell1_test/manifest.json \
  --limit 100 \
  --repeat 8 \
  --batch-size 8 \
  --max-new-tokens 64 \
  --cache-max-entry-count 0.2 \
  --output artifacts/aishell1_test/v100_tm_native_100x8_bs8_greedyfix_ownlinear_clean.json \
  --stdout-summary
```

Run the PyTorch baseline:

```bash
PYTHONPATH=/path/to/lmdeploy \
python benchmark/qwen3_asr_smoke.py \
  --backend pytorch \
  --manifest artifacts/aishell1_test/manifest.json \
  --limit 100 \
  --repeat 4 \
  --batch-size 8 \
  --max-new-tokens 64 \
  --output artifacts/aishell1_test/v100_pytorch_100x4_bs8_tf4576.json \
  --stdout-summary
```

## Additional Checks

- `git diff --check` passed on the touched TurboMind files.
- `python -m compileall -q lmdeploy/turbomind/turbomind.py lmdeploy/vl/model/qwen3_asr.py benchmark/qwen3_asr_smoke.py` passed.
- Local `pytest tests/test_lmdeploy/test_vl/test_qwen3_asr_processor.py` could
  not collect because the local Python environment lacked `mmengine`.
- Remote pytest could not run because the remote conda environment lacked
  `pytest`.
