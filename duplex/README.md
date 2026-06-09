# Duplex LMDeploy Gateway

This package integrates the duplex business decision logic with an LMDeploy
TurboMind gRPC backend through a Sanic server.

## Request Flow

1. `POST /infer` receives the original duplex request schema.
2. The Sanic handler calls `duplex.full_duplex.get_duplex_response`.
3. `DuplexLmdeployClient.infer(..., decoding_type=0)` maps to TurboMind
   `infer_type=0` for the validity decision.
4. If the validity decision is `<valid>`, `decoding_type=1` maps to TurboMind
   `infer_type=1` for the completion decision.
5. Both decision calls use `max_new_tokens=1` by default.

`POST /infer/end_turn` preserves the original turn-end history flushing logic.

## Important Environment

Sanic gateway:

```bash
export DUPLEX_HOST=0.0.0.0
export DUPLEX_PORT=18080
export DUPLEX_MAX_CONCURRENCY=50
export DUPLEX_GRPC_CLIENT_POOL_SIZE=1
export DUPLEX_GRPC_CLIENT_CHANNELS=50
export DUPLEX_REQUEST_TIMEOUT=0.25
export DUPLEX_MAX_NEW_TOKENS=1
```

LMDeploy gRPC target and token decisions:

```bash
export DUPLEX_GRPC_TARGET=127.0.0.1:50051
export DUPLEX_VALID_ID="${TM_VALID_ID}"
export DUPLEX_INVALID_ID="${TM_INVALID_ID}"
export DUPLEX_END_ID="${TM_END_ID}"
export DUPLEX_CERTAINTY_THRESHOLD=0.05
export DUPLEX_COMPLETION_THRESHOLD=0.1
export DUPLEX_INVALID_BIAS=0.1
```

When the gRPC server already has `TM_VALID_ID`, `TM_INVALID_ID`, and
`TM_END_ID` configured, the gateway can inherit those values directly.

## Run

Start the TurboMind gRPC server first, then run:

```bash
python -m duplex.server --host 0.0.0.0 --port 18080
```

Pressure test the Sanic endpoint:

```bash
python tests/test_lmdeploy/duplex_sanic_pressure.py \
  --url http://127.0.0.1:18080/infer \
  --requests 1000 \
  --concurrency 50 \
  --channels 50 \
  --warmup 100 \
  --repeat 3 \
  --rate-qps 250 \
  --min-chars 1 \
  --max-chars 256 \
  --top-slow 5
```
