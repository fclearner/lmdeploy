#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from google.protobuf import json_format, struct_pb2


GRPC_SERVICE = "lmdeploy.turbomind.TurboMindService"


def dict_to_struct(data: dict) -> struct_pb2.Struct:
    msg = struct_pb2.Struct()
    json_format.ParseDict(data, msg)
    return msg


def struct_to_dict(msg: struct_pb2.Struct) -> dict:
    return json_format.MessageToDict(msg, preserving_proto_field_name=True)


async def amain(args: argparse.Namespace) -> int:
    import grpc

    channel = grpc.aio.insecure_channel(args.target)
    try:
        generate = channel.unary_unary(
            f"/{GRPC_SERVICE}/Generate",
            request_serializer=struct_pb2.Struct.SerializeToString,
            response_deserializer=struct_pb2.Struct.FromString,
        )
        payload = {
            "request_id": f"check-grpc-once-{uuid.uuid4()}",
            "prompt": args.prompt,
            "max_new_tokens": args.max_tokens,
            "infer_type": args.infer_type,
            "include_text": args.include_text,
            "include_token_ids": True,
        }
        response = await generate(dict_to_struct(payload), timeout=args.timeout_sec)
        body = struct_to_dict(response)
        print(json.dumps(body, ensure_ascii=False, sort_keys=True))
        if args.expected_token_id is not None:
            token_ids = body.get("token_ids") or []
            first_token = int(token_ids[0]) if token_ids else None
            if first_token != args.expected_token_id:
                print(
                    f"expected first token {args.expected_token_id}, got {first_token}",
                    file=sys.stderr,
                )
                return 1
        return 0
    finally:
        await channel.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--prompt", default="请判断这句话是否结束，只返回一个 token。")
    parser.add_argument("--infer-type", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--expected-token-id", type=int)
    raise SystemExit(asyncio.run(amain(parser.parse_args())))


if __name__ == "__main__":
    main()
