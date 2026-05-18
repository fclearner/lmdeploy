from typing import Any

from google.protobuf import json_format, struct_pb2


GRPC_SERVICE_NAME = "lmdeploy.turbomind.TurboMindService"


def dict_to_struct(data: dict[str, Any]) -> struct_pb2.Struct:
    msg = struct_pb2.Struct()
    json_format.ParseDict(data, msg)
    return msg


def struct_to_dict(msg: struct_pb2.Struct) -> dict[str, Any]:
    return json_format.MessageToDict(msg, preserving_proto_field_name=True)
