"""
Wire protocol for coordinator <-> volunteer communication.

Messages are length-prefixed: 4-byte big-endian payload length, then the payload.
The payload is msgpack for metadata + raw tensor bytes appended after.

Message types:
  coordinator -> volunteer:
    WELCOME         — model config, full state_dict, current iter_num
    BATCH           — token batch (x, y) for the volunteer to train on
    WEIGHT_UPDATE   — updated state_dict after aggregation
    HEARTBEAT_PING  — keepalive check
    SHUTDOWN        — graceful disconnect

  volunteer -> coordinator:
    REGISTER        — volunteer capabilities (device, dtype, etc.)
    GRADIENTS       — computed gradients for a batch
    HEARTBEAT_PONG  — keepalive response
    STATUS          — volunteer status update (loss, compute time)
"""

import io
import struct
import zlib
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any

import torch
import msgpack


class MsgType(IntEnum):
    WELCOME = 1
    BATCH = 2
    WEIGHT_UPDATE = 3
    HEARTBEAT_PING = 4
    SHUTDOWN = 5
    REGISTER = 10
    GRADIENTS = 11
    HEARTBEAT_PONG = 12
    STATUS = 13


def _serialize_state_dict(state_dict: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return zlib.compress(buf.getvalue(), level=1)


def _deserialize_state_dict(data: bytes) -> dict:
    raw = zlib.decompress(data)
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location='cpu', weights_only=True)


def _serialize_tensors(tensors: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(tensors, buf)
    return zlib.compress(buf.getvalue(), level=1)


def _deserialize_tensors(data: bytes) -> dict:
    raw = zlib.decompress(data)
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location='cpu', weights_only=True)


def encode_message(msg_type: MsgType, metadata: dict, tensor_payload: bytes = b'') -> bytes:
    meta_bytes = msgpack.packb(metadata, use_bin_type=True)
    header = struct.pack('>BII', int(msg_type), len(meta_bytes), len(tensor_payload))
    return header + meta_bytes + tensor_payload


def decode_message(data: bytes) -> tuple:
    msg_type_val, meta_len, tensor_len = struct.unpack('>BII', data[:9])
    msg_type = MsgType(msg_type_val)
    metadata = msgpack.unpackb(data[9:9 + meta_len], raw=False)
    tensor_payload = data[9 + meta_len:9 + meta_len + tensor_len]
    return msg_type, metadata, tensor_payload


async def send_msg(writer, msg_type: MsgType, metadata: dict, tensor_payload: bytes = b''):
    payload = encode_message(msg_type, metadata, tensor_payload)
    length_prefix = struct.pack('>I', len(payload))
    writer.write(length_prefix + payload)
    await writer.drain()


async def recv_msg(reader) -> tuple:
    length_bytes = await reader.readexactly(4)
    length = struct.unpack('>I', length_bytes)[0]
    payload = await reader.readexactly(length)
    return decode_message(payload)
