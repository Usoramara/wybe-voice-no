"""Binary WebSocket message protocol.

Each message: 1 byte type + payload.

Types:
  0x01  HANDSHAKE     client→server  JSON config
  0x02  AUDIO_IN      client→server  raw audio bytes (WebM/Opus from browser)
  0x03  AUDIO_OUT     server→client  PCM s16le audio chunk
  0x04  TEXT_ASR      server→client  JSON {"text": "transcribed text"}
  0x05  TEXT_LLM      server→client  JSON {"text": "token", "done": false}
  0x06  VAD_EVENT     server→client  JSON {"event": "speech_start"|"speech_end"}
  0x07  ERROR         server→client  JSON {"error": "message"}
  0x08  STATUS        server→client  JSON {"status": "ready"|"listening"|"thinking"|"speaking"}
"""

import json
import struct
from enum import IntEnum


class MsgType(IntEnum):
    HANDSHAKE = 0x01
    AUDIO_IN = 0x02
    AUDIO_OUT = 0x03
    TEXT_ASR = 0x04
    TEXT_LLM = 0x05
    VAD_EVENT = 0x06
    ERROR = 0x07
    STATUS = 0x08


def pack_binary(msg_type: MsgType, payload: bytes) -> bytes:
    return struct.pack("B", msg_type) + payload


def pack_json(msg_type: MsgType, data: dict) -> bytes:
    return pack_binary(msg_type, json.dumps(data).encode())


def unpack(raw: bytes) -> tuple[MsgType, bytes]:
    msg_type = MsgType(raw[0])
    payload = raw[1:]
    return msg_type, payload


def status_msg(status: str) -> bytes:
    return pack_json(MsgType.STATUS, {"status": status})


def asr_msg(text: str) -> bytes:
    return pack_json(MsgType.TEXT_ASR, {"text": text})


def llm_msg(text: str, done: bool = False) -> bytes:
    return pack_json(MsgType.TEXT_LLM, {"text": text, "done": done})


def vad_msg(event: str) -> bytes:
    return pack_json(MsgType.VAD_EVENT, {"event": event})


def audio_out_msg(pcm_bytes: bytes) -> bytes:
    return pack_binary(MsgType.AUDIO_OUT, pcm_bytes)


def error_msg(message: str) -> bytes:
    return pack_json(MsgType.ERROR, {"error": message})
