"""FastAPI application with WebSocket voice conversation endpoint."""

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import settings
from server.models.manager import models
from server.pipeline import ConversationSession
from server.protocol import MsgType, error_msg, status_msg, unpack

log = logging.getLogger(__name__)

app = FastAPI(title="Wybe Voice NO", version="0.1.0")


@app.on_event("startup")
async def startup():
    models.load_all()


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "models_loaded": models.tts is not None})


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws/conversation")
async def ws_conversation(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected: %s", ws.client)

    async def send_bytes(data: bytes):
        await ws.send_bytes(data)

    session = ConversationSession(send_fn=send_bytes)
    await ws.send_bytes(status_msg("ready"))

    try:
        while True:
            data = await ws.receive_bytes()
            msg_type, payload = unpack(data)

            if msg_type == MsgType.AUDIO_IN:
                await session.handle_audio(payload)
            elif msg_type == MsgType.HANDSHAKE:
                log.info("Handshake received")
                await ws.send_bytes(status_msg("ready"))
            else:
                log.warning("Unknown message type: %s", msg_type)

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", ws.client)
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        try:
            await ws.send_bytes(error_msg(str(e)))
        except Exception:
            pass
