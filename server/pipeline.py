"""Conversation pipeline: VAD → ASR → LLM → TTS orchestration.

Each WebSocket connection gets a ConversationSession that manages
the full audio-in → text → audio-out loop.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator

import numpy as np

from server.audio import decode_webm_opus, pcm_f32_to_s16le
from server.config import settings
from server.models.manager import models
from server.protocol import (
    asr_msg,
    audio_out_msg,
    error_msg,
    llm_msg,
    status_msg,
    vad_msg,
)

log = logging.getLogger(__name__)

# Sentence boundary pattern for TTS flushing
SENTENCE_END = re.compile(r"[.!?;:]\s*$")


class ConversationSession:
    """Manages a single voice conversation over WebSocket."""

    def __init__(self, send_fn):
        """
        Args:
            send_fn: async callable that sends bytes over WebSocket.
        """
        self.send = send_fn
        self.history: list[dict[str, str]] = [
            {"role": "system", "content": settings.system_prompt}
        ]
        self._audio_buffer: list[bytes] = []

    async def handle_audio(self, data: bytes):
        """Process incoming audio data from the browser.

        Decodes WebM/Opus, runs VAD, and triggers the conversation
        pipeline when speech ends.
        """
        try:
            audio = decode_webm_opus(data, target_sr=settings.sample_rate)
        except RuntimeError as e:
            log.warning("Audio decode failed: %s", e)
            await self.send(error_msg(f"Audio decode error: {e}"))
            return

        vad_result = models.vad.process_chunk(audio)

        if vad_result is None:
            return

        if vad_result["event"] == "speech_start":
            await self.send(vad_msg("speech_start"))
            await self.send(status_msg("listening"))

        elif vad_result["event"] == "speech_end":
            await self.send(vad_msg("speech_end"))
            speech_audio = vad_result["audio"]
            # Process the complete utterance
            await self._process_utterance(speech_audio)

    async def _process_utterance(self, audio: np.ndarray):
        """Run ASR → LLM → TTS on a complete speech segment."""
        # Step 1: ASR
        await self.send(status_msg("thinking"))
        text = await asyncio.get_event_loop().run_in_executor(
            None, models.asr.transcribe, audio
        )

        if not text.strip():
            await self.send(status_msg("ready"))
            return

        log.info("User: %s", text)
        await self.send(asr_msg(text))

        # Add to conversation history
        self.history.append({"role": "user", "content": text})

        # Step 2: LLM streaming → Step 3: TTS at sentence boundaries
        await self.send(status_msg("speaking"))

        full_response = ""
        sentence_buffer = ""

        for token in models.llm.generate_stream(self.history):
            full_response += token
            sentence_buffer += token
            await self.send(llm_msg(token))

            # Flush to TTS at sentence boundaries
            if SENTENCE_END.search(sentence_buffer):
                await self._synthesize_and_send(sentence_buffer.strip())
                sentence_buffer = ""

        # Flush remaining text
        if sentence_buffer.strip():
            await self._synthesize_and_send(sentence_buffer.strip())

        await self.send(llm_msg("", done=True))

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": full_response})
        log.info("Wybe: %s", full_response)

        await self.send(status_msg("ready"))

    async def _synthesize_and_send(self, text: str):
        """Synthesize text to audio and send over WebSocket."""
        if not text:
            return

        def _synth():
            chunks = []
            for chunk in models.tts.synthesize_stream(text):
                chunks.append(chunk)
            return chunks

        audio_chunks = await asyncio.get_event_loop().run_in_executor(None, _synth)

        for chunk in audio_chunks:
            pcm_bytes = pcm_f32_to_s16le(chunk)
            await self.send(audio_out_msg(pcm_bytes))
