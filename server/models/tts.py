"""Chatterbox TTS wrapper for Norwegian speech synthesis.

Uses the Chatterbox multilingual model with Norwegian language support.
Supports both batch and streaming generation.
"""

import logging

import numpy as np
import torch

from server.config import settings

log = logging.getLogger(__name__)


class TTS:
    def __init__(self):
        log.info("Loading TTS model (multilingual with Norwegian support)...")
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        self.sr = self.model.sr  # 24000 Hz
        log.info("TTS model loaded (sr=%d).", self.sr)

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize Norwegian text to audio.

        Args:
            text: Norwegian text to synthesize.

        Returns:
            float32 numpy array at 24kHz.
        """
        kwargs = {
            "exaggeration": settings.tts_exaggeration,
            "cfg_weight": settings.tts_cfg_weight,
        }
        if settings.tts_speaker_wav:
            kwargs["audio_prompt_path"] = settings.tts_speaker_wav

        wav = self.model.generate(text, **kwargs)

        # wav is a torch tensor [1, T]
        audio = wav.squeeze(0).cpu().numpy()
        return audio

    def synthesize_stream(self, text: str):
        """Stream audio chunks for the given text.

        Yields (audio_chunk, metrics) tuples where audio_chunk is a torch tensor.
        Falls back to batch synthesis if streaming is not available.
        """
        kwargs = {
            "exaggeration": settings.tts_exaggeration,
            "cfg_weight": settings.tts_cfg_weight,
        }
        if settings.tts_speaker_wav:
            kwargs["audio_prompt_path"] = settings.tts_speaker_wav

        if hasattr(self.model, "generate_stream"):
            for audio_chunk, metrics in self.model.generate_stream(text, **kwargs):
                yield audio_chunk.squeeze(0).cpu().numpy()
        else:
            # Fallback: return the whole audio as a single chunk
            yield self.synthesize(text)
