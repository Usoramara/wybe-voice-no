"""NB-Whisper ASR wrapper using faster-whisper.

Loads NbAiLab/nb-whisper-large-distil-turbo-beta with CTranslate2 backend
for fast Norwegian speech-to-text. Auto-converts from HuggingFace format
on first load.
"""

import logging

import numpy as np
from faster_whisper import WhisperModel

from server.config import settings

log = logging.getLogger(__name__)


class ASR:
    def __init__(self):
        log.info("Loading ASR model: %s (compute_type=%s)", settings.asr_model, settings.asr_compute_type)
        self.model = WhisperModel(
            settings.asr_model,
            device="cuda",
            compute_type=settings.asr_compute_type,
        )
        log.info("ASR model loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to Norwegian text.

        Args:
            audio: float32 numpy array, mono, 16kHz

        Returns:
            Transcribed text string.
        """
        segments, info = self.model.transcribe(
            audio,
            language=settings.asr_language,
            beam_size=settings.asr_beam_size,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        log.debug("ASR result (lang=%s, prob=%.2f): %s", info.language, info.language_probability, text)
        return text
