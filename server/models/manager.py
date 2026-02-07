"""Model manager — loads all models at startup for deterministic VRAM usage."""

import logging
import time

from server.models.asr import ASR
from server.models.llm import LLM
from server.models.tts import TTS
from server.models.vad import VAD

log = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.vad: VAD | None = None
        self.asr: ASR | None = None
        self.llm: LLM | None = None
        self.tts: TTS | None = None

    def load_all(self):
        """Load all models eagerly. Call at server startup."""
        t0 = time.time()
        log.info("Loading all models...")

        self.vad = VAD()
        log.info("VAD loaded.")

        self.asr = ASR()
        log.info("ASR loaded.")

        self.llm = LLM()
        log.info("LLM loaded.")

        self.tts = TTS()
        log.info("TTS loaded.")

        elapsed = time.time() - t0
        log.info("All models loaded in %.1fs.", elapsed)

        self._log_gpu_usage()

    def _log_gpu_usage(self):
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                log.info("GPU memory: %.1f GB allocated, %.1f GB reserved", allocated, reserved)
        except Exception:
            pass


models = ModelManager()
