"""NB-Whisper ASR wrapper using faster-whisper.

Loads NbAiLab/nb-whisper-large-distil-turbo-beta with CTranslate2 backend.
The model must be converted to CT2 format first (done by setup script or
auto-converted on first load if ctranslate2 is installed).
"""

import logging
import os
import subprocess
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from server.config import settings

log = logging.getLogger(__name__)

# Default CT2 model directory (on RunPod persistent volume or local)
CT2_MODEL_DIR = os.environ.get(
    "ASR_CT2_DIR",
    os.path.join(os.environ.get("HF_HOME", "/root/.cache"), "nb-whisper-ct2"),
)


def _ensure_ct2_model(model_id: str, output_dir: str, compute_type: str) -> str:
    """Convert HuggingFace Whisper model to CTranslate2 format if not already done."""
    output_path = Path(output_dir)
    if (output_path / "model.bin").exists():
        log.info("CT2 model already exists at %s", output_dir)
        return output_dir

    log.info("Converting %s to CTranslate2 format at %s...", model_id, output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    quantization = "float16" if "float" in compute_type else "int8"

    cmd = [
        "ct2-transformers-converter",
        "--model", model_id,
        "--output_dir", output_dir,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--quantization", quantization,
    ]

    token = settings.hf_token
    if token:
        cmd.extend(["--model_token", token])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CT2 conversion failed: {result.stderr}")

    log.info("CT2 conversion complete.")
    return output_dir


class ASR:
    def __init__(self):
        log.info("Loading ASR model: %s (compute_type=%s)", settings.asr_model, settings.asr_compute_type)

        # Ensure CT2 model is available
        ct2_path = _ensure_ct2_model(
            settings.asr_model,
            CT2_MODEL_DIR,
            settings.asr_compute_type,
        )

        self.model = WhisperModel(
            ct2_path,
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
