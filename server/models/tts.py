"""Chatterbox TTS wrapper for Norwegian speech synthesis.

Loads the Norwegian fine-tuned Chatterbox model (akhbar/chatterbox-tts-norwegian)
via from_local() after downloading weights from HuggingFace.
The Norwegian model doesn't handle long text well, so we split at sentence boundaries.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from server.config import settings

log = logging.getLogger(__name__)

# Files required for the Norwegian fine-tuned model
MODEL_FILES = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]


class TTS:
    def __init__(self):
        log.info("Loading Norwegian TTS model: %s", settings.tts_model)
        from chatterbox.tts import ChatterboxTTS

        # Download model files from HuggingFace
        local_path = None
        for fname in MODEL_FILES:
            local_path = hf_hub_download(
                repo_id=settings.tts_model,
                filename=fname,
                token=settings.hf_token or None,
            )

        # Load from the directory containing the downloaded files
        model_dir = Path(local_path).parent
        log.info("Loading TTS from local dir: %s", model_dir)
        self.model = ChatterboxTTS.from_local(model_dir, device="cuda")
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

        Uses generate_stream if available (chatterbox-streaming fork),
        otherwise falls back to batch synthesis.
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
