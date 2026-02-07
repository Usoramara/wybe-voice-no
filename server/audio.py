"""Audio decoding utilities — WebM/Opus from browser to PCM numpy arrays."""

import io
import subprocess
import tempfile

import numpy as np


def decode_webm_opus(data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode WebM/Opus audio bytes to mono float32 numpy array at target sample rate.

    Uses ffmpeg for reliable WebM/Opus decoding.
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        result = subprocess.run(
            [
                "ffmpeg",
                "-i", tmp.name,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(target_sr),
                "-loglevel", "error",
                "pipe:1",
            ],
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()}")

        pcm = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return pcm


def pcm_f32_to_s16le(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] numpy array to s16le bytes for browser playback."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def concat_audio_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate list of audio arrays into a single array."""
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)
