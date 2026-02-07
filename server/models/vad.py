"""Silero VAD wrapper — detects speech boundaries in audio stream."""

import numpy as np
import torch


class VAD:
    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 250, min_silence_ms: int = 700):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.sample_rate = 16000

        self.model, self.utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self.model.eval()

        self._is_speaking = False
        self._speech_buffer: list[np.ndarray] = []
        self._silence_samples = 0
        self._speech_samples = 0

    def reset(self):
        self.model.reset_states()
        self._is_speaking = False
        self._speech_buffer = []
        self._silence_samples = 0
        self._speech_samples = 0

    def process_chunk(self, audio: np.ndarray) -> dict | None:
        """Process an audio chunk (float32, 16kHz, mono).

        Returns:
            None if no event,
            {"event": "speech_start"} when speech begins,
            {"event": "speech_end", "audio": np.ndarray} when speech ends (includes full utterance).
        """
        tensor = torch.from_numpy(audio).float()
        # Silero VAD expects chunks of 512 samples at 16kHz
        chunk_size = 512
        result = None

        for i in range(0, len(tensor), chunk_size):
            chunk = tensor[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))

            prob = self.model(chunk, self.sample_rate).item()

            if prob >= self.threshold:
                self._silence_samples = 0
                self._speech_samples += len(chunk)

                if not self._is_speaking:
                    min_samples = int(self.min_speech_ms * self.sample_rate / 1000)
                    if self._speech_samples >= min_samples:
                        self._is_speaking = True
                        result = {"event": "speech_start"}

                if self._is_speaking:
                    self._speech_buffer.append(audio[i : i + chunk_size])
            else:
                if self._is_speaking:
                    self._silence_samples += len(chunk)
                    self._speech_buffer.append(audio[i : i + chunk_size])

                    min_silence = int(self.min_silence_ms * self.sample_rate / 1000)
                    if self._silence_samples >= min_silence:
                        full_audio = np.concatenate(self._speech_buffer)
                        result = {"event": "speech_end", "audio": full_audio}
                        self._is_speaking = False
                        self._speech_buffer = []
                        self._silence_samples = 0
                        self._speech_samples = 0
                else:
                    self._speech_samples = 0

        return result
