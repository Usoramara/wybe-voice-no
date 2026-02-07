"""Configuration for Wybe Voice NO — all settings with env var overrides."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # HuggingFace
    hf_token: str = ""

    # ASR — Whisper large-v3-turbo (great Norwegian support, pre-built CT2)
    asr_model: str = "large-v3-turbo"
    asr_compute_type: str = "float16"
    asr_beam_size: int = 1
    asr_language: str = "no"

    # LLM — NorMistral
    llm_model: str = "norallm/normistral-7b-warm-instruct"
    llm_gguf_file: str = "normistral-7b-warm-instruct.Q4_K_M.gguf"
    llm_gpu_layers: int = -1  # -1 = all layers on GPU
    llm_context_length: int = 4096
    llm_max_tokens: int = 256
    llm_temperature: float = 0.3
    llm_repeat_penalty: float = 1.0  # Disabled — NorMistral is sensitive to repeat penalty
    llm_top_p: float = 0.9

    # TTS — Chatterbox Norwegian
    tts_model: str = "akhbar/chatterbox-tts-norwegian"
    tts_speaker_wav: str = ""  # Path to reference speaker wav for voice cloning
    tts_exaggeration: float = 1.0  # Norwegian model works best at 1.0
    tts_cfg_weight: float = 0.5

    # VAD — Silero
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 700

    # Pipeline
    sample_rate: int = 16000
    tts_sample_rate: int = 24000

    # System prompt
    system_prompt: str = (
        "Du er en vennlig og hjelpsom norsk assistent som heter Wybe. "
        "Du svarer alltid på norsk. Hold svarene korte og naturlige, "
        "som i en vanlig samtale. Bruk bokmål."
    )


settings = Settings()
