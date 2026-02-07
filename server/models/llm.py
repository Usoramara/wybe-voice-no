"""NorMistral LLM wrapper using llama-cpp-python.

Loads norallm/normistral-7b-warm-instruct Q4_K_M GGUF for Norwegian
conversation with streaming token generation. Uses ChatML format.
"""

import logging
from collections.abc import Iterator

from llama_cpp import Llama

from server.config import settings

log = logging.getLogger(__name__)

# ChatML stop token
STOP_TOKENS = ["<|im_end|>"]


class LLM:
    def __init__(self):
        log.info("Loading LLM: %s (%s)", settings.llm_model, settings.llm_gguf_file)
        self.model = Llama.from_pretrained(
            repo_id=settings.llm_model,
            filename=settings.llm_gguf_file,
            n_gpu_layers=settings.llm_gpu_layers,
            n_ctx=settings.llm_context_length,
            verbose=False,
        )
        log.info("LLM loaded (%d GPU layers).", settings.llm_gpu_layers)

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream tokens from NorMistral given a conversation history.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."} dicts.

        Yields:
            Token strings as they are generated.
        """
        stream = self.model.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            repeat_penalty=settings.llm_repeat_penalty,
            stop=STOP_TOKENS,
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a complete response (non-streaming)."""
        return "".join(self.generate_stream(messages))
