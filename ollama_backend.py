# ollama_backend.py — Ollama 本地后端（独立模块，按需启用）
# ============================================================================
# 2026-09-06 从 llm_provider.py 分离。默认不启用：
#   config.py "enable_ollama": false 时本模块不会被导入，
#   provider "ollama" 在所有入口不可见/不可选。
# 启用方式：config.py 改 "enable_ollama": true 后重启。
# ============================================================================

import os
import logging
from llm_provider import LLMBackend, BoundBackend

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""

    def __init__(self, model: str = "qwen2.5:7b-instruct-q5_k_m",
                 temperature: float = 0.1, format: str = "json",
                 num_predict: int = 1024, top_p: float = 0.9):
        from langchain_ollama import ChatOllama
        self._model = model
        self._temperature = temperature
        self._format = format
        self._num_predict = num_predict
        self._top_p = top_p
        self._llm = ChatOllama(
            model=model, temperature=temperature,
            format=format, num_predict=num_predict, top_p=top_p
        )

    def invoke(self, prompt) -> str:
        return self._llm.invoke(prompt)

    def bind(self, **kwargs):
        return BoundBackend(self, kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    def with_temperature(self, temp: float) -> "OllamaBackend":
        """Create a variant with different temperature (for expansion tasks)."""
        from langchain_ollama import ChatOllama
        clone = OllamaBackend.__new__(OllamaBackend)
        clone._model = self._model
        clone._temperature = temp
        clone._format = self._format
        clone._num_predict = self._num_predict
        clone._top_p = self._top_p
        clone._llm = ChatOllama(
            model=self._model, temperature=temp,
            format=self._format, num_predict=self._num_predict, top_p=self._top_p
        )
        return clone
