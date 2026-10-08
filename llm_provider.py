# llm_provider.py — Multi-provider LLM backend for Fascinator
# ============================================================================
# Providers: MiMo / DeepSeek (cloud); Ollama 分离至 ollama_backend.py（按需启用）.
# All backends expose a consistent interface: invoke(prompt) -> str
# ============================================================================

import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    @abstractmethod
    def invoke(self, prompt) -> str:
        """Invoke the LLM with a LangChain prompt and return raw output."""
        pass

    @abstractmethod
    def bind(self, **kwargs):
        """Return a copy of this backend with bound kwargs (for JSON mode etc)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class OpenAICompatBackend(LLMBackend):
    """OpenAI 兼容协议 API 后端基类（raw openai client，不依赖 langchain_openai）。

    子类通过类属性声明默认模型 / Base URL / 环境变量名。
    """
    _DEFAULT_MODEL = ""
    _DEFAULT_BASE_URL = ""
    _ENV_KEY = ""
    _ENV_URL = ""
    _LABEL = "OpenAICompat"
    # MiMo 等新式推理模型要求 max_completion_tokens 参数名
    _USE_MAX_COMPLETION_TOKENS = False

    def __init__(self, model: str = None,
                 api_key: str = None, base_url: str = None,
                 temperature: float = 0.1, max_tokens: int = 1024,
                 top_p: float = 0.9, json_mode: bool = False):
        self._model = model or self._DEFAULT_MODEL
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._json_mode = json_mode
        # response_format 支持探测缓存：服务端不支持 json_object 时降级
        self._json_ok = True

        self._key = api_key or os.environ.get(self._ENV_KEY, "")
        self._url = base_url or os.environ.get(self._ENV_URL, self._DEFAULT_BASE_URL)

        # Use raw openai client (avoids langchain_openai version issues)
        self._llm = None
        import openai
        self._raw_client = openai.OpenAI(api_key=self._key, base_url=self._url)

    def _create(self, msgs: list):
        """统一发起 chat.completions.create，处理参数名与 json 降级。

        观测（fas_log，2026-09-22）：这里是一切云端 LLM 调用的唯一出口，
        逐调用记 provider/model/purpose/latency/tokens/重试/失败——
        纯旁路记录，返回值与异常语义与接入前逐字节一致。
        """
        import time as _time
        import fas_log as _fl
        kwargs = dict(
            model=self._model,
            messages=msgs,
            temperature=self._temperature,
            top_p=self._top_p,
        )
        if self._USE_MAX_COMPLETION_TOKENS:
            kwargs["max_completion_tokens"] = self._max_tokens
        else:
            kwargs["max_tokens"] = self._max_tokens
        if self._json_mode and self._json_ok:
            kwargs["response_format"] = {"type": "json_object"}
        _t0 = _time.perf_counter()

        def _evt(event, level, **extra):
            try:
                _fl.emit(_fl.LLM, level, event,
                         f"{self._LABEL} {self._model} {event}",
                         provider=self._LABEL, model=self._model,
                         msgs=len(msgs),
                         purpose=_fl.current_purpose() or "unlabeled",
                         caller=_caller_hint(),
                         latency_ms=round((_time.perf_counter() - _t0) * 1000),
                         **extra)
            except Exception:
                pass

        def _done(resp, retried=False):
            try:
                u = getattr(resp, "usage", None)
                fin = (resp.choices[0].finish_reason
                       if getattr(resp, "choices", None) else None)
                clen = (len(resp.choices[0].message.content or "")
                        if getattr(resp, "choices", None) else 0)
                _fl.bump("llm_calls")
                _fl.bump_trace("llm_calls")
                _evt("llm_call_finished", "INFO",
                     prompt_tokens=getattr(u, "prompt_tokens", None),
                     completion_tokens=getattr(u, "completion_tokens", None),
                     total_tokens=getattr(u, "total_tokens", None),
                     finish_reason=fin, resp_chars=clen, retried=retried)
            except Exception:
                pass
            return resp

        try:
            return _done(self._raw_client.chat.completions.create(**kwargs))
        except Exception as e:
            # 服务端不支持 response_format 时去掉重试一次
            if "response_format" in kwargs:
                logger.warning(f"[{self._LABEL}] response_format 不支持，降级重试: {e}")
                _evt("llm_call_downgrade", "WARNING",
                     error=str(e)[:200], json_mode=True)
                self._json_ok = False
                del kwargs["response_format"]
                try:
                    return _done(self._raw_client.chat.completions.create(**kwargs),
                                 retried=True)
                except Exception as e2:
                    _evt("llm_call_failed", "ERROR", error=str(e2)[:300],
                         retried=True, _with_exc=False)
                    raise
            _evt("llm_call_failed", "ERROR", error=str(e)[:300],
                 json_mode=("response_format" in kwargs), _with_exc=False)
            raise

    @property
    def _lc_model(self):
        """Return a LangChain Runnable for use in chains."""
        if self._llm:
            return self._llm
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(self._raw_invoke)

    @staticmethod
    def _to_msgs(prompt) -> list:
        if hasattr(prompt, 'to_messages'):
            msgs = []
            for m in prompt.to_messages():
                role = m.type
                if role == 'human': role = 'user'
                elif role == 'ai': role = 'assistant'
                msgs.append({"role": role, "content": str(m.content)})
        elif isinstance(prompt, str):
            msgs = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            msgs = []
            for m in prompt:
                if hasattr(m, 'type'):
                    role = m.type
                    if role == 'human': role = 'user'
                    elif role == 'ai': role = 'assistant'
                    msgs.append({"role": role, "content": str(m.content)})
                elif isinstance(m, dict):
                    msgs.append(m)
                else:
                    msgs.append({"role": "user", "content": str(m)})
        else:
            msgs = [{"role": "user", "content": str(prompt)}]
        return msgs

    def _raw_invoke(self, prompt):
        """Invoke via raw OpenAI client, returns AIMessage."""
        from langchain_core.messages import AIMessage
        resp = self._create(self._to_msgs(prompt))
        content = resp.choices[0].message.content or ""
        finish = resp.choices[0].finish_reason
        if not content.strip():
            logger.warning(f"[{self._LABEL}] 空响应 (finish_reason={finish}, model={self._model})")
        return AIMessage(content=content)

    def invoke(self, prompt) -> str:
        if self._llm:
            return self._llm.invoke(prompt)
        raw = self._create([{"role": "user", "content": str(prompt)}])
        content = raw.choices[0].message.content or ""
        finish = raw.choices[0].finish_reason
        if not content.strip():
            logger.warning(f"[{self._LABEL}] invoke 空响应 (finish_reason={finish}, model={self._model})")
        return content

    def bind(self, **kwargs):
        return BoundBackend(self, kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    def with_temperature(self, temp: float) -> "OpenAICompatBackend":
        clone = object.__new__(type(self))
        clone._model = self._model
        clone._temperature = temp
        clone._max_tokens = self._max_tokens
        clone._top_p = self._top_p
        clone._json_mode = self._json_mode
        clone._json_ok = self._json_ok
        clone._raw_client = self._raw_client
        clone._key = self._key
        clone._url = self._url
        clone._llm = None
        return clone


def _caller_hint(max_frames: int = 14) -> str:
    """向上找第一个不在本模块/langchain 栈里的调用者（module.func）。
    只在 LLM 调用（百毫秒级）路径上付出这点栈成本。
    注意：必须定义在类外——留在类方法之间会把类体切成死代码。"""
    import sys as _sys
    try:
        f = _sys._getframe(2)
        here = __file__.replace("\\", "/")
        for _ in range(max_frames):
            if f is None:
                break
            fn = f.f_code.co_filename.replace("\\", "/")
            if (fn != here and "langchain" not in fn
                    and "fas_log" not in fn and "openai" not in fn):
                mod = fn.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                return f"{mod}.{f.f_code.co_name}"
            f = f.f_back
    except Exception:
        pass
    return "?"


class DeepSeekBackend(OpenAICompatBackend):
    """DeepSeek API backend."""
    _DEFAULT_MODEL = "deepseek-v4-flash"
    _DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    _ENV_KEY = "DEEPSEEK_API_KEY"
    _ENV_URL = "DEEPSEEK_BASE_URL"
    _LABEL = "DeepSeek"

    @property
    def model_name(self) -> str:
        return self._model


class MiMoBackend(OpenAICompatBackend):
    """小米 MiMo API backend（OpenAI 兼容协议，https://mimo.mi.com）。

    文档要求用 max_completion_tokens 传输出预算；json_object 模式
    未在文档中承诺，失败时自动降级为纯文本输出（模板已强制 JSON 格式）。
    """
    _DEFAULT_MODEL = "mimo-v2.6-flash"
    _DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
    _ENV_KEY = "MIMO_API_KEY"
    _ENV_URL = "MIMO_BASE_URL"
    _LABEL = "MiMo"
    _USE_MAX_COMPLETION_TOKENS = True

    @property
    def model_name(self) -> str:
        return self._model


class _DeepSeekLCChat:
    """Minimal LangChain-compatible chat model wrapper around raw OpenAI client.
    Implements invoke() so it can be used in Runnable chains."""

    def __init__(self, client, model: str, temperature: float, max_tokens: int):
        self._client = client
        self.model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, prompt, config=None, **kwargs):
        from langchain_core.messages import AIMessage
        if hasattr(prompt, 'to_messages'):
            msgs = []
            for m in prompt.to_messages():
                role = m.type
                if role == 'human': role = 'user'
                elif role == 'ai': role = 'assistant'
                msgs.append({"role": role, "content": str(m.content)})
        elif isinstance(prompt, str):
            msgs = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            msgs = []
            for m in prompt:
                if hasattr(m, 'type'):
                    role = m.type
                    if role == 'human': role = 'user'
                    elif role == 'ai': role = 'assistant'
                    msgs.append({"role": role, "content": str(m.content)})
                elif isinstance(m, dict):
                    msgs.append(m)
                else:
                    msgs.append({"role": "user", "content": str(m)})
        else:
            msgs = [{"role": "user", "content": str(prompt)}]
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=msgs,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return AIMessage(content=resp.choices[0].message.content)

    def bind(self, **kwargs):
        return self


class BoundBackend(LLMBackend):
    """Wraps a backend with extra bound parameters."""

    def __init__(self, backend: LLMBackend, kwargs: dict):
        self._backend = backend
        self._kwargs = kwargs

    def invoke(self, prompt) -> str:
        return self._backend.invoke(prompt)

    def bind(self, **kwargs):
        merged = {**self._kwargs, **kwargs}
        return BoundBackend(self._backend, merged)

    @property
    def model_name(self) -> str:
        return self._backend.model_name


def ollama_enabled() -> bool:
    """Ollama 开关（config.py enable_ollama）。默认关闭=不可见。"""
    try:
        from config import DEFAULT_CONFIG
        return bool(DEFAULT_CONFIG.get("enable_ollama"))
    except Exception:
        return False


def create_backend(provider: str = "mimo", **kwargs) -> LLMBackend:
    """Factory to create an LLM backend.

    Args:
        provider: "ollama", "deepseek" or "mimo"
        **kwargs: passed to the backend constructor

    Returns:
        LLMBackend instance
    """
    if provider == "deepseek":
        return DeepSeekBackend(**kwargs)
    elif provider in ("mimo", "xiaomi"):
        return MiMoBackend(**kwargs)
    elif provider == "ollama":
        if not ollama_enabled():
            raise ValueError("Ollama 未启用（config.py enable_ollama=false）")
        from ollama_backend import OllamaBackend  # 按需导入
        return OllamaBackend(**kwargs)
    else:
        raise ValueError(f"未知 LLM provider: {provider}")
