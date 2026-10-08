# Fascinator NLP Processor - Direct LLM Mode with Speech Act Routing
import logging
from datetime import datetime
from typing import List, Dict, Any
import sys
import re
import json as _json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import LLM provider for multi-backend support
try:
    from llm_provider import create_backend, LLMBackend, ollama_enabled
except ImportError:
    create_backend = None
    LLMBackend = None

# Architecture Refactor 0.1: 统一 Prompt 模板
from prompt_templates import NLP_PARSE, ANSWER_GENERATE, MEMORY_EXTRACT, CONCEPT_EXPAND

# 输入复杂度自适应（2026-09）：Level 1 轻认知走短模板 + 快速小模型
from prompt_templates import PARSE_FAST, ANSWER_SHORT, INTENTION_EXTRACT
# Cognitive Context（2026-09-20）：证据状态归一化渲染——语言层拿到的
# 永远是"executing/queued/done/failed"级别的硬事实，不许把没做完的说成做完
from cognitive_context import (classify_action_evidence, render_action_evidence,
                               EV_NONE)

# 配置日志，方便查看调试信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义常量（模块级别保留给其他模块引用）
SPEECH_ACTS = ["断言类", "指令类", "承诺类", "表达类", "宣告类"]

SYSTEM_PROMPT = NLP_PARSE  # Architecture Refactor 0.1: 统一模板

RECONSTRUCT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "【当前日期】{{current_date}}\n\n请分析以下文本：\n\n{{text}}")
], template_format="mustache")

ANSWER_SYSTEM_PROMPT = ANSWER_GENERATE  # Architecture Refactor 0.1: 统一模板（合并知识+社交）

SOCIAL_ANSWER_SYSTEM_PROMPT = ANSWER_GENERATE  # 已合并到 ANSWER_GENERATE，保留变量名向后兼容


class NLPProcessor:
    MEMORY_TYPE_ACTIVATION_CHAIN = {
        "episodic": ["情境记忆", "事件记录", "经历存储"],
        "semantic": ["语义记忆", "事实存储", "概念关联"],
    }

    def __init__(self, model=None,
                 provider: str = None, api_key: str = None):
        self._provider_name = provider or "mimo"
        self._api_key = api_key

        # 智能默认模型：未指定时按 provider 选择
        if model:
            self.model = model
        elif self._provider_name == "deepseek":
            self.model = "deepseek-v4-flash"
        elif self._provider_name in ("mimo", "xiaomi"):
            self.model = "mimo-v2.6-flash"
        else:
            self.model = "qwen2.5:7b-instruct-q5_k_m"
        # 对话回答模型（2026-09-22 全档统一 mimo-v2.6-flash，不再有 pro/便宜档之分）
        self._chat_model = "mimo-v2.6-flash" if self._provider_name in ("mimo", "xiaomi") else self.model
        self._apply_fast_parse()

        logger.info(f"[NLP] 初始化开始 (provider={self._provider_name}, model={self.model})")

        self.logs = []
        self.chain = None

        # Initialize LLM backend
        self._init_backend()

        self.chain = RECONSTRUCT_PROMPT | self.llm | JsonOutputParser()

        # Chat LLM for answer generation (non-JSON)
        self._init_chat_backend()

        # Level 1 轻认知后端：快速小模型 + JSON 输出（复杂度自适应 2026-09）。
        # 单独于主解析链：短输入不值得动用整条长 prompt 管线。
        self._init_fast_backend()

        logger.info(f"[NLP] 初始化完成 (provider={self._provider_name}, model={self.model_name()})")

    def _fast_model_name(self) -> str:
        """Level 1 模型：与主链同一思考模型（2026-09-22 全场景统一
        mimo-v2.6-flash）；快是靠短 prompt + 短输入，不是靠换非推理档。"""
        if self._provider_name in ("mimo", "xiaomi"):
            return "mimo-v2.6-flash"
        if self._provider_name == "deepseek":
            return "deepseek-v4-flash"
        return self.model

    def _init_fast_backend(self):
        """Level 1 后端：短 prompt + 快速模型 + JSON 模式。"""
        fm = self._fast_model_name()
        try:
            if self._provider_name == "deepseek":
                from llm_provider import DeepSeekBackend
                self._fast_backend = DeepSeekBackend(
                    model=fm, api_key=self._api_key,
                    temperature=0.1, max_tokens=1024, json_mode=True)
            elif self._provider_name in ("mimo", "xiaomi"):
                from llm_provider import MiMoBackend
                self._fast_backend = MiMoBackend(
                    model=fm, api_key=self._api_key,
                    temperature=0.1, max_tokens=4096, json_mode=True)
            elif self._provider_name == "ollama":
                from ollama_backend import OllamaBackend
                self._fast_backend = OllamaBackend(
                    model=fm, temperature=0.1, format="json", num_predict=512)
            else:
                self._fast_backend = None
        except Exception as e:
            logger.warning(f"[NLP] Level1 快速后端初始化失败（降级为主链）: {e}")
            self._fast_backend = None

    def _init_backend(self):
        """Initialize the primary LLM backend. Stores raw LangChain model for chains."""
        if self._provider_name == "deepseek":
            from llm_provider import DeepSeekBackend
            self._backend = DeepSeekBackend(
                model=self.model, api_key=self._api_key,
                # 4096 会触发 finish_reason=length：deepseek-v4-flash 的
                # 推理阶段消耗同一份 token 预算，推理一长输出就为空。
                temperature=0.1, max_tokens=8192, json_mode=True)
            self._expand_backend = DeepSeekBackend(
                model=self.model, api_key=self._api_key,
                temperature=0.35, max_tokens=8192, json_mode=True)
        elif self._provider_name in ("mimo", "xiaomi"):
            from llm_provider import MiMoBackend
            # mimo-v2.6-flash 是思考模型：max_completion_tokens 把「思考 token +
            # 可见输出」合并计算（API 上限 ~131k）。抽取/expand 的推理可能很长，
            # 预算给到 16384，避免思考挤占 JSON 输出（历史 8192 对重推理偏紧）。
            self._backend = MiMoBackend(
                model=self.model, api_key=self._api_key,
                temperature=0.1, max_tokens=16384, json_mode=True)
            self._expand_backend = MiMoBackend(
                model=self.model, api_key=self._api_key,
                temperature=0.35, max_tokens=16384, json_mode=True)
        elif self._provider_name == "ollama":
            if not ollama_enabled():
                raise ValueError("Ollama 未启用（config.py enable_ollama=false）")
            from ollama_backend import OllamaBackend  # 按需导入
            self._backend = OllamaBackend(
                model=self.model, temperature=0.1, format="json",
                num_predict=1024, top_p=0.9)
            self._expand_backend = OllamaBackend(
                model=self.model, temperature=0.35, format="json",
                num_predict=1536, top_p=0.9)
        # Raw LangChain models for Runnable chains
        self.llm = getattr(self._backend, '_lc_model', self._backend._llm)
        self.llm_expand = getattr(self._expand_backend, '_lc_model', self._expand_backend._llm)

    def _init_chat_backend(self):
        """Initialize chat backend for answer generation."""
        if self._provider_name == "deepseek":
            from llm_provider import DeepSeekBackend
            self._chat_backend = DeepSeekBackend(
                model=self.model, api_key=self._api_key,
                temperature=0.7, max_tokens=512)
        elif self._provider_name in ("mimo", "xiaomi"):
            from llm_provider import MiMoBackend
            # 对话回答同样走 mimo-v2.6-flash（全场景统一思考模型）。思考 token
            # 与回复共享预算，2048（旧便宜档取值）偏紧 → 4096。
            self._chat_backend = MiMoBackend(
                model=self._chat_model, api_key=self._api_key,
                temperature=0.7, max_tokens=4096)
        elif self._provider_name == "ollama":
            if not ollama_enabled():
                raise ValueError("Ollama 未启用（config.py enable_ollama=false）")
            from ollama_backend import OllamaBackend  # 按需导入
            self._chat_backend = OllamaBackend(
                model=self.model, temperature=0.7, num_predict=512,
                top_p=0.95, format=None)
        # Raw LangChain model for chain usage
        self.chat_llm = getattr(self._chat_backend, '_lc_model', self._chat_backend._llm)

    # ── 温度调制（2026-09-20 网络调制层接线）────────────────
    # 回合内由 app 调 set_answer_temperature（llm.temperature_answer /
    # llm.temperature_expand 参数：DMN 高→升温放开发散，CEN 高→降温收紧）。
    # None = 用后端构造时的静态温度（未接调制的场合行为不变）。

    def set_answer_temperature(self, answer_temp=None, expand_temp=None):
        self._chat_temp = answer_temp
        self._expand_temp = expand_temp

    def _chat(self):
        t = getattr(self, "_chat_temp", None)
        if t is None:
            return self.chat_llm
        try:
            # 用后端官方克隆接口（bind kwargs 会漏进自建 backend 的
            # _raw_invoke 签名；with_temperature 克隆后 _lc_model 按新温度重建）
            b = self._chat_backend.with_temperature(float(t))
            return getattr(b, "_lc_model", b._llm)
        except Exception:
            return self.chat_llm

    def _expand(self):
        t = getattr(self, "_expand_temp", None)
        if t is None:
            return self.llm_expand
        try:
            b = self._expand_backend.with_temperature(float(t))
            return getattr(b, "_lc_model", b._llm)
        except Exception:
            return self.llm_expand

    def switch_provider(self, provider: str, api_key: str = None, model: str = None):
        """Hot-switch the LLM provider at runtime. Always persists model choice."""
        if provider == "ollama" and not ollama_enabled():
            raise ValueError("Ollama 未启用（config.py enable_ollama=false）")
        self._provider_name = provider
        if api_key is not None:
            self._api_key = api_key
        # Always update model: use provided model, or sensible default for this provider
        if model:
            self.model = model
        elif provider == "deepseek":
            self.model = "deepseek-v4-flash"
        elif provider in ("mimo", "xiaomi"):
            self.model = "mimo-v2.6-flash"
        elif provider == "ollama":
            self.model = "qwen2.5:7b-instruct-q5_k_m"
        self._chat_model = "mimo-v2.6-flash" if provider in ("mimo", "xiaomi") else self.model
        self._apply_fast_parse()
        self._init_backend()
        self._init_chat_backend()
        self._init_fast_backend()
        # self.llm is now the raw LangChain model from _init_backend
        self.chain = RECONSTRUCT_PROMPT | self.llm | JsonOutputParser()
        logger.info(f"[NLP] Switched to provider={provider}, model={self.model_name()}")

    def _apply_fast_parse(self):
        """延迟优化开关（2026-09-19 设）：历史上 true 时把解析/抽取降到非推理档。

        2026-09-22：全场景统一思考模型 mimo-v2.6-flash——两档同名，本开关
        不再切换模型（保留仅为配置兼容）；实测旧 pro 档单次解析 14~73 秒、
        非推理档延迟约减半的结论只属于 v2.5 时代。只影响 mimo 提供者。
        """
        if self._provider_name in ("mimo", "xiaomi"):
            try:
                import config as _cfg
                if getattr(_cfg.DEFAULT_CONFIG, "get", lambda k, d=None: None)(
                        "nlp_fast_parse", False):
                    self.model = "mimo-v2.6-flash"
                    logger.info("[NLP] fast_parse：全场景统一 mimo-v2.6-flash（思考模型），此开关不再换档")
            except Exception:
                pass

    def model_name(self) -> str:
        """Get current model name — always returns self.model for consistency."""
        return self.model

    def process(self, text: str) -> Dict[str, Any]:
        start = datetime.now()
        logger.info(f"[NLP] 输入: {text}")

        # 默认返回结构
        parsed = {
            "nodes": [],
            "edges": [],
            "assertion_type": "semantic"
        }

        try:
            clean_text = text.strip()

            if not clean_text:
                logger.warning("[NLP] 输入为空，返回默认结构")
                return parsed

            logger.info(f"[LLM] 请求模型 {self.model}")
            import fas_log
            with fas_log.llm_purpose("dialogue_decomposition"):
                current_date = datetime.now().strftime("%Y年%m月%d日 %A")
                raw_result = self.chain.invoke({
                    "text": clean_text,
                    "current_date": current_date,
                })
            logger.info("[LLM] 完成")

            parsed = self._sanitize_result(raw_result)

        except Exception as e:
            logger.warning(f"[NLP] 首次解析失败: {e}，重试中...")
            try:
                current_date = datetime.now().strftime("%Y年%m月%d日 %A")
                raw_result = self.chain.invoke({
                    "text": clean_text,
                    "current_date": current_date,
                })
                parsed = self._sanitize_result(raw_result)
                logger.info("[NLP] 重试成功")
            except Exception as e2:
                logger.exception(f"[NLP] 重试仍失败: {e2}")

        end = datetime.now()
        duration = end - start

        self.logs.append({
            "time": start.strftime("%Y-%m-%d %H:%M:%S"),
            "input": text,
            "result": parsed,
            "duration": str(duration)
        })

        logger.info(f"[NLP] 完成耗时: {duration}")
        return parsed

    # ── Level 1 轻认知解析（复杂度自适应 2026-09）──────────

    def _fast_chain(self):
        """Level 1 JSON 链。快速后端不可用时降级为主链（同一条认知管线的
        慢档，功能不缺失，只是慢）。"""
        if self._fast_backend is not None:
            lc = getattr(self._fast_backend, "_lc_model",
                         getattr(self._fast_backend, "_llm", None))
            if lc is not None:
                from langchain_core.output_parsers import JsonOutputParser as _JOP
                fast_prompt = ChatPromptTemplate.from_messages([
                    ("system", PARSE_FAST),
                    ("human", "{{input}}"),
                ], template_format="mustache")
                return fast_prompt | lc | _JOP()
        return self.chain

    def process_fast(self, text: str) -> Dict[str, Any]:
        """Level 1：短输入/低复杂度 → 短 prompt + 快速模型 + 一次调用。

        输出仍是认知产物（speech_act / intent / target / urgency /
        needs_reply / needs_action + 少量节点），由 app 接线进同一条
        感知 → 图激活 → 行为竞争管路。不是旁路，是变轻的认知。
        """
        start = datetime.now()
        t = str(text or "").strip()
        parsed = {
            "nodes": [], "edges": [],
            "illocutionary_act": "assertive",
            "dialogue_act": "information_statement",
            "response_expectation": "low",
            "suggested_reply_goals": ["acknowledge"],
            "memory_type": "episodic",
            "level": 1,
        }
        try:
            raw = self._fast_chain().invoke({"input": t})
            parsed = self._sanitize_fast_result(raw)
        except Exception as e:
            logger.warning(f"[NLP] Level1 快速解析失败: {e}")
        self.logs.append({
            "time": start.strftime("%Y-%m-%d %H:%M:%S"),
            "input": f"[L1] {text}",
            "result": parsed,
            "duration": str(datetime.now() - start),
        })
        logger.info(f"[NLP] Level1 解析完成: intent={parsed.get('intent')} "
                    f"({datetime.now() - start})")
        return parsed

    def _sanitize_fast_result(self, raw: Any) -> Dict[str, Any]:
        """Level 1 输出 → 与主解析兼容的结构（下游代码零分支）。"""
        result = {
            "nodes": [], "edges": [],
            "illocutionary_act": "assertive",
            "dialogue_act": "information_statement",
            "response_expectation": "medium",
            "suggested_reply_goals": ["acknowledge"],
            "memory_type": "episodic",
            "level": 1,
        }
        if not isinstance(raw, dict):
            logger.warning("[NLP] L1 快速解析输出不是 JSON 对象")
            result["intent"] = None
            return result
        intent = str(raw.get("intent") or "none").strip().lower()
        result["intent"] = intent if intent and intent != "none" else None
        result["target"] = raw.get("target") or None
        try:
            result["count"] = int(raw["count"]) if raw.get("count") else None
        except (TypeError, ValueError):
            result["count"] = None
        try:
            result["urgency"] = max(0.0, min(1.0, float(raw.get("urgency", 0.5))))
        except (TypeError, ValueError):
            result["urgency"] = 0.5
        result["needs_reply"] = bool(raw.get("needs_reply", True))
        result["needs_action"] = bool(raw.get("needs_action", False))

        ill = str(raw.get("speech_act") or "").strip()
        result["illocutionary_act"] = {
            "指令": "directive", "断言": "assertive", "承诺": "commissive",
            "表达": "expressive", "宣告": "declaration"}.get(ill, "assertive")
        da = str(raw.get("dialogue_act") or "").strip().lower()
        VALID_DA = {"greeting", "question", "answer", "sharing", "request",
                    "suggestion", "opinion", "agreement", "disagreement",
                    "thanking", "apology", "comfort", "congratulation",
                    "invitation", "farewell", "backchannel",
                    "information_statement", "emotion_expression",
                    "curiosity_expression"}
        result["dialogue_act"] = da if da in VALID_DA else "information_statement"
        exp = str(raw.get("response_expectation") or "medium").lower()
        result["response_expectation"] = exp if exp in ("high", "medium", "low", "none") else "medium"
        nodes = []
        for n in (raw.get("nodes") or []):
            if isinstance(n, str) and n.strip():
                n = n.strip()
                n = {"我": "用户", "我的": "用户", "你": "Fascinator",
                     "你的": "Fascinator"}.get(n, n)
                nodes.append(n)
        result["nodes"] = list(dict.fromkeys(nodes))[:4]
        mem = str(raw.get("memory_type") or "episodic").lower()
        result["memory_type"] = mem if mem in ("episodic", "semantic") else "episodic"
        if result["illocutionary_act"] == "expressive":
            result["assertion_type"] = "social"
        elif result["memory_type"] == "episodic":
            result["assertion_type"] = "episodic"
        else:
            result["assertion_type"] = "semantic"
        if result["needs_action"] and result["intent"]:
            result["suggested_reply_goals"] = ["acknowledge"]
        return result

    # ── Level 2 多意图抽取（复杂指令的目标分解）──────────

    def extract_intentions(self, text: str) -> Dict[str, Any]:
        """Level 2：复杂输入的意图分解（高层目标，不是操作序列）。

        返回 {intentions: [{type,target,params,note}], ...}。
        产物进入 Action 层的目标队列，由行动系统逐步执行。
        """
        prompt_tpl = ChatPromptTemplate.from_messages([
            ("system", INTENTION_EXTRACT),
            ("human", "{{input}}"),
        ], template_format="mustache")
        try:
            chain = prompt_tpl | self.llm | JsonOutputParser()
            raw = chain.invoke({"input": str(text or "").strip()})
        except Exception as e:
            logger.warning(f"[NLP] 意图分解失败: {e}")
            return {"intentions": []}
        out = {"intentions": [], "nodes": [],
               "dialogue_act": "", "response_expectation": "medium"}
        if not isinstance(raw, dict):
            return out
        VALID_TYPES = {
            "go_to", "explore_direction", "follow_user", "come_here",
            "stop_action", "mine_block", "gather_resource", "collect_food",
            "eat_food", "craft_item", "smelt_item", "equip_tool",
            "attack_entity", "retreat", "seek_safety", "place_block",
            "build_shelter", "place_light", "sleep", "inspect_target",
            "remember_location", "return_home", "give_item", "wait",
            "converse",
        }
        for it in (raw.get("intentions") or []):
            if not isinstance(it, dict):
                continue
            typ = str(it.get("type") or "").strip()
            if typ not in VALID_TYPES:
                continue
            out["intentions"].append({
                "type": typ,
                "target": it.get("target") or None,
                "params": it.get("params") or {},
                "note": it.get("note") or None,
            })
        out["nodes"] = [str(n) for n in (raw.get("nodes") or []) if str(n).strip()][:8]
        da = str(raw.get("dialogue_act") or "").strip().lower()
        if da:
            out["dialogue_act"] = da
        exp = str(raw.get("response_expectation") or "medium").lower()
        if exp in ("high", "medium", "low", "none"):
            out["response_expectation"] = exp
        logger.info(f"[NLP] 意图分解: {len(out['intentions'])} 个意图 "
                    f"{[i['type'] for i in out['intentions']]}")
        return out

    # ── Level 1 短回应生成 ────────────────────────────────

    def answer_short(self, text: str, action_result: dict = None,
                     cognitive_context: dict = None) -> str:
        """Level 1 短回应：动作已开始 → 一两句应答（先行动，后回复）。

        cognitive_context 是 Context Compiler 的 L1 编译产物：
        decision（应答姿态/长度约束）+ evidence（动作真实状态——
        executing/queued 不许说成完成）+ mood/最近对话底色。
        action_result 既接受归一化证据（带 status），也兼容旧式结果字典。
        """
        from cognitive_context import (classify_action_evidence,
                                        render_action_evidence, EV_NONE)
        cog = cognitive_context or {}
        blocks = []
        ev = action_result or {}
        if ev and not ev.get("status"):
            ev = classify_action_evidence(ev, source="turn:l1")
        if ev and ev.get("status") and ev["status"] != EV_NONE:
            line = render_action_evidence(ev)
            if line:
                blocks.append(line)
        # 决定（语言层不许翻案：是否回应/是否提问/长度——裁决已定）
        dec = cog.get("decision") or {}
        if dec:
            dl = []
            if dec.get("ask_question") is False:
                dl.append("本轮不向用户提问")
            if dec.get("length_constraint"):
                dl.append(f"长度：{dec['length_constraint']}")
            if dl:
                blocks.append("【交流决定】" + "；".join(dl) + "（已裁决，遵守）")
        _ot = cog.get("outline") or {}
        if _ot.get("move"):
            _ol = [f"姿态={_ot.get('move')}"
                   + (f"；长度={_ot['length']}" if _ot.get("length") else "")]
            for _m in (_ot.get("must") or [])[:4]:
                _ol.append("必须交代：" + _m)
            for _n in (_ot.get("must_not") or [])[:3]:
                _ol.append("禁止：" + _n)
            blocks.append("【发言提纲】" + "；".join(_ol) +
                          "（你只决定措辞，不改变计划）")
        _sa1 = cog.get("speech_act") or {}
        if _sa1.get("concept"):
            blocks.append(f"【言外行为】本轮话语在图谱中被归类为「{_sa1['concept']}」"
                          f"（这是认知投影，回应姿态仍按已裁决的决定）")
        cons = [c for c in (cog.get("constraints") or []) if isinstance(c, str)]
        if any("场合约束" in c for c in cons):
            blocks.append("场合：游戏内一句话聊天，回复短、单行、无 Markdown。")
        mood = cog.get("mood")
        if mood:
            blocks.append(f"【当前心情】{mood}")
        recent = cog.get("recent_dialogue") or []
        if recent:
            dlg = "\n".join(
                ("用户：" if x.get("user_input") else "FAS：")
                + str(x.get("user_input") or x.get("system_response"))[:60]
                for x in recent[-3:])
            blocks.append(f"【近期对话】\n{dlg}")
        prompt_tpl = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SHORT),
            ("human", "{{input}}"),
        ], template_format="mustache")
        user_prompt = "\n\n".join(blocks + [f"【用户的话】\n{text}",
                                            "请回应："])
        try:
            chain = prompt_tpl | self._chat()
            resp = chain.invoke({"input": user_prompt})
            ans = resp.content if hasattr(resp, "content") else str(resp)
            out = (ans or "").strip()
            if out in ("空字符串", '""', "''", "无"):
                out = ""
            return out
        except Exception as e:
            logger.warning(f"[NLP] 短回应生成失败: {e}")
            return ""

    def _sanitize_result(self, raw_data: Any) -> Dict[str, Any]:
        result = {
            "nodes": [],
            "edges": [],
            "illocutionary_act": "assertive",
            "dialogue_act": "information_statement",
            "response_expectation": "low",
            "suggested_reply_goals": ["acknowledge"],
            "memory_type": "semantic",
        }

        if not isinstance(raw_data, dict):
            logger.warning("[Sanitize] LLM 输出不是字典格式")
            return result

        DROP = {"他","她","它","他们","她们","他的","她的","它的","他们的"}
        SELF_MAP = {"我":"用户","我的":"用户","你":"Fascinator","你的":"Fascinator"}

        raw_nodes = raw_data.get("nodes", [])
        if isinstance(raw_nodes, list):
            clean = []
            for node in raw_nodes:
                if not node or not isinstance(node, (str, int)):
                    continue
                n = str(node).strip()
                if not n:
                    continue
                if n in DROP:
                    continue
                n = SELF_MAP.get(n, n)
                clean.append(n)
            result["nodes"] = list(dict.fromkeys(clean))

        raw_edges = raw_data.get("edges", [])
        valid_edges = []

        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue

            src = str(edge.get("src", "")).strip()
            dst = str(edge.get("dst", "")).strip()
            rel_type = str(edge.get("type", "")).strip()
            weight = edge.get("weight", 0.0)

            src = SELF_MAP.get(src, src)
            dst = SELF_MAP.get(dst, dst)
            if not src or not dst or src == dst or src in DROP or dst in DROP:
                continue

            if not rel_type:
                rel_type = "关联"
            # 关系归一（架构对齐 2026-09-19）：自由文本关系在抽取层收敛到规范词
            import graph_schema as _gs
            rel_type = _gs.normalize_relation(rel_type)

            try:
                weight = float(weight)
                weight = max(-1.0, min(1.0, weight))
            except (ValueError, TypeError):
                weight = 0.0

            valid_edges.append({
                "src": src,
                "dst": dst,
                "type": rel_type,
                "relation": rel_type,
                "weight": weight
            })

        result["edges"] = valid_edges

        # ── Layer 1: Illocutionary Act ──
        VALID_ILLOCUTIONARY = {"assertive", "directive", "commissive", "expressive", "declaration"}
        ill = str(raw_data.get("illocutionary_act", "")).strip().lower()
        if ill in VALID_ILLOCUTIONARY:
            result["illocutionary_act"] = ill

        # ── Layer 2: Dialogue Act ──
        VALID_DIALOGUE = {
            "greeting", "question", "answer", "sharing", "request",
            "suggestion", "opinion", "agreement", "disagreement",
            "thanking", "apology", "comfort", "congratulation",
            "invitation", "farewell", "backchannel",
            "information_statement", "emotion_expression", "curiosity_expression",
        }
        da = str(raw_data.get("dialogue_act", "")).strip().lower()
        if da in VALID_DIALOGUE:
            result["dialogue_act"] = da

        # ── Layer 3: Response Expectation ──
        VALID_EXPECTATION = {"high", "medium", "low", "none"}
        re_val = str(raw_data.get("response_expectation", "")).strip().lower()
        if re_val in VALID_EXPECTATION:
            result["response_expectation"] = re_val

        # ── Layer 4: Reply Goals ──
        VALID_GOALS = {
            "acknowledge", "continue_conversation", "ask_followup",
            "answer", "explain", "comfort", "congratulate", "encourage",
            "express_curiosity", "clarify", "correct", "accept", "reject",
            "end_conversation",
        }
        raw_goals = raw_data.get("suggested_reply_goals", [])
        if isinstance(raw_goals, list):
            result["suggested_reply_goals"] = [
                g for g in raw_goals
                if isinstance(g, str) and str(g).strip().lower() in VALID_GOALS
            ]
        if not result["suggested_reply_goals"]:
            result["suggested_reply_goals"] = ["acknowledge"]

        # ── Memory Type ──
        raw_mem = str(raw_data.get("memory_type", "")).strip().lower()
        if raw_mem in ("episodic", "semantic"):
            result["memory_type"] = raw_mem
        else:
            # 兜底：有"用户"节点 → episodic；否则 semantic
            node_set = set(result["nodes"])
            result["memory_type"] = "episodic" if "用户" in node_set else "semantic"

        # ── Backward compat: assertion_type ──
        # Derived from the new layers for code that still checks assertion_type
        da = result["dialogue_act"]
        ill = result["illocutionary_act"]
        if ill == "expressive" and da in ("greeting", "farewell", "thanking", "apology", "emotion_expression"):
            result["assertion_type"] = "social"
        elif result["memory_type"] == "episodic":
            result["assertion_type"] = "episodic"
        else:
            result["assertion_type"] = "semantic"

        return result

    COORD_PATTERNS = [
        re.compile(r'点击\s*\(\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*\)'),
        re.compile(r'[Cc]lick\s*\(\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*\)'),
        re.compile(r'[Xx]\s*[=＝]\s*(-?\d+)\s*[,，]?\s*[Yy]\s*[=＝]\s*(-?\d+)'),
        re.compile(r'坐标\s*[（(]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[）)]'),
        re.compile(r'点击\s+(-?\d+)\s+(-?\d+)'),
        re.compile(r'屏幕\s+(-?\d+)\s+(-?\d+)'),
    ]

    def extract_coordinates(self, text: str) -> dict:
        for pat in self.COORD_PATTERNS:
            m = pat.search(text)
            if m:
                x = int(m.group(1))
                y = int(m.group(2))
                logger.info(f"[Coord] 检测到坐标: X={x}, Y={y}")
                return {"action": "点击", "x": x, "y": y}
        return {}


    def extract_assertion_graph(self, text: str, context_nodes: list = None,
                                focus_events: list = None,
                                now: "datetime" = None) -> dict:
        logger.info("[Memory] 开始从断言抽取知识图谱")

        # Architecture Refactor 0.1: 使用统一 MEMORY_EXTRACT 模板
        # 运行时上下文注入到用户消息中，系统提示使用统一模板
        context_str = ""
        if context_nodes:
            context_str = "\n".join(
                f"- {getattr(n, 'id', str(n))}" for n in context_nodes[:15]
            )
            context_str = f"\n【当前图谱中已激活的上下文节点】\n{context_str}\n"
        # 论文 §3.3：自认知子图维护的"当前焦点事件"注入提取上下文，
        # 让延续陈述（如"没走"≈"出发日"）无需扩散命中也能识别 parent_event
        if focus_events:
            focus_str = "、".join(str(x) for x in focus_events[:4])
            context_str += (
                f"\n【进行中的事件（若当前陈述是其中某事件的延续，parent_event 填该事件名）】"
                f"{focus_str}\n"
            )

        # 注入当前日期：事件框架需要将"今天/明天"解析为具体日期锚点。
        # now 可覆盖：历史轮次重放时必须用那一轮的真实日期，否则一周的事会被
        # 全部锚到今天——情景记忆的时间轴就废了（恢复场景，见 chat_log_replay.py）
        current_date = (now or datetime.now()).strftime("%Y年%m月%d日 %A")
        prompt_text = (
            context_str
            + f"\n【当前日期】{current_date}\n"
            + "\n【用户的陈述句】\n" + text
        )

        # Architecture Refactor 0.1: 使用统一 MEMORY_EXTRACT 模板
        # template_format="mustache" 避免 JSON 示例中的 {} 被 f-string 解析
        prompt_tpl = ChatPromptTemplate.from_messages([
            ("system", MEMORY_EXTRACT),
            ("human", "{{input}}")
        ], template_format="mustache")
        chain = prompt_tpl | self.llm | JsonOutputParser()

        # ── LLM 调用重试：JSON 解析失败时重试一次 ──
        result = None
        for _attempt in range(2):
            try:
                result = chain.invoke({"input": prompt_text})
                if isinstance(result, dict):
                    break
                logger.warning(f"[Memory] LLM 返回非字典格式 (attempt {_attempt+1})")
            except Exception as _retry_e:
                if _attempt == 0:
                    logger.warning(f"[Memory] 首次提取失败: {_retry_e}，重试中...")
                else:
                    logger.exception(f"[Memory] 重试仍失败: {_retry_e}")
                    return {"nodes": [], "edges": [], "error": str(_retry_e)}

        if not isinstance(result, dict):
            return {"nodes": [], "edges": [], "error": "LLM 返回格式异常"}

        try:
            PRONOUNS = {"我","我的","你","你的","他","她","它","我们","你们","他们","她们","他的","她的","它的","他们的"}
            VERB_FRAGMENTS = {"是","了","的","在","和","与","或","到","对","从","把","被","让","给","为","以","就","也","都","还","着","过"}

            nodes = []
            for n in result.get("nodes", []):
                if isinstance(n, str):
                    n = n.strip()
                    if n == "我" or n == "我的":
                        n = "用户"
                    elif n in ("你","你的"):
                        n = "Fascinator"
                    elif n in PRONOUNS or n in VERB_FRAGMENTS:
                        continue
                    if n:
                        nodes.append({"id": n, "node_type": "概念"})
                elif isinstance(n, dict):
                    nid = str(n.get("id", n.get("name", ""))).strip()
                    if nid == "我" or nid == "我的":
                        nid = "用户"
                    elif nid in ("你","你的"):
                        nid = "Fascinator"
                    elif nid in PRONOUNS or nid in VERB_FRAGMENTS:
                        continue
                    if nid:
                        nodes.append({
                            "id": nid,
                            "node_type": str(n.get("node_type", "概念"))
                        })

            edges = []
            for e in result.get("edges", []):
                if not isinstance(e, dict):
                    continue
                src = str(e.get("src", e.get("from", ""))).strip()
                dst = str(e.get("dst", e.get("to", ""))).strip()
                if not src or not dst or src == dst:
                    continue
                if src == "我" or src == "我的":
                    src = "用户"
                if dst == "我" or dst == "我的":
                    dst = "用户"
                # Map second-person to Fascinator, discard ambiguous third-person
                if src in ("你","你的"):
                    src = "Fascinator"
                elif src in ("他","她","它","我们","你们","他们","她们","他的","她的","它的","他们的"):
                    continue
                if dst in ("你","你的"):
                    dst = "Fascinator"
                elif dst in ("他","她","它","我们","你们","他们","她们","他的","她的","它的","他们的"):
                    continue
                if src in VERB_FRAGMENTS or dst in VERB_FRAGMENTS:
                    continue
                w = e.get("weight", 0.7)
                try:
                    w = max(0.1, min(1.0, float(w)))
                except Exception:
                    w = 0.7
                # 关系归一（架构对齐 2026-09-19）：LLM 的自由文本关系在抽取层
                # 就收敛到规范原子关系（graph_schema 词表），不再原样进图。
                import graph_schema as _gs
                _rel = _gs.normalize_relation(
                    str(e.get("type", e.get("relation", "关联"))).strip())
                edges.append({
                    "src": src, "dst": dst,
                    "type": _rel,
                    "weight": w,
                    "reason": str(e.get("reason", ""))[:200]
                })

            logger.info(f"[Memory] 提取节点: {len(nodes)}, 边: {len(edges)}")

            assertion_type = result.get("assertion_type", "")
            if assertion_type not in ("episodic", "semantic", "social"):
                has_user = any(n.get("id") == "用户" for n in nodes)
                assertion_type = "episodic" if has_user else "semantic"

            # ── 事件框架（论文 §3.2）：episodic 记忆以事件节点为枢纽 ──
            # 架构对齐（2026-09-19）：禁止 X—Y—Z 命名式事件节点（把谓词+槽位
            # 烤进 id）。命中即拆解：事件节点用谓词短语，槽位（方面/实体）转为
            # 对象/地点 论元边——槽位由结构表达，不由节点名表达。
            event_info = None
            if isinstance(result.get("event"), dict):
                import graph_schema as _gs
                ev = result["event"]
                ev_summary = str(ev.get("summary", "")).strip()
                ev_time = str(ev.get("event_time", "")).strip()
                parent = str(ev.get("parent_event", "")).strip()
                # LLM 偶发输出字符串 "None"/"null" 而非 JSON null，统一归一为空
                if parent.lower() in ("none", "null", "无", "无父事件"):
                    parent = ""
                slot_edges = []
                if ev_summary and _gs.COMPOUND_NODE_RE.match(ev_summary):
                    parts = [p.strip() for p in ev_summary.split("—") if p.strip()]
                    if len(parts) >= 3 and len(parts[0]) >= 2:
                        aspect, entity = parts[1], "—".join(parts[2:])
                        logger.info(
                            f"[Memory] 事件命名式拆解: {ev_summary} → "
                            f"事件[{parts[0]}] + {aspect}[{entity}]")
                        ev_summary = parts[0]
                        # 槽位边：实体本身也进 nodes（原子实体），关系按方面归类
                        if entity:
                            nodes.append({"id": entity, "node_type":
                                          "地点" if aspect in ("地点", "位置") else "对象"})
                            slot_edges.append({
                                "src": parts[0], "dst": entity,
                                "type": "位于" if aspect in ("地点", "位置") else "涉及",
                                "weight": 0.8,
                                "reason": f"事件命名式拆解槽位: {aspect}",
                            })
                    # 部件过短拆不出语义 → 保留原 id（守卫不毁语义）
                if ev_summary:
                    event_info = {
                        "summary": ev_summary,
                        "event_time": ev_time,
                        "parent_event": parent or None,
                    }
                    # LLM 若漏了事件节点，补进 nodes，保证边端点存在
                    if not any(n.get("id") == ev_summary for n in nodes):
                        nodes.append({"id": ev_summary, "node_type": "事件"})
                    edges.extend(slot_edges)
                    logger.info(
                        f"[Memory] 事件框架: '{ev_summary}' "
                        f"time={ev_time or '?'} parent={parent or '-'}"
                    )

            return {
                "nodes": nodes,
                "edges": edges,
                "assertion_type": assertion_type,
                "event": event_info,
            }

        except Exception as e:
            logger.exception(f"[Memory] 节点/边处理异常: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}

    @staticmethod
    def split_narrative(text: str, chunk_size: int = 600) -> list:
        """长叙事按句子边界切分为 ≤chunk_size 的段。"""
        import re as _re
        sents = _re.split(r'(?<=[。！？!?\n])', text.strip())
        chunks, cur = [], ''
        for s in sents:
            if not s.strip():
                continue
            if len(cur) + len(s) > chunk_size and cur:
                chunks.append(cur)
                cur = s
            else:
                cur += s
        if cur.strip():
            chunks.append(cur)
        return chunks

    def extract_narrative(self, text: str) -> dict:
        """叙事分段抽取：长文本按段调用 NARRATIVE_EXTRACT，合并事件序列。"""
        from langchain_core.prompts import ChatPromptTemplate
        from prompt_templates import build_prompt
        chunks = self.split_narrative(text)
        story, characters, events = None, [], []
        system = build_prompt("narrative_extract")
        for ci, chunk in enumerate(chunks):
            try:
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system),
                     ("human", "【第 {{idx}} 段（共 {{total}} 段）】\n{{chunk}}")],
                    template_format="mustache")
                import fas_log
                with fas_log.llm_purpose("narrative_extract"):
                    resp = (prompt | self.chat_llm).invoke(
                        {"idx": str(ci + 1), "total": str(len(chunks)), "chunk": chunk})
                raw = resp.content if hasattr(resp, 'content') else str(resp)
                if raw.startswith("```"):
                    raw = "\n".join(raw.split("\n")[1:-1])
                data = _json.loads(raw.strip())
                if not isinstance(data, dict):
                    continue
            except Exception as e:
                logger.warning(f"[Narrative] 段 {ci+1} 抽取失败: {e}")
                continue
            if data.get("story") and not story:
                story = data["story"]
            for c in data.get("characters") or []:
                if c and c not in characters:
                    characters.append(c)
            for ev in data.get("events") or []:
                if isinstance(ev, dict) and ev.get("title"):
                    ev["_chunk"] = ci
                    events.append(ev)
        return {"story": story, "characters": characters, "events": events,
                "chunks": len(chunks)}

    def answer_question(self, original_text: str, topk_nodes: list, topk_edges: list = None,
                        context_type: str = None, cognitive_context: dict = None,
                        mode: str = None, attention_context: dict = None) -> str:
        logger.info("[Answer] 开始生成回答")
        try:
            if cognitive_context is None:
                cognitive_context = {}

            topk_node_ids = set()
            topk_str = ""
            for i, n in enumerate(topk_nodes[:15]):
                act_val = getattr(n, 'activation', 0.0)
                nid = getattr(n, 'id', str(n))
                lbl = getattr(n, 'label', '')
                # 能力节点自带"我能做什么"的说明——读成能力本身，而不是一个裸标签
                _extra = getattr(n, 'extra_attrs', {}) or {}
                _cap = _extra.get("description", "") if _extra.get("self_capability") else ""
                _cap_str = f" ｜ 我能做的事: {_cap}" if _cap else ""
                # 状态节点的**值**必须一起给：只给"Haru的位置"这个标签，语言层
                # 只能猜坐标（实测她答成"我在Haru这里，和你一起"）。通用规则：
                # 凡是带 extra_attrs.value 的节点都渲染成「槽位 = 值」。
                _val = _extra.get("value")
                _val_str = f" = {_val}" if _val not in (None, "") else ""
                topk_str += f"{i+1}. {nid}{_val_str} (激活度:{act_val:.4f}, 标签:{lbl}{_cap_str})\n"
                topk_node_ids.add(nid)

            topk_edge_str = ""
            if topk_edges:
                knowledge_edges = [
                    e for e in topk_edges
                    if (getattr(e, 'src', e.get('src', '') if isinstance(e, dict) else '') in topk_node_ids
                        and getattr(e, 'dst', e.get('dst', '') if isinstance(e, dict) else '') in topk_node_ids)
                ]
                if knowledge_edges:
                    topk_edge_str = "\n【Top-K 节点间关系边】\n"
                    for i, e in enumerate(knowledge_edges[:20]):
                        src = getattr(e, 'src', e.get('src', '') if isinstance(e, dict) else '')
                        dst = getattr(e, 'dst', e.get('dst', '') if isinstance(e, dict) else '')
                        rel = getattr(e, 'relation', e.get('relation', '') if isinstance(e, dict) else '')
                        w = getattr(e, 'weight', e.get('weight', 0) if isinstance(e, dict) else 0)
                        act = getattr(e, 'activation', e.get('activation', 0) if isinstance(e, dict) else 0)
                        topk_edge_str += f"{i+1}. {src} -[{rel}]-> {dst} (边权:{w:.2f}, 激活:{act:.2f})\n"

            # ── 构建认知上下文块 ──
            cog_str = f"""【交流认知分析】
言外行为: {cognitive_context.get('illocutionary_act', 'assertive')}
对话行为: {cognitive_context.get('dialogue_act', 'information_statement')}
回应期待: {cognitive_context.get('response_expectation', 'medium')}
建议目标: {', '.join(cognitive_context.get('suggested_reply_goals', ['acknowledge']))}
"""

            # ── MODE（LLM 参与模式）与注意力上下文 ──
            # （Bug 修复 2026-09-19：此块原来在 cog_str 定义之前执行 `cog_str +=`，
            #  mode 或 attention_context 非空时必然 UnboundLocalError。移到定义后。）
            # A1 接线（2026-09-30,CON-1/CON-2）：历史上门控参数只认形参，而
            #  生产调用经 compile_for_language 只在 cognitive_context 里带
            #  attention_context/mode 键（4477 构建 → L2 编译 399 行）→ 形参
            #  恒空 → 本块永不渲染，七分区上下文与 demand 三层结构成死数据。
            #  现在：形参缺失时回退读 cognitive_context；config 门
            #  nlp_render_cognitive_context=False 整块跳过（精确回滚）。
            if mode or attention_context or cognitive_context.get("attention_context") \
                    or cognitive_context.get("mode"):
                try:
                    import config as _cfg
                    _render_cc = bool(getattr(_cfg.DEFAULT_CONFIG, "get",
                                              lambda k, d=None: None)(
                        "nlp_render_cognitive_context", True))
                except Exception:
                    _render_cc = True
                if _render_cc:
                    mode = mode or cognitive_context.get("mode") or "language"
                    attention_context = attention_context or \
                        cognitive_context.get("attention_context") or {}
                    mode_line = f"模式: {mode or 'language'}"
                    att_lines = []
                    att_keys = ("active_core", "recent_episodic", "self",
                                "emotion", "goal", "relevant_semantic",
                                "unknowns")
                    for k in att_keys:
                        v = attention_context.get(k)
                        if v:
                            if isinstance(v, (list, tuple)):
                                if k == "active_core":
                                    # 带激活度的核心激活（active_core 项是
                                    # {"id","act"} dict）——激活是有量纲的
                                    # 运行时状态量，与 topk 同一呈现风格；
                                    # 其余分区只给名（机制不外显原则）
                                    att_lines.append(
                                        f"{k}: " + ", ".join(
                                            (f"{x.get('id')}({x.get('act')})"
                                             if isinstance(x, dict)
                                             else str(x)) for x in v[:8]))
                                else:
                                    att_lines.append(f"{k}: " + ", ".join(
                                        (x if isinstance(x, str)
                                         else x.get("id", "")) for x in v[:8]))
                            else:
                                att_lines.append(f"{k}: {v}")
                    att_block = "\n".join(att_lines)
                    cog_str += (
                        "\n【认知状态】（图谱运行时状态；你的回答需基于此"
                        "作为语言实现）\n"
                        + mode_line + "\n" + att_block + "\n"
                    )
                    # ── 认知资源路由：demand/gap 三层结构（analyze_
                    # cognitive_demand 的产物,量化路由依据）。意识层消费
                    # 的是定性观感而非数值——这里只有 top 维度名,不暴露
                    # 强度数值（机制不外显原则）。
                    _cd = cognitive_context.get("cognitive_demand") or {}
                    _cg = cognitive_context.get("cognitive_gap") or {}
                    _cr = cognitive_context.get("cognitive_resource") or {}
                    _dim_zh = {"memory": "记忆", "knowledge": "知识",
                               "reasoning": "推理", "emotion": "情绪",
                               "social": "社交", "action": "行动",
                               "curiosity": "好奇", "self_model": "自我",
                               "language": "语言"}
                    if _cd or _cg:
                        _dtop = sorted(
                            ((k, v) for k, v in _cd.items()
                             if isinstance(v, (int, float)) and v >= 0.3),
                            key=lambda kv: -float(kv[1]))[:3]
                        _gtop = sorted(
                            ((k, v) for k, v in _cg.items()
                             if isinstance(v, (int, float)) and v >= 0.3),
                            key=lambda kv: -float(kv[1]))[:2]
                        _route_lines = []
                        if _dtop:
                            _route_lines.append(
                                "认知资源需求: " + "、".join(
                                    _dim_zh.get(k, k) for k, _ in _dtop))
                        if _gtop:
                            _route_lines.append(
                                "缺口: " + "、".join(
                                    _dim_zh.get(k, k) for k, _ in _gtop))
                        if _cr.get("llm_reasoning", 0) >= 0.55:
                            _route_lines.append("资源定档: 高阶推理")
                        elif _cr.get("llm_interpret", 0) >= 0.45:
                            _route_lines.append("资源定档: LLM 解释")
                        elif _cr.get("llm_language", 0) >= 0.4:
                            _route_lines.append("资源定档: 语言实现")
                        elif _cr:
                            _route_lines.append("资源定档: 图谱自足")
                        if _route_lines:
                            cog_str += (
                                "\n【认知资源路由】（认知层先行分析,投影"
                                "参考——回答仍以实际交流情境为准,不被它"
                                "锁死）\n" + "\n".join(_route_lines) + "\n"
                            )

            # ── 言外行为的图谱投影（speech_act_graph：本轮话语在认知场
            # 点亮了什么——状态呈现，不是"指令→必须行动"的硬映射）──
            _sa = cognitive_context.get("speech_act_landscape") or {}
            if _sa.get("concept"):
                _co = "、".join(
                    f"{c.get('id')}(激活 {c.get('act')})"
                    for c in (_sa.get("coactivated") or [])[:6])
                cog_str += f"""
【言外行为·图谱】本轮话语在图谱中的言外行为=「{_sa['concept']}」
（概念激活 {_sa.get('activation')}）；由它扩散共激活：{_co or '暂无明显共激活'}。
这是当前认知状态的投影——回应姿态由全部激活共同参与竞争后决定，
概念被点亮不等于必须照做。
"""

            # ── 发言提纲（认知层定型的说话计划；措辞自由、内容边界不自由）──
            _ot = cognitive_context.get("outline") or {}
            if _ot.get("move"):
                _ol = [f"- 姿态：{_ot.get('move')}"
                       + (f"；意图：{_ot.get('intent')}" if _ot.get("intent") else "")
                       + (f"；长度：{_ot['length']}" if _ot.get("length") else "")]
                for _m in (_ot.get("must") or [])[:5]:
                    _ol.append(f"- 必须交代：{_m}")
                for _n in (_ot.get("must_not") or [])[:4]:
                    _ol.append(f"- 禁止：{_n}")
                _ol.append("- 语气：以当前心情为自然依据（禁止报告内部机制）")
                cog_str += chr(10) + "【发言提纲】（这轮话的计划已由认知层定型；你只决定措辞）" + chr(10) + chr(10).join(_ol) + chr(10)

            # ── Reflection Evolution R3: 行为倾向注入 ──
            # §十六 机制不外显：只给中文行为名与定性强度词，不暴露
            # strength/activation 数值、不出现"disposition/倾向边权"等术语——
            # 倾向的影响靠图激活实现，这里是让她"自然地更倾向这么做"，
            # 而不是"根据我的 disposition 我应该…"。
            tendencies = cognitive_context.get("behavior_tendencies") or []
            if tendencies:
                _bhv_zh = {"respond": "回应", "ask": "追问", "acknowledge": "确认",
                           "elaborate": "详述", "empathize": "共情", "share": "分享",
                           "continue": "延续话题", "end": "收尾", "silence": "安静",
                           "explore": "主动探索"}

                def _qual(t):
                    s = float(t.get("strength") or 0)
                    if t.get("status") == "stable" and s >= 0.6:
                        return "是她比较自然的做法"
                    if s >= 0.4:
                        return "她最近常这么做"
                    return "她可以顺手这么做"

                def _t_name(t):
                    # 图谱行动概念名优先（提议出的新行动有中文名），
                    # 其次 LEGACY 中文表，最后原样
                    return (t.get("label")
                            or _bhv_zh.get(t.get("behavior"),
                                           t.get("behavior")))
                tend_lines = "、".join(
                    f"{_t_name(t)}"
                    f"（{_qual(t)}）" for t in tendencies)
                cog_str += f"""
【她此刻的自然倾向】（从经历中形成，自然地体现，不要复述或解释这个倾向本身）
{tend_lines}
"""

            # ── 网络搜索动作结果注入（网络搜索节点执行产物）──
            web_results = cognitive_context.get("web_results") or []
            if web_results:
                web_lines = "\n".join(
                    f"{i+1}. {r.get('title','')}\n   {r.get('snippet','')[:200]}\n   来源: {r.get('url','')[:100]}"
                    for i, r in enumerate(web_results[:5]))
                cog_str += f"""
【网络搜索结果】（来自互联网的实时信息，非图谱记忆；回答时可引用，需与图谱知识区分）
{web_lines}
"""

            # ── 文件操作动作结果注入（文件操作节点执行产物）──
            fa = cognitive_context.get("file_action")
            if fa:
                if fa.get("ok"):
                    _fa_verb = "覆盖写入" if fa.get("overwritten") else "创建"
                    cog_str += f"""
【动作执行结果】你刚刚完成了文件操作：已{_fa_verb}文件 {fa.get('path')}（{fa.get('size', 0)} 字符）。请在回答中用第一人称自然告知用户完成情况。
"""
                else:
                    cog_str += f"""
【动作执行结果】你尝试执行文件操作但失败了：{fa.get('error', '未知原因')}。请在回答中诚实告知用户失败与原因。
"""

            # ── 屏幕感知结果注入（看屏幕节点执行产物）──
            eye_result = cognitive_context.get("eye_result")
            if eye_result:
                items = eye_result.get("items") or []
                eye_lines = "\n".join(
                    f"- 「{it.get('text','')[:60]}」 位于屏幕 ({it['center'][0]},{it['center'][1]})"
                    for it in items[:12])
                cog_str += f"""
【你刚看到的屏幕内容】（你的眼睛刚截屏识别，共 {eye_result.get('count', 0)} 条文字；回答时用第一人称描述你看到了什么）
{eye_lines}
"""

            # ── 回应约束（交流决策层生成，图谱/行为信号）──
            rc = cognitive_context.get("response_constraints") or []
            if rc:
                rc_map = {
                    "no_question": "不要向用户提问——当前没有真实的信息需求；用共鸣/确认/联想回应分享",
                    "ack_only": "只需简短确认（例如：好。两个字级别），不展开，不开启新话题",
                    "no_new_topic": "不要开启或提议任何新话题",
                    "minimal_len": "回应一两句话以内即可",
                }
                rc_lines = "\n".join(f"- {rc_map.get(c, c)}" for c in rc)
                cog_str += f"""
【回应约束】（来自你的交流决策，必须遵守）
{rc_lines}
"""

            # ── 近期对话（语境/指代依据；含 FAS 主动发言）──
            recent = cognitive_context.get("recent_dialogue") or []
            recent = [x for x in recent if x.get("user_input") or x.get("system_response")]
            if recent:
                dlg_lines = "\n".join(
                    ("用户：" if x.get("user_input") and not x.get("user_input").startswith("(FAS") else "FAS：")
                    + str(x.get("user_input") or x.get("system_response"))[:80]
                    for x in recent[-6:])
                cog_str += f"""
【近期对话】（就在刚才发生，'你这话/刚才'等指代以此为准）
{dlg_lines}
"""

            # ── Minecraft 动作结果注入（状态感知：pending/queued 不得说成完成）──
            _act_ev = cognitive_context.get("action_evidence") or {}
            if not _act_ev.get("status") or _act_ev.get("status") == EV_NONE:
                _raw_act = cognitive_context.get("mc_action")
                _act_ev = (classify_action_evidence(_raw_act, source="turn:mc_action")
                           if _raw_act else {})
            if _act_ev.get("status") and _act_ev["status"] != EV_NONE:
                _line = render_action_evidence(_act_ev)
                if _line:
                    cog_str += chr(10) + _line + chr(10)

            # ── Minecraft 会话动作（端口请求/连接结果/取消）──
            mc_sess = cognitive_context.get("mc_session") or {}
            if mc_sess.get("request_message"):
                cog_str += f"""
【动作执行结果】你想进入 Minecraft 世界，请向用户请求局域网端口号：
"{mc_sess['request_message']}"
（自然地说出来，等待用户提供端口）
"""
            elif mc_sess.get("cancelled"):
                cog_str += """
【动作执行结果】用户决定暂不进入，自然地确认即可（如"好，那先不进了"），不追问。
"""
            elif mc_sess:
                _sess_ev = cognitive_context.get("session_evidence") or {}
                if not _sess_ev.get("status"):
                    _sess_ev = classify_action_evidence(mc_sess, source="turn:mc_session")
                _line = render_action_evidence(_sess_ev, default_desc="进入 Minecraft 世界")
                if _line:
                    cog_str += chr(10) + _line + chr(10)

            # ── 交流决定（decision 分区：已经形成的裁决，语言层不许翻案）──
            _dec = cognitive_context.get("decision") or {}
            if _dec:
                _dl = [f"裁决={_dec.get('response_mode')}",
                       f"应回应={'是' if _dec.get('should_respond', True) else '否'}",
                       f"提问={'是' if _dec.get('ask_question') else '否'}"]
                if _dec.get("length_constraint"):
                    _dl.append(f"长度={_dec['length_constraint']}")
                _ai = _dec.get("action_intent")
                if _ai and _ai.get("action_type"):
                    _dl.append(f"行动意图={_ai['action_type']}({_ai.get('outcome') or '已受理'})")
                cog_str += f"""
【交流决定】（认知层已经做出的裁决，回答必须与之一致；是否回应、是否提问、
行动与结果都以这里为准，不要自行改变）
{'；'.join(_dl)}
"""

            # ── 世界状态（她在具身环境里的真实状态；未接入时不渲染）──
            # 与【当前心情】同级：都是"她此刻的状态"，不是知识、也不是能力清单。
            # 位置/状态类问题必须以此为准——不许猜。
            _ws = cognitive_context.get("world_state") or {}
            if _ws:
                _ws_lines = chr(10).join(f"- {k} = {v}" for k, v in _ws.items())
                cog_str += f"""
【世界状态】（你在游戏世界里的真实状态，来自感知；被问到位置/血量/手持物/附近有什么时，必须用这里的数值回答，不许猜测或含糊）
{_ws_lines}
"""

            # ── 当前心情（瞬态内部状态→语气依据，禁止报告机制）──
            mood = cognitive_context.get("mood")
            if mood:
                cog_str += f"""
【当前心情】{mood}——语气的自然依据，不直接向用户报告
"""

            # 自我能力不在这里渲染：能力是图谱节点（self_capability），
            # 随语义召回与扩散进入 Top-K，已由上方节点列表呈现。

            # ── 认知触发事件（触发器产物，llm_mode>=1 才到这里）──
            # 只呈现"发生了什么"和可用的动作结果；是否提及、怎么处理由你决定。
            _cog_events = cognitive_context.get("cognitive_events") or []
            if _cog_events:
                lines = []
                for ev in _cog_events[:3]:
                    _desc = (f"{ev.get('monitor_name') or ev.get('monitor_id')} "
                             f"{ev.get('previous')} → {ev.get('current')}")
                    _act = (ev.get("action_result") or {}).get("type")
                    _act_ok = (ev.get("action_result") or {}).get("ok")
                    lines.append(f"- {_desc}；触发动作: {_act}（{'完成' if _act_ok else '未完成'}）")
                cog_str += f"""
【认知触发】（你的状态检测器刚刚捕到的变化，已按触发规则处理；是否向用户提及由你决定）
{chr(10).join(lines)}
"""

            # ── 社交表达场景 ──
            is_social = (
                context_type == "social"
                or cognitive_context.get("illocutionary_act") == "expressive"
            )
            if is_social:
                # A 修复：社交路径同样注入图谱记忆——共情归共情，
                # 但输入含事实内容时（如"害怕离合器"），模型需要知识依据，
                # 否则只能靠参数知识裸猜（C2 事件根因）。
                _mem_block = ""
                if topk_str.strip():
                    _mem_block = f"""
【相关记忆】（以情感回应为主，事实以记忆为依据）
{topk_str}
{topk_edge_str}"""
                user_prompt = f"""{cog_str}{_mem_block}
【用户原始输入】
{original_text}

请根据以上认知分析自然回应："""

                answer_prompt = ChatPromptTemplate.from_messages([
                    ("system", SOCIAL_ANSWER_SYSTEM_PROMPT),
                    ("human", "{input}")
                ])

                chain = answer_prompt | self._chat()
                response = chain.invoke({"input": user_prompt})
                answer_text = response.content if hasattr(response, 'content') else str(response)
                logger.info(f"[Answer] (社交) 生成完成: {answer_text[:120]}...")
                return answer_text.strip()

            # ── 知识/经历回应 ──
            user_prompt = f"""{cog_str}
【相关记忆】
{topk_str}
{topk_edge_str}
【用户原始输入】
{original_text}

请根据以上认知分析及你的记忆回应："""

            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", ANSWER_SYSTEM_PROMPT),
                ("human", "{input}")
            ])

            chain = answer_prompt | self._chat()
            response = chain.invoke({"input": user_prompt})
            answer_text = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"[Answer] 生成完成: {answer_text[:120]}...")
            return answer_text.strip()

        except Exception as e:
            logger.exception(f"[Answer异常] {e}")
            return f"[回答生成失败: {e}]"

    def _sanitize_expand_result(self, raw_data: Any) -> Dict[str, Any]:
        result = {
            "nodes": [],
            "edges": [],
            "node_meta": [],
            "edge_meta": []
        }

        if not isinstance(raw_data, dict):
            return result

        raw_nodes = raw_data.get("nodes", [])
        if isinstance(raw_nodes, list):
            nodes = []
            for x in raw_nodes:
                if isinstance(x, dict):
                    v = x.get("id", x.get("name", x.get("node", "")))
                else:
                    v = x
                v = str(v).strip()
                if v:
                    nodes.append(v)
            result["nodes"] = nodes
        result["nodes"] = [x for x in result["nodes"] if x]
        result["nodes"] = list(dict.fromkeys(result["nodes"]))

        raw_edges = raw_data.get("edges", [])
        if isinstance(raw_edges, list):
            edges = []
            for e in raw_edges:
                if not isinstance(e, dict):
                    continue
                src = str(e.get("src", e.get("from", e.get("source", "")))).strip()
                dst = str(e.get("dst", e.get("to", e.get("target", "")))).strip()
                rel = str(e.get("type", e.get("relation", e.get("predicate", "关联")))).strip() or "关联"
                import graph_schema as _gs_rel
                rel = _gs_rel.normalize_relation(rel)
                w = e.get("weight", 0.5)
                try:
                    w = float(w)
                    w = max(0.0, min(1.0, w))
                except Exception:
                    w = 0.5
                item = {
                    "src": src,
                    "dst": dst,
                    "type": rel,
                    "relation": rel,
                    "weight": w
                }
                conf = e.get("confidence", None)
                if conf is not None:
                    try:
                        item["confidence"] = max(0.0, min(1.0, float(conf)))
                    except (ValueError, TypeError):
                        pass
                reason = e.get("reason", None)
                if reason is not None:
                    item["reason"] = str(reason)[:300]
                if src and dst and src != dst:
                    edges.append(item)
            result["edges"] = edges

        node_meta = raw_data.get("node_meta", raw_data.get("nodes_meta", []))
        if isinstance(node_meta, list):
            nm = []
            for it in node_meta:
                if not isinstance(it, dict):
                    continue
                nid = str(it.get("id", it.get("node", ""))).strip()
                if not nid:
                    continue
                item = {"id": nid}
                conf = it.get("confidence", None)
                if conf is not None:
                    try:
                        item["confidence"] = max(0.0, min(1.0, float(conf)))
                    except (ValueError, TypeError):
                        pass
                reason = it.get("reason", None)
                if reason is not None:
                    item["reason"] = str(reason)[:300]
                nm.append(item)
            result["node_meta"] = nm

        edge_meta = raw_data.get("edge_meta", [])
        if isinstance(edge_meta, list):
            em = []
            for it in edge_meta:
                if not isinstance(it, dict):
                    continue
                src = str(it.get("src", "")).strip()
                dst = str(it.get("dst", "")).strip()
                rel = str(it.get("type", it.get("relation", ""))).strip()
                import graph_schema as _gs_rel
                rel = _gs_rel.normalize_relation(rel) if rel else rel
                if not (src and dst and rel):
                    continue
                item = {"src": src, "dst": dst, "type": rel, "relation": rel}
                conf = it.get("confidence", None)
                if conf is not None:
                    try:
                        item["confidence"] = max(0.0, min(1.0, float(conf)))
                    except (ValueError, TypeError):
                        pass
                reason = it.get("reason", None)
                if reason is not None:
                    item["reason"] = str(reason)[:300]
                em.append(item)
            result["edge_meta"] = em

        return result

    def expand_concept(self, word: str, context: str = "") -> dict:
        """
        【学习模块专属方法】利用大模型围绕指定核心词推理生成高质量的邻居实体与关系链路。
        context 为可选的附加说明，帮助 LLM 消歧（如'这是一款Steam生存游戏'），
        但不影响图谱中节点的名字。
        """
        # Architecture Refactor 0.1: 使用统一 CONCEPT_EXPAND 模板
        system_prompt = CONCEPT_EXPAND
        # 构建带可选上下文的用户消息
        if context:
            user_msg = f'请围绕核心词汇进行高质量关联知识扩充：{word}\n\n【附加说明】{context}'
        else:
            user_msg = f'请围绕核心词汇进行高质量关联知识扩充：{word}'

        prompt = ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            ('user', '{{input}}')
        ], template_format="mustache")

        try:
            chain = prompt | self._expand() | JsonOutputParser()
            raw = chain.invoke({'input': user_msg})
            return self._sanitize_expand_result(raw)
        except Exception as e:
            logger.error(f'LLM 扩充概念失败: {e}')
            return {'nodes': [], 'edges': []}

    def get_logs(self):
        return self.logs

    def ask(self, task: str, user_input: str) -> str:
        """Architecture Refactor 0.1: 通用 LLM 问答接口。

        供 self_graph、curiosity_engine 等模块使用，通过统一 Prompt 模板
        调用 LLM，避免各模块自行拼接 prompt。
        """
        from prompt_templates import build_prompt
        from langchain_core.prompts import ChatPromptTemplate

        system_prompt = build_prompt(task)
        if not system_prompt:
            logger.warning(f"[Ask] 未知任务模板: {task}，回退到纯文本模式")
            system_prompt = "你是 Fascinator 认知图谱系统。请简洁回答。"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{{input}}")
        ], template_format="mustache")
        chain = prompt | self.chat_llm
        response = chain.invoke({"input": user_input})
        answer_text = response.content if hasattr(response, 'content') else str(response)
        return answer_text.strip()

# =========================
# Main 函数用于测试
# =========================
def main():
    print("🚀 初始化 NLP 处理器...")
    nlp = NLPProcessor(model="qwen2.5:7b-instruct-q5_k_m")

    print("💡 输入句子进行测试 (输入 'quit' 退出):")

    while True:
        try:
            text = input("\n📝 请输入: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            if not text:
                continue

            result = nlp.process(text)

            print("\n" + "=" * 50)
            print("🧠 解析结果:")
            print(f"🗣️  言语行为: {result['speech_act']}")

            if result['nodes']:
                print(f"📌 节点 ({len(result['nodes'])}):")
                for node in result['nodes']:
                    print(f"   - {node}")
            else:
                print("📌 节点: 无")

            if result['edges']:
                print(f"🔗 边 ({len(result['edges'])}):")
                for edge in result['edges']:
                    print(f"   {edge['src']} --[{edge['type']} ({edge['weight']})]--> {edge['dst']}")
            else:
                print("🔗 边: 无")
            print("=" * 50)

        except KeyboardInterrupt:
            print("\n\n👋 检测到退出信号，再见！")
            break
        except Exception as e:
            logger.error(f" 发生错误: {e}")
            continue


if __name__ == "__main__":
    main()



