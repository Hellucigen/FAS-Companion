# app.py — Fascinator Flask 后端（修复增强完整版）
# ============================================================================
# Fascinator 认知图谱系统的 Web 服务入口。
#
# 功能概述：
#   1. 加载知识图谱（主图谱 + 外部知识库）
#   2. 初始化扩散引擎、NLP 处理器、情景缓冲区、Embedding 管理器
#   3. 提供 RESTful API 供前端 UI 调用（见图文 README 的 API 文档）
#   4. 支持自动图谱持久化（每次修改后保存到 JSON）
#
# 启动方式：
#   python app.py                    # 默认端口 5000，只监听 127.0.0.1
#   PORT=8080 python app.py          # 自定义端口
#   FAS_HOST=0.0.0.0 python app.py   # 开放到局域网（非回环来源需 token，见 api_guard）
#   FAS_NO_BROWSER=1 python app.py   # 不自动打开浏览器
#   FASCINATOR_GRAPH=my_graph.json python app.py  # 自定义图谱路径
# ============================================================================

import os
import time  # 收尾修复 2026-09-22（P0 冷启动崩溃）：persona 在本文件上方区段
# 就经 _state_clock 的 fallback 调 time.time()，而旧代码的模块级 import time
# 在其后数百行——任何冷启动必炸 NameError。提到导入块顶部，原位置只留说明。
# bge 模型已本地化：禁止 sentence-transformers 联网校验（HF 不可达时每次请求阻塞数分钟）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 模型权重加载进度条（"Loading weights: 100%|███| 408/408"）在服务器日志里是噪音，
# 它是 transformers 在 import 时按本环境变量决定是否启用，所以必须在这里设——
# 一旦 transformers 被导入（sentence-transformers 加载 bge 时）就改不了了。
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import logging
import re
import traceback
import random
import threading
import atexit
from collections import deque

# ── 仓库拆分 bootstrap（2026-10-08，docs/ARCHITECTURE_PLAN.md 两轨制） ──────
# 本仓只装陪伴层（LLM 语言栈/UI/外设）；认知核心模块在同级仓库 FAS-Cognitive。
# 认知↔陪伴的边界契约在 FAS-Cognitive/fas/（contracts/protocols），依赖方向
# 单向：陪伴 → 认知。环境变量 FAS_COG_ROOT 可指到别处。
import sys
_FAS_COG = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FAS-Cognitive")
if os.path.isdir(_FAS_COG):
    if _FAS_COG not in sys.path:
        sys.path.insert(0, _FAS_COG)
else:
    raise ImportError(
        "找不到 FAS-Cognitive 同级仓库：" + _FAS_COG +
        "\n请把两个仓放在同一父目录，或设置环境变量 FAS_COG_ROOT。")

from flask import Flask, jsonify, request, send_from_directory, g
from flask_cors import CORS

from graph_model import KnowledgeGraph, Node, Edge, is_cognitive_visible, now_str
from diffusion_engine import DiffusionEngine
from nlp_processor import NLPProcessor
from config import DEFAULT_CONFIG
from knowledge_pack_manager import KnowledgePackManager, get_pack_manager
from self_graph import (
    set_preference, adjust_preference,
    inject_emotion, get_current_emotion, get_preferences,
    set_goal, get_goals, connect_experience_to_self,
    add_experience, self_diffusion_for_answer, get_self_state,
    decay_preferences, get_self_model_summary, get_self_model_audit
)
from graph_evolution_log import get_evolution_log
from llm_provider import ollama_enabled
from conversation_gap_detector import ConversationGapDetector
import curiosity_engine
from chat_log import get_chat_log
from self_model import SelfMemoryUpdater  # Phase 1: Self Model
from reflection_engine import ReflectionEngine  # Phase 2: Reflection
from narrative_trigger import narrative_score  # 事件结构化叙事触发
from cognition_modes import (  # LLM 参与模式：认知需求→预算→注意力上下文
    BudgetManager, attention_context,
    MODE0_GRAPH_ONLY, MODE1_LANGUAGE, MODE2_INTERPRET, MODE3_REASON,
    MODE4_REFLECT, MODE5_DEEP, MODE_COST)
from temporal_awareness import (  # 时间感知锚定
    ground as time_ground, link_event_to_bucket, ensure_bucket_nodes)
from dialogue_decision import (  # 对话行为层：交流决策
    dialogue_decide, mark_termination, inhibit_topics, inhibition_active,
    bind_lock_registry, release_termination, termination_state)
from disposition_store import (  # Reflection Evolution: 行为倾向
    DispositionStore, classify_behavior,
    context_for_dialogue_act, MODIFIER_EMOTION_LOW, MODIFIER_FAMILIAR,
)
from drive_engine import DriveEvaluator  # Phase 3 MVP: Drive System
from actions.action_engine import ActionSelector  # Phase 4 MVP: Action System
from compound_phrase_handler import CompoundPhraseHandler  # Phase 2: Compound Phrase Handling

# ── Ear 听觉感知模块 ──────────────────────────────────────
from ear.ear_processor import EarProcessor, get_ear_processor

# ── Vision 视觉感知模块 ───────────────────────────────────
from vision.vision_processor import VisionProcessor, get_vision_processor

# ── Action 动作注册表 ─────────────────────────────────────
from Action import discover_actions, list_actions, get_action_func

# =========================================================
# 日志
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ── 统一运行日志（fas_log，2026-09-22）──
# 纯 Observability：只加 logs/ 文件流 + 结构化事件，控制台与认知行为不变。
# 配置在 config.py "统一运行日志" 段；logging_enabled=false 时完全旁路。
import fas_log
logger = logging.getLogger(__name__)

# =========================================================
# 初始化
# =========================================================

app = Flask(
    __name__,
    static_folder="."
)

CORS(app)

config = dict(DEFAULT_CONFIG)

# 统一运行日志装配（config 加载后立刻做；总开关 logging_enabled=false 即全旁路）
if config.get("logging_enabled", True):
    fas_log.setup(log_dir=str(config.get("log_dir", "logs")),
                  config=config,
                  summary_interval_s=float(
                      config.get("log_summary_interval_s", 300)))
    fas_log.install_excepthooks()

# 旧图谱路径（向后兼容）
GRAPH_PATH = os.environ.get("FASCINATOR_GRAPH", "knowledge_graph.json")

# ── execution 写入闸门 ──
# 为什么需要：节点 execution 会被 diffusion_engine.execute_action() 的 exec() 执行，
# 而这三处 HTTP 写入口无鉴权——任何能发请求的客户端都能借此注入代码。
# 两道闸（缺一不可）：
#   1. config.allow_api_execution_write（默认 False，可用 PUT /api/config 热更新）
#   2. 来源必须是回环地址——开关打开也不给局域网留口子
# 只拦"写入非空代码"；写 null/空串是清除旧代码，属降级操作，照常放行
# （前端建点/改点总是带 execution 键，值为 null，不能因此误伤）。
_LOOPBACK_ADDRS = ("127.0.0.1", "::1", "localhost")


def _execution_write_denied(data) -> str:
    """返回拒绝原因；允许写入时返回空串。

    判据用值而不是键：前端无论建点还是改点都会带 execution 键。
    """
    _exe = data.get("execution", None) if isinstance(data, dict) else None
    if not (isinstance(_exe, str) and _exe.strip()):
        return ""
    if not config.get("allow_api_execution_write", False):
        return ("execution 写入已禁用（config.allow_api_execution_write=false）："
                "该字段会被 exec() 执行，属代码注入面。"
                "确需从前端写动作代码，请先 PUT /api/config 打开开关。")
    try:
        _addr = (request.remote_addr or "").strip()
    except Exception as _rae:
        logger.warning(f"[Security] remote_addr 读取失败(按非回环拒绝): {_rae}")
        _addr = ""
    if _addr not in _LOOPBACK_ADDRS:
        return f"execution 写入只接受回环地址请求（当前来源 {_addr or '未知'}）"
    return ""


# ── API 访问守卫（§18.2 #19，2026-09-22）──
# 两道独立防线，判定逻辑全在 api_guard.py（纯函数，单测不启 Flask）：
#   1. Origin/Referer 核验：**常开**。恶意网页以 127.0.0.1 为"同源"
#      发 XHR 的 CSRF/DNS rebinding 通道在这里被截——与 token 无关。
#   2. token 核验：token 生效时（config/FAS_API_TOKEN/非回环绑定自动
#      生成）检查**非回环来源**；回环默认不受影响（本机零摩擦不变，
#      api_token_require_loopback=true 可收紧）。
from api_guard import evaluate_request as _evaluate_request
from graph_rotation import maybe_rotate as _rotate_graph_snapshot

_API_TOKEN_PATH = os.path.join("data", "api_token.json")


def _resolved_api_token() -> str:
    """解析优先级：运行时 config（可热更新）> FAS_API_TOKEN > token 文件。"""
    try:
        t = str(config.get("api_token") or "").strip()
    except Exception as _ce:
        logger.warning(f"[Security] api_token 读取失败(回落空 token): {_ce}")
        t = ""
    if t:
        return t
    t = (os.environ.get("FAS_API_TOKEN") or "").strip()
    if t:
        return t
    try:
        with open(_API_TOKEN_PATH, "r", encoding="utf-8") as f:
            return str(_json.load(f).get("token") or "").strip()
    except Exception:
        return ""


def _ensure_api_token_for_bind(host_is_local: bool) -> str:
    """app.run 前调用：非回环监听且无 token → 生成一次并持久化。"""
    tok = _resolved_api_token()
    if host_is_local or tok:
        return tok
    import secrets
    tok = secrets.token_urlsafe(24)
    try:
        os.makedirs(os.path.dirname(_API_TOKEN_PATH) or ".", exist_ok=True)
        with open(_API_TOKEN_PATH, "w", encoding="utf-8") as f:
            _json.dump({"token": tok,
                        "note": "FAS 非回环绑定自动生成；删除文件重启=轮换"},
                       f, ensure_ascii=False, indent=1)
    except Exception as e:
        logger.warning(f"[SECURITY] token 持久化失败（仅本进程有效）: {e}")
    return tok


@app.before_request
def _api_guard_before():
    try:
        supplied = str(request.headers.get("X-FAS-Token") or "").strip()
        _auth = str(request.headers.get("Authorization") or "").strip()
        if not supplied and _auth.lower().startswith("bearer "):
            supplied = _auth[7:].strip()
        via_query = False
        if not supplied:
            supplied = str(request.cookies.get("fas_token") or "").strip()
        if not supplied:
            supplied = str(request.args.get("token") or "").strip()
            via_query = bool(supplied)
        expected = _resolved_api_token()
        out = _evaluate_request(
            remote_addr=request.remote_addr or "",
            path=request.path or "/",
            origin=request.headers.get("Origin") or "",
            referer=request.headers.get("Referer") or "",
            request_host=request.host or "",
            token_expected=expected,
            token_supplied=supplied,
            token_via_query=via_query,
            require_token_on_loopback=bool(
                config.get("api_token_require_loopback", False)))
        if out["mode"] == "bootstrap":
            g._fas_set_cookie = expected
        if out["allow"]:
            return None
        return jsonify({"success": False, "error": out["reason"],
                        "status": out["status"]}), out["status"]
    except Exception as e:
        # 守卫**自身**异常不锁死服务（放行并大声记日志）；注意 token
        # 不匹配走上面的正常拒绝分支，不经过这里——fail-open 只覆盖
        # 判定代码的意外错误，不覆盖任何一次"明确判定为拒绝"。
        logger.error(f"[SECURITY] 请求守卫判定异常（放行）: {e}")
        return None


@app.after_request
def _api_guard_cookie(resp):
    tok = getattr(g, "_fas_set_cookie", None)
    if tok:
        resp.set_cookie("fas_token", tok, httponly=True, samesite="Lax")
    return resp

# =========================================================
# [V4 Phase 1] 加载 Knowledge Pack → 生成 Runtime Graph
# =========================================================

logger.info("=" * 60)
logger.info("[Pack] 初始化 Knowledge Pack Manager...")
logger.info("=" * 60)

# ── 节点重命名追踪 ──
RENAME_MAP_PATH = os.path.join("data", "rename_map.json")

def _load_rename_map() -> dict:
    """加载重命名映射：{new_id: old_id}"""
    try:
        if os.path.exists(RENAME_MAP_PATH):
            with open(RENAME_MAP_PATH, "r", encoding="utf-8") as f:
                return _json.load(f)
    except Exception as _rne:
        logger.warning(f"[RenameMap] 加载失败(回落空映射): {_rne}")
    return {}

def _save_rename_map(rmap: dict):
    """保存重命名映射"""
    try:
        os.makedirs(os.path.dirname(RENAME_MAP_PATH), exist_ok=True)
        with open(RENAME_MAP_PATH, "w", encoding="utf-8") as f:
            _json.dump(rmap, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[RenameMap] 保存失败: {e}")

_rename_map = _load_rename_map()

pack_mgr = get_pack_manager()
pack_mgr.scan_packs()
kg = pack_mgr.merge_runtime_graph()

# ── 应用重命名映射：将 Pack 中的旧节点 ID 改名为新 ID ──
if _rename_map:
    _applied_renames = 0
    for new_id, old_id in list(_rename_map.items()):
        if old_id in kg.nodes and new_id not in kg.nodes:
            if not kg.rename_node(old_id, new_id):
                logger.warning(f"[Rename] 改名失败（目标已存在？）: {old_id} → {new_id}")
                continue
            # 更新 pack_mgr 的节点映射
            if old_id in pack_mgr.runtime_node_map:
                pack_mgr.runtime_node_map[new_id] = pack_mgr.runtime_node_map.pop(old_id)
            _applied_renames += 1
    if _applied_renames > 0:
        logger.info(f"[Rename] 已应用 {_applied_renames} 个重命名映射")
        # 写回 runtime_graph 使其与当前状态一致
        pack_mgr.runtime_graph = kg
        pack_mgr.save_runtime_graph()

# 加载运行时持久化的修改（如新增节点）
_runtime_backup = os.path.join("data", "runtime_graph.json")
if os.path.exists(_runtime_backup):
    try:
        saved_kg = KnowledgeGraph.load(_runtime_backup)
        for nid, node in saved_kg.nodes.items():
            if nid not in kg.nodes:
                kg.nodes[nid] = node
                pack_mgr.runtime_node_map[nid] = "_runtime"
            else:
                # 更新已有节点的属性（label, weight 等，但不覆盖重命名后的 ID）
                if node.label:
                    kg.nodes[nid].label = node.label
                if hasattr(node, 'weight') and node.weight != 0.5:
                    kg.nodes[nid].weight = node.weight
        for edge in saved_kg.edges:
            if edge.src in kg.nodes and edge.dst in kg.nodes:
                exists = any(e.src == edge.src and e.dst == edge.dst and e.relation == edge.relation for e in kg.edges)
                if not exists:
                    kg.edges.append(edge)
        logger.info(f"[System] 已加载运行时持久化 ({len(saved_kg.nodes)} nodes)")
    except Exception as e:
        logger.warning(f"[System] 加载运行时备份失败: {e}")

# overlay 期间走了裸容器直写（节点/边批量灌入）——重建派生邻接索引
kg.rebuild_indexes()

logger.info(
    f"[系统] Runtime Graph 生成完成 "
    f"(节点: {len(kg.nodes)}, "
    f"边: {len(kg.edges)}, "
    f"启用 Pack: {len(pack_mgr.get_enabled_packs())})"
)

# ═══════════════════════════════════════════════════════════════
# Architecture Refactor 0.1: Bootstrap 认知类型化
# ═══════════════════════════════════════════════════════════════

_bootstrap_log = {"self_fixed": False, "spaces_assigned": 0, "categories_assigned": 0}

# ── 1. Self 节点修复 ──
_self_node = kg.get_node("Self")
if _self_node:
    if _self_node.label != "self":
        _self_node.label = "self"
        _bootstrap_log["self_fixed"] = True
        logger.info("[Bootstrap] Self.label: declarative-semantic → self")
    if getattr(_self_node, 'graph_space', 'semantic') != "self":
        _self_node.graph_space = "self"
        logger.info("[Bootstrap] Self.graph_space → self")

# ── 2. 节点 graph_space 补全 ──
_fallback_map = config.get("graph_space_fallback_map", {})
for _nid, _node in kg.nodes.items():
    _current_space = getattr(_node, 'graph_space', '') or ''
    if _current_space not in ("semantic", "episodic", "cognitive", "self"):
        _mapped = _fallback_map.get(_node.label, "semantic")
        _node.graph_space = _mapped
        _bootstrap_log["spaces_assigned"] += 1

if _bootstrap_log["spaces_assigned"] > 0:
    logger.info(f"[Bootstrap] 为 {_bootstrap_log['spaces_assigned']} 个节点补全了 graph_space")

# ── 3. 边 relation_category 补全 ──
_relation_ontology = config.get("relation_ontology", {})
for _edge in kg.edges:
    _current_cat = getattr(_edge, 'relation_category', '') or ''
    if _current_cat not in ("semantic_relation", "causal_relation", "temporal_relation",
                            "emotional_relation", "social_relation", "cognitive_relation",
                            "procedural_relation"):
        _mapped_cat = _relation_ontology.get(_edge.relation, "semantic_relation")
        _edge.relation_category = _mapped_cat
        _bootstrap_log["categories_assigned"] += 1

if _bootstrap_log["categories_assigned"] > 0:
    logger.info(f"[Bootstrap] 为 {_bootstrap_log['categories_assigned']} 条边补全了 relation_category")

# ── 4. 统计各空间分布 ──
_space_counts = {"semantic": 0, "episodic": 0, "cognitive": 0, "self": 0}
for _n in kg.nodes.values():
    _sp = getattr(_n, "graph_space", "semantic")
    _space_counts[_sp] = _space_counts.get(_sp, 0) + 1
logger.info(f"[Bootstrap] 认知空间分布: {dict(_space_counts)}")

# ═══════════════════════════════════════════════════════════════
# 激活是瞬时认知状态，不应随图谱结构持久化。
# 服务器重启 = 认知重置，所有节点从零开始。
_reset_count = 0
for _n in kg.nodes.values():
    if _n.activation > 0:
        _n.activation = 0.0
        _reset_count += 1
for _e in kg.edges:
    if _e.activation > 0:
        _e.activation = 0.0
if _reset_count > 0:
    logger.info(f"[系统] 启动时重置了 {_reset_count} 个节点的激活值")

# =========================================================
# 初始化扩散引擎
# =========================================================

engine = DiffusionEngine(kg, config)

# ── Self Model Phase 1: 持续性主体模型 ──
self_updater = SelfMemoryUpdater(kg, config)
logger.info("[SelfModel] SelfMemoryUpdater 已初始化")

# ── 当前焦点事件指针（论文 §3.3 自认知子图）─────────────────────
# 自认知子图存储"当前认知焦点事件框架"。这里落地为：
#   当前焦点事件 -[指向]→ 进行中的头部事件（cognitive_relation）
# 提取路径把指向的事件及其子事件注入 LLM 上下文，用于延续识别。

_FOCUS_POINTER_ID = "当前焦点事件"


def _update_focus_event(kg: KnowledgeGraph, ev_id: str, parent_id: str = None):
    """把焦点指针移动到目标事件（有父事件时聚焦父事件）。"""
    from graph_model import Edge as _GEdge, Node as _GNode
    target = parent_id or ev_id
    if target not in kg.nodes:
        return
    if _FOCUS_POINTER_ID not in kg.nodes:
        kg.add_node(_GNode(
            id=_FOCUS_POINTER_ID, weight=0.6, label="infrastructure",
            graph_space="self", extra_attrs={"type": "focus_event_pointer"}))
        engine.name_to_node[_FOCUS_POINTER_ID] = kg.nodes[_FOCUS_POINTER_ID]
        logger.info("[Focus] 创建焦点事件指针节点")
    # 清掉全部旧指向边（不看 dst——可能存在多条历史指针），再挂新指针
    kg.remove_edges(src=_FOCUS_POINTER_ID, relation="指向")
    kg.add_edge(_GEdge(
        src=_FOCUS_POINTER_ID, dst=target, relation="指向", weight=0.9,
        relation_category="cognitive_relation"))
    # Haru -[认知焦点]-> 指针（架构对齐 2026-09-19）：自我对"当前在关注什么"
    # 的指针边，幂等一条。焦点语义由指针节点承载，不随事件切换抖动 Haru。
    if "Haru" in kg.nodes and not kg.get_edge("Haru", _FOCUS_POINTER_ID, "认知焦点"):
        kg.add_edge(_GEdge(src="Haru", dst=_FOCUS_POINTER_ID, relation="认知焦点",
                           weight=0.8, relation_category="cognitive_relation"))
    logger.info(f"[Focus] 当前焦点事件 → {target}")


def _get_focus_context(kg: KnowledgeGraph, max_sub: int = 3) -> list:
    """取焦点头部事件 + 其最近子事件（时间顺序链的源头端），供提取上下文注入。"""
    head = None
    for e in kg.edges:
        if e.src == _FOCUS_POINTER_ID and e.relation == "指向":
            head = e.dst
            break
    if head is None or head not in kg.nodes:
        return []
    subs = [e.src for e in kg.edges if e.relation == "时间顺序" and e.dst == head]
    return [head] + subs[:max_sub]

# =========================================================
# 好奇心认知机制 — 注入图谱基础设施
# =========================================================

curiosity_engine.bootstrap_curiosity(kg)
logger.info("[Curiosity] 好奇心基础设施已注入图谱")

# ── 情绪节点确保 ──
# 万物皆图：情绪词汇必须作为图谱节点存在，以供情绪共振机制检测。
# 情绪节点带 type=emotion 标记，由 inject_emotion() 在图谱层面验证。
# valence（效价属性，2026-09-19 反馈图谱化）：概念自身的图谱知识，
# 反馈观察读"本轮共激活情绪的 valence 合计"判断反馈方向——
# 取代旧 _POSITIVE/_NEGATIVE 词表直接决定 outcome 的机制。
from self_graph import EMOTION_NODES, EMOTION_VALENCE
_emo_added = 0
for _emo in EMOTION_NODES:
    if _emo not in kg.nodes:
        kg.add_node(Node(id=_emo, weight=0.3, label="declarative-semantic",
                        graph_space="cognitive",
                        extra_attrs={"type": "emotion",
                                     "valence": EMOTION_VALENCE.get(_emo, 0.0)}))
        engine.name_to_node[_emo] = kg.nodes[_emo]
        _emo_added += 1
    else:
        # 修复存量：情绪节点必须有 type=emotion 标记（情绪共振的识别依据），
        # 统一落 cognitive 空间（情绪是当前状态，不是客观事实），
        # 并补齐 valence 属性（旧图自愈）。
        _en = kg.nodes[_emo]
        _changed = False
        if (_en.extra_attrs or {}).get("type") != "emotion":
            _en.extra_attrs = _en.extra_attrs or {}
            _en.extra_attrs["type"] = "emotion"
            _changed = True
        if _en.graph_space != "cognitive":
            _en.graph_space = "cognitive"
            _changed = True
        if _en.extra_attrs.get("valence") is None:
            _en.extra_attrs["valence"] = EMOTION_VALENCE.get(_emo, 0.0)
            _changed = True
        if _changed:
            engine.name_to_node.setdefault(_emo, _en)
            _emo_added += 1
if _emo_added > 0:
    logger.info(f"[Emotion] 情绪节点新建/修复标记: {_emo_added} 个")

# ── 社交互动认知基础设施 ──
# 万物皆图：社交互动也走图谱扩散，不设旁路。
# 流程: NLP解析 → 注入"社交互动"节点 → 扩散 → 社交互动 -(触发)→ LLM回答 → 生成回复
_social_infra_nodes = [
    ("社交互动", "infrastructure", {"category": "social_interaction",
     "description": "认知语境标记：当前事件涉及人与人之间的互动关系。"
                    "不决定是否回应——回应资格由行为竞争产生"}),
    ("LLM回答", "procedural", {"category": "cognitive_action",
     "description": "LLM回答生成节点，激活度超过阈值时触发自然语言回答生成"}),
]
for _sn_id, _sn_label, _sn_attrs in _social_infra_nodes:
    if _sn_id not in kg.nodes:
        kg.add_node(Node(id=_sn_id, weight=0.4, label=_sn_label, extra_attrs=_sn_attrs))
        # 名字索引补注册（与行动留痕同坑）：engine 在播种前已扫描建索引，
        # 空图首启时这些节点若不入 name_to_node，activate_from_inputs 与
        # 回答门（LLM回答激活判定）会找不到它们——实测临时图冒烟里
        # 首轮 L2 回答分支因 llm_node=None 被静默跳过。
        with engine._lock:
            engine.name_to_node[_sn_id] = kg.nodes[_sn_id]
        logger.info(f"[Social] 创建图谱节点: {_sn_id}")

_social_infra_edges = [
    # 架构修正（2026-09-19c，检测≠行为决定）：
    # 移除 社交互动-[触发]->LLM回答/文件操作/看屏幕 三条行为导向直连边——
    # 社交信号只能改变认知场，回应与工具使用资格由 dialogue_decision
    # 行为竞争与现有动作通路产生，不允许"社交输入天然获得回应资格"。
    ("需要确认", "社交互动", "触发", 0.7),  # 不确定的事件 → 涉及互动的语境
    # 网络搜索动作通路：未知信息 → 搜索 → 辅助回答（万物皆图，无旁路）
    ("未知信息", "网络搜索", "触发", 0.8),
    ("网络搜索", "LLM回答", "辅助", 0.8),
    ("文件操作", "LLM回答", "辅助", 0.8),
    ("看屏幕", "LLM回答", "辅助", 0.8),
]
for _src, _dst, _rel, _w in _social_infra_edges:
    if not kg.get_edge(_src, _dst, _rel):
        kg.add_edge(Edge(src=_src, dst=_dst, relation=_rel, weight=_w))
        logger.info(f"[Social] 创建图谱边: {_src} -[{_rel}]→ {_dst}")

# 旧图自愈：历史版本曾把 社交互动 直连行为节点，启动时幂等移除
import dialogue_signals as _dsig
_dsig.repair_social_wiring(kg)
# 主体枢纽卫生（Haru 治理推广）：清理 思考_*/反思_* 往 用户/Self/Haru
# 堆的共现边（源头已封堵：枢纽需思考文本佐证才连边）
import graph_schema as _gschema
_gschema.repair_hub_pollution(kg)

# 表达反馈的图谱内机制（outcome 架构升级 2026-09-19）：
# 表达事件入图 → 后验观察回应关系（共激活回流）→ Feedback Event →
# 派生解释层 social 标签供 reward/日志。文本→outcome 分类器已退役。
import expression_feedback

# ── 网络搜索认知动作节点 ──
_web_search_nodes = [
    ("未知信息", "infrastructure", {"category": "info_need",
     "description": "图谱未覆盖的外部信息需求，触发网络搜索动作"}),
    ("网络搜索", "procedural", {"category": "cognitive_action",
     "description": "网络搜索动作节点，被激活时执行外部搜索并辅助LLM回答"}),
    ("文件操作", "procedural", {"category": "cognitive_action",
     "description": "文件操作动作节点，被激活时创建/写入本地文件并汇报结果"}),
    ("看屏幕", "procedural", {"category": "cognitive_action",
     "description": "屏幕感知动作节点，被激活时截屏识别屏幕文字与方位并汇报"}),
]
for _sn_id, _sn_label, _sn_attrs in _web_search_nodes:
    if _sn_id not in kg.nodes:
        kg.add_node(Node(id=_sn_id, weight=0.4, label=_sn_label, extra_attrs=_sn_attrs))
        logger.info(f"[WebSearch] 创建图谱节点: {_sn_id}")

# ── 自认知能力节点（"我能做什么"住在图里，不住在 prompt 里）──
# 万物皆图：能力是节点，被输入经语义相似度点亮、被扩散传播，
# 随 TopK 进入回答区——不是每轮硬塞给 LLM 的一段能力清单。
# self_capability 标记让能力节点区别于引擎零件（触发节点/动作编译产物/
# 执行留痕）：零件不参与召回，能力参与（见 graph_model.is_cognitive_visible）。
_self_capability_nodes = [
    ("网络搜索", {"description": "联网搜索图谱未覆盖的外部信息，返回标题、摘要与链接",
                "aliases": ["上网查", "搜一下", "查资料", "百度", "搜索"]}),
    ("文件操作", {"description": "在本地创建并写入文本文件，用户给文件名和内容",
                "aliases": ["建文件", "写文件", "创建txt", "保存文件"]}),
    ("看屏幕", {"description": "截屏识别屏幕上的文字与方位，回答屏幕上有什么",
              "aliases": ["截屏", "看我屏幕", "屏幕上写了什么", "屏幕识别"]}),
    ("进入Minecraft世界", {
        "description": "进入用户的 Minecraft 世界一起玩：先向用户要局域网端口号，"
                       "连接成功后可以在世界里移动、跟随用户、挖矿、装备物品、攻击",
        "aliases": ["玩Minecraft", "玩我的世界", "进入游戏", "陪你玩游戏", "MC", "我的世界"]}),
]
for _cid, _cattrs in _self_capability_nodes:
    _cn = kg.nodes.get(_cid)
    if _cn is None:
        kg.add_node(Node(id=_cid, weight=0.6, label="procedural",
                         graph_space="semantic",
                         extra_attrs={"category": "cognitive_action",
                                      "self_capability": True, **_cattrs}))
        logger.info(f"[SelfCapability] 创建能力节点: {_cid}")
    else:
        _cn.extra_attrs["self_capability"] = True
        for _ck, _cv in _cattrs.items():
            _cn.extra_attrs.setdefault(_ck, _cv)
        _cn.touch()

# 能力 ↔ 自我 / 对象的接线：Self 能→ 能力，语义对象 相关能力→ 能力
_self_capability_edges = [
    ("Self", "网络搜索", "能", 0.9),
    ("Self", "文件操作", "能", 0.9),
    ("Self", "看屏幕", "能", 0.9),
    ("Self", "进入Minecraft世界", "能", 0.9),
    ("Minecraft", "进入Minecraft世界", "相关能力", 0.9),
    ("进入Minecraft世界", "Minecraft会话", "实现于", 0.8),
]
for _cs, _cd, _cr, _cw in _self_capability_edges:
    if _cs in kg.nodes and _cd in kg.nodes and not kg.get_edge(_cs, _cd, _cr):
        kg.add_edge(Edge(src=_cs, dst=_cd, relation=_cr, weight=_cw))
        logger.info(f"[SelfCapability] 创建图谱边: {_cs} -[{_cr}]→ {_cd}")

# ── 动作概念层（Action Concept 架构升级 2026-09-19）──
# 稳定的动作语义节点（FOLLOW/SEARCH/SCREEN_OBSERVE/FILE_CREATE/…）与
# "表达方式"边种进图谱（幂等；复用上面已有的 网络搜索/文件操作/看屏幕
# procedural 节点作为概念节点）。图谱提供语义概念、关系与候选激活——
# 执行必须经 action_resolver 形成结构化 Action Intent（不是关键词词典）。
try:
    from action_concepts import ensure_action_concepts
    ensure_action_concepts(kg, engine)
except Exception as _ace:
    logger.warning(f"[ActionConcept] 动作概念种子失败（不阻塞启动）: {_ace}")

# ── 行动空间（Action Concept 一等公民 2026-09-20）─────────────
# 候选宇宙=图谱行动节点（激活 ∪ 活跃供性源），概念可被提议/去重/
# 生命周期成长；供性边（Drive/情绪→行动）与最小行动本体在 Drive
# bootstrap 之后种入（见 drive_evaluator.bootstrap_drives() 后）。
from action_space import ActionSpace
action_space = ActionSpace(kg, engine, config)

# ── 用户教学知识与节点别名 ──
# P0-1: 用户教学知识（9 个 taught_by_user 节点 + 9 条边）已迁出源代码，
# 位于 knowledge_packs/user_teaching_2026-07/，随 Pack 加载进入图谱。
# Minecraft/Fascinator 的别名长在节点 extra_attrs.aliases 上
# （minecraft pack 与基础图已携带），不再需要启动补丁。
# 经历类知识（喝奶茶 等）按事件框架走情景记忆，不做启动注入。

# ── 焦点事件指针引导（论文 §3.3）──
# 指针不存在时，聚焦 episodic 空间里 event_timestamp 最新的头部事件
# （无 时间顺序 出边指向其他事件、且非前瞻性计划）。
_focus_exists = kg.get_node(_FOCUS_POINTER_ID) is not None
if not _focus_exists:
    _best_ts, _best_id = "", None
    for _n in kg.nodes.values():
        if getattr(_n, "graph_space", "") != "episodic":
            continue
        _attrs = getattr(_n, "extra_attrs", {}) or {}
        if _attrs.get("prospective"):
            continue
        _ts = _attrs.get("event_timestamp") or ""
        if not _ts:
            continue
        _has_parent = any(
            e.src == _n.id and e.relation == "时间顺序" for e in kg.edges)
        if _has_parent:
            continue
        if _ts > _best_ts:
            _best_ts, _best_id = _ts, _n.id
    if _best_id:
        _update_focus_event(kg, _best_id)
        logger.info(f"[Focus] 启动引导: 焦点事件 → {_best_id} ({_best_ts})")

# =========================================================
# NLP — 先加载配置再初始化，避免重复创建 Backend
# =========================================================

import json as _json

LLM_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "llm_config.json")
# API Key 单独存放于 gitignored 文件，避免密钥随 llm_config.json 进 git
LLM_SECRET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "llm_secret.json")

def _load_llm_secret() -> dict:
    try:
        with open(LLM_SECRET_PATH, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {}

def _save_llm_secret(api_key: str):
    """API Key 只写入 gitignored 的密钥文件，支持环境变量 MIMO_API_KEY 覆盖读取。"""
    try:
        os.makedirs(os.path.dirname(LLM_SECRET_PATH), exist_ok=True)
        with open(LLM_SECRET_PATH, "w", encoding="utf-8") as f:
            _json.dump({"api_key": api_key or ""}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[LLM Secret] Save failed: {e}")

def _load_llm_config():
    try:
        os.makedirs(os.path.dirname(LLM_CONFIG_PATH), exist_ok=True)
        with open(LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        # 密钥来源优先级：llm_config.json（历史遗留，已不再写入）> 密钥文件 > 环境变量
        if not cfg.get("api_key"):
            cfg["api_key"] = (_load_llm_secret().get("api_key", "")
                              or os.environ.get("MIMO_API_KEY", "")
                              or os.environ.get("DEEPSEEK_API_KEY", ""))
        return cfg
    except Exception:
        return {"provider": "mimo", "model": "", "api_key": ""}

def _save_llm_config():
    try:
        os.makedirs(os.path.dirname(LLM_CONFIG_PATH), exist_ok=True)
        cfg = {
            "provider": getattr(nlp, '_provider_name', 'mimo'),
            "model": nlp.model_name(),
            # 注意：api_key 永不写入本文件；本文件已被 .gitignore 排除（模板见 llm_config.example.json），见 _save_llm_secret
            "api_key": "",
        }
        with open(LLM_CONFIG_PATH, "w", encoding="utf-8") as f:
            _json.dump(cfg, f, ensure_ascii=False, indent=2)
        _save_llm_secret(getattr(nlp, '_api_key', ''))
        logger.info(f"[LLM Config] Saved: provider={cfg['provider']}, model={cfg['model']}")
    except Exception as e:
        logger.warning(f"[LLM Config] Save failed: {e}")

# 读取保存的 LLM 配置，传入 NLPProcessor 避免默认 ollama 初始化
_saved_cfg = _load_llm_config()
_initial_provider = _saved_cfg.get("provider", "mimo")
_initial_api_key = _saved_cfg.get("api_key", "") or None
_initial_model = _saved_cfg.get("model", "") or None

# Ollama 关闭时强制回落 MiMo（saved 配置里遗留的 ollama 不可见）
from llm_provider import ollama_enabled as _ollama_on
if _initial_provider == "ollama" and not _ollama_on():
    logger.info("[LLM Config] ollama 未启用，回落 mimo")
    _initial_provider, _initial_model = "mimo", None

logger.info(f"[LLM Config] 启动配置: provider={_initial_provider}, model={_initial_model}")

nlp = NLPProcessor(
    provider=_initial_provider,
    api_key=_initial_api_key,
    model=_initial_model if _initial_model else None
)

# 确保配置文件与实际运行状态一致
_save_llm_config()

# ── 架构解耦（2026-10-08，docs/ARCHITECTURE_PLAN.md）：语言后端显式化 ──
# nlp（NLPProcessor）是临时语言实现（temporary/external，槽位登记见 fas/registry.py）。
# 认知侧持有的语言对象上限 = CognitiveLanguageAccess 最小面（llm/chat_llm/ask，
# 恰好覆盖现状全部触点：reflection .llm／curiosity+cc._express .chat_llm／
# self_graph.generate_thought .ask——逐点核对见 fas/companion/temporary_llm.py 头注释）。
# 换语言方案 = 只换这一个注入对象，认知模块不动。
# 陪伴主管道（本文件的 /api/nlp 等）继续直接用 nlp：那是陪伴侧特权，不经门面。
from fas.companion.temporary_llm import TemporaryLLMBackend
_language_backend = TemporaryLLMBackend(nlp)
cog_language = _language_backend.access()
engine._nlp_ref = cog_language

# =========================================================
# Episodic Buffer
# =========================================================

from episodic_buffer import EpisodicBuffer
buffer = EpisodicBuffer(capacity=100)

# ── Reflection Phase 2: 认知反思循环 ──
# Reflection Evolution R2: 行为倾向存储（封闭词表，self 空间）
disposition_store = DispositionStore(kg, config)
# 持续认知循环（Continuous Cognition）：无输入期图谱活动 + 交流意图
from continuous_cognition import ContinuousCognition
cc = ContinuousCognition(kg, engine, cog_language, buffer, config,
                         chat_log=get_chat_log())
cc._save_fn = lambda: _save(force=True)
cc.start()

# 基线人格层：出厂倾向先验 + 心情瞬态（长在现有 self 图与 disposition 骨架上）
# R2 P11：config 是心情参数（衰减/事件偏移）的出厂真源；时基与调制器共用
# （`internal_state` 在下方才构造，所以取时刻要走这个惰性闭包，而不是各自
#  `time.time()`——注入时基做回放/实验时（§26）两者会分家）。
def _state_clock():
    _is = globals().get("internal_state")
    try:
        return float(_is.now())
    except Exception:                               # noqa: BLE001
        return time.time()


from personality_baseline import PersonalityBaseline
persona = PersonalityBaseline(kg, disposition_store, config=config,
                              clock=_state_clock)
persona.bootstrap()
reflection_engine = ReflectionEngine(kg, cog_language, config, episodic_buffer=buffer,
                                     disposition_store=disposition_store)
cc.reflection = reflection_engine  # 离线反思自问（论文 §3.3 反思性自问）
ensure_bucket_nodes(kg)  # 时间感知：时段节点+睡眠常识幂等播种
llm_budget = BudgetManager(profile="balanced")  # LLM 全局预算（平衡档）
cc.llm_budget = llm_budget

# ── 认知调节：锁 / 状态检测器 / 触发器 ──────────────────────────
# 三个基础机制的装配点。锁注册表绑到扩散引擎（介入点见 diffusion_engine
# 的 5 处注释 + 2 处输出侧过滤）；检测器状态变化 → 触发器评估 → 认知事件
# 派发，全程确定性，LLM 只在 llm_mode>=1 时由认知层按需取用。
from cognitive_regulation import CognitiveRegulation
regulation = CognitiveRegulation(kg=kg, engine=engine, config=config,
                                 data_dir="data",
                                 save_graph_fn=lambda: _save())
engine.set_lock_registry(regulation.locks)
cc.regulation = regulation   # 检测器轮询挂在既有认知循环的 tick 上（不新建线程）


def _on_cognitive_event(cognitive_event, action_result):
    """认知事件进入认知循环的接线点。

    只做两件确定性的事：留日志、把触发结果挂到最近认知事件环（前端与
    回答层按 llm_mode 决定是否取用）。这里**不调用 LLM**。
    """
    logger.info(
        f"[Regulation] 认知事件 {cognitive_event.get('trigger_id')} "
        f"← {cognitive_event.get('monitor_id')} "
        f"[{cognitive_event.get('condition_explain')}] "
        f"动作={action_result.get('type')} ok={action_result.get('ok')}"
    )


regulation.register_handler(_on_cognitive_event)
# 话题终止抑制期改由 system 锁承载（§18.2 #10）：到期审计、显式解除、
# /api/regulation 可见都走 LockRegistry；注册表缺席时自动回落浮点通道。
bind_lock_registry(regulation.locks)
logger.info("[Regulation] 认知调节已装配（锁定=%d 检测器=%d 触发器=%d）" % (
    len(regulation.locks.list()), len(regulation.monitors.list()),
    len(regulation.triggers.list())))
# Minecraft 桥（Mineflayer）：Haru 在游戏中的实体状态
from minecraft.bridge import get_state as mc_get_state, say as mc_say, move as mc_move
import minecraft.bridge as bridge_module
from minecraft.perception import world_state_snapshot
import minecraft.actions as mc_actions
from minecraft.session import MinecraftSession, BOT_DIR
# （update_perception 的对话裸调用已删——世界写图统一走 minecraft.embodiment.
#  perceive 门控路径，收尾审计 B2；裸调用曾是敌对标注被抹平的病灶。）
mc = {"get_state": mc_get_state, "say": mc_say, "move": mc_move}
# （原 826 行的 `import time` 已上移至顶部导入块——收尾修复 2026-09-22 P0：
#  它原本晚于 _state_clock/persona 构造的调用点，冷启动必炸。）


def _resolve_node_binary() -> str:
    """找 node 可执行文件：配置 > PATH > 常见安装路径（本机 node 不在 PATH 里）。"""
    import shutil as _shutil
    cand = config.get("minecraft_bot_node_path")
    if cand and os.path.exists(cand):
        return cand
    found = _shutil.which("node")
    if found:
        return found
    for guess in (r"E:/Nodejs/node.exe", r"C:/Program Files/nodejs/node.exe",
                  r"C:/Program Files (x86)/nodejs/node.exe"):
        if os.path.exists(guess):
            return guess
    return "node"


def _dispatch_mc_reflex(reflex: dict) -> dict:
    """MODE0 反射指令分发（游戏在线时）。快路径与正常路径共用，防两套逻辑。"""
    _r_act = reflex.get("action")
    _rp = reflex.get("params") or {}
    _game_user = config.get("minecraft_user_name") or "Hellucigen"
    if _r_act in ("follow", "approach"):
        _rf = mc_actions.follow_player(_game_user)
        # B4/§4 邀请→承诺：反射先动身（响应性），随后落**持久目标**，
        # 让这次邀请经 透传→候选→评分→阈值 全链参与之后的每一次决策；
        # 兑现/被叫停时分别由 settle 销账与 stop 撤账收尾。
        if isinstance(_rf, dict) and _rf.get("success"):
            try:
                autonomy.register_invitation(
                    "follow" if _r_act == "follow" else "approach",
                    _game_user)
            except Exception as _rie:
                logger.debug(f"[Reflex] 邀请承诺登记跳过: {_rie}")
        return _rf
    if _r_act == "stop":
        result = mc_actions.stop()
        try:
            _sf = mc_actions.stop_follow()
            if isinstance(result, dict):
                result["followed_seconds"] = _sf.get("followed_seconds", 0)
        except Exception as _sfe:
            logger.warning(f"[MCReflex] stop_follow 状态合并失败: {_sfe}")
        try:
            # 撤账："别跟了"之后承诺不能复活（否则透传下一拍又把她叫回去）
            autonomy.cancel_invitation()
        except Exception as _cie:
            logger.warning(f"[MCReflex] 撤账失败(承诺可能残留复活): {_cie}")
        return result
    if _r_act == "jump":
        return mc_actions.jump()
    if _r_act in ("dig", "attack"):
        # 具身图谱化 2026-09-21：破坏性反射不再直连 /dig、/attack——
        # 统一经 ActionManager → Safety Kernel（资产授权/攻击对象核验）。
        try:
            from action_intents import reflex_to_action
            _spec = reflex_to_action({"action": _r_act, "params": _rp,
                                      "raw": ""}, user_name=_game_user)
        except Exception as _rte:
            logger.warning(f"[Reflex] 反射→动作意图转换失败(按 missing_target 报): {_rte}")
            _spec = None
        if not _spec:
            return {"success": False, "action": _r_act,
                    "reason": "missing_target"}
        _pr = action_manager.propose(_spec, source="user")
        _res = _pr.get("result") or {
            "success": bool(_pr.get("started")),
            "reason": _pr.get("reason") or ""}
        _res.setdefault("action", _r_act)
        return _res
    if _r_act == "move":
        _secs = float(_rp.get("seconds") or 1.0)
        _dir = _rp.get("direction") or "forward"
        return (mc_actions.move_forward(_secs) if _dir == "forward"
                else mc_move(_dir, _secs))
    return {"success": False, "action": _r_act, "reason": "unsupported"}


def _record_user_action_intent(spec: dict, text: str, cycle_id=None):
    """用户的行动请求 → 图谱认知事件（万物皆图，不建旁路）。

    建 ActionIntent 请求节点（cognitive 空间）+ 用户 -[请求]-> 节点 边，
    并注入激活——动作经 ActionManager 执行后，结果留痕节点会连回这里的
    basis。零 LLM、确定性。
    """
    try:
        import time as _t
        atype = str(spec.get("action_type") or "act")
        nid = f"ActionRequest_{atype}_{int(_t.time() * 1000) % 10**9}"
        with kg._lock:
            kg.add_node(Node(
                id=nid, weight=0.6, label="declarative-episodic", graph_space="cognitive",
                extra_attrs={
                    "type": "user_action_request",
                    "action_type": atype,
                    "target": spec.get("target"),
                    "params": spec.get("params", {}),
                    "utterance": str(text or "")[:120],
                    "cycle_id": cycle_id,
                    "created": now_str(),
                }))
            if "用户" in kg.nodes:
                kg.add_edge(Edge(src="用户", dst=nid, relation="请求",
                                 weight=0.8,
                                 relation_category="cognitive_relation"))
        engine.mark_active([nid])
        logger.info(f"[UserAction] 请求入图: {nid}")
    except Exception as _ue:
        logger.debug(f"[UserAction] 请求入图跳过: {_ue}")


# 反射快路径的游戏内即时回执（她已经在游戏里用行动+短句回答，不用等 LLM）
_MC_REFLEX_ACK = {"follow": "好，跟着你。", "approach": "过来啦。",
                  "stop": "好，停下了。", "jump": "跳！", "dig": "这就挖。",
                  "attack": "好。", "move": "走一个。"}


def _restart_mc_bot() -> bool:
    """换世界专用（connect 每次调用）：让旧 bot 退出（/quit），拉起新进程读新 config。

    旧进程只会永远重试它启动时的端口（实测 54766 关闭后 ECONNREFUSED 循环），
    bot.js 只在启动时读 config.json——所以换端口必须重启进程。
    """
    import subprocess as _sp
    import urllib.request as _ur
    try:
        req = _ur.Request("http://127.0.0.1:5010/quit", data=b"{}", method="POST")
        with _ur.urlopen(req, timeout=2) as r:
            r.read()
        logger.info("[MCBot] 已请求旧 bot 退出")
    except Exception as _qe:
        logger.warning(f"[MCBot] 旧 bot 退出请求失败(拉新进程照旧): {_qe}")
    time.sleep(1.5)
    node = _resolve_node_binary()
    bot_dir = BOT_DIR
    log_path = os.path.join(bot_dir, "haru.log")
    try:
        log = open(log_path, "a", encoding="utf-8")
        flags = 0
        if os.name == "nt":
            flags = getattr(_sp, "CREATE_NO_WINDOW", 0x08000000)
        _sp.Popen([node, "bot.js"], cwd=bot_dir, stdout=log, stderr=log,
                  creationflags=flags)
        logger.info("[MCBot] bot 已重启（读取最新 config）")
    except Exception as e:
        logger.warning(f"[MCBot] bot 重启失败: {e}")
        return False
    for _ in range(20):
        time.sleep(1)
        if bridge_module.health():
            logger.info("[MCBot] bot 重启完成，桥就绪")
            return True
    logger.warning("[MCBot] bot 重启后 20 秒桥未就绪")
    return False


def _ensure_mc_bot() -> bool:
    """确保 bot 进程在跑。桥在线就不动；不在线才启动，并等桥就绪。

    一个桥端口只能有一个 bot：抢占失败时 bot.js 会自己退出（避免同名互踢）；
    这里绝不做"先杀再起"——那正是之前反复进出的成因。
    """
    import subprocess as _sp
    if bridge_module.health():
        return True
    node = _resolve_node_binary()
    bot_dir = BOT_DIR
    log_path = os.path.join(bot_dir, "haru.log")
    try:
        log = open(log_path, "a", encoding="utf-8")
        flags = 0
        if os.name == "nt":
            flags = getattr(_sp, "CREATE_NO_WINDOW", 0x08000000)
        _sp.Popen([node, "bot.js"], cwd=bot_dir, stdout=log, stderr=log,
                  creationflags=flags)
        logger.info(f"[MCBot] 已启动 bot 进程 ({node} bot.js)，等待桥就绪…")
    except Exception as e:
        logger.warning(f"[MCBot] 启动 bot 失败: {e}")
        return False
    for _ in range(20):
        time.sleep(1)
        if bridge_module.health():
            logger.info("[MCBot] 桥已就绪")
            return True
    logger.warning("[MCBot] 桥在 20 秒内未就绪（可能端口被占用或 node 路径不对）")
    return False


mc_session = MinecraftSession(kg, bridge_get_state=mc["get_state"],
                              ensure_bot_process=_ensure_mc_bot,
                              restart_bot_process=_restart_mc_bot)


def _wire_bridge_liveness():
    """桥活性回写（§18.2 #7 + #11，一次机制两处收口）。

    在检测器注册表上登记一个 mode=poll 的 boolean 检测器，数据源就是
    bridge health()；CC 既有的 monitor_poll_every_ticks 节拍调用
    poll_due() 驱动它（**不新建线程/定时器**）。observe() 的值去重让
    它天然是边沿触发：只有 health True→False 的翻转才发 state_changed
    事件，订阅者据此把 connected 的 session 节点回写 disconnected，
    并在经历时间线记一条观察事件。
    """
    try:
        mons = regulation.monitors.list(target_type="external")
        mon = next((m for m in mons
                    if m.get("target_id") == "minecraft_bridge"), None)
        if mon is None:
            out = regulation.monitors.create(
                name="MC 桥活性", target_type="external",
                target_id="minecraft_bridge", state_type="bool",
                mode="poll",
                poll_interval_s=float(
                    config.get("mc_bridge_poll_interval_s", 30) or 30),
                description="Mineflayer 桥 /health 可达性轮询；"
                            "True→False 翻转回写 session disconnected（§18.2 #7）")
            mon = out.get("monitor") or {}
        mid = mon.get("id") or ""
        if not mid:
            logger.warning("[MCSession] 桥活性检测器登记失败，跳过接线")
            return
        regulation.monitors.register_poller(
            mid, lambda: bool(bridge_module.health()))

        # 腿死自愈（2026-09-24 夜）：pathfinder 静默冻结一天三连，bot.js 侧
        # 看门狗会自杀退出——但复活必须是**电平触发**：边沿事件只在翻转那拍
        # 存在，复活的 bot 若再死就永远没人管了。轮询每拍都判，120s 节流，
        # 且只在"世界端口还开着"时拉起（世界关了就别造连接拒绝刷屏）。
        _heal = {"ts": 0.0}

        def _world_port_open():
            try:
                import json as _j
                cfgp = os.path.join(BOT_DIR, "config.json")
                port = int((_j.load(open(cfgp, encoding="utf-8")) or {}).get("port") or 0)
                import socket as _s
                c = _s.socket(); c.settimeout(1.5)
                try:
                    c.connect(("127.0.0.1", port)); return True
                finally:
                    c.close()
            except Exception:
                return False

        def _poll_bridge_and_heal():
            ok = bool(bridge_module.health())
            if not ok and time.time() - _heal["ts"] > 120.0:
                _heal["ts"] = time.time()
                if mc_session.get_state() in ("connected", "disconnected") \
                        and _world_port_open():
                    logger.info("[MCSession] 桥不健康而世界仍开——拉起 bot（腿死自愈）")
                    import threading as _th
                    _th.Thread(target=_ensure_mc_bot, daemon=True).start()
            return ok

        regulation.monitors.register_poller(mid, _poll_bridge_and_heal)

        def _on_bridge_state(ev):
            if ev.get("monitor_id") != mid:
                return
            # 只处理"活性消失"翻转；首轮 None→True 的"出现"事件不动作
            if ev.get("current") is not False:
                return
            if mc_session.on_bridge_lost(reason="monitor_poll"):
                logger.warning("[MCSession] 检测器判定桥失联，session 已回写")
                try:
                    timeline.append(make_event(
                        EVENT_OBSERVATION, actor="self",
                        subject="minecraft_bridge_lost",
                        content={"via": "state_monitor",
                                 "previous": ev.get("previous")},
                        source="mc_session"))
                    _save()
                except Exception as e:
                    logger.warning(f"[MCSession] 失联事件记录失败: {e}")

        regulation.monitors.subscribe(_on_bridge_state)
        logger.info(f"[MCSession] 桥活性检测器已接线（{mid}，"
                    f"轮询间隔 {mon.get('poll_interval_s')}s，CC 节拍驱动）")
    except Exception as e:
        logger.warning(f"[MCSession] 桥活性接线失败（不影响其余功能）: {e}")


_wire_bridge_liveness()

# ── 自主认知与行动闭环（Phase D）────────────────────────────────
# 通用自主层（autonomy.AutonomousLoop）+ Minecraft 具身适配器。
# 循环不新建线程：由持续认知循环按节拍驱动 autonomy.tick()。
# 默认模式由 autonomy 的 mode_default 决定（现为 on，2026-09-19 应要求）；
# bot 未连接/安静期/无认知依据时它每个 tick 都如实返回"没做"并写明原因
# （前端"自主日志"面板可见）。/api/autonomy/mode 可随时 on/off。
from autonomy import AutonomousLoop
from minecraft.embodiment import MinecraftEmbodiment
from minecraft.reflex import parse_reflex_command

autonomy = AutonomousLoop(
    kg=kg, engine=engine, config=config, regulation=regulation,
    llm_budget=llm_budget, disposition_store=disposition_store, persona=persona,
    data_dir="data", save_graph_fn=lambda: _save(),
    note_outcome_fn=lambda negative: cc.note_outcome(negative))
mc_embodiment = MinecraftEmbodiment(
    kg=kg, engine=engine, perceive_into_graph=True, config=config)
# 屏幕观察者（2026-09-22）："看屏幕"从对话 inline 升级为真实可执行行动。
# EmbodimentRouter 是薄适配层：Minecraft 具身仍是默认子执行器（属性全透传），
# 只把 screen_observe 路由给眼睛。嵌入器延迟取（emb_mgr 在本对象之后构造）。
from eye.observer import ScreenObserver, CameraObserver, EmbodimentRouter
screen_observer = ScreenObserver(
    kg, engine=engine,
    embedder_fn=lambda: emb_mgr if emb_mgr.ready() else None)
# 摄像头观察者（2026-10-01）：核心移植自 The Institute Eyes（YOLOv8-Pose
# 姿态/行为检测）。依赖（ultralytics）缺失时能力如实为空，不影响其余路由。
# 模型文件存在则直接用（省去 ultralytics 自动下载）。
import os as _os
_camera_model = "E:/Models/vision/yolov8n-pose.pt" \
    if _os.path.exists("E:/Models/vision/yolov8n-pose.pt") else None
camera_observer = CameraObserver(kg, engine=engine, model_path=_camera_model)
embodiment_router = EmbodimentRouter(
    default=mc_embodiment,
    routes={"screen_observe": screen_observer,
            "camera_observe": camera_observer})
autonomy.register_embodiment(embodiment_router)
cc.autonomy = autonomy   # 自主行动节拍挂在既有认知循环上（不新建线程）

# ── 内部状态层（Phase 1）：需求 / 神经调制 / 性格 的唯一状态源 ──
# 三层分工：图谱只放可解释状态节点（镜子节点，in-place）；运行时数值在本模块
# 落盘；控制器逻辑按 Phase 2–5 逐步接入同一写入口（apply_delta/set_value）。
from internal_state import InternalState

internal_state = InternalState(kg=kg, config=config, persona=persona,
                               engine=engine, data_dir="data",
                               save_graph_fn=lambda: _save())
internal_state.sync_graph()
cc.internal_state = internal_state   # Phase 2+ 的需求/奖励更新挂在既有循环上

# ── 奖赏评估中间层（人格学习架构改造 2026-09）──────────────
# outcome → RewardEvent → 激素释放(internal_state) → 学习调制 → disposition。
# reward.py 不落文件、不写图：它只是 internal_state 奖赏接口的第一个真实
# 消费者 + disposition 学习率的计算者。长期人格仍只写 self 图。
from reward import (RewardSystem, classify_self_outcome)

reward_system = RewardSystem(internal_state=internal_state, config=config)
autonomy.reward_system = reward_system   # 自主行动结果经奖赏层形成自我证据
cc.reward_system = reward_system          # 激素衰减 tick 挂在既有循环上
logger.info("[Reward] 奖赏评估层已装配（social/self 双路 + 激素调制学习率）")

# ── 经验时间轴 + 保守因果发现（Experience Timeline 改造 2026-09）──
# 通用经验底座：所有来源（具身/用户语言/内部状态/认知事件）统一成
# ExperienceEvent 进入同一条时间轴；CausalLearner 只做 候选关联→重复聚合
# →假设→（够稳才）晋升 KG。零 LLM、零具体环境字段。
from experience import ExperienceTimeline, CausalLearner, EVENT_ACTION, EVENT_OBSERVATION, \
    EVENT_SELF_STATE, EVENT_COGNITIVE, make_event

timeline = ExperienceTimeline(path="data/experience_timeline.json", config=config)
causal_learner = CausalLearner(timeline, config=config, kg=kg, engine=engine)
autonomy.timeline = timeline
autonomy.causal = causal_learner
mc_embodiment.timeline = timeline   # 具身只报观察，不归因（§五）
# 时间感知重构（2026-09-20）：世界时钟是持续状态——CC 分钟级写入图谱
# （时段/昼夜地板激活 + 迁移事件入统一时间轴），与用户是否提到时间无关。
cc.timeline = timeline
try:
    from temporal_awareness import update_clock_state as _clock_init
    _clock_init(kg, engine, timeline=timeline)
except Exception as _cke:
    logger.debug(f"[Time] 初始时钟更新跳过: {_cke}")
logger.info("[Experience] 经验时间轴已装配（事件级、去重合并、跨来源统一）")

# ── Action 节点系统（具身改造 2026-09）──────────────────────────
# 所有具身动作（用户指令/自主/生存紧急）的统一闸门：
# 当前动作承诺 + 优先级中断 + 目标队列 + 因果记录。
# ActionNode 从这里进 Skill Library（minecraft.embodiment → skills/）。
from action_system import ActionManager
from action_intents import l1_to_action, intention_to_action, reflex_to_action
from action_resolver import (resolve_minecraft_intent, resolve_turn_intents,
                             intent_to_minecraft_flow, describe_intent)
# Cognitive Context（认知上下文重构 2026-09-20）：七分区构建 + L1/L2 编译
from cognitive_context import (build_cognitive_context, compile_for_language,
                               debug_view as _cog_debug_view)
from cognitive_demand import analyze_cognitive_demand

action_manager = ActionManager(
    embodiment=embodiment_router, kg=kg, engine=engine, config=config,
    internal_state=internal_state, reward_system=reward_system,
    disposition_store=disposition_store, persona=persona,
    timeline=timeline, causal=causal_learner, llm_budget=llm_budget,
    save_graph_fn=lambda: _save(),
    note_outcome_fn=lambda negative: cc.note_outcome(negative))
autonomy.actions = action_manager          # 自主候选经 ActionManager 竞争执行
autonomy.internal_state = internal_state   # 事件 → 需求变化的写入口
cc.action_manager = action_manager         # 动作推进挂在既有认知循环 tick 上
action_manager.cc = cc                     # 行动失败→反思压力（统一认知循环）
logger.info("[Action] ActionManager 已装配（当前动作承诺/优先级中断/目标队列）")


# ── 世界接入异步化（2026-09-20）：连上之后要做的事在这里发生 ──
# 旧行为：connect() 在请求线程里轮询 25 秒（bot 重启+握手的真实耗时），
# 一轮"来玩我的世界"整轮卡死在接入上。现在回合只花一次写配置的时间，
# 拿 pending 证据（语言层如实说"正在连"）；后台连上后：图谱会话节点
# 已由 _set 更新，这里补上行动（走向用户）、游戏内播报与经历入轴。
def _mc_on_connected(result):
    logger.info(f"[MCSession] 后台连接完成: port={result.get('port')}")
    try:
        timeline.append(make_event(
            EVENT_COGNITIVE, actor="self", subject="minecraft_connected",
            content={"port": result.get("port"), "via": "async"},
            source="mc_session"))
    except Exception as _cse:
        logger.warning(f"[MCSession] 连接成功事件未入轴: {_cse}")
    # B5/§5：原先这里硬编码 mc_say("我进来啦…")——一句没有认知来源的台词。
    # 移交 CI 链：连接事件已入时间轴，若值得说，表达由
    # CI→communicate→通道产生并带真实投递回执；过不了富基底门就不说（诚实）。
    try:
        _guser = config.get("minecraft_user_name") or "Hellucigen"
        action_manager.propose({
            "action_type": "navigate_to_entity", "target": _guser,
            "params": {"entity": _guser, "player": _guser,
                       "keep_distance": 2.0},
            "urgency": 0.7, "priority": 0.65,
            "motivation": "user_commitment",
            "expected_effect": "走到用户身边，开始一起玩",
            "reason": ["connect_minecraft_success"]}, source="user")
        # B4/§4：邀请落成持久承诺（propose 被 busy 拒掉也不会丢——
        # 下一拍透传继续竞争；settle 成功自动销账）
        autonomy.register_invitation("approach", _guser)
    except Exception as _ape:
        logger.debug(f"[MCSession] 后台走向用户跳过: {_ape}")


def _mc_on_failed(result):
    logger.info(f"[MCSession] 后台连接失败: {result.get('reason')}")
    try:
        timeline.append(make_event(
            EVENT_COGNITIVE, actor="self", subject="minecraft_connect_failed",
            content={"reason": str(result.get("reason") or "")[:80]},
            source="mc_session"))
    except Exception as _cfe:
        logger.warning(f"[MCSession] 连接失败事件未入轴: {_cfe}")
    try:
        mc_say(f"没连上世界：{result.get('reason') or '不知道什么原因'}")
    except Exception as _mse:
        logger.warning(f"[MCSession] 连接失败提醒未送达: {_mse}")


mc_session.on_connected = _mc_on_connected
mc_session.on_failed = _mc_on_failed

# ── 对话通道层：世界里的文字聊天也能直接走认知管线 ──────────────
# 通用设计：任何"能收消息/发消息"的环境都能注册成通道（游戏内聊天、
# 外部 IM 桥、未来的语音转写…），回复回到消息来的那个通道。
# 认知逻辑不复制：通道消息走应用自己的 /api/nlp 入口（同一条管线）。
from conversation_channel import ChannelHub
from minecraft.channel import MinecraftChatChannel

channel_hub = ChannelHub(config=config, poll_interval_s=2.0)
minecraft_chat = MinecraftChatChannel(
    user_names=[mc["username"]] if False else None,
    enabled=bool(config.get("channel_minecraft_chat", True)))
channel_hub.register(minecraft_chat)
# B5/§5：具身发言口接通道统一出口（清洗+节流+诚实回执都在这一个点）。
# 注入在注册之后：_communicate 经 channel_hub 查 minecraft_chat。
mc_embodiment.channel_hub = channel_hub


def _channel_turn_runner(text, meta):
    """通道消息 → 认知管线。走应用自己的 HTTP 入口：逻辑只有一份。
    meta 里的身份事实（sender/kind/mention）随 payload 一起进管线——
    通道不再决定回应与否，认知需要知道"这是谁在说话、有没有点名"。"""
    with app.test_client() as _client:
        resp = _client.post("/api/nlp", json={
            "text": text,
            "channel": meta.get("channel"),
            "sender": meta.get("sender") or "",
            "sender_kind": meta.get("kind") or "user",
            "mentioned": bool(meta.get("mention")),
        })
        data = resp.get_json() or {}
    return {"answer": data.get("answer"), "cycle_id": data.get("cycle_id"),
            "behavior": data.get("behavior")}


def _channel_observer(msg):
    """他人消息的通知钩子（日志/前端观察簿）。注意：自 2026-09-21 起
    这不再是"只看不说"的旁路——消息本身已经随队列进入统一认知管线，
    是否回应由 dialogue_decide 决定。"""
    logger.info(f"[Channel] 他人消息（已进认知）{msg.get('sender')}: {msg.get('text')[:40]}")


channel_hub.set_turn_runner(_channel_turn_runner)
channel_hub.set_observer(_channel_observer)
channel_hub.set_busy_check(lambda: cc._busy.is_set())
# channel_hub.start() 延后到全部路由注册完成之后（见文件末尾）——
# 否则启动窗口期进来的游戏聊天会经 test_client 触发"第一个请求"，
# 导致后续 @app.route 全部报错崩溃（实测竞态，2026-09-19）。
logger.info("[Channel] 对话通道已装配（游戏内聊天 %s）",
            "启用" if minecraft_chat.enabled else "停用")
logger.info("[InternalState] 内部状态层已装配（cycle_seq=%d）",
            internal_state.state()["cycle_seq"])
logger.info("[Autonomy] 自主闭环已装配（mode=%s，tick 挂持续认知循环）",
            autonomy.mode)
logger.info("[Reflection] ReflectionEngine 已初始化")

# ── 学习实验模式（§16/§20）：装配前翻现成消音开关，装配后注入结构化目标 ──
# mode=off 时本段是纯 no-op。屏蔽全部借既有语义（config 开关 / 一行级门），
# 不删数据不改拓扑；消费者清单见 experiment_mode.py 头注释。
import experiment_mode as _xm
_xm.configure(config)
try:
    import world_prior as _wp
    _wp.bind_config(config)
except Exception as _wpe:
    logger.warning(f"[Experiment] world_prior 绑定失败: {_wpe}")
if _xm.enabled():
    _sh = _xm.shield
    _ms = config.setdefault("modulator_system", {})
    if _sh("graph_projection"):
        _ms["subgraph"] = False         # 投影/图偏置/心情投影/漂移 一并旁路
    if _sh("graph_pulse"):
        _ms["graph_pulse"] = False
    if _sh("drift"):
        _ms["need_drift_rate_per_min"] = 0.0
        _ms["circadian_drift_rate_per_min"] = 0.0
    if _sh("social_drive"):
        config.setdefault("drive_field", {}).setdefault(
            "drives", {}).setdefault("social", {})["enabled"] = False
    if _sh("personality_tendency"):
        config.setdefault("autonomy", {}).setdefault(
            "weights", {})["tendency"] = 0.0
        _lm = _ms.setdefault("learning_modulation", {})
        _lm.update({"base": 1.0, "phasic_coef": 0.0, "tonic_coef": 0.0,
                    "stress_coef": 0.0, "surprise_coef": 0.0,
                    "min": 1.0, "max": 1.0})   # 学习率恒定 1.0×：激素不改记账
    logger.warning("[Experiment] learning_closed_loop 已装配（shield 表见 "
                   "config.experiment.shield；LLM 事实/因果通道本就为零）")

# ── Drive System Phase 3 MVP: 内部驱动力 ──
drive_evaluator = DriveEvaluator(kg, config)
drive_evaluator.engine = engine  # 直写 Drive 激活后由引擎同步活跃前沿
engine.modulation = drive_evaluator.modulation()  # 网络/张力→扩散参数闭环
cc.drive_evaluator = drive_evaluator  # Bug 修复：自主期驱动力刷新（否则 CuriosityDrive 被衰减吃光）
drive_evaluator.bootstrap_drives()
autonomy.modulation = drive_evaluator.modulation()  # 行动候选的探索权重接调制层
try:
    action_space.ensure_action_space()   # Drive/情绪→行动 供性边 + 行动族本体
except Exception as _ase:
    logger.warning(f"[ActionSpace] 供性边种子失败（候选退回图谱激活通道）: {_ase}")
logger.info("[Drive] DriveEvaluator 已初始化（CognitiveField 四层动力学）")

# ── 神经调制子图（R2 P6）─────────────────────────────────────
# 12 个调制器镜像点在图上原本是孤岛（审计 D-15）：种入 调制目标:* 靶点 +
# 调制边 + Self-[处于]-> 锚定，再把边**投影**成 ModulationLayer 的系数行。
# 种入后图是权威：改一条边=改一份效力，删一条边=收回一份效力（不再有
# per-hormone 的 if 代码）。表在 config.modulator_system.edges（§24 集中管理）。
try:
    import modulator_subgraph
    _ms_boot = modulator_subgraph.ensure_modulator_subgraph(kg, internal_state, config)
    _ms_proj = modulator_subgraph.project_coefficients(
        kg, internal_state, drive_evaluator.modulation(), config)
    logger.info("[Modulation] 调制子图：靶点+%s 边+%s 锚定+%s 跳过%s；"
                "投影 %s 条边→%s 行系数（回收 %s）",
                _ms_boot.get("targets"), _ms_boot.get("edges"),
                _ms_boot.get("anchors"), _ms_boot.get("edges_skipped"),
                _ms_proj["projection"]["edges_read"], _ms_proj["applied"],
                _ms_proj["removed"])
    if _ms_proj["projection"].get("no_positive_out"):
        logger.warning("[Modulation] 这些调制器只有负权出边，扩散里不会发射（纯抑制"
                       "节点在引擎侧被 `total_w>0` 拦下）：%s",
                       _ms_proj["projection"]["no_positive_out"])
except Exception as _mse:
    logger.warning(f"[Modulation] 调制子图装配失败（退回 P5 状态：调制器仍是孤岛）: "
                   f"{_mse}")

# ── 心情的调制器包络（R2 P11）────────────────────────────────
# 心情的**数值在 persona 里**，"哪些调制器以多大权重进底色"是**图上的边**
# （`调制器 -[w]-> 心情`，出厂表同上）。这里只做一件事：把编译器
# `mood_projection` 作为只读取值闭包交给 persona。
# 方向是**单向**的：personality_baseline 没有任何调制器写入口，`心情` 节点也不建
# 出边（闸门测试 tests/test_mood_one_way.py 从数值指纹/图结构/源码三处钉住）。
try:
    if _xm.shield("mood"):
        logger.info("[Experiment] mood 屏蔽：心情调制包络不接线（modulator_term≡0）")
    else:
        persona.set_mood_source(
            lambda: modulator_subgraph.mood_projection(kg, internal_state, config))
    _mf = modulator_subgraph.mood_facts(kg, internal_state, config)
    logger.info("[Modulation] 心情包络已接线：节点 %s（入边 %s / 出边 %s），"
                "cap %s，本拍 term %s",
                _mf.get("node"), _mf.get("in_edges"), _mf.get("out_edges"),
                (_mf.get("projection") or {}).get("cap"),
                (_mf.get("projection") or {}).get("term"))
except Exception as _mpe:
    logger.warning(f"[Modulation] 心情包络接线失败（心情仍是纯事件余韵）: {_mpe}")

# ── 调制事件表（R2 P7）───────────────────────────────────────
# `事件类型:* -[影响 w]-> 调制器` 决定"什么事件动哪个调制器、动多少"，
# 取代 reward.release() 里的逐激素 if/else（六个常数迁入边权重）。
# 与子图同一约定：种入后图是权威。应用器是 modulation_events.ModulatorEngine，
# reward_system.mod_engine() 就是这个实例（惰性构造 + 自动 attach 图谱）。
try:
    _me = reward_system.mod_engine()
    _me_boot = _me.ensure() if _me is not None else {"skipped": "no_engine"}
    logger.info("[Modulation] 事件表：%s 类事件，影响边 +%s，跳过 %s",
                _me_boot.get("types"), _me_boot.get("edges"),
                _me_boot.get("edges_skipped"))
    if _me_boot.get("conflicts"):
        logger.warning("[Modulation] 事件表冲突 %s 处（同一事件+调制器写了两条规则）",
                       len(_me_boot["conflicts"]))
except Exception as _mee:
    logger.warning(f"[Modulation] 事件表装配失败（奖赏释放将退回无调制）: {_mee}")

# ── 能力图谱装配（Capability–Action–Executor 2026-09-20）────────
# 能力/概念/诱发边/抑制边种入图谱（表在 config.capability_graph，种入后
# 图是权威）；CapabilityIndex 是"概念→能力→executor"的路径查询器。
# autonomy.cap_index 注入后候选发现走图谱；摘除（或 enabled=false）
# 精确回滚到旧 if 链。embodiment.capabilities() 供 executor 连接态。
from capability_graph import CapabilityIndex
cap_index = CapabilityIndex(kg, engine, config)
cap_index.embodiment = embodiment_router   # caps=连接态∪屏幕（测试同款）
try:
    cap_index.ensure_graph()
except Exception as _cge:
    logger.warning(f"[CapabilityGraph] 种子失败（autonomy 走回滚路径）: {_cge}")
autonomy.cap_index = cap_index
action_manager.cap_index = cap_index
mc_embodiment.cap_index = cap_index
# MC 世界语义入图 + Safety Kernel（具身图谱化 2026-09-21）：
# 物种/保护/工具/产出/危险因果全部上图（config.mc_world 种子）；
# kernel 只做不可逾越的执行不变量（资产授权/攻击核验/防抖/濒死停手）。
import mc_knowledge as _mck
_mck.init_protection_node(config)
try:
    _mck.ensure_mc_world(kg, config)
except Exception as _mke:
    logger.warning(f"[MCKnowledge] 世界语义种图失败（kernel 降级名单外放行）: {_mke}")
# P3/§2-4 先验知识层（具身学习闭环地基，2026-09-23）：通用概念/关系/
# 分类桥/探索缺口与配方 hub 幂等种图——扩散从此有"桥"可走。零 LLM。
import prior_knowledge as _pk
try:
    _pk.ensure_prior(kg, config)
except Exception as _pke:
    logger.warning(f"[Prior] 先验层种图失败（缺口机制本拍不可用）: {_pke}")
from safety_kernel import SafetyKernel
safety_kernel = SafetyKernel(kg=kg, engine=engine, config=config)
action_manager.kernel = safety_kernel
# 具身状态映射器：观测事实→图激活/生存压力（感知节拍调用，无 if→action）
from embodied_mapper import EmbodiedStateMapper
autonomy.mapper = EmbodiedStateMapper(kg=kg, engine=engine, config=config,
                                      causal=causal_learner)
logger.info("[CapabilityGraph] 能力层已装配（概念→需要→能力→executors 路径查询）"
            "+ MC 世界语义入图 + Safety Kernel")

# ── 言语行为图谱化（speech_act_graph 2026-09-20）────────────────
# 概念层（断言/指令/承诺/表达/宣告）→ 细化边合并既有 情境:X 层 →
# 倾向边连 回应方式 → 行为竞争词表。每轮话语建短期事件节点入图，
# 概念作为统一激活入口参与既有 diffusion（不建第二套决策系统）。
from speech_act_graph import (bootstrap_speech_acts, inject_utterance,
                              projection as speech_act_projection)
bootstrap_speech_acts(kg, engine)
logger.info("[SpeechAct] 言语行为概念层已播种（万物皆图：言外行为入认知场）")

# ── Action System Phase 4 MVP: 认知层 Capability ──
action_selector = ActionSelector(kg, config,
                                 drive_evaluator=drive_evaluator,
                                 self_updater=self_updater)
action_selector.bootstrap_capabilities()
logger.info("[Action] ActionSelector 已初始化")

# ── 预测误差基线（V2 补账）：本轮认知焦点 vs 上轮预期 ──
from prediction_baseline import PredictionBaseline
prediction_baseline = PredictionBaseline(config=config)

# ── 认知信号接线（需求控制器 + 四驱 provider，2026-09-20 补账）──────
# 全部是"读既有系统"的闭包：具身状态、行为竞争、奖赏环、反思队列、
# 聊天时间戳——没有新状态源，只有新消费口。
from skills.observation import HOSTILE_ENTITIES as _HOSTILE_SET


def _cs_mc():
    try:
        return mc_embodiment.raw_state() or {}
    except Exception:
        return {}


def _cs_last_interaction_min():
    try:
        last = get_chat_log().recent(n=1)
        if not last:
            return 60.0
        import datetime as _dt
        ts = str((last[0] or {}).get("time") or "")
        t = _dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
        idle = max(0.0, (time.time() - t) / 60.0)
        # 陪伴缓释孤独（2026-09-24 "跟随占大头"诊断）：同伴在场时按 ×0.25
        # 计——贴着挂机的人胜过独处，但真正的满足只来自交流；不缓释的话
        # SocialDrive 永远压满，跟随每轮挤掉一切其他行为。
        if (_cs_mc() or {}).get("playersNearby"):
            idle *= 0.25
        return idle
    except Exception:
        return 0.0


def _cs_safety_target():
    st = _cs_mc()
    if not st.get("connected"):
        return internal_state.need_level("safety")   # 断开不猜：保持原值
    try:
        hp = float(st.get("health", 20) or 20)
        fd = float(st.get("food", 20) or 20)
        hd = min([float(e.get("dist") or 99) for e in (st.get("nearbyEntities") or [])
                  if str(e.get("name") or "") in _HOSTILE_SET] + [99.0])
        tod = st.get("timeOfDay")
        night = tod is not None and 13000 <= int(tod) % 24000 < 23000
    except Exception:
        return internal_state.need_level("safety")
    t = 0.15
    if fd <= 6:
        t = max(t, 0.8)
    if hd <= 8:
        t = max(t, 0.7)
    if hp <= 8:
        t = max(t, 0.95)
    if night and hd <= 16:
        t = max(t, 0.5)
    return t


def _cs_social_target():
    st = _cs_mc()
    presence = 1.0 if (st.get("playersNearby") and st.get("connected")) else 0.0
    idle = min(1.0, _cs_last_interaction_min() / 60.0)
    oxy = internal_state.modulator_level("oxytocin") or 0.0
    return min(1.0, 0.15 + 0.45 * idle + 0.25 * presence
               + 0.15 * min(1.0, oxy * 1.5))


def _cs_exploration_target():
    try:
        drive = float(getattr(kg.nodes.get("CuriosityDrive"), "activation", 0) or 0)
    except Exception:
        drive = 0.0
    return min(1.0, 0.25 + 0.5 * (drive / 5.0))


def _cs_recent_failures(n=12, window_s=600.0):
    try:
        _today = time.strftime("%Y/%m/%d")
        return float(sum(1 for a in action_manager.status()["recent"]
                         if not a.get("success") and not a.get("cancelled")
                         and str(a.get("ts", "")).startswith(_today)))
    except Exception:
        return 0.0


def _cs_blockers():
    try:
        cnt = 0
        for agg in getattr(causal_learner, "_aggregations", {}).values():
            for osig, o in (agg.get("outcomes") or {}).items():
                if ":failed:" in str(osig) and int(o.get("support", 0)) >= 2:
                    cnt += 1
        return float(min(4, cnt))
    except Exception:
        return 0.0


def _cs_novel_objects():
    try:
        with kg._lock:
            return float(sum(1 for n, nd in kg.nodes.items()
                             if str(n).startswith("Unknown")
                             and float(getattr(nd, "activation", 0) or 0) > 0.5))
    except Exception:
        return 0.0


def _cs_pending_candidates():
    try:
        return float(len(reflection_engine.get_pending_candidates()))
    except Exception:
        return 0.0


def _cs_negative_ratio():
    try:
        ev = reward_system.recent_events(n=20)
        if not ev:
            return 0.0
        # blocked = "世界此刻没让过"（暂态：夜拦/腿卡/不可见/失联），不是她
        # 决策的错——不进负回报比。否则一天腿坏会把负比灌到 0.75，心情拖进
        # 沮丧态（mood_low 0.42）、皮质醇抬高行动阈值，越挫越不动（2026-09-25
        # 实测的瘫痪循环第二环）。
        neg = sum(1 for e in ev if float(e.get("valence", 0) or 0) < 0
                  and e.get("outcome") != "blocked")
        return neg / float(len(ev))
    except Exception:
        return 0.0


def _cs_mood_valence():
    try:
        return float(persona.current_mood().get("valence", 0.0) or 0.0)
    except Exception:
        return 0.0


internal_state.set_need_signal_provider("safety", _cs_safety_target)
internal_state.set_need_signal_provider("social", _cs_social_target)
internal_state.set_need_signal_provider("exploration", _cs_exploration_target)
internal_state.set_need_signal_provider(
    "competence",
    lambda: min(1.0, 0.2 + 0.4 * min(1.0, (_cs_recent_failures() + _cs_blockers()) / 4.0)))

drive_evaluator.set_signal_provider("social_idle_minutes", _cs_last_interaction_min)
# 刺激剥夺钟（2026-09-24，与 social_idle_minutes 同构）：距最近一次
# "好奇心动机成功结算"的分钟数；provider 容错取属性，重启后从进程锚点走。
drive_evaluator.set_signal_provider(
    "stimulation_idle_minutes",
    lambda: max(0.0, (time.time() - float(getattr(
        autonomy, "_last_novel_ts", time.time()))) / 60.0))
drive_evaluator.set_signal_provider("social_need", lambda: internal_state.need_level("social"))
if not _xm.shield("hormone_to_params"):
    drive_evaluator.set_signal_provider("oxytocin", lambda: internal_state.modulator_tonic("oxytocin"))
drive_evaluator.set_signal_provider(
    "players_present",
    lambda: float(len((_cs_mc().get("playersNearby") or []))
                  if _cs_mc().get("connected") else 0))
drive_evaluator.set_signal_provider("recent_goal_failures", _cs_recent_failures)
drive_evaluator.set_signal_provider("causal_blockers", _cs_blockers)
drive_evaluator.set_signal_provider("novel_objects", _cs_novel_objects)
drive_evaluator.set_signal_provider("competence_need",
                                    lambda: internal_state.need_level("competence"))
drive_evaluator.set_signal_provider("prediction_surprise",
                                   lambda: prediction_baseline.recent_avg_surprise())
drive_evaluator.set_signal_provider("pending_reflection_candidates", _cs_pending_candidates)
drive_evaluator.set_signal_provider("recent_negative_reward_ratio", _cs_negative_ratio)
drive_evaluator.set_signal_provider("mood_valence", _cs_mood_valence)
logger.info("[Signals] 需求控制器与四驱信号 provider 已接线（读既有系统，无新状态源）")

# ── 认知网络调制层接线（2026-09-20 Drive 重构；R2 P3 推广到 12 个调制器）──
# 激素走调制器通道（Drive 速率/亲和增益 + hormone.* 信号），不进张力加数。
# 读的是 **modulator_tonic**（慢分量 + 受体曲线）：设计决定 D2——tonic→参数通道，
# phasic→图谱激活。瞬时惊喜不该把长期参数瞬间推走。
for _horm in internal_state.modulator_names():
    if _xm.shield("hormone_to_params"):
        break   # ①：不注册 tonic 读数 ⇒ 场侧"缺读数=无调制"，hormone.*≡0
    drive_evaluator.set_signal_provider(
        _horm, (lambda h=_horm: internal_state.modulator_tonic(h)))

# ── R2 P9：需求进信号空间 + 图上的稳态偏置 ──
# ① `need.*` 信号（审计 D-4：effects 里有 `need.social` 这行却从不生产）——
#    注册的是 `need_salience()`（紧迫度与离靶距离里较大的那个），不是裸 level：
#    在理想区间内波动的需求不该被当饥饿去压低表达阈值。
for _nd in internal_state.need_names():
    drive_evaluator.register_need(
        _nd, (lambda n=_nd: internal_state.need_salience(n)))
# ② 图上的 `调制器 -[w]-> 网络/驱动` 边 → 场稳态的连续偏置（每拍 step 时取一次）。
#    名册由场自己给（`node_names()`），所以图上的 `CENetwork` 与字段 `CEN`
#    只有这一处换算，不存在第二份对应表。
try:
    drive_evaluator.set_graph_bias_source(
        lambda: modulator_subgraph.graph_biases(
            kg, internal_state, config, drive_evaluator.field.node_names()))
    logger.info("[Modulation] 需求信号与图上偏置已接线（R2 P9）")
except NameError:
    logger.warning("[Modulation] modulator_subgraph 不可用：图上偏置未接线"
                   "（场仍按出厂动力学运行，只是少一条稳态通道）")


def _cf_task_engaged():
    """任务投入度：用户回合 5 分钟内线性衰减；进行中动作=任务占线。"""
    eng = 0.0
    try:
        eng = max(eng, max(0.0, 1.0 - _cs_last_interaction_min() / 5.0))
    except Exception as _lie:
        logger.warning(f"[CF] 最后交互时间读取失败(eng 回落 0.0): {_lie}")
    try:
        if action_manager.status().get("current"):
            eng = max(eng, 0.8)
    except Exception as _ase:
        logger.warning(f"[CF] action_manager.status 失败(eng 不再抬 0.8): {_ase}")
    return min(1.0, eng)


def _cf_time_since_task():
    try:
        return min(1.0, _cs_last_interaction_min() / 30.0)
    except Exception:
        return 0.0


def _cf_player_near():
    try:
        st = _cs_mc()
        return 1.0 if (st.get("connected")
                       and (st.get("playersNearby") or [])) else 0.0
    except Exception:
        return 0.0


drive_evaluator.set_context_provider("task_engaged", _cf_task_engaged)
drive_evaluator.set_context_provider("time_since_task", _cf_time_since_task)
drive_evaluator.set_context_provider("player_near", _cf_player_near)
drive_evaluator.set_context_provider(
    "idle_minutes", lambda: min(1.0, _cs_last_interaction_min() / 60.0))
logger.info("[CognitiveField] 激素调制器与网络语境读数已接线")

# ── Compound Phrase Handling: 复合短语判定系统 ──
compound_handler = CompoundPhraseHandler(kg, config)
logger.info(f"[CompoundPhrase] Handler 已初始化 (slots={compound_handler.phrase_table.slot_count}, "
            f"surface_forms={compound_handler.phrase_table.surface_form_count})")

# =========================================================
# Embedding Manager (FAISS)
# =========================================================

from embedding_manager import EmbeddingManager
emb_mgr = EmbeddingManager()

# 图写入 → 索引跟随：新节点诞生时登记待索引（只记 id，编码交给刷新点）。
# 图谱是唯一真源，索引是派生缓存——她每长出一个新认知，语义召回就能找到它。
kg._cb_node_added.append(emb_mgr.mark_dirty)
cc._emb_mgr = emb_mgr   # 持续认知循环周期性把待索引节点编入 FAISS

# =========================================================
# Ear 听觉感知处理器
# =========================================================

ear_processor = get_ear_processor()
# 默认禁用，需要前端手动开启
logger.info("[Ear] 听觉感知模块已加载 (默认禁用)")

# =========================================================
# Vision 视觉感知处理器
# =========================================================

vision_processor = get_vision_processor()
vision_processor.set_graph(kg, engine)
# 默认禁用，需要前端手动开启
logger.info("[Vision] 视觉感知模块已加载 (默认禁用)")

# =========================================================
# Action 动作注册表 — 自动发现
# =========================================================

try:
    discover_actions()
    logger.info(f"[Action] 已注册动作: {len(list_actions())}")
except Exception as _action_e:
    logger.warning(f"[Action] 动作发现失败: {_action_e}")

# =========================================================
# Conversation Gap Detector
# =========================================================
gap_detector = ConversationGapDetector()

# =========================================================
# Capability Registry — 能力注册表（发育阶段核心）
# =========================================================

from capability_registry import get_capability_registry

cap_registry = get_capability_registry()
cap_registry.bootstrap_development_stage(kg)
# 统一 Capability 层（2026-09-20）：旧 能力_* 引擎节点幂等补
# type=capability/channel=engine 标记（status 真相源已图优先，JSON 降缓存）
try:
    cap_registry.sync_all_to_graph()
except Exception as _crs:
    logger.debug(f"[CapabilityRegistry] 图同步跳过: {_crs}")
logger.info(f"[Capability] 能力注册表已初始化 ({len(cap_registry.list_all())} 项能力)")

# 保存能力节点到运行时图谱
pack_mgr.runtime_graph = kg
pack_mgr.save_runtime_graph()

# =========================================================
# Intent Tree 提取
# =========================================================


# =========================================================
# 保存 & 模糊匹配工具
# =========================================================

_last_save_time = 0


def _ingest_narrative(kg: KnowledgeGraph, text: str) -> dict:
    """叙事分段抽取入图：故事节点+按序事件实例+角色原子节点。

    结构：用户-[讲述]->故事X；故事X-[包含情节]->事件i；
          事件i-1 -[时间顺序]-> 事件i；角色-[参与]->事件i。
    返回 {story, events(节点id列表), characters}。
    """
    import time as _t
    nd = nlp.extract_narrative(text)
    if not nd.get("events"):
        return {}
    story_name = nd.get("story") or ("剧情_" + text.strip()[:10])
    now = now_str()

    def _add(nid, extra, space="semantic", label="declarative-semantic", w=0.5):
        if nid in kg.nodes:
            n0 = kg.nodes[nid]
            for k, v in extra.items():
                n0.extra_attrs.setdefault(k, v)
            return n0
        n0 = Node(id=nid, weight=w, label=label, graph_space=space,
                  extra_attrs=extra)
        kg.add_node(n0)
        engine.name_to_node[nid] = n0
        return n0

    if story_name not in kg.nodes:
        _add(story_name, {"canon": "concept", "source": "narrative",
                          "created": now}, w=0.6)
    if not any(e.src == "用户" and e.dst == story_name and e.relation == "讲述"
               for e in kg.edges):
        kg.add_edge(Edge(src="用户", dst=story_name, relation="讲述",
                         weight=0.8, relation_category="social_relation"))
    event_ids = []
    prev = None
    for i, ev in enumerate(nd["events"], 1):
        title = str(ev.get("title", ""))[:20] or f"情节{i}"
        nid = f"情节{i}—{story_name}" if nid_ok(title) else f"情节{i}—{story_name}"
        if nid in kg.nodes:
            nid = f"{nid}({int(_t.time()) % 10000})"
        _add(nid, {"canon": "event_instance", "sequence": i,
                   "summary": str(ev.get("summary", ""))[:150],
                   "source": "narrative", "created": now},
             space="episodic", label="declarative-episodic", w=0.5)
        if not any(e.src == story_name and e.dst == nid and e.relation == "包含情节"
                   for e in kg.edges):
            kg.add_edge(Edge(src=story_name, dst=nid, relation="包含情节",
                             weight=0.8, relation_category="semantic_relation"))
        if prev:
            if not kg.get_edge(prev, nid, "时间顺序"):
                kg.add_edge(Edge(src=prev, dst=nid, relation="时间顺序",
                                 weight=0.8, relation_category="temporal_relation"))
        for actor in (ev.get("actors") or [])[:4]:
            actor = str(actor).strip()[:20]
            if not actor:
                continue
            _add(actor, {"canon": "concept"})
            if not any(e.src == actor and e.dst == nid and e.relation == "参与"
                       for e in kg.edges):
                kg.add_edge(Edge(src=actor, dst=nid, relation="参与",
                                 weight=0.7, relation_category="social_relation"))
        if "用户" in (ev.get("actors") or []):
            if not any(e.src == "用户" and e.dst == nid and e.relation == "参与"
                       for e in kg.edges):
                kg.add_edge(Edge(src="用户", dst=nid, relation="参与",
                                 weight=0.8, relation_category="social_relation"))
        for ent in (ev.get("entities") or [])[:3]:
            ent = str(ent).strip()[:20]
            if ent and ent in kg.nodes:
                if not kg.get_edge(nid, ent, "涉及"):
                    kg.add_edge(Edge(src=nid, dst=ent, relation="涉及",
                                     weight=0.5))
        prev = nid
        event_ids.append(nid)
    # 用户讲述关系 + 角色归属故事
    for c in nd.get("characters", [])[:8]:
        c = str(c).strip()[:20]
        if c and c in kg.nodes:
            if not kg.get_edge(story_name, c, "涉及"):
                kg.add_edge(Edge(src=story_name, dst=c, relation="涉及",
                                 weight=0.5))
    return {"story": story_name, "events": event_ids,
            "characters": nd.get("characters", [])[:8]}


def nid_ok(title):
    return bool(re.match(r"^[\u4e00-\u9fa5A-Za-z0-9—]+$", title))


def _inject_eye_result_to_graph(kg: KnowledgeGraph, eye_result: dict):
    """Eye 感知结果入图。2026-09-22 起真身搬到 eye/screen_ocr.py
    （自主路径 ActionManager 与对话路径共用同一注入器，不复制逻辑）；
    这里保留名字给既有调用点，并补上 engine 前沿点亮。"""
    from eye.screen_ocr import inject_observation
    return inject_observation(kg, eye_result, engine=engine)


def _inject_ear_result_to_graph(kg: KnowledgeGraph, ear_result: dict):
    """将 Ear 听觉感知结果注入知识图谱"""
    summary = ear_result.get("summary", {})
    audio_type = ear_result.get("audio_type", "")

    # 确保音频类型节点存在
    type_names = {
        "speech": "语音",
        "music": "音乐",
        "sound": "环境声",
        "mixed": "混合音频",
    }
    type_name = type_names.get(audio_type, "音频")

    # 创建/更新音频类型节点
    if type_name not in kg.nodes:
        kg.add_node(Node(id=type_name, weight=0.5, label="declarative-semantic",
                          extra_attrs={"category": "audio_type"}))

    # 处理转录文本
    transcript = summary.get("transcript", "")
    if transcript:
        # 确保"转录"节点存在
        for node_id in ["语音转录", "语音识别"]:
            if node_id not in kg.nodes:
                kg.add_node(Node(id=node_id, weight=0.4, label="declarative-semantic",
                                  extra_attrs={"category": "ear_perception"}))

        # 创建文本内容节点
        text_id = f"ear_text_{hash(transcript) % 100000}"
        if text_id not in kg.nodes:
            kg.add_node(Node(id=text_id, weight=0.3, label="declarative-episodic",
                              graph_space="episodic",
                              extra_attrs={"transcript": transcript[:200]}))

    # 处理环境声分类
    top_sounds = summary.get("top_sounds", [])
    for sound_info in top_sounds[:3]:
        sound_name = sound_info.get("class", "")
        if sound_name and sound_name not in kg.nodes:
            kg.add_node(Node(id=sound_name, weight=0.3, label="declarative-semantic",
                              extra_attrs={"category": "sound_class", "count": sound_info.get("count", 0)}))


def _fuzzy_match_node(query_name: str) -> str:
    """将 NLP 解析出的名称模糊匹配到图谱中实际存在的节点 ID。
    规则：完全一致 > 别名匹配 > 前缀+边界 > 包含+边界，均不匹配则返回原名称。
      木斧 → 木斧(Minecraft)  ✓  (查询词后紧跟 '(' )
      浏览 → 浏览器         ✗  (查询词后紧跟 '器'，字母，无边界)
      Mc → Minecraft         ✓  (别名匹配: Minecraft.aliases 包含 "Mc")
    """
    q = query_name.strip()
    if not q:
        return q
    qlow = q.lower()

    # 1. 完全匹配（大小写不敏感）
    # engine.name_to_node 的 key 是原始节点 ID，需要两边都 lower 比较
    for _nid in engine.name_to_node:
        if _nid.lower() == qlow:
            return _nid

    # 2. 别名匹配：检查图谱节点的 extra_attrs.aliases
    #    节点可以在 extra_attrs 中声明 aliases 列表，如 Minecraft.aliases=["Mc","MC","我的世界"]
    with kg._lock:
        for nid, node in kg.nodes.items():
            aliases = node.extra_attrs.get("aliases", []) if hasattr(node, 'extra_attrs') else []
            if isinstance(aliases, list):
                for alias in aliases:
                    if str(alias).strip().lower() == qlow:
                        logger.info(f"[Alias] '{query_name}' → '{nid}' (别名匹配: {alias})")
                        return nid

    # 2.5 包含匹配（中文无词边界）：查询词与节点名互为子串即视为同一实体。
    #     例：收获日2 ⊂ 收获日2游玩（LLM 抽取措辞波动产生的近似重复）。
    #     防护：较短方长度 ≥3 且占比 ≥60%——避免"两个礼拜"被长事件名
    #     （"事件: 我和高中同学的收获日2游玩结束了..."）误吞。
    if len(q) >= 3:
        _best_len = 0
        _best_id = None
        for _nid in engine.name_to_node:
            _nlen = len(_nid)
            if min(len(q), _nlen) < 3:
                continue
            _longer = q if len(q) >= _nlen else _nid
            _shorter = _nid if _longer == q else q
            if _shorter not in _longer:
                continue
            if len(_shorter) / len(_longer) < 0.6:
                continue
            if _nlen > _best_len:  # 同含时优先更长的候选（更具体的实体名）
                _best_len = _nlen
                _best_id = _nid
        if _best_id:
            logger.info(f"[Fuzzy] '{query_name}' → '{_best_id}' (包含匹配)")
            return _best_id

    # 3. 边界匹配：按匹配边界数量打分
    best_score = 0
    best_id = None

    for nid in engine.name_to_node:
        nlow = nid.lower()
        idx = nlow.find(qlow)
        while idx != -1:
            prev_char = nlow[idx - 1] if idx > 0 else ' '
            next_char = nlow[idx + len(qlow)] if idx + len(qlow) < len(nlow) else ' '
            boundaries = 0
            if not prev_char.isalnum():
                boundaries += 1
            if not next_char.isalnum():
                boundaries += 1
            # 两端都是边界（最高优先级）或完全匹配开头且后面是边界
            if boundaries == 2 or (idx == 0 and (not next_char.isalnum())):
                score = boundaries + (10 if idx == 0 else 0)
                if score > best_score:
                    best_score = score
                    best_id = nid
            idx = nlow.find(qlow, idx + 1)

    if best_id:
        logger.info(f"[Fuzzy] '{query_name}' → '{best_id}' (score={best_score})")
        return best_id

    return q


# ── 情景记忆事件框架（论文 §3.2/§3.4/§6.3）──────────────────────────
# episodic 记忆以事件节点为枢纽入库：
#   用户 -[参与/态度]→ 事件节点 -[涉及/参与者/发生时间/引发]→ 实体/时间/情绪
# 事件节点落 episodic 空间；时间/地点节点是语义空间的共享参考实体，
# 不再因 node_type 被误标为情景记忆。

_NODE_TYPE_SPEC = {
    "实体": ("declarative-semantic", "semantic"),
    "事件": ("declarative-episodic", "episodic"),
    "时间": ("declarative-semantic", "semantic"),
    "地点": ("declarative-semantic", "semantic"),
    "概念": ("declarative-semantic", "semantic"),  # 概念/情绪词不是事件，落语义空间
}


def _node_label_space(node_type: str, default_label: str):
    """按节点类型决定 (label, graph_space)。未列出的类型沿用默认标签。"""
    spec = _NODE_TYPE_SPEC.get(str(node_type).strip())
    if spec is None:
        return (default_label, "semantic")
    return spec


def _edge_category(rel: str) -> str:
    """按关系词推断边类别。

    架构对齐（2026-09-19）：归一到 graph_schema（规范词表 → 类别），
    不再使用本文件曾有的 8 条目局部表——两张表漂移是类别错配的来源。
    """
    import graph_schema as _gs
    return _gs.category_for(_gs.normalize_relation(rel))


def _wire_event_structure(kg: KnowledgeGraph, draft: dict, added: dict, tag: str = "Consolidate"):
    """事件框架收尾：时间属性 + 父子事件挂接 + 焦点指针。

    节点/边循环把草稿写入图谱后调用。draft["event"] 由 LLM 生成
    （MEMORY_EXTRACT 模板），含 summary/event_time/parent_event。

    架构对齐（2026-09-19）：
      - 时间不再落成日期节点（2026-09-19 那类节点曾是 25 入度的枢纽）——
        时间是事件的属性（extra_attrs["event_time"]），图上的时间索引只保留
        固定的时段桶（凌晨/清晨/…，发生于时段 边，见 link_event_to_bucket）。
      - 事件 summary 不允许 X—Y—Z 命名式（把谓词+槽位烤进节点 id）；
        命中守卫时拆出尾部实体改为 对象/地点 关联，summary 保留谓词短语。
    """
    event_info = draft.get("event")
    if not isinstance(event_info, dict):
        return
    import graph_schema as _gs
    from graph_model import Edge as _GEdge
    ev_id = str(event_info.get("summary", "")).strip()
    # ── X—Y—Z 命名式守卫（兜底）──
    # 正常路径下 extract_assertion_graph 已把命名式拆解为"谓词短语 + 槽位边"，
    # 这里只处理绕过该漏斗的草稿（API 手工审批等）。优先找拆解后的谓词节点，
    # 找不到再回退原名（存量兼容：不因守卫丢事件）。
    _original_ev_id = ev_id
    _compound_slots = list(draft.get("_compound_slots") or [])
    if _gs.COMPOUND_NODE_RE.match(ev_id):
        parts = [p.strip() for p in ev_id.split("—") if p.strip()]
        if len(parts) >= 3 and len(parts[0]) >= 2:
            _aspect = parts[1]
            _entity = "—".join(parts[2:])
            ev_id = parts[0]
            _compound_slots.append((_aspect, _entity))
    ev_node = kg.nodes.get(ev_id)
    if ev_node is None and ev_id != _original_ev_id:
        # 拆解后的谓词节点不存在 → 回退原名（兼容，不丢事件）
        ev_id = _original_ev_id
        _compound_slots = []
        ev_node = kg.nodes.get(ev_id)
    if ev_node is None:
        # LLM 的事件摘要与草稿节点名可能有措辞差异，先模糊归并再找
        _merged = _fuzzy_match_node(ev_id)
        ev_node = kg.nodes.get(_merged)
        if ev_node is not None:
            ev_id = _merged
    if ev_node is None:
        logger.warning(f"[{tag}] 事件节点 '{event_info.get('summary')}' 未找到，跳过事件收尾")
        return

    # 1. 事件节点必须落 episodic 空间，并带时间戳属性（论文 §3.2 中心节点）
    if ev_node.label != "declarative-episodic":
        ev_node.label = "declarative-episodic"
        engine.note_action_dirty(ev_node.id)  # label 变更影响行动队列资格
    ev_node.graph_space = "episodic"
    ev_time = str(event_info.get("event_time", "")).strip()
    # LLM 偶发输出字符串 "None"/"null" 表示无时间，归一为空，
    # 避免落出字面量 "None" 时间锚点节点（与 parent_event 同源问题）
    if ev_time.lower() in ("none", "null", "n/a"):
        ev_time = ""
    if ev_time:
        ev_node.extra_attrs = ev_node.extra_attrs or {}
        if not str(ev_node.extra_attrs.get("event_timestamp", "")).strip():
            ev_node.extra_attrs["event_timestamp"] = ev_time

    # 1.5 命名式守卫拆出的槽位 → 对象/地点 边（槽位由结构表达，不进节点 id）
    for _aspect, _entity in _compound_slots:
        if _entity in kg.nodes and ev_id in kg.nodes:
            _rel = "位于" if _aspect in ("地点", "位置") else "涉及"
            if not kg.get_edge(ev_id, _entity, _rel):
                kg.add_edge(_GEdge(src=ev_id, dst=_entity, relation=_rel,
                                   weight=0.8,
                                   relation_category=_gs.category_for(_rel)))
                added["edges"] += 1
                logger.info(f"[{tag}] +槽位边: {ev_id} -[{_rel}]→ {_entity}")

    # 2. 时间属性（不再建日期节点）：发生时间 语义保留在事件属性里；
    #    需要图结构的时间检索走时段桶（link_event_to_bucket，固定 5 桶）。
    if ev_time:
        logger.debug(f"[{tag}] 事件时间已入属性: {ev_id}.event_time={ev_time}")

    # 3. 父子事件挂接（论文 §3.4 层级化事件建模）：
    #    子事件 -[时间顺序]→ 头部事件。父事件应来自上下文节点（图内已有），
    #    缺失时不硬造，避免无中生有。
    parent = str(event_info.get("parent_event") or "").strip()
    if parent and parent != ev_id:
        if parent not in kg.nodes:
            logger.warning(f"[{tag}] parent_event '{parent}' 不在图谱中，跳过挂接")
        elif not kg.get_edge(ev_id, parent, "时间顺序"):
            kg.add_edge(_GEdge(
                src=ev_id, dst=parent, relation="时间顺序", weight=0.9,
                relation_category="temporal_relation"))
            added["edges"] += 1
            logger.info(f"[{tag}] +子事件边: {ev_id} -[时间顺序]→ {parent}")

    # 4. 当前焦点事件指针（论文 §3.3 自认知子图）：延续事件聚焦父事件，
    #    新事件聚焦自身。下游提取据此注入【进行中的事件】上下文。
    _parent = parent if (parent and parent in kg.nodes) else None
    _update_focus_event(kg, ev_id, _parent)


def _auto_consolidate_curiosity_knowledge(
    kg: KnowledgeGraph, memory_draft: dict, buffer, original_text: str
) -> dict:
    """好奇心驱动的自动知识沉淀。

    当 FAS 主动提问并获得用户回答后，自动将用户教的知识写入图谱。
    与 /api/memory/approve 不同：此路径跳过人工审批，因为用户明确在教 FAS。

    Returns:
        {"nodes": N, "edges": N} 或 None（无可沉淀内容时）
    """
    if not memory_draft or not memory_draft.get("nodes"):
        return None

    added = {"nodes": 0, "edges": 0}
    assertion_type = memory_draft.get("assertion_type", "semantic")
    default_label = (
        "declarative-episodic" if assertion_type == "episodic"
        else "declarative-semantic"
    )

    for n in memory_draft.get("nodes", []):
        nid = str(n.get("id", "")).strip()
        if not nid:
            continue
        # 名称归并：LLM 每次抽取措辞可能有波动（收获日2 vs 收获日2游玩），
        # 先模糊映射到已有节点，避免近似重复节点。
        _merged = _fuzzy_match_node(nid)
        if _merged != nid:
            logger.info(f"[CuriosityAuto] 名称归并: '{nid}' → '{_merged}'")
        nid = _merged
        if nid in kg.nodes:
            continue
        label = str(n.get("node_type", "概念"))
        kg_label, kg_space = _node_label_space(label, default_label)
        from graph_model import Node as _GNode
        kg.add_node(_GNode(
            id=nid, weight=0.6, label=kg_label, graph_space=kg_space,
            extra_attrs={"source": "curiosity_auto", "taught_by_user": True}
        ))
        # 更新扩散引擎的 name→node 索引
        engine.name_to_node[nid] = kg.nodes[nid]
        added["nodes"] += 1
        logger.info(f"[CuriosityAuto] +节点: {nid} ({kg_label}/{kg_space})")

    for e in memory_draft.get("edges", []):
        src = str(e.get("src", "")).strip()
        dst = str(e.get("dst", "")).strip()
        if not src or not dst or src == dst:
            continue
        # 边端点同样走归并，避免指向被合并掉的原名称节点
        src = _fuzzy_match_node(src)
        dst = _fuzzy_match_node(dst)
        if src == dst:
            continue
        rel = str(e.get("type", e.get("relation", "关联"))).strip()
        w = float(e.get("weight", 0.6))
        w = max(0.1, min(1.0, w))
        from graph_model import Edge as _GEdge
        existing = kg.get_edge(src, dst, rel)
        if existing:
            if w > existing.weight:
                existing.weight = w
            continue
        _edge_ok = kg.add_edge(_GEdge(
            src=src, dst=dst, relation=rel, weight=w,
            relation_category=_edge_category(rel)))
        if not _edge_ok:
            # L1-CON-3:add_edge 端点缺失返回 False——知识草稿边在漏斗处
            # 无声消失,必须留痕("和朋友"这类归并后不存在的端点)
            logger.warning(
                "[CuriosityAuto] 边被丢弃(端点可能不在图内): %s-[%s]->%s",
                src, rel, dst)
            continue
        added["edges"] += 1
        logger.info(f"[CuriosityAuto] +边: {src} -[{rel}]→ {dst}")

    # 事件框架收尾：时间锚点 + 父子事件挂接（论文 §3.2/§3.4）
    _wire_event_structure(kg, memory_draft, added, "CuriosityAuto")

    # 标记 EpisodicBuffer 中对应经历已处理
    if original_text:
        exp = buffer.find_by_text(original_text)
        if exp:
            exp.importance = min(1.0, exp.importance + 0.3)
            buffer.promote_to_long_term(exp)

    if added["nodes"] == 0 and added["edges"] == 0:
        return None
    return added


_temp_node_re = None


def _should_hide_node(node_id: str) -> bool:
    """根据配置决定是否隐藏自动生成的临时节点"""
    if not config.get("hide_temp_nodes", False):
        return False
    global _temp_node_re
    if _temp_node_re is None:
        import re
        pattern = config.get("temp_node_pattern", r"^(思考|记忆|反思|情绪)_\d+|ear_text_\d+|UnknownObject\d+$")
        _temp_node_re = re.compile(pattern, re.IGNORECASE)
    return bool(_temp_node_re.match(str(node_id)))


def _filter_temp_nodes(nodes: list) -> list:
    """过滤掉临时节点，保留非临时节点"""
    if not config.get("hide_temp_nodes", False):
        return nodes
    return [n for n in nodes if not _should_hide_node(n.get("id", ""))]


def _filter_temp_edges(edges: list, visible_ids: set) -> list:
    """过滤掉引用临时节点的边"""
    if not config.get("hide_temp_nodes", False):
        return edges
    return [e for e in edges if e.get("src", "") in visible_ids and e.get("dst", "") in visible_ids]


def _apply_temp_filter(graph_dict: dict) -> dict:
    """对图谱结果应用临时节点过滤"""
    if not config.get("hide_temp_nodes", False):
        return graph_dict
    nodes = _filter_temp_nodes(graph_dict.get("nodes", []))
    visible_ids = {n.get("id", "") for n in nodes}
    edges = _filter_temp_edges(graph_dict.get("edges", []), visible_ids)
    return {"nodes": nodes, "edges": edges}


# =========================================================
# 异步单飞保存写入器
# ---------------------------------------------------------
# 持久化走单写入器后台线程：请求只折叠标志位，写盘（含 kg.to_dict()
# 全量拍平）不阻塞请求线程；写盘中再触发保存 → 折叠为一次补写，且
# 补写重新取图快照（不复用旧快照，防止把旧状态写回去）。
# force=True = 提交并等待本轮（含折叠的补写）写完成——调用方"返回即
# 落盘"的既有约定。写失败只记 ERROR，_last_save_time 不更新（30 秒
# 节流从上次成功起算，失败后窗口内可重试）。
# =========================================================

_save_cv = threading.Condition()
_save_again = False     # 有待写请求（写盘中到达的请求也折叠到这里）
_save_busy = False      # 写盘进行中
_snap_last_ts = 0.0     # 图谱快照时间闸（单写入器线程持有，§18.3 P1）


def _write_graph_once():
    """执行一次完整落盘（写失败不更新 _last_save_time）。"""
    global _last_save_time, _snap_last_ts
    import time as _time
    try:
        logger.info("[SAVE] 正在保存图谱...")
        pack_mgr.runtime_graph = kg
        kg.save(pack_mgr.runtime_graph_path)  # 原子替换（graph_model.save）
        _last_save_time = _time.time()
        logger.info("[SAVE] 图谱保存完成")
        # 落盘成功后按时间闸留一份快照副本（§18.3 P1：缩小守卫拦骤减，
        # 这个补"任意时刻都回得去"；轮转只保留最近 N 份）。快照失败
        # 只记日志——绝不让它影响主落盘的返回值。
        try:
            _snap_last_ts = _rotate_graph_snapshot(
                pack_mgr.runtime_graph_path,
                os.path.join(
                    os.path.dirname(pack_mgr.runtime_graph_path) or "data",
                    "graph_snapshots"),
                interval_s=float(config.get("graph_snapshot_interval_s", 1800) or 0),
                keep=int(config.get("graph_snapshot_keep", 12)),
                last_ts=_snap_last_ts)[-1]
        except Exception as e:
            logger.warning(f"[Rotation] 快照接线异常（忽略）: {e}")
        return True
    except Exception as e:
        logger.exception(f"[SAVE异常] {e}")
        return False


def _save_writer_loop():
    global _save_again, _save_busy
    while True:
        with _save_cv:
            while not _save_again:
                _save_cv.wait()
        while True:
            with _save_cv:
                _save_again = False
            _write_graph_once()
            with _save_cv:
                if not _save_again:
                    _save_busy = False
                    _save_cv.notify_all()  # 唤醒 force 等待者
                    break
                # 写盘期间又有请求 → 循环补写最新快照


def _save(force=False):
    global _save_again, _save_busy

    import time

    now = time.time()

    # 每30秒最多保存一次（force=True 时无视限制）
    if not force and now - _last_save_time < 30:
        return

    with _save_cv:
        _save_again = True
        if not _save_busy:
            # 预占 busy：并发 force 等待者据此阻塞到本轮写完
            _save_busy = True
        _save_cv.notify()
        if force:
            while _save_again or _save_busy:
                _save_cv.wait()


def _start_save_writer():
    global _save_thread
    # 启动时清理上次崩溃残留的 temp 文件（os.replace 原子替换的伴生垃圾）
    _data_dir = os.path.dirname(pack_mgr.runtime_graph_path)
    try:
        if os.path.isdir(_data_dir):
            _prefix = "." + os.path.basename(pack_mgr.runtime_graph_path) + ".tmp"
            for _name in os.listdir(_data_dir):
                if _name.startswith(_prefix):
                    try:
                        os.remove(os.path.join(_data_dir, _name))
                        logger.info(f"[SAVE] 清理残留临时文件: {_name}")
                    except OSError as _oe:
                        logger.warning(f"[SAVE] 残留临时文件清理失败(将留到下次): {_oe}")
    except Exception:
        logger.exception("[SAVE] 临时文件清理失败")
    _save_thread = threading.Thread(target=_save_writer_loop,
                                    daemon=True, name="graph-save-writer")
    _save_thread.start()


_BOOT_REJECTED = False


def _flush_save_at_exit():
    """进程退出前把未落盘的修改写掉（全库原无 shutdown 钩子）。"""
    if _BOOT_REJECTED:
        # 端口守卫拒启的实例：它的图谱只是启动瞬间的旧副本，落盘会
        # 覆盖真正在跑的实例更新过的文件——拒启即静默，一个字都不写。
        return
    try:
        _save(force=True)
    except Exception as e:
        # §14：图谱落盘失败=状态同步丢失，退出前必须留痕（不许静默）
        logger.warning("[shutdown] 图谱落盘失败，最后修改可能丢失: %s", e)
    # 内驱场状态（张力/驱动/网络/调制）关机强制落盘——平时靠每 24 拍
    # 自动存，SIGTERM/崩溃会丢最后一段；下次启动继承（2026-09-25 用户
    # 指令：驱动能量是持续变量，关机不清零）。
    try:
        drive_evaluator.field.save_state()
    except Exception as e:
        # §14：驱动能量持久化是用户点名要保的状态，失败必须可见
        logger.warning("[shutdown] 内驱场状态落盘失败: %s", e)


_save_thread = None

# 启动即起写入器 + 注册退出冲洗（kg/pack_mgr 在此之前均已就绪）
_start_save_writer()
atexit.register(_flush_save_at_exit)

# =========================================================
# 静态文件
# =========================================================

@app.route("/")
def index():

    logger.info("[HTTP] GET /")

    return send_from_directory(
        ".",
        "index.html"
    )

# =========================================================
# 图谱总览
# =========================================================

@app.route("/api/graph", methods=["GET"])
def get_graph():

    logger.info("[API] /api/graph")

    try:

        show_full = (
                request.args.get(
                    "full",
                    "false"
                ).lower() == "true"
        )

        if show_full:

            logger.warning(
                "[API] 返回完整图谱"
            )

            return jsonify(
                kg.to_dict()
            )

        threshold = float(
            request.args.get(
                "threshold",
                0.0
            )
        )

        logger.info(
            f"[API] 激活阈值: {threshold}"
        )

        with kg._lock:

            active_nodes = [
                n.to_dict()
                for n in kg.nodes.values()
                if n.activation > threshold
            ]

            active_node_ids = {
                n["id"]
                for n in active_nodes
            }

            active_edges = [
                e.to_dict()
                for e in kg.edges
                if (
                        e.src in active_node_ids
                        and e.dst in active_node_ids
                )
            ]

        filtered = _apply_temp_filter({
            "nodes": active_nodes,
            "edges": active_edges
        })

        logger.info(
            f"[API] 返回 "
            f"{len(filtered['nodes'])} 节点 "
            f"{len(filtered['edges'])} 边"
        )

        return jsonify(filtered)

    except Exception as e:

        logger.exception(
            f"[API异常] /api/graph: {e}"
        )

        return jsonify({
            "error": str(e)
        }), 500


def _active_graph_dict(threshold: float = 0.0):
    with kg._lock:
        active_nodes = [
            n.to_dict()
            for n in kg.nodes.values()
            if n.activation > threshold
        ]

        active_node_ids = {n["id"] for n in active_nodes}

        active_edges = [
            e.to_dict()
            for e in kg.edges
            if (
                e.src in active_node_ids
                and e.dst in active_node_ids
            )
        ]

    return _apply_temp_filter({
        "nodes": active_nodes,
        "edges": active_edges
    })


def _resolve_center(name: str):
    """邻域查询中心解析：精确 → 大小写 → 别名 → 包含（取最短名）。
    修"查询邻居子图异常"：以前只认字面 ID，输入"haru"或简称直接查不到。"""
    name = str(name or "").strip()
    if not name:
        return None
    with kg._lock:
        if name in kg.nodes:
            return name
        low = name.lower()
        for nid in kg.nodes:
            if nid.lower() == low:
                return nid
        for nid, n in kg.nodes.items():
            for a in ((n.extra_attrs or {}).get("aliases") or []):
                if str(a).strip().lower() == low:
                    return nid
        cands = [nid for nid in kg.nodes
                 if low in nid.lower() or nid.lower() in low]
        if cands:
            return min(cands, key=len)
    return None


def _neighborhood_graph_dict(center: str, depth: int = 2, max_result_nodes: int = 800):
    center = _resolve_center(center)
    depth = max(0, int(depth))
    max_result_nodes = max(10, int(max_result_nodes))

    with kg._lock:
        if not center or center not in kg.nodes:
            return {"nodes": [], "edges": []}, None

        adj = {}
        for e in kg.edges:
            adj.setdefault(e.src, set()).add(e.dst)
            adj.setdefault(e.dst, set()).add(e.src)

        visited = {center}
        q = deque([(center, 0)])

        while q and len(visited) < max_result_nodes:
            nid, d = q.popleft()
            if d >= depth:
                continue
            for nb in adj.get(nid, set()):
                if nb in visited:
                    continue
                visited.add(nb)
                q.append((nb, d + 1))
                if len(visited) >= max_result_nodes:
                    break

        nodes = [kg.nodes[nid].to_dict() for nid in visited if nid in kg.nodes]
        edges = [
            e.to_dict()
            for e in kg.edges
            if e.src in visited and e.dst in visited
        ]

    return _apply_temp_filter({"nodes": nodes, "edges": edges}), center


@app.route("/api/graph/auto", methods=["GET"])
def get_graph_auto():
    try:
        max_nodes = int(request.args.get("max_nodes", 2500))
        depth = int(request.args.get("depth", 2))
        center = request.args.get("center", "").strip()

        total_nodes = len(kg.nodes)
        total_edges = len(kg.edges)

        if total_nodes <= max_nodes and not center:
            data = kg.to_dict()
            filtered = _apply_temp_filter(data)
            return jsonify({
                "mode": "full",
                "center": None,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "nodes": filtered.get("nodes", []),
                "edges": filtered.get("edges", [])
            })

        if not center:
            center = random.choice(list(kg.nodes.keys())) if kg.nodes else ""

        graph, picked = _neighborhood_graph_dict(center, depth=depth)
        filtered = _apply_temp_filter(graph)
        return jsonify({
            "mode": "neighborhood",
            "center": picked,
            "depth": depth,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "nodes": filtered.get("nodes", []),
            "edges": filtered.get("edges", [])
        })
    except Exception as e:
        logger.exception(f"[API异常] /api/graph/auto: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/nodes", methods=["POST"])
def create_node():
    try:
        data = request.json or {}
        _deny = _execution_write_denied(data)
        if _deny:
            logger.warning(f"[SECURITY] 拒绝 POST /api/nodes 写入 execution: {_deny}")
            return jsonify({"error": _deny}), 403
        nid = str(data.get("id", "")).strip()
        if not nid:
            return jsonify({"error": "节点 id 不能为空"}), 400

        weight = float(data.get("weight", 0.5))
        label = data.get("label", "declarative-semantic")
        execution = data.get("execution", None)

        node = Node(
            id=nid,
            weight=weight,
            label=label,
            execution=execution
        )

        ok = kg.add_node(node)
        if not ok:
            return jsonify({"error": f"节点已存在: {nid}"}), 400

        with engine._lock:
            engine.name_to_node[nid] = node

        try: emb_mgr.add_node(node)
        except Exception as _emb_e:
            logger.warning(f"[Embedding] add_node failed: {_emb_e}")

        _save()
        return jsonify({"success": True})
    except Exception as e:
        logger.exception(f"[API异常] /api/nodes: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/nodes/<node_id>", methods=["PUT"])
def update_node(node_id):
    try:
        nid = str(node_id).strip()
        data = request.json or {}
        _deny = _execution_write_denied(data)
        if _deny:
            logger.warning(f"[SECURITY] 拒绝 PUT /api/nodes/{node_id} 写入 execution: {_deny}")
            return jsonify({"error": _deny}), 403
        new_id = data.get("new_id", None)
        if new_id is not None:
            new_id = str(new_id).strip()

        with kg._lock:
            node = kg.nodes.get(nid)
            if not node:
                return jsonify({"error": f"节点不存在: {nid}"}), 404

            # 处理节点重命名
            if new_id and new_id != nid:
                if new_id in kg.nodes:
                    return jsonify({"error": f"节点已存在: {new_id}"}), 409
                old_nid = nid
                pack_name = pack_mgr.runtime_node_map.get(old_nid)
                if not kg.rename_node(old_nid, new_id):
                    return jsonify({"error": f"改名失败: {old_nid} → {new_id}"}), 409
                if pack_name:
                    pack_mgr.runtime_node_map[new_id] = pack_name
                    pack_mgr.runtime_node_map.pop(old_nid, None)
                # ── 持久化重命名映射 ──
                _rename_map[new_id] = old_nid
                # 清理旧的反向映射
                for k, v in list(_rename_map.items()):
                    if v == old_nid and k != new_id:
                        del _rename_map[k]
                _save_rename_map(_rename_map)
                logger.info(f"[Rename] 记录重命名: {old_nid} → {new_id}")
                nid = new_id
            if "weight" in data:
                node.weight = float(data.get("weight", node.weight))
            _label_changed = False
            if "label" in data:
                _label_changed = (node.label != data.get("label", node.label))
                node.label = data.get("label", node.label)
            if "execution" in data:
                node.execution = data.get("execution", None)
            node.touch()

        with engine._lock:
            engine.name_to_node.pop(node_id, None)
            engine.name_to_node[nid] = node
        if _label_changed:
            engine.note_action_dirty(nid)  # label 变更影响行动队列资格

        _save(force=True)
        return jsonify({"success": True, "new_id": nid})
    except Exception as e:
        logger.exception(f"[API异常] PUT /api/nodes/{node_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/nodes/<node_id>", methods=["DELETE"])
def delete_node(node_id):
    try:
        nid = str(node_id).strip()
        ok = kg.remove_node(nid)
        if not ok:
            return jsonify({"error": f"节点不存在: {nid}"}), 404

        with engine._lock:
            engine.name_to_node.pop(nid, None)

        try: emb_mgr.remove_node(nid)
        except Exception as _emb_e:
            logger.warning(f"[Embedding] remove_node failed: {_emb_e}")

        # 清理重命名映射
        if nid in _rename_map:
            del _rename_map[nid]
            _save_rename_map(_rename_map)
        # 反向清理：如果某个旧ID映射到此节点
        for k, v in list(_rename_map.items()):
            if v == nid:
                del _rename_map[k]
                _save_rename_map(_rename_map)

        _save(force=True)
        return jsonify({"success": True})
    except Exception as e:
        logger.exception(f"[API异常] DELETE /api/nodes/{node_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/edges", methods=["POST"])
def create_edge():
    try:
        data = request.json or {}
        src = str(data.get("src", "")).strip()
        dst = str(data.get("dst", "")).strip()
        rel = str(data.get("relation", data.get("type", ""))).strip()
        if not src or not dst or not rel:
            return jsonify({"error": "src/dst/relation 不能为空"}), 400

        weight = float(data.get("weight", 0.5))

        with kg._lock:
            if src not in kg.nodes or dst not in kg.nodes:
                return jsonify({"error": "src 或 dst 节点不存在"}), 400

            existing = kg.get_edge(src, dst, rel)
            if existing:
                existing.weight = weight
                existing.touch()
            else:
                kg.add_edge(Edge(src=src, dst=dst, relation=rel, weight=weight))

        _save()
        return jsonify({"success": True})
    except Exception as e:
        logger.exception(f"[API异常] POST /api/edges: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/edges", methods=["PUT"])
def update_edge():
    try:
        data = request.json or {}
        src = str(data.get("src", "")).strip()
        dst = str(data.get("dst", "")).strip()
        rel = str(data.get("relation", data.get("type", ""))).strip()
        if not src or not dst or not rel:
            return jsonify({"error": "src/dst/relation 不能为空"}), 400

        weight = float(data.get("weight", 0.5))
        ok = kg.update_edge(src, dst, rel, weight=weight)
        if not ok:
            return jsonify({"error": "边不存在"}), 404

        _save()
        return jsonify({"success": True})
    except Exception as e:
        logger.exception(f"[API异常] PUT /api/edges: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/edges", methods=["DELETE"])
def delete_edge():
    try:
        data = request.json or {}
        src = str(data.get("src", "")).strip()
        dst = str(data.get("dst", "")).strip()
        rel = str(data.get("relation", data.get("type", ""))).strip()
        if not src or not dst or not rel:
            return jsonify({"error": "src/dst/relation 不能为空"}), 400

        ok = kg.remove_edge(src, dst, rel)
        if not ok:
            return jsonify({"error": "边不存在"}), 404

        _save()
        return jsonify({"success": True})
    except Exception as e:
        logger.exception(f"[API异常] DELETE /api/edges: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/activate/reset", methods=["POST"])
def activate_reset():
    """专用激活重置端点 — 强制清空所有节点/边激活值并停止自动扩散。"""
    try:
        # 停止自动扩散（防止重置后立即重新激活）
        if engine._running:
            engine.stop_auto()
            logger.info("[Reset] 已停止自动扩散")

        with kg._lock:
            reset_count_n = 0
            reset_count_e = 0
            for n in kg.nodes.values():
                if n.activation > 0:
                    n.activation = 0.0
                    reset_count_n += 1
            for e in kg.edges:
                if e.activation > 0:
                    e.activation = 0.0
                    reset_count_e += 1

        # 全图清零后活跃前沿必须同步清空（否则残留零激活项参与 top-k/衰减）
        with engine._lock:
            engine._active_nodes.clear()
            engine._active_edges.clear()
        engine._refresh_action_queue()
        _save()

        logger.info(f"[Reset] 重置了 {reset_count_n} 个节点 + {reset_count_e} 条边")
        return jsonify({
            "success": True,
            "message": f"重置了 {reset_count_n} 个节点 + {reset_count_e} 条边",
            "reset_nodes": reset_count_n,
            "reset_edges": reset_count_e,
            "graph": _active_graph_dict(0.0)
        })
    except Exception as e:
        logger.exception(f"[Reset异常] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/activate", methods=["POST"])
def activate():
    try:
        data = request.json or {}
        node_ids = data.get("nodes", []) or []
        edges = data.get("edges", []) or []

        node_ids = [str(x).strip() for x in node_ids if str(x).strip()]

        rel_only = []
        edge_specs = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            src = str(e.get("src", "")).strip()
            dst = str(e.get("dst", "")).strip()
            rel = str(e.get("relation", e.get("type", e.get("rel", "")))).strip()
            if src and dst and rel:
                edge_specs.append({"src": src, "dst": dst, "type": rel})
            elif rel:
                rel_only.append(rel)

        if not node_ids and not edge_specs and not rel_only:
            with kg._lock:
                for n in kg.nodes.values():
                    n.activation = 0.0
                for e in kg.edges:
                    e.activation = 0.0
            # 全图清零后活跃前沿必须同步清空（否则残留零激活项参与 top-k/衰减）
            with engine._lock:
                engine._active_nodes.clear()
                engine._active_edges.clear()
            engine._refresh_action_queue()
            _save()
            return jsonify({
                "success": True,
                "graph": _active_graph_dict(0.0)
            })

        if node_ids or edge_specs:
            engine.activate_from_inputs(node_ids, edge_specs)

        if rel_only:
            factor = float(config.get("edge_relation_factor", 0.75))
            _bumped_edges = []
            _bumped_nodes = []
            with kg._lock:
                for e in kg.edges:
                    if e.relation in rel_only:
                        e.activation = max(0.0, e.activation + 1.0 * factor)
                        _bumped_edges.append(e)
                        if e.src in kg.nodes:
                            kg.nodes[e.src].activation = max(0.0, kg.nodes[e.src].activation + 0.2 * factor)
                            _bumped_nodes.append(e.src)
                        if e.dst in kg.nodes:
                            kg.nodes[e.dst].activation = max(0.0, kg.nodes[e.dst].activation + 0.6 * factor)
                            _bumped_nodes.append(e.dst)
            # 直写激活后同步活跃前沿（绕过 activate_from_inputs 的路径）
            engine.mark_edges_active(_bumped_edges)
            engine.mark_active(_bumped_nodes)
            engine._refresh_action_queue()

        _save()
        return jsonify({
            "success": True,
            "graph": _active_graph_dict(0.0)
        })
    except Exception as e:
        logger.exception(f"[API异常] /api/activate: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# NLP
# =========================================================

@app.route("/api/nlp", methods=["POST"])
def process_nlp():

    logger.info("=" * 60)
    logger.info("[API] /api/nlp")
    logger.info("=" * 60)

    try:
        # 持续认知循环在请求处理期间暂停 tick 扩散，避免双重扩散
        cc.set_busy(True)
        # 用户活动 → 自主行动让位（并打断在飞的自主动作）：用户永远优先
        try:
            autonomy.notify_user_activity()
        except Exception as _nae:
            logger.debug(f"[Autonomy] 用户活动通知失败: {_nae}")
        # ── MODE 决策：认知需求评分 → 预算检查 → 选择 LLM 参与模式 ──
        # 回合开始：预算回合计数重置（matters for max_tokens_per_turn）
        llm_budget.turn_reset()
        # Minecraft 桥：世界状态 → 图。**统一走具身的 perceive 门控路径**
        # （签名门 + hostile_names 完整实参 + 背包低频刷拍）。旧版在这里
        # 裸调 update_perception 是第二写者：缺 hostile_names 实参 → 每轮
        # 对话都把"附近的生物"的敌对标注抹平（2026-09-22 收尾审计 B2）。
        try:
            _mc_p = mc_embodiment.perceive() or {}
            _uns = ([e.get("name") for e in _mc_p.get("unknown_entities") or []]
                    + [b.get("name") for b in _mc_p.get("unknown_blocks") or []])
            if _uns:
                fas_log.get_logger(fas_log.PERCEPTION).info(
                    "mc_unknown_objects", "MC 感知到图外新对象",
                    unknowns=_uns[:8])
        except Exception as _me:
            logger.debug(f"[Minecraft] 状态轮询跳过: {_me}")

        data = request.json or {}

        text = str(
            data.get("text", "")
        ).strip()

        logger.info(
            f"[NLP输入] {text}"
        )

        # 认知周期开始：本轮所有内部状态写入都带这个 cycle_id（可解释性）。
        # 必须放在 text 解析之后；这里的异常不该被静默——用 warning 暴露真实故障。
        _channel = str((data.get("channel") or "")).strip() or "web"
        # 通道层随消息带来的**身份事实**（web 直连时缺省为 user/无点名）。
        # 这里只接收为事件/周期记录的一部分——不新增任何行为分支，
        # "是否回应"仍然且只由 dialogue_decide 决定。
        _sender = str((data.get("sender") or "")).strip()
        _sender_kind = str((data.get("sender_kind") or "").strip() or "user")
        _mentioned = bool(data.get("mentioned"))
        # ── 观测：trace 在输入入口开启，贯穿认知→决策→LLM→动作→回复 ──
        fas_log.new_trace("nlp")
        _turn_t0 = time.time()   # 回合端到端延迟锚点（收尾 §性能）
        fas_log.get_logger(fas_log.INPUT).info(
            "input_received", "输入到达认知管线入口",
            source=("channel" if _channel != "web" else "web"),
            channel=_channel, sender=_sender or "user",
            sender_kind=_sender_kind, mentioned=_mentioned,
            text=fas_log.text(text), text_len=len(text))
        _cycle_id = None
        try:
            _cycle_id = internal_state.begin_cycle(
                "turn", {"text_len": len(text), "channel": _channel,
                         "sender": _sender or None, "sender_kind": _sender_kind,
                         "mentioned": _mentioned})
        except Exception as _ise:
            logger.warning(f"[InternalState] 周期开启失败: {_ise}")
        fas_log.get_logger(fas_log.COGNITION).info(
            "cycle_start", f"回合周期开始（channel={_channel}）",
            trigger="user_message", kind="turn")

        if not text:

            logger.warning(
                "[NLP] 输入为空"
            )
            fas_log.get_logger(fas_log.INPUT).warning(
                "input_filtered", "空输入被拒绝（未进入认知）",
                channel=_channel, reason="empty_text")

            # L1-DGR-01:早退路径同样收周期/解 busy/清锚点,防 CC 永久停摆
            try:
                if _cycle_id:
                    fas_log.get_logger(fas_log.COGNITION).info(
                        "cycle_end", "回合周期结束（空输入早退）",
                        outcome="early_exit", reason="empty_text")
                    internal_state.end_cycle(_cycle_id, outcome={"early_exit": True})
            except Exception as _ce:
                # L1-CON-6 留痕:周期结算失败会悬挂 current_cycle,
                # 不能只靠早退逻辑自身兜底——失败要可见。
                logger.warning(f"[NLP] 空输入早退周期结算失败(cycle 可能悬挂): {_ce}")
            try:
                engine.clear_anchors()
            except Exception as _ae:
                logger.warning(f"[NLP] 空输入早退清锚失败(激活来源可能残留): {_ae}")
            cc.set_busy(False)

            return jsonify({
                "error": "输入文本不能为空"
            }), 400

        # 经验时间轴：一句话是一条 OBSERVATION（§四：所有来源同一条轴）。
        # actor 写**实际说话者**（通道带来 sender 时）——"别人的话记成
        # user 说的"曾是通道把 other 当好消息吞掉才没暴露的失真。
        try:
            timeline.append(make_event(
                EVENT_OBSERVATION, actor=_sender or "user", subject="utterance",
                content={"text": text[:200], "channel": _channel,
                         "sender_kind": _sender_kind, "mentioned": _mentioned},
                source="nlp", meta={"cycle_id": _cycle_id}))
        except Exception as _tle:
            logger.debug(f"[Experience] 用户话语入轴跳过: {_tle}")

        # ── 反射快路径（Bug 修复 2026-09-19；具身改造 2026-09 接入 Action 系统）──
        # 游戏在线 + 明确动作指令（"过来/跟着我/停"）→ 零 LLM 立即决策。
        # 注意：这不是绕过认知——解析结果先映射成 ActionNode 规格，经
        # ActionManager（优先级/中断/承诺）执行，并注入图谱激活；认知照常
        # 继续（记忆/奖赏/经验），只是 LLM 答案生成被"行动+游戏内短回执"替代。
        _reflex_fast = False
        _mc_action_result = None
        _mc_intent = None
        _mc_info = {}
        _fb_pre = None   # ExprFB 激活基线（在 activate_from_inputs 前采样）
        # B5 收尾修复（冒烟 500 实锤）：旧版 `_mc_state` 靠已删除的感知双写
        # 块顺带赋值（B2），函数后段仍有一处赋值 → 整函数视为局部变量，
        # 前面所有读点 UnboundLocalError。此处每轮**一次**早期读取。
        try:
            _mc_state = mc["get_state"]()
        except Exception as _msg:
            logger.warning(f"[NLP] MC 状态读取失败(按未连接处理): {_msg}")
            _mc_state = None
        if _mc_state and _mc_state.get("connected"):
            # Action Concept 快路径（2026-09-19 架构升级）：
            #   fast pattern（现有正则原文，否定优先在 minecraft.reflex 内）
            #   → miss → 语义层（seed 表面形式 → 图谱/embedding 召回）
            # 全程零 LLM（低延迟不变）；命令性门拦截叙事句——
            # "我刚才看到有人跟着我"是讲述，不是命令（§十七 误触发修复）。
            try:
                _mc_intent, _mc_info = resolve_minecraft_intent(
                    text, kg=kg,
                    embedder=emb_mgr if emb_mgr.ready() else None,
                    action_space=action_space)
            except Exception as _mie:
                logger.debug(f"[MC Intent] 解析失败（回退旧反射路径）: {_mie}")
                _mc_intent, _mc_info = None, {}
            if _mc_intent is None and _mc_info.get("refused"):
                # 识别到指令但拒绝执行（否定句/未点名目标）：不执行，但把
                # "拒绝了"作为证据带进认知上下文（语言层要能如实解释没做）
                _mc_action_result = {"success": False, "action": "refused",
                                     "reason": str(_mc_info["refused"].get("reason")
                                                   or "refused")}
                logger.info(f"[MC Reflex] 否定优先拒绝: "
                            f"{_mc_info['refused'].get('reason', '')}")
            elif _mc_intent is None and _mc_info.get("rejected"):
                logger.info(f"[MC Intent] 命令性门拦截: "
                            f"{_mc_info['rejected'].get('concept')} "
                            f"evidence={_mc_info['rejected'].get('evidence')}")
            if _mc_intent is not None:
                try:
                    _game_user = config.get("minecraft_user_name") or "Hellucigen"
                    _early = _mc_intent.get("reflex") or {
                        "action": "none",
                        "params": _mc_intent.get("parameters") or {}}
                    _spec, _flow = intent_to_minecraft_flow(
                        _mc_intent, user_name=_game_user,
                        current_action_type=(action_manager.current or {}).get(
                            "action_type") if action_manager.current else None)
                    if (_spec is not None and _flow.get("flow")
                            in ("execute", "negation_stop",
                                "negation_stop_current")):
                        # 万物皆图：用户动作请求是认知事件（激活 + 留痕）
                        _record_user_action_intent(_spec, text, _cycle_id)
                        _prop = action_manager.propose(_spec, source="user")
                        if _prop.get("started"):
                            _mc_action_result = {
                                "success": True, "action": _spec["action_type"],
                                "describe": _prop.get("describe") or _spec["action_type"],
                                "target": _spec.get("target"),
                                "pending": bool(_prop.get("pending"))}
                            if str(_flow.get("flow")).startswith("negation_stop"):
                                # B4 撤账（真机实锤 23:03:28："别跟了"走概念
                                # 快路径，旧版只在 _dispatch_mc_reflex 里撤，
                                # 承诺永不被撤=失约幽灵目标）。
                                try:
                                    autonomy.cancel_invitation()
                                except Exception as _cie:
                                    logger.debug(f"[Reflex] 邀请撤账失败: {_cie}")
                        else:
                            _mc_action_result = {
                                "success": False, "action": _spec["action_type"],
                                "reason": _prop.get("reason") or "action_rejected"}
                            # B4 落诺（真机实锤 23:02:21："跟着我"撞上连接承诺
                            # 在忙被拒后邀请蒸发）：follow/approach 没即刻启动
                            # （被拒或在队）都落持久目标——queued 者兑现时由
                            # settle 销账；rejected 者经透传→评分参与后续决策，
                            # 说到必须做到（或如实失败）。
                            if (_flow.get("flow") == "execute"
                                    and _early.get("action") in ("follow",
                                                                 "approach")):
                                try:
                                    autonomy.register_invitation(
                                        str(_early.get("action")),
                                        config.get("minecraft_user_name")
                                        or "Hellucigen")
                                    _mc_action_result["committed"] = True
                                except Exception as _rie:
                                    logger.debug(f"[Reflex] 邀请落诺失败: {_rie}")
                    elif _flow.get("flow") == "unmapped":
                        # 旧分发兜底（未映射的反射类型）
                        _mc_action_result = _dispatch_mc_reflex(_early)
                    else:
                        # refused（否定挖/攻击且不在执行中）：不执行、不承诺
                        _mc_action_result = None
                    if _mc_action_result is not None:
                        _ok = _mc_action_result.get("success", True)
                        _act = _early.get("action")
                        if _ok:
                            _ack = _MC_REFLEX_ACK.get(_act)
                            if _ack:
                                mc_say(_ack)
                        else:
                            # 失败必须如实说（最坏的是"说好然后没做"）：
                            # 最常见原因是目标不在 bot 视野里（no_target）
                            _tgt = str(_early.get("params", {}).get("target") or "")
                            mc_say(f"我没看到{_tgt}，走近一点再试试？"
                                   if _mc_action_result.get("reason") == "no_target"
                                   else f"做不到：{_mc_action_result.get('reason') or '失败了'}")
                        _reflex_fast = True
                    logger.info(
                        f"[MC Reflex] 快路径: {describe_intent(_mc_intent)} "
                        f"flow={_flow.get('flow')} fast={_reflex_fast}（零 LLM 等待）")
                except Exception as _rfe:
                    logger.warning(f"[MC Reflex] 快路径失败（转正常管线）: {_rfe}")
                    _mc_action_result = None
                    _mc_intent = None   # 解析失败 → MODE0 旧路径兜底

        # ── Phase C：Minecraft 世界接入生命周期（2026-09-21 概念化改造）──
        # 与反射快路径同层（先于 NLP/LLM，零 LLM 延迟保证不变），但裁决
        # 已从"app 持正则→直接行为"改为"输入→动作概念层→意图/槽位事实→
        # 会话状态机"：进游戏/缺端口索取/数字填槽/否定取消/"刚才那个"回指
        # 全部住在 action_resolver + action_concepts（CONNECT 概念），
        # 这里只分发 resolver 的结构化产物。旧实现长在 LLM 回答分支里
        # 一次要跑 1~2 分钟，"来玩Minecraft吧，端口号是XXXXX"经常死在半路，
        # 这就是"刚刚进不去世界"的直接原因——快，仍然是这条路径的第一要求。
        _mc_session_action = None
        try:
            _mc_conn_state = mc_session.get_state()
            # 会话状态过期检测：图内说 connected 但桥已断 → 状态是
            # 上一个世界的残留，按 disconnected 处理（桥是真相）
            if (_mc_conn_state == "connected"):
                _br = mc["get_state"]()
                if not (_br and _br.get("connected")):
                    _mc_conn_state = "disconnected"
                    logger.info("[MCSession] 图内 connected 但桥已断"
                                " → 视为 disconnected（世界已关闭/更换）")
            if _mc_conn_state not in ("connected", "connecting"):
                # 世界未在线：这一轮可能是在"请求进入"，也可能是在"回答端口"
                # （槽位事实 state=awaiting_port 在图谱节点上，resolver 读它）
                _si, _sinfo = resolve_minecraft_intent(
                    text, kg=kg,
                    embedder=emb_mgr if emb_mgr.ready() else None,
                    action_space=action_space)
                if not _mc_info:
                    _mc_info = _sinfo   # 继承否定/缺参裁决（_mc_neg_refused 门）
                if _si is not None and _si.get("action") == "CONNECT":
                    # 端口已绑定（这句就给了 / 槽位被数字填上 / 图谱回指复用）
                    _mc_session_action = mc_session.connect_async(
                        int(_si["parameters"]["port"]))
                elif _sinfo.get("need_param"):
                    # 槽位缺失且可问：向用户索取（request_port 把"等待"事实
                    # 写进节点）；绝不猜端口——猜是上一版正则分支的罪状
                    _mc_session_action = mc_session.request_port()
                elif _sinfo.get("refused"):
                    # 取消只有一个来源：显式否定（NEGATION_RE 单一真源）。
                    # "等一下/还没开世界"不再触发取消——它们就是继续等待。
                    _mc_session_action = mc_session.cancel_awaiting(
                        str(_sinfo["refused"].get("reason") or ""))
                elif _sinfo.get("invalid_port"):
                    # 格式裁决是工程约束：如实反馈，槽位保持等待可重试
                    _mc_session_action = {
                        "success": False, "action": "invalid_port",
                        "port": _sinfo["invalid_port"],
                        "reason": "port_out_of_range",
                        "describe": "用户给的端口数字超出有效范围（1024~65535），"
                                    "已如实请对方重新看局域网提示里的端口",
                        "message": "这个数字不像端口号（有效范围 1024~65535），"
                                   "麻烦再看一下局域网提示里的端口"}
                # awaiting_port / 其它概念命中 / 无命中：不在这里行动，
                # "继续等待"这一事实本来就记在图谱节点上
            # 连接成功后同步会话状态（轮询块也会更新）
            if _mc_state and _mc_state.get("connected"):
                if mc_session.get_state() != "connected":
                    mc_session._set("connected", port=_mc_state.get("port"))
            # 进入世界 ≠ 完事："来玩我的世界"的自然收束是走到用户身边。
            # 异步连接（pending）的收束在 _mc_on_connected 回调里；这里只
            # 兜住同步即成的 rare path。动作门 = 只对"连接成功"收束，
            # 不再让 request_port/cancel 的 success 字段误触发（旧 bug）。
            # 这是用户命令的后续步骤（SRC_USER），经 ActionManager 承诺/
            # 中断规则执行；失败如实（bot 会在回执里说没找到人），不编造。
            if (_mc_session_action and _mc_session_action.get("success")
                    and not _mc_session_action.get("pending")
                    and _mc_session_action.get("action") == "connect_minecraft"):
                try:
                    _guser = config.get("minecraft_user_name") or "Hellucigen"
                    _prop_ap = action_manager.propose({
                        "action_type": "navigate_to_entity",
                        "target": _guser,
                        "params": {"entity": _guser, "player": _guser,
                                   "keep_distance": 2.0},
                        "urgency": 0.7, "priority": 0.65,
                        "motivation": "user_commitment",
                        "expected_effect": "走到用户身边，开始一起玩",
                        "reason": ["connect_minecraft_success"]},
                        source="user")
                    try:
                        autonomy.register_invitation("approach", _guser)
                    except Exception as _rie:
                        logger.warning(f"[MCSession] 走近承诺注册失败(承诺丢失): {_rie}")
                    logger.info(
                        f"[MCSession] 进入世界→走向用户: "
                        f"started={_prop_ap.get('started')} "
                        f"queued={_prop_ap.get('queued')} "
                        f"{_prop_ap.get('reason') or ''}")
                except Exception as _ape:
                    logger.debug(f"[MCSession] 走向用户跳过: {_ape}")
            if _mc_session_action is not None:
                logger.info(f"[MCSession] 接入处理: {_mc_session_action}")
        except Exception as _mse:
            logger.warning(f"[MCSession] 异常: {_mse}")

        # ── 输入复杂度自适应（2026-09）：选轻认知还是全认知 ──
        # 不是跳过认知，是让认知本身变轻：Level1 短 prompt + 快速模型 +
        # 一次调用；输出结构化结果后照常进图激活/行为竞争/行动。
        _complexity = {"level": 2, "components": {}, "reasons": []}
        _l1_fast = False
        try:
            from cognitive_complexity import estimate_complexity as _est_cx
            # 只有真正需要重计算的路径才强制全管线：网络搜索/文件操作
            # （参数抽取要 LLM）、坐标点击。Minecraft 会话状态机是纯代码
            # （在复杂度估计**之前**就已运行，连接/走向用户的动作已经发出），
            # 不再把含"minecraft/端口"的短输入拖进全管线——那曾让
            # "来玩minecraft吧，端口号是51027" 走 3~4 次串行大模型调用。
            _force_full = bool(re.search(
                r"搜一?下|搜索|上网|网上查|帮我查|查一下|"
                r"点击\s*\(?\s*\d|创建.{0,6}文件|删除文件|"
                r"看看.{0,4}屏幕|屏幕|截屏", text, re.I))
            _complexity = _est_cx(text)
            if _force_full:
                _complexity["level"] = 2
                _complexity["reasons"].append("含搜索/文件/坐标/屏幕类语义，走全管线")
            _l1_fast = (_complexity["level"] == 1
                        and bool(config.get("cognitive_complexity", {}).get("enable", True)))
        except Exception as _cxe:
            logger.warning(f"[Complexity] 估计失败（按 Level2 处理）: {_cxe}")

        # -------------------------------------------------
        # NLP（复杂度自适应：L1 轻认知 / L2 全认知）
        # -------------------------------------------------

        logger.info(f"[NLP] 开始解析（level={_complexity.get('level')}）"
                    f"{'' if not _complexity.get('reasons') else ' — ' + '; '.join(_complexity['reasons'][:3])}")

        if _l1_fast:
            try:
                parsed = nlp.process_fast(text)
            except Exception as _pfe:
                logger.warning(f"[NLP] L1 快速解析失败（降级全解析）: {_pfe}")
                parsed = nlp.process(text)
                _l1_fast = False
        else:
            parsed = nlp.process(text)

        logger.info(
            f"[NLP] 解析完成: "
            f"{parsed}"
        )

        # 对 NLP 解析出的节点名做模糊匹配，映射到图谱中实际存在的节点ID
        raw_nodes = parsed.get("nodes", [])
        parsed_nodes = [_fuzzy_match_node(n) for n in raw_nodes]
        parsed_nodes = list(dict.fromkeys(parsed_nodes))  # 去重保序
        parsed_edges = list(parsed.get("edges", []))

        # -------------------------------------------------
        # 0. 检测坐标点击指令
        # -------------------------------------------------

        coords = nlp.extract_coordinates(text)
        click_data = None
        if coords:
            coord_node_id = f"坐标(X={coords['x']},Y={coords['y']})"
            if "点击" not in parsed_nodes:
                parsed_nodes.append("点击")
            if coord_node_id not in parsed_nodes:
                parsed_nodes.append(coord_node_id)
            if not any(
                e.get("src") == "点击" and e.get("dst") == coord_node_id
                for e in parsed_edges
            ):
                parsed_edges.append({
                    "src": "点击", "dst": coord_node_id,
                    "type": "目标", "relation": "目标", "weight": 1.0
                })
            click_data = coords
            logger.info(f"[Click] 注入点击坐标: {coord_node_id}")

        # -------------------------------------------------
        # 0a. 社交互动检测 → 认知语境信号（检测 ≠ 行为决定）
        # -------------------------------------------------
        # 社交互动节点语义 = "当前事件涉及人与人之间的互动关系"。
        # 它只作为普通种子改变认知场：能被想起什么由既有图谱路径决定，
        # 是否回应由 dialogue_decision 行为竞争决定——问候/感谢/道歉
        # 都不天然保证回应，允许"检测到社交 → 无候选过阈值 → 沉默"。
        # 判据来自 NLP 语义解析（dialogue_signals.is_social_input）。
        _ctrl_directive = _dsig.is_topic_control(text)
        _is_social = _dsig.is_social_input(parsed, text)
        _just_terminated = False
        if _ctrl_directive:
            # 抑制时长来自运行时 config（可热更新）；dialogue_decision 不硬编码
            mark_termination(config.get("topic_termination_inhibit_sec", 600))
            _just_terminated = True
            logger.info("[DialogueDecision] 话题终止标记")

        if _is_social:
            if "社交互动" not in parsed_nodes:
                parsed_nodes.append("社交互动")
            logger.info(f"[Social] 注入 社交互动 节点 (illoc={parsed.get('illocutionary_act')}, da={parsed.get('dialogue_act')})")

        # -------------------------------------------------
        # 0a1. 言语行为图谱化（speech_act_graph 2026-09-20）
        # -------------------------------------------------
        # 本轮话语 → episodic 事件节点（短期）；言外行为 → 概念节点
        # （长期，semantic）。概念与话语节点作为**种子**并入 parsed_nodes，
        # 与 社交互动/情境 走同一扇激活门：指令 → 行动回应 → 行为:回应 →
        # Minecraft 能力 全部由既有 diffusion 传播——这里没有也不允许有
        # "directive→行动" 的 if/else。单次激活不会沉淀为长期倾向
        # （episodic 快衰减 + 概念激活随回合间衰减回落；长期学习仍只走
        # disposition/outcome/因果 既有通道）。
        _sa_info = None
        try:
            _sa_info = inject_utterance(kg, engine, text, parsed,
                                        cycle_id=_cycle_id)
            for _sid in (_sa_info or {}).get("seeds", []):
                if _sid and _sid not in parsed_nodes:
                    parsed_nodes.append(_sid)
        except Exception as _sae:
            logger.debug(f"[SpeechAct] 话语入图跳过: {_sae}")

        # -------------------------------------------------
        # 0a2. 情绪共振 → 认知状态变化（检测 ≠ 共情行为）
        # -------------------------------------------------
        # 分层（dialogue_signals.emotion_resonance）：
        #   关键词命中  → 只建长期结构 事件-[引发]->情绪-[感受]->Self
        #                （"角色叫焦虑"这类误报不会改变当前认知场）
        #   语义佐证命中 → 额外注入有限的当前激活（emotion_resonance_boost，
        #                source=emotion，走正常扩散/衰减/cap 体系）
        # 情绪被激活只说明内部状态变了；是否共情/回应由行为竞争决定。
        _resonance = _dsig.emotion_resonance(kg, engine, text,
                                             parsed_nodes, parsed_edges)
        # 调制事件桥（R2 P7）：情绪 → 结构化事件 → 图上的 `影响` 边决定扇出。
        # 幅度与靶点不再是这里读的表（config.emotion_hormone_modulation 现在只是
        # 事件表的**出厂种子**，由 modulation_events.ensure_event_types 种进图）；
        # 数值写入仍只经 internal_state 唯一写入口（I1），调制绝不指定行为。
        if _resonance.get("activated") and internal_state is not None:
            for _emo in _resonance["activated"]:
                try:
                    reward_system.mod_engine().emit(
                        f"emotion:{_emo}", source="emotion_resonance",
                        valence=0.0, intensity=1.0, cycle_id=_cycle_id, ref=_emo)
                except Exception as _ehe:
                    logger.debug(f"[EmotionResonance] 调制事件跳过: {_ehe}")

        # -------------------------------------------------
        # 0b. FAISS 语义召回 → 注入命中节点
        # -------------------------------------------------

        faiss_hits = []
        try:
            if emb_mgr.ready():
                # 召回广度：每个输入词取多少语义近邻。5 太窄——"minecraft"
                # 的语义邻域里，Minecraft(0.80) / Minecraft会话(0.79) /
                # 游玩—对象—Minecraft(0.76) 之后才是 进入Minecraft世界(0.72)
                # 这类能力节点，取 5 会把"我能玩Minecraft"这件事直接截掉，
                # 于是输入点亮了对象却点亮不了能力。广度可配。
                _recall_k = int(config.get("faiss_recall_topk", 8))
                for raw_node in raw_nodes:
                    hits = emb_mgr.search(str(raw_node), top_k=_recall_k, min_similarity=0.3)
                    for h in hits:
                        nid = h["node_id"]
                        # 只注入图谱中实际存在的节点，避免 FAISS 召回不存在的节点
                        # 引发虚假的好奇心触发（如 "点击" "回答" 等语义相近但不在图谱中的词）。
                        # 同时排除基础设施/程序节点：语义召回的是知识，不是引擎内部零件。
                        _fnode = engine.name_to_node.get(nid)
                        if (
                            _fnode is not None
                            and is_cognitive_visible(_fnode)
                            and nid not in parsed_nodes
                            and nid not in raw_nodes
                        ):
                            parsed_nodes.append(nid)
                            faiss_hits.append({"query": str(raw_node), "hit": nid, "sim": h["similarity"]})
                            logger.info(f"[FAISS] {raw_node} → {nid} ({h['similarity']:.3f})")
                if faiss_hits:
                    logger.info(f"[FAISS] 语义召回: {len(faiss_hits)} 命中")
                    # ── A4 晋升 dry-run 观测（2026-09-30）───────────────
                    # FAISS 语义命中 = 情景经历的真实访问路径。只对引用命中
                    # 节点的经历做访问计数（mark_accessed），去重后每轮至多
                    # +1；**不触发任何自动晋升**——晋升仍仅由手动
                    # /api/buffer/promote 与 _auto_consolidate_curiosity_knowledge
                    # （importance 驱动）消费。观测目的 = 采集高频访问经历的
                    # 激活轨迹，与手动审批分布对照，确认候选质量后再接驱动。
                    _a4_seen = set()
                    _a4_cnt = 0
                    try:
                        for _hit in faiss_hits:
                            for _exp in buffer.find_by_node(_hit["hit"]):
                                if id(_exp) in _a4_seen:
                                    continue
                                _a4_seen.add(id(_exp))
                                buffer.mark_accessed(_exp)
                                _a4_cnt += 1
                    except Exception as _a4e:
                        logger.warning(f"[A4-Obs] 访问计数观测跳过: {_a4e}")
                    if _a4_cnt:
                        logger.info(f"[A4-Obs] 经历访问计数 +{_a4_cnt}（FAISS 命中 → 经历）")
        except Exception as e:
            logger.warning(f"[FAISS] 搜索跳过: {e}")

        # -------------------------------------------------
        # 0c. 探索提问结算（可回答、可忽略）
        # -------------------------------------------------
        # 若上一轮她提出了探索性提问（pending_inquiry），本轮用户输入要经过
        # 相关性判定才被视为回答：答非所问 → 提问过期、兴趣保留、用户输入
        # 照常走自己的处理流程（好奇心不是强制打断，P8：允许不行动）。
        curiosity_resolved = None
        if curiosity_engine.is_curiosity_active(kg):
            curiosity_resolved = curiosity_engine.settle_inquiry(
                kg, text, parsed_nodes=parsed_nodes, engine=engine, config=config)
            if curiosity_resolved.get("settled"):
                logger.info(f"[Curiosity] 探索提问被回答: {curiosity_resolved.get('question')}")
                # ── 自我奖赏：提问得到回答 = 她自己挣到知识（§十）──
                # 这是 self_outcome，不是"用户夸了她"——即便用户语气平淡，
                # 学到东西本身就是正向奖赏，强化 ask@主动发起 倾向。
                try:
                    _ev = reward_system.evaluate(
                        behavior="ask", context="情境:主动发起",
                        social_outcome=None, self_outcome="knowledge_gain",
                        ref=curiosity_resolved.get("question") or "",
                        cycle_id=_cycle_id)
                    _rel = reward_system.release(_ev)
                    _mod = reward_system.modulation(
                        key="ask@情境:主动发起",
                        surprise=_rel.get("surprises", {}).get("ask@情境:主动发起"))
                    for _rev in _ev:
                        try:
                            timeline.append(make_event(
                                EVENT_SELF_STATE, actor="self", subject="reward",
                                content={"change": "changed",
                                         "source": _rev["source"],
                                         "outcome": _rev["outcome"],
                                         "valence": _rev["valence"]},
                                source="reward", meta={"cycle_id": _cycle_id}))
                        except Exception as _rrc:
                            logger.warning(f"[Experience] 奖赏入轴失败(curiosity 分支): {_rrc}")
                    reward_trace = disposition_store.apply_experience(
                        "ask", "情境:主动发起", social_outcome=None,
                        self_outcome="knowledge_gain", hormone=_mod,
                        evidence_ref=curiosity_resolved.get("target") or "",
                        allow_create=True)
                    logger.info(
                        f"[Reward] self=knowledge_gain → ask@主动发起 "
                        f"sig={reward_trace.get('learning_signal')} "
                        f"{reward_trace.get('old')}→{reward_trace.get('new')} "
                        f"[{reward_trace.get('state')}] mod={_mod.get('factor')}")
                except Exception as _re:
                    logger.debug(f"[Reward] 知识奖赏回写跳过: {_re}")
            elif curiosity_resolved.get("dismissed"):
                logger.info(
                    f"[Curiosity] 用户没有回答探索提问（{curiosity_resolved.get('reason')}），"
                    f"过期并保留兴趣")

        # -------------------------------------------------
        # 0d. 复合短语判定 — 关系型 vs 词汇化 vs 绝对名词
        # -------------------------------------------------
        # 在好奇心检测之前，识别并过滤关系型短语（如"代表产品"、
        # "创始人"等），避免其被误判为未知概念触发打断性反问。
        compound_result = compound_handler.process(
            text, parsed_nodes, parsed_edges,
            cognitive_context={
                "illocutionary_act": parsed.get("illocutionary_act", ""),
                "dialogue_act": parsed.get("dialogue_act", ""),
            }
        )
        curiosity_nodes = compound_result.get("filtered_nodes", parsed_nodes)
        frame_queries = compound_result.get("frame_queries", [])
        if frame_queries:
            logger.info(
                f"[CompoundPhrase] Frame 查询生成: "
                f"{[(fq['frame'], fq['attribute_type'], fq.get('possessor')) for fq in frame_queries]}"
            )
        if compound_result.get("relational_nodes"):
            logger.info(
                f"[CompoundPhrase] 过滤关系型节点: {compound_result['relational_nodes']} "
                f"(剩余 curiosity 候选: {curiosity_nodes})"
            )

        # ── Compound Phrase: Frame 查询 → 边激活 boost ──
        # 关系短语（如"代表产品"→FLAGSHIP）已被过滤不掉进 curiosity，
        # 但其 frame 查询结果需要反馈到边激活系统，使匹配的边
        # （如 麦当劳-[代表产品]->巨无霸）获得更强的运行时信号。
        if frame_queries:
            with kg._lock:
                for fq in frame_queries:
                    possessor = fq.get("possessor")
                    slot_id = fq.get("slot_id", "")
                    frame_rel = fq.get("frame", "")
                    attr_type = fq.get("attribute_type", "")

                    if not possessor:
                        continue

                    possessor_node = kg.get_node(possessor)
                    if not possessor_node:
                        continue

                    # 收集该槽位的所有 surface_forms 用于边匹配
                    surface_forms = compound_handler.phrase_table.get_surface_forms(slot_id)
                    # 同时加入 frame relation 和 attribute_type 作为匹配候选项
                    match_labels = set(surface_forms)
                    match_labels.add(frame_rel)
                    match_labels.add(attr_type)

                    boosted = 0
                    _boosted_edges = []
                    for edge in kg.get_out_edges(possessor_node.id):
                        rel = getattr(edge, 'relation', '')
                        if rel in match_labels:
                            boost_amount = 1.2  # 强 boost，使其在扩散中传递更强信号
                            edge.activation = min(3.0, edge.activation + boost_amount)
                            _boosted_edges.append(edge)
                            boosted += 1
                            logger.info(
                                f"[CompoundPhrase] 边激活 boost +{boost_amount}: "
                                f"{edge.src}-[{rel}]->{edge.dst} "
                                f"(slot={slot_id}, act={edge.activation:.3f})"
                            )
                    engine.mark_edges_active(_boosted_edges)

                    if boosted == 0:
                        logger.info(
                            f"[CompoundPhrase] 未找到可 boost 的边: "
                            f"possessor={possessor} slot={slot_id} match_labels={match_labels}"
                        )

        # -------------------------------------------------
        # 0e. 探索信号检测 — 认知 discrepancy（只报告，不决定行为）
        # -------------------------------------------------
        # 检测 NLP 解析结果中的未知概念/未知关系/情绪确认需求，经调制
        # （novelty/relevance/场合/satiation）后注入图信号。
        # 未知 ≠ 好奇：信号注入后能否形成探索行为，由扩散（CuriosityDrive）
        # 和后面的行为竞争决定——弱信号允许自然湮灭（什么都不做是合法的）。
        # 注意：curiosity_nodes 已经过滤了关系型短语。
        curiosity_info = {"triggered": False}
        if not curiosity_resolved or not curiosity_resolved.get("settled"):
            _cog_for_curiosity = {
                "illocutionary_act": parsed.get("illocutionary_act", ""),
                "dialogue_act": parsed.get("dialogue_act", ""),
                "response_expectation": parsed.get("response_expectation", ""),
            }
            curiosity_info = curiosity_engine.detect_cognitive_signals(
                kg, engine, curiosity_nodes, parsed_edges, text,
                cognitive_context=_cog_for_curiosity, config=config
            )
            if curiosity_info.get("triggered"):
                logger.info(
                    f"[Curiosity] discrepancy 信号: type={curiosity_info.get('trigger_type')} "
                    f"signals={[(s['type'], s['target'], s['injected']) for s in curiosity_info.get('signals', [])]}")
                # 经验时间轴：好奇触发是重要认知事件（§三 COGNITIVE_EVENT，
                # 只记触发的"事件级"事实，不记内部激活细节）
                try:
                    timeline.append(make_event(
                        EVENT_COGNITIVE, actor="self", subject="curiosity",
                        content={"trigger_type": curiosity_info.get("trigger_type"),
                                 "targets": [s.get("target") for s in curiosity_info.get("signals", [])[:3]]},
                        source="curiosity_engine", meta={"cycle_id": _cycle_id}))
                except Exception as _tle:
                    logger.debug(f"[Experience] 好奇事件入轴跳过: {_tle}")
        else:
            # 本轮已结算探索提问，不再注入新信号
            logger.info("[Curiosity] 本轮已结算探索提问，跳过新信号检测")

        # -------------------------------------------------
        # 1. 回合间衰减 → 清旧噪声 → 激活 + 扩散
        # -------------------------------------------------

        logger.info("[NLP] 开始激活扩散引擎")

        # 回合间衰减：新输入前清掉旧激活噪声，防止跨轮次累积
        # 比 lambda_decay 更强（默认 40%），但不影响扩散内部节奏
        # （引擎内只衰减活跃前沿——前沿外激活恒为 0，数学语义与全图循环一致）
        _inter_decay = float(engine._mod(
            "diffusion.inter_round_decay",
            config.get("inter_round_decay", 0.4)))
        # 激活地板：衰减后低于此值的节点/边直接归零。
        # 默认 0.5 — 一个节点从 cap(5.0) 开始，经过 ~5 轮未被激活后归零。
        # 这比原来的 0.001 大大截断了无关节点（如苹果、药）的噪声传播。
        _decay_floor = float(config.get("inter_decay_floor", 0.5))
        if _inter_decay > 0:
            engine.apply_inter_round_decay(_inter_decay, _decay_floor)
            logger.info(f"[Decay] 回合间衰减: {_inter_decay*100:.0f}% (地板={_decay_floor})")

        # 构建 semantic_similarity map (从 FAISS 命中)
        sim_map = {}
        for h in faiss_hits:
            existing = sim_map.get(h["hit"], 0)
            sim_map[h["hit"]] = max(existing, h["sim"])

        # ── disposition → attention 通路（人格学习架构 §十一）──
        # 点亮当前对话情境节点：`情境-[激活 weight=倾向强度]->行为` 的边把学到的
        # 倾向转成行为节点的实时激活，参与本轮扩散/Top-K/竞争——倾向不只是查表，
        # 而是走图激活。行为节点 activation 会被 tendencies_for 混入有效强度。
        # 在 activate_from_inputs 之前注入，使其随本轮一起扩散。
        try:
            _ctx_now = context_for_dialogue_act(parsed.get("dialogue_act", ""))
            _ctx_node = kg.get_node(_ctx_now)
            if _ctx_node is not None:
                _ctx_node.activation = min(5.0, _ctx_node.activation + 1.2)
                _ctx_node.touch()
                engine.mark_active([_ctx_node.id])
                engine.register_activation_source([_ctx_node.id], "internal_drive")
                logger.info(f"[Attention] 情境种子激活: {_ctx_now} → 行为节点经倾向边扩散")
        except Exception as _cse:
            logger.debug(f"[Attention] 情境种子激活跳过: {_cse}")

        # ExprFB：本轮激活前采样"被注视表达"的激活基线（observe 在扩散后对比）
        fas_log.get_logger(fas_log.COGNITION).info(
            "context_built", "解析/检索/种子构建完成",
            parsed_nodes=len(parsed_nodes or []),
            parsed_edges=len(parsed_edges or []),
            dialogue_act=str(parsed.get("dialogue_act") or ""),
            faiss_hits=len(faiss_hits or []))
        _fb_pre = expression_feedback.pre_baseline(kg, engine)
        # L1-CON-4:解析双失败时种子为空——零种子扩散后 TopK 取自
        # 上轮残留激活,回答会基于陈旧"依据"。留痕,回答区照此标注。
        _zero_seed = not (parsed_nodes or [])
        if _zero_seed:
            logger.warning(
                "[NLP] 零种子回合: 解析节点为空(LLM 解析失败?),本轮扩散 "
                "无图侧输入——回答将基于残留激活")
        engine.activate_from_inputs(parsed_nodes, parsed_edges, similarity_map=sim_map)

        # ── 探索信号整合相（显式 API，P0-4 遗产）──
        # 仅当本轮检测到 discrepancy 信号时：先跑少量扩散步让信号整合到
        # 探索行为节点（好奇/生成好奇问题）与 CuriosityDrive，随后衰减信号
        # 节点防过冲，再跑主扩散。
        # 关键语义变化（好奇心机制重构 2026-09）：信号到达探索行为节点
        # **不再直接生成问题**——那只赋予探索行为候选资格，是否探索由
        # dialogue_decide 的统一行为竞争裁决（P3：提问不是默认行为）。
        curiosity_question = None
        _exploration = {"eligible": False, "candidates": [], "reason": "no_signal",
                        "drive": 0.0, "target": None}
        _configured_max_depth = int(round(engine._mod(
            "diffusion.max_depth", int(config.get("max_depth", 6)))))

        if curiosity_info.get("triggered") and not curiosity_engine.is_curiosity_active(kg):
            logger.info("[Curiosity] 探索信号整合相（3步）")
            engine.diffuse_round(max_steps=3)

            # 图驱动门槛：信号是否真的经扩散到达了探索行为节点
            _exploration_gate = curiosity_engine.should_generate_question(kg, config)
            if not _exploration_gate:
                logger.info("[Curiosity] 信号未在扩散中到达探索行为节点"
                            "（好奇心不够强，不产生候选——合法状态）")

            # Drive 刷新：候选评分要读新鲜驱动力（含本轮信号 + 兴趣水位）
            try:
                drive_evaluator.evaluate(force=True)
            except Exception as _dve:
                logger.warning(f"[Drive] 探索相刷新失败: {_dve}")

            # 衰减探索信号节点，防止信号继续传播干扰主扩散
            curiosity_engine.cleanup_curiosity_activation(kg)

            # 主扩散（剩余步数）
            remaining_steps = _configured_max_depth - 3
            if remaining_steps > 0:
                logger.info(f"[Curiosity] 继续主扩散（剩余{remaining_steps}步）")
                engine.diffuse_round(max_steps=remaining_steps)
        else:
            # 正常扩散（无探索信号）
            logger.info("[NLP] 激活完成，开始一轮扩散")
            diffuse_info = engine.diffuse_round()
            logger.info(f"[NLP] 扩散完成: {diffuse_info}")
            _exploration_gate = False

        # ── 言外行为图谱投影（OGCTX 消费的状态摘要）──
        _sa_projection = {}
        try:
            if _sa_info:
                _sa_projection = speech_act_projection(kg, engine, _sa_info)
                if _sa_projection:
                    logger.info(
                        f"[SpeechAct] 投影: {_sa_projection.get('concept')} "
                        f"act={_sa_projection.get('activation')} 共激活="
                        f"{[c['id'] for c in _sa_projection.get('coactivated') or []][:6]}")
        except Exception as _sape:
            logger.debug(f"[SpeechAct] 投影跳过: {_sape}")

        # ── 调试：扩散后 Top-K ──
        _dbg_topk, _ = engine.get_topk(k=10)
        _dbg_after = [f"{_n.id}({_n.activation:.3f})" for _n in _dbg_topk[:10]]
        logger.info(f"[DEBUG] 扩散后Top10: {', '.join(_dbg_after)}")
        # ── 观测：激活小结（种子/深度/Top 前沿——不落全图，任务书 §9）──
        fas_log.get_logger(fas_log.ACTIVATION).info(
            "diffusion_summary", "种子激活与主扩散完成",
            seed_count=len(parsed_nodes or []),
            seeds=[str(n) for n in (parsed_nodes or [])[:12]],
            max_depth=_configured_max_depth,
            top=[_n.id for _n in _dbg_topk[:10]],
            top_energy=[round(float(_n.activation or 0.0), 3) for _n in _dbg_topk[:10]])

        # ── 预测误差基线（V2 补账 2026-09-20）──
        # 本轮焦点 vs 上轮预期：surprise 是"世界没按我以为的演化"的度量，
        # 进时间轴（经历层）+ 经 provider 抬 Learning/Consistency 驱动力。
        try:
            _pb = prediction_baseline.observe([_n.id for _n in _dbg_topk[:8]])
            if _pb.get("surprise") is not None and _pb["surprise"] >= 0.35:
                timeline.append(make_event(
                    EVENT_COGNITIVE, actor="self", subject="prediction_error",
                    content={"surprise": _pb["surprise"],
                             "unexpected": _pb["actual"][:4]},
                    source="prediction_baseline", meta={"cycle_id": _cycle_id}))
                logger.info(f"[PredictionBaseline] surprise={_pb['surprise']} "
                            f"本轮焦点偏离预期: {_pb['actual'][:4]}")
                # 预测误差 → 反思压力（世界没按预期演化 = 攒"为什么"）
                try:
                    cc.note_pressure(
                        "prediction_error",
                        min(0.5, float(_pb["surprise"]) * 0.3))
                except Exception as _ppe:
                    logger.warning(f"[PredictionBaseline] 反思压力注入失败(学习闭环断一环): {_ppe}")
                # R2 P7：意外也是调制事件（去甲肾上腺素 phasic = 场景 14 的抢占，
                # 乙酰胆碱 = 保真度需求上升）。幅度/靶点都在图上的 `影响` 边里。
                try:
                    reward_system.mod_engine().emit(
                        "prediction_violation", source="prediction_baseline",
                        uncertainty=float(_pb["surprise"]),
                        rpe=float(_pb["surprise"]),
                        intensity=float(_pb["surprise"]),
                        novelty=float(_pb["surprise"]), cycle_id=_cycle_id,
                        ref="prediction_error")
                except Exception as _mee:
                    logger.debug(f"[Modulation] 意外事件跳过: {_mee}")
        except Exception as _pbe:
            logger.debug(f"[PredictionBaseline] 观察跳过: {_pbe}")

        # ── 探索候选构建（只造候选不裁决；胜负在 dialogue_decide）──
        if _exploration_gate:
            _exploration = curiosity_engine.build_exploration_candidates(
                kg, config, gate=True, signal_result=curiosity_info,
                cognitive_context={
                    "illocutionary_act": parsed.get("illocutionary_act", ""),
                    "dialogue_act": parsed.get("dialogue_act", ""),
                    "response_expectation": parsed.get("response_expectation", ""),
                })
            if _exploration.get("candidates"):
                logger.info(
                    f"[Exploration] 候选入场: "
                    f"{[(c['action'], c['target'], c['score']) for c in _exploration['candidates']]} "
                    f"(drive={_exploration.get('drive')})")
            else:
                logger.info(f"[Exploration] 好奇但不行动: {_exploration.get('reason')}")

        # -------------------------------------------------
        # 2a-pre.0 显式动作意图统一解析（Action Concept 架构 2026-09-19）
        #   搜索/文件/屏幕三个概念走同一 resolver：
        #   fast pattern（现有触发正则原文，降级为快速入口）
        #   → miss → 图谱语义召回（Action Concept 长在图上）
        #   → 歧义（仅 web 通道）→ LLM 消歧。否定优先 + 命令性门。
        #   下面各动作块只消费 Action Intent；执行与认知回注代码不变。
        # -------------------------------------------------
        try:
            _action_intents, _action_info = resolve_turn_intents(
                text, kg=kg,
                embedder=emb_mgr if emb_mgr.ready() else None,
                llm=(lambda _p: cog_language.ask("action_resolve", _p))
                    if _channel in ("", "web") else None,
                action_space=action_space)
        except Exception as _aire:
            logger.debug(f"[ActionIntent] 统一解析失败（本轮按无显式动作处理）: {_aire}")
            _action_intents, _action_info = {}, {}

        # -------------------------------------------------
        # 2a-pre. 网络搜索动作：触发判定 + 执行（万物皆图）
        #   触发：显式搜索意图（Action Intent），或提问型输入引入≥2个图外概念
        # -------------------------------------------------
        _web_results = []
        _web_query = ""
        try:
            import re as _re
            # 显式命令路径：Action Intent（fast pattern/语义召回统一在 resolver）
            _si = _action_intents.get("SEARCH")
            _search_by_explicit = _si is not None
            # 自主信息需求路径（非用户命令）：图外概念/技术名词 → 搜索辅助。
            # §十七 误触发修复：显式"搜索"词被命令性门拦下时（"我不知道该不该
            # 搜索这个东西"），本轮不得再经启发式路径补触发搜索。
            _heuristic_ok = ("SEARCH" not in (_action_info.get("rejected") or {})
                             and not _action_info.get("refused", {}).get("SEARCH"))
            _new_concepts = 0
            if _heuristic_ok and parsed.get("dialogue_act") == "question":
                for _pn in (parsed.get("nodes") or []):
                    _pid = _pn if isinstance(_pn, str) else _pn.get("id", "")
                    if (_pid and _pid not in ("用户", "Fascinator")
                            and _pid not in kg.nodes):
                        _new_concepts += 1
            # C 修复：情绪/陈述中含图外"技术名词"（字母数字型概念，如 C2/5G）也触发
            # 情绪表达常带事实内容（"害怕离合器"），社交路径无知识注入，
            # 没有外部信息就只剩模型裸猜
            _tech_unknown = False
            if (_heuristic_ok and parsed.get("dialogue_act")
                    in ("emotion_expression", "information_statement", "opinion")):
                for _pn in (parsed.get("nodes") or []):
                    _pid = _pn if isinstance(_pn, str) else _pn.get("id", "")
                    if (_pid and _pid not in ("用户", "Fascinator")
                            and _pid not in kg.nodes
                            and _re.search(r"[A-Za-z0-9]", _pid)):
                        _tech_unknown = True
                        break
            if _search_by_explicit or _new_concepts >= 2 or _tech_unknown:
                if _search_by_explicit:
                    # 参数绑定已在 resolver 完成（查询清洗/指示词归一）
                    _web_query = str(_si["parameters"].get("query") or "")[:30]
                else:
                    _web_query = _re.sub(
                        r"上网|网上|帮我|一下|搜搜|搜索|搜|查查|查|【.*?】|[。？！?，,.!~\s]",
                        "", text)
                    # 指示词归一：搜索引擎对"这首歌"这类口语指代召回差
                    _web_query = _re.sub(r"[这那一]首歌", "歌曲", _web_query)
                    _web_query = _re.sub(r"[这那一]部电影", "电影", _web_query)[:30]
                from actions.web_search import do_web_search
                _web_results = do_web_search(_web_query, top_n=5)
                with kg._lock:
                    _ws_node = kg.get_node("网络搜索")
                    if _ws_node:
                        _ws_node.activation = min(5.0, _ws_node.activation + 3.0)
                        _ws_node.touch()
                        engine.mark_active([_ws_node.id])
                logger.info(
                    f"[WebSearch] 触发({'意图词' if _search_by_explicit else '图外概念'}) "
                    f"query='{_web_query}' 命中 {len(_web_results)} 条")
        except Exception as _we:
            logger.warning(f"[WebSearch] 动作失败: {_we}")

        # -------------------------------------------------
        # 2a-pre.4. 文件操作动作：触发 → LLM 抽参 → 执行（万物皆图）
        # -------------------------------------------------
        _file_result = None
        try:
            # 文件类三个概念走 Action Intent（原触发正则已降级为概念 fast pattern；
            # 读取/删除过去正则能命中但执行器一律拒绝、且口语变体根本不触发——
            # 现在能理解、如实告知缺口）
            _fi = (_action_intents.get("FILE_CREATE")
                   or _action_intents.get("FILE_READ")
                   or _action_intents.get("FILE_DELETE"))
            if _fi is not None:
                with kg._lock:
                    _fa_node = kg.get_node("文件操作")
                    if _fa_node:
                        _fa_node.activation = min(5.0, _fa_node.activation + 3.0)
                        _fa_node.touch()
                        engine.mark_active([_fa_node.id])
                if _fi["action"] != "FILE_CREATE":
                    _gap = "读取文件" if _fi["action"] == "FILE_READ" else "删除文件"
                    _file_result = {
                        "ok": False,
                        "error": f"暂不支持{_gap}（当前文件能力只支持创建/写入），"
                                 f"已理解你的请求",
                        "recognized": _fi["action"]}
                    logger.info(f"[FileAction] {describe_intent(_fi)} → 能力缺口如实拒绝")
                else:
                    # 动作识别与参数抽取解耦：文件名/内容/目录由 LLM 在**执行前**抽取
                    _spec = nlp.ask("file_action_extract", text)
                    try:
                        if _spec.startswith("```"):
                            _spec = "\n".join(_spec.split("\n")[1:-1])
                        _spec = _json.loads(_spec.strip())
                    except Exception:
                        _spec = {}
                    if _spec.get("action"):
                        from actions.file_action import execute_file_action
                        _file_result = execute_file_action(
                            _spec["action"], _spec.get("name", ""),
                            _spec.get("content", ""), _spec.get("dir", "桌面"))
                        logger.info(f"[FileAction] 执行结果: {_file_result}")
        except Exception as _fe:
            logger.warning(f"[FileAction] 动作失败: {_fe}")

        # -------------------------------------------------
        # 2a-pre.45. 屏幕感知动作：触发 → 截屏OCR → 注入图谱
        # -------------------------------------------------
        _eye_result = None
        # Minecraft 反射动作/会话结果：本块之后才可能执行，但对话决策层
        # （has_action_result）在此之前就要读——先声明，避免未赋值引用。
        # （反射快路径已在 turn 开头赋值 → 不在此重置，否则抹掉快路径结果；
        #   _mc_session_action 已前移至 turn 开头的 Phase C 块赋值，勿重置）
        if not _reflex_fast:
            _mc_action_result = None
        try:
            # 看屏幕走 SCREEN_OBSERVE Action Intent（原双正则=概念 fast pattern；
            # "看下画面/帮我看看现在是什么情况"等口语经语义层召回命中）
            if _action_intents.get("SCREEN_OBSERVE") is not None:
                # 与自主路径共用同一条观察流水线（eye/observer.run_observation：
                # recognize_text → 图谱驱动 salient_texts → inject_observation）
                from eye.observer import run_observation
                _eye_result = run_observation(
                    kg, engine=engine,
                    embedder=emb_mgr if emb_mgr.ready() else None)
                logger.info(f"[Eye] 对话触发屏幕识别: {_eye_result['count']} 条")
        except Exception as _ee:
            logger.warning(f"[Eye] 动作失败: {_ee}")

        # -------------------------------------------------
        # 2a-pre.45.5. 摄像头感知动作：触发 → 抓帧姿态推理 → 注入图谱
        # （2026-10-01，The Institute Eyes 移植；与屏幕块同构。同拍既看
        #   屏幕又看摄像头时屏幕优先——_eye_result 已有结果则跳过）
        # -------------------------------------------------
        try:
            if _action_intents.get("CAMERA_OBSERVE") is not None \
                    and _eye_result is None:
                _cam_res = camera_observer.execute(
                    {"action_type": "camera_observe"})
                if _cam_res.get("success"):
                    # 摄像头结果借用 eye_result 通道进认知上下文：
                    # count=人数, salient=行为列表（键形与 screen 一致）
                    _obs = _cam_res.get("observation") or {}
                    _eye_result = {
                        "source": "camera",
                        "describe": _cam_res.get("describe", ""),
                        "count": _obs.get("people", 0),
                        "salient": [{"text": b}
                                    for b in (_obs.get("behaviors") or [])],
                        "graph_node": _obs.get("graph_node"),
                    }
                    logger.info(f"[Eye] 对话触发摄像头识别: "
                                f"{_obs.get('people', 0)} 人")
                else:
                    logger.info(f"[Eye] 摄像头动作未执行: "
                                f"{_cam_res.get('reason')}")
        except Exception as _ce:
            logger.warning(f"[Eye] 摄像头动作失败: {_ce}")

        # -------------------------------------------------
        # 2a-pre.5.用户指令 → ActionNode（具身改造 2026-09：先行动，后回复）
        # -------------------------------------------------
        # 认知解析（L1 快速解析 / L2 意图分解）的产物映射成 ActionNode：
        #   L1：单意图 → 立即 propose（优先级/中断由 ActionManager 裁决）
        #   L2：多意图 → 第一个立即执行，其余进目标队列逐步执行
        # 动作结果随后进入回答生成的认知上下文（她一边开始做一边说话）。
        _user_action_result = None
        _user_intent_specs = []
        _mc_online = bool(_mc_state and _mc_state.get("connected"))
        _l1_wants_action = _l1_fast and bool(parsed.get("needs_action", False))
        # 否定优先（§十）：动作概念层已判定"否定挖/攻击"并拒绝执行的句子，
        # 绝不允许再被 LLM 意图分解旁路成执行（"别挖这个"≠挖）
        _mc_neg_refused = bool(_mc_info.get("refused"))
        if (not _reflex_fast and _mc_online and not _mc_neg_refused
                and ((_l1_fast and _l1_wants_action) or not _l1_fast)):
            _game_user = config.get("minecraft_user_name") or "Hellucigen"
            if _l1_fast:
                _spec = l1_to_action(parsed, _game_user)
                if _spec is not None:
                    _user_intent_specs = [_spec]
            else:
                # L2：复杂指令 → 意图分解（有行动语义才调；不直接生成操作序列）
                if _complexity.get("level") == 2 and re.search(
                        r"跟着|过来|停下|别动|挖|采|砍|收集|拿|放|建造|搭|合成|"
                        r"攻击|打|杀|吃|睡|回|去|走|探索|看看|找|烧|熔|种|收", text):
                    try:
                        _intention_result = nlp.extract_intentions(text)
                        _l2_intentions = _intention_result.get("intentions") or []
                        parsed["intentions"] = _l2_intentions
                    except Exception as _iex:
                        logger.warning(f"[Intention] 意图分解失败: {_iex}")
                        _l2_intentions = []
                else:
                    _l2_intentions = parsed.get("intentions") or []
                for _it in _l2_intentions[:6]:
                    _spec = intention_to_action(_it, _game_user)
                    if _spec is not None:
                        _user_intent_specs.append(_spec)
            if _user_intent_specs:
                # 多步指令 = 一个任务：挂自我图谱 当前目标 指针（排空自动摘除）
                if len(_user_intent_specs) > 1:
                    action_manager.set_goal_context(text, source="user")
                # 万物皆图：用户的行动请求先成为认知事件
                _record_user_action_intent(_user_intent_specs[0], text, _cycle_id)
                _prop = action_manager.propose(_user_intent_specs[0], source="user")
                if _prop.get("started"):
                    _user_action_result = {
                        "success": True, "action": _user_intent_specs[0]["action_type"],
                        "describe": _prop.get("describe")
                        or _user_intent_specs[0]["action_type"],
                        "target": _user_intent_specs[0].get("target"),
                        "pending": bool(_prop.get("pending"))}
                elif _prop.get("queued"):
                    _user_action_result = {
                        "success": True, "action": _user_intent_specs[0]["action_type"],
                        "queued": True,
                        "describe": str(_user_intent_specs[0]["action_type"]),
                        "target": _user_intent_specs[0].get("target")}
                else:
                    _user_action_result = {
                        "success": False, "action": _user_intent_specs[0]["action_type"],
                        "reason": _prop.get("reason") or "action_rejected"}
                # 其余意图 → 目标队列（由行动系统逐步执行，不一次生成操作序列）
                for _rest_spec in _user_intent_specs[1:]:
                    action_manager.queue_goal(_rest_spec)
                _mc_action_result = _mc_action_result or _user_action_result
                logger.info(f"[UserAction] 指令→Action: "
                            f"{[s['action_type'] for s in _user_intent_specs]} "
                            f"→ {_user_action_result}")

        # ── 时间感知锚定：时间指示语 → 当前时段节点激活 ──
        _tod = None
        try:
            _tod = time_ground(text, kg, engine)
        except Exception as _te:
            logger.warning(f"[Time] 锚定失败: {_te}")

        # -------------------------------------------------
        # 2a-pre.5. 记忆提取与沉淀（B 修复：先于回答生成）
        #   当轮新实体先入图并激活，回答时 TopK 才能带上本轮语境
        # -------------------------------------------------
        # ── 叙事分段抽取触发：事件结构评分（Event-Structure Trigger）──
        _nt = narrative_score(text, profile=llm_budget.profile)
        _narrative_result = None
        if _nt["decision"] == "narrative":  # 事件结构化触发（评分见 narrative_trigger）
            try:
                if llm_budget.can_call("reason"):
                    llm_budget.register("reason", tokens=1200)
                    _narrative_result = _ingest_narrative(kg, text)
                    if _narrative_result:
                        logger.info(f"[Narrative] 入图: 故事={_narrative_result['story']} "
                                    f"情节={len(_narrative_result['events'])}个")
                        # （2026-09-22 修复：此处原有局部 `import fas_log`，与
                        #  函数内更早的使用共同把 fas_log 变成本函数局部名 →
                        #  UnboundLocalError。fas_log 已在模块顶导入。）
                        fas_log.get_logger(fas_log.MEMORY).info(
                            "narrative_extracted", "叙事分段入图（只记计数）",
                            story=str(_narrative_result.get("story", ""))[:40],
                            events=len(_narrative_result.get("events") or []),
                            characters=len(_narrative_result.get("characters") or []))
                        _save(force=True)
                else:
                    logger.info("[LLMBudget] 叙事抽取预算不足，降级为常规抽取")
            except Exception as _ne:
                logger.warning(f"[Narrative] 抽取失败: {_ne}")

        memory_draft = None
        _turn_memory_draft = None
        _auto_added = None

        _da = parsed.get("dialogue_act", "")
        _skip_regular_extract = bool(_narrative_result)  # 叙事已分段入图
        _non_teaching_das = {
            "greeting", "farewell", "thanking", "apology", "emotion_expression",
            "question", "request", "opinion", "agreement", "disagreement",
            "comfort", "congratulation", "invitation", "backchannel", "suggestion",
            "curiosity_expression", "answer",
        }
        if _da:
            _should_extract_memory = (
                not _skip_regular_extract
                and (_da not in _non_teaching_das
                     or parsed.get("memory_type") in ("episodic", "semantic"))
            )
        else:
            _should_extract_memory = parsed.get("memory_type") in ("episodic", "semantic")

        logger.info(
            f"[Memory] da={_da}, extract={_should_extract_memory}, "
            f"illoc={parsed.get('illocutionary_act', '')}"
        )

        if _should_extract_memory:
            logger.info("[Memory] 断言类输入, 触发 LLM 知识图谱抽取")
            with engine._lock:
                topk_check, _ = engine.get_topk(k=20)
                _focus_events = _get_focus_context(kg)
            memory_draft = nlp.extract_assertion_graph(
                text, topk_check, focus_events=_focus_events)
            _turn_memory_draft = memory_draft
            if memory_draft and not memory_draft.get("error") and (memory_draft.get("nodes") or memory_draft.get("edges")):
                assertion_type = memory_draft.get("assertion_type", "episodic")
                buffer.add_experience(
                    raw_text=text,
                    nodes=memory_draft.get("nodes", []),
                    edges=memory_draft.get("edges", []),
                    assertion_type=assertion_type
                )
                candidates = buffer.get_promotion_candidates()
                if candidates:
                    logger.info(f"[Memory] 晋升候选: {len(candidates)} 条经历满足晋升条件")
                    # A4 观测：access 驱动候选（仅计数提示，不自动晋升）
                    _acc_driven = [
                        c for c in candidates
                        if c.access_count >= buffer.promote_threshold_access
                    ]
                    if _acc_driven:
                        logger.info(
                            "[A4-Obs] access 驱动候选 %d 条: %s",
                            len(_acc_driven), "; ".join(
                                f'"{c.raw_text[:22]}"(acc={c.access_count},imp={c.importance:.2f})'
                                for c in _acc_driven[:8]))
                logger.info(
                    f"[Memory] 草稿: {len(memory_draft.get('nodes',[]))} 节点, "
                    f"{len(memory_draft.get('edges',[]))} 边"
                )

                # ── 自动知识沉淀（人工审核门槛已屏蔽：agent 代管对话模式，
                #    无人点审批，等审核=数据丢失；抽取守卫已排除纯寒暄）──
                _auto_added = _auto_consolidate_curiosity_knowledge(
                    kg, memory_draft, buffer, text
                )
                if _auto_added:
                    _trigger = "curiosity" if (curiosity_resolved and curiosity_resolved.get("resolved")) else "casual_qa"
                    logger.info(
                        f"[AutoConsolidate] ({_trigger}) +{_auto_added.get('nodes',0)}节点 "
                        f"+{_auto_added.get('edges',0)}边"
                    )
                    _save(force=True)

                # 本轮新实体补激活：抽取时它们还不在图中，扩散期错过；
                # 不补激活的话，TopK（按激活度取）仍带不上本轮语境
                _seeds = []
                for _dn in memory_draft.get("nodes", []):
                    _did = _dn.get("id", "") if isinstance(_dn, dict) else str(_dn)
                    _did = _did.strip()
                    if _did and _did in kg.nodes:
                        _seeds.append(_did)
                if _seeds:
                    engine.activate_from_inputs(_seeds, [])
                    logger.info(f"[Memory] 新实体补激活: {len(_seeds)} 个")
                    # 补一步"只从新实体发射"的传播：新实体在本轮扩散后才拿到
                    # 激活，只补激活不补传播的话它的邻域（关联能力、时间桶、
                    # 相关实体）照旧是暗的——语境进得了 TopK，语境的关系进不去。
                    # 必须走 diffuse_from：否则整片前沿会重新发射一轮，把邻域
                    # 泵到饱和，回答区退化成一堆 cap 并列（旧节点靠入图序取胜）。
                    try:
                        engine.diffuse_from(_seeds, steps=1)
                    except Exception as _dse:
                        logger.warning(f"[Memory] 新实体补扩散失败: {_dse}")
            else:
                logger.warning(f"[Memory] 抽取无结果: {memory_draft}")

            # 时间感知：为本次事件补 发生于时段 边（去重轮也补——事件可能已存在）
            if _tod and _turn_memory_draft:
                try:
                    _evt = _fuzzy_match_node(_turn_memory_draft["event"]["summary"])
                    link_event_to_bucket(kg, _evt, _tod)
                except Exception as _tee:
                    logger.warning(f"[Time] 事件挂接失败: {_tee}")

        # -------------------------------------------------
        # 2a. Reflection Evolution: 上一轮表达事件的 outcome 标注
        #     + 本轮行为倾向计算（激活竞争）+ 沉默判定
        # -------------------------------------------------
        _behavior_contexts = [context_for_dialogue_act(
            parsed.get("dialogue_act", ""))]
        _neg_markers = ("不开心", "难过", "失落", "难受", "烦", "伤心", "委屈")
        if any(m in text for m in _neg_markers):
            _behavior_contexts.append(MODIFIER_EMOTION_LOW)
        try:
            if len(get_chat_log().all()) > 50:
                _behavior_contexts.append(MODIFIER_FAMILIAR)
        except Exception as _fle:
            logger.warning(f"[NLP] 聊天记录读取失败(熟稔修饰缺失,低影响): {_fle}")

        # ── 表达反馈观察（ExprFB：outcome 图谱化 2026-09-19）──
        # "上一句产生了什么结果"由图谱后验关系决定：本轮输入经话题/实体边
        # 把上轮表达事件回流点亮（共激活增量≥阈值），或出现会话控制指令
        # （既有检测，语言理解层）→ 建立回应关系 → SocialFeedback 事件节点。
        # 派生的 social 标签只是 reward/日志的解释层输入（方向：图谱→派生，
        # 不再文本→分类）。回应未建立则什么都不记：未观察到结果 ≠ neutral
        # （旧 ">4h→neutral" 与积极/抵触词表已退役）。
        try:
            _fb = expression_feedback.observe_response(
                kg, engine, timeline, pre=_fb_pre, user_text=text,
                parsed=parsed, cycle_id=_cycle_id, config=config)
            if _fb is not None:
                # buffer 兼容视图回填（reflection 消费的数据形态：四态+detail）
                try:
                    _prev_expr = buffer.last_unannotated_expression()
                    if _prev_expr is not None:
                        buffer.annotate_expression(
                            _prev_expr, _fb["legacy"], _fb["detail"])
                except Exception as _aee:
                    logger.warning(f"[ExprFB] 上轮表达 outcome 标注丢失(Reflection 闭环断一环): {_aee}")
                # 用户对 FAS 表达的反馈 → CC 表达抑制信号（叫停/抵触 → 抑制）
                cc.note_outcome(_fb["legacy"] == "negative")
                if _fb["legacy"] in ("positive", "negative"):
                    persona.mood_event(_fb["legacy"])
                # ── 社会反馈经奖赏层进 disposition（红线不变：只缩放学习率，
                #    不做"用户满意最大化"）──
                _b, _c = _fb.get("behavior", ""), _fb.get("context", "")
                _social = _fb["social"]
                if _b and _c:
                    _ev = reward_system.evaluate(
                        behavior=_b, context=_c, social_outcome=_social,
                        self_outcome=None, ref=text[:60], cycle_id=_cycle_id)
                    _rel = reward_system.release(_ev)
                    _mod = reward_system.modulation(
                        key=f"{_b}@{_c}",
                        surprise=_rel.get("surprises", {}).get(f"{_b}@{_c}"))
                    # 经验时间轴：奖赏是自身状态变化（SELF_STATE_CHANGE）
                    for _rev in _ev:
                        try:
                            timeline.append(make_event(
                                EVENT_SELF_STATE, actor="self", subject="reward",
                                content={"change": "changed",
                                         "source": _rev["source"],
                                         "outcome": _rev["outcome"],
                                         "valence": _rev["valence"]},
                                source="reward", meta={"cycle_id": _cycle_id}))
                        except Exception as _rrte:
                            logger.warning(f"[Experience] 奖赏入轴失败(反馈分支): {_rrte}")
                    _trace = disposition_store.apply_experience(
                        _b, _c, social_outcome=_social, self_outcome=None,
                        hormone=_mod, evidence_ref=text[:60], allow_create=True)
                    logger.info(
                        f"[ExprFB] 上轮 {_b}@{_c} ← "
                        f"delta={_fb['response_delta']}"
                        f"{'(control)' if _fb['via_control'] else ''} "
                        f"dt={_fb['dt_seconds']}s "
                        f"emo_v={_fb['emotion_valence']} "
                        f"social={_social} ({_fb['detail']}) "
                        f"sig={_trace.get('learning_signal')} "
                        f"{_trace.get('old')}→{_trace.get('new')} "
                        f"[{_trace.get('state')}] hormone_mod={_mod.get('factor')}")
        except Exception as _e:
            logger.warning(f"[ExprFB] 回应观察失败（不影响本轮）: {_e}")

        # ── 交流决策层：回应欲望认知竞争（替代原单一沉默倾向判定）──
        try:
            _last_expr = buffer.last_unannotated_expression() or {}
            from datetime import datetime as _dt2
            try:
                _gap_s = (_dt2.now() - _dt2.strptime(
                    _last_expr.get("ts", ""), "%Y-%m-%d %H:%M:%S")).total_seconds()
            except Exception:
                _gap_s = None
            if _just_terminated:
                _sup = inhibit_topics(kg, engine)
                logger.info(f"[DialogueDecision] 终止后抑制话题节点: {_sup}")
            fas_log.get_logger(fas_log.DECISION).info(
                "decision_started", "统一行为竞争开始裁决",
                exploration_candidates=[
                    c.get("action") for c in (_exploration.get("candidates") or [])][:8],
                curiosity_active=curiosity_engine.is_curiosity_active(kg))
            _dd = dialogue_decide(
                parsed=parsed, text=text, kg=kg, engine=engine,
                tendencies=disposition_store.tendencies_for(_behavior_contexts),
                curiosity_active=curiosity_engine.is_curiosity_active(kg),
                last_outcome_negative=cc._last_outcome_negative,
                last_expr_gap_s=_gap_s,
                has_action_result=bool(_web_results or _file_result or _eye_result or _mc_action_result),
                exploration=_exploration,
                hormone=(reward_system.modulation() if reward_system else None),
                modulation=drive_evaluator.modulation(),
                action_tendencies=drive_evaluator.field.action_tendencies(),
                action_space=action_space)
            if _dd.get("exploration", {}).get("selected"):
                logger.info(
                    f"[Exploration] 竞争胜出: {_dd['exploration']['selected']} "
                    f"target={_dd['exploration'].get('target')} (desire={_dd.get('desire')})")
            elif _exploration.get("candidates"):
                logger.info(
                    f"[Exploration] 探索落选（好奇但不行动）: 候选="
                    f"{[(c['action'], c['score']) for c in _exploration['candidates']]} "
                    f"desire={_dd.get('desire')}")
        except Exception as _dde:
            logger.warning(f"[DialogueDecision] 决策失败默认回应: {_dde}")
            _dd = {"decision": "respond", "desire": 0.6, "constraints": [],
                   "factors": {}, "tendencies": [],
                   "exploration": {"candidates": [], "selected": None, "target": None}}
        _tendencies = _dd["tendencies"]
        # ── 观测：裁决结果（为什么 A 没选而 B 被选，任务书 §12）──
        fas_log.get_logger(fas_log.DECISION).info(
            "decision_finished",
            f"裁决={_dd.get('decision')} desire={_dd.get('desire')}",
            decision=_dd.get("decision"), desire=_dd.get("desire"),
            factors=_dd.get("factors"),
            constraints=_dd.get("constraints"),
            rejected=[c.get("action")
                      for c in ((_dd.get("exploration") or {}).get("candidates") or [])
                      if isinstance(c, dict)][:8],
            selected_exploration=(_dd.get("exploration") or {}).get("selected"))
        _silence_chosen = (_dd["decision"] == "silence")
        if _silence_chosen:
            logger.info(f"[DialogueDecision] 沉默: desire={_dd['desire']} {_dd['factors']}")

        # ─────────────────────────────────────────────────────────
        # 1.5 Cognitive Context：回合级一次构建，L1/L2 各自编译（cognitive_context.py）
        #   状态/动机/决定/证据 在此定型；语言实现层只拿编译结果。
        #   reflex 轮与 L1 轮不消费 cognitive_events（不 drain——
        #   取用即推进水位，只有真要把事件交给 LLM 的轮次才消耗）。
        # ─────────────────────────────────────────────────────────
        # ── 认知需求三层分析（cognitive_demand 重构 2026-09-20）──
        # Demand（九维需求）→ Gap（需求−内部能力）→ Resource Routing（资源
        # 调度）→ LLM Resource Level（MODE0~5 预算语义不变）。三条硬规则：
        # emotion 不进 LLM 路由；unknown 只有"阻塞理解"才成缺口；
        # 本分析只**读取** dialogue_decision 的 should_speak，绝不反向
        # 决定"是否回应"。_demand/_mode 字段保留（legacy 兼容 + 预算门）。
        try:
            with kg._lock:
                _unknown_now = [str(n) for n in (raw_nodes or [])
                                if str(n) and str(n) not in kg.nodes
                                and str(n) not in ("用户", "Fascinator")]
        except Exception:
            _unknown_now = [str(n) for n in (raw_nodes or [])]
        _demand_analysis = analyze_cognitive_demand(
            text=text, parsed=parsed, kg=kg, engine=engine,
            unknown_nodes=_unknown_now,
            faiss_hits=faiss_hits,
            resonance_activated=(_resonance or {}).get("activated"),
            exploration=_exploration,
            intentions=parsed.get("intentions"),
            action_result=(_user_action_result or _mc_action_result),
            current_action=action_manager.current,
            executor_ready=(action_manager.current is None),
            players_nearby=[pl.get("name")
                            for pl in ((_mc_state or {}).get("playersNearby") or [])],
            should_speak=(_dd.get("decision") != "silence"),
            decision=_dd,
        )
        _demand = _demand_analysis["legacy"]
        _mode = _demand_analysis["mode"]
        # ── 观测：需求→缺口→模式（认知资源路由决策依据）──
        fas_log.get_logger(fas_log.COGNITION).info(
            "demand_evaluated",
            f"LLM 模式初选={_mode}",
            demand={k: round(float(v), 2)
                    for k, v in (_demand_analysis.get("demand") or {}).items()
                    if isinstance(v, (int, float)) and v >= 0.2},
            gap={k: round(float(v), 2)
                 for k, v in (_demand_analysis.get("gap") or {}).items()
                 if isinstance(v, (int, float)) and v >= 0.2},
            mode=_mode)
        # ── 网络态微调 mode（Demand 定档位=要多少资源；Network 定组织方式）──
        # 只允许 ±1 档、下限 MODE1（语言仍可实现）/上限 MODE3；有迟滞
        # （nudge 来自带惯性的网络场，不逐拍翻转）。
        try:
            _nudge = int(round(drive_evaluator.modulation().get(
                "llm.mode_nudge", 0.0)))
            if _nudge:
                _order = [MODE0_GRAPH_ONLY, MODE1_LANGUAGE, MODE2_INTERPRET,
                          MODE3_REASON]
                if _mode in _order:
                    _idx = min(len(_order) - 1,
                               max(1, _order.index(_mode) + _nudge))
                    if _order[_idx] != _mode:
                        logger.info(f"[CognitiveField] mode 微调 {_nudge:+d}: "
                                    f"{_mode} → {_order[_idx]}")
                        _mode = _order[_idx]
        except Exception as _ne:
            logger.warning(f"[CognitiveField] mode nudge 读取/换算失败(mode 保持原档): {_ne}")
        # 网络态 → 语言实现温度（DMN 高→发散，CEN 高→收紧）
        try:
            _ml = drive_evaluator.modulation()
            nlp.set_answer_temperature(
                answer_temp=float(_ml.get("llm.temperature_answer", 0.7)),
                expand_temp=float(_ml.get("llm.temperature_expand", 0.35)))
        except Exception as _te:
            logger.warning(f"[CognitiveField] 温度设置失败(温度保持旧值): {_te}")
        logger.info(
            "[CognitiveDemand] "
            + " demand={" + ", ".join(
                f"{k}:{v:.2f}" for k, v in _demand_analysis["demand"].items()
                if v >= 0.2) + "}"
            + " gap={" + ", ".join(
                f"{k}:{v:.2f}" for k, v in _demand_analysis["gap"].items()
                if v >= 0.2) + "}"
            + f" → LLM mode={_mode}")
        for _whyline in _demand_analysis["why"]:
            logger.info(f"[CognitiveDemand]   · {_whyline}")
        if _web_results or _file_result or _eye_result:
            if _mode == MODE0_GRAPH_ONLY:
                _mode = MODE1_LANGUAGE
        # ── 观测：资源定档（网络微调/证据提升之后的最终 LLM 档位）──
        fas_log.get_logger(fas_log.COGNITION).info(
            "resources_routed", f"LLM 资源档位定型 mode={_mode}",
            mode=_mode,
            budget_profile=getattr(llm_budget, "profile", None),
            evidence=bool(_web_results or _file_result or _eye_result
                          or _mc_action_result))
        _cog_full = {}
        try:
            _mc_online_now = bool(_mc_state and _mc_state.get("connected"))
            _sess_ev = dict(_mc_session_action) if isinstance(_mc_session_action, dict) else None
            if _sess_ev:
                if _sess_ev.get("request_message"):
                    _sess_ev.setdefault("describe", str(_sess_ev["request_message"])[:120])
                elif _sess_ev.get("cancelled"):
                    _sess_ev.setdefault("describe", str(_sess_ev.get("message")
                                                        or "用户决定先不进来了"))
                elif _sess_ev.get("success"):
                    _sess_ev.setdefault(
                        "describe",
                        f"已进入 Minecraft 世界（端口 {_sess_ev.get('port')}），"
                        f"并出发走向用户")
                elif _sess_ev.get("reason"):
                    _sess_ev.setdefault("describe", "连接 Minecraft 没成功")
            _drives = {}
            try:
                with kg._lock:
                    for _dn in ("CuriosityDrive", "SocialDrive",
                                "LearningDrive", "ConsistencyDrive"):
                        _node = kg.nodes.get(_dn)
                        if _node is not None:
                            _drives[_dn] = round(
                                float(getattr(_node, "activation", 0.0) or 0.0), 3)
            except Exception as _dfe:
                logger.warning(f"[NLP] 驱动器激活快照读取失败(_drives 留空,仅影响观测): {_dfe}")
            _intent_outcome = None
            if _user_intent_specs:
                if _user_action_result is None:
                    _intent_outcome = None
                elif _user_action_result.get("queued"):
                    _intent_outcome = "queued"
                elif _user_action_result.get("success"):
                    _intent_outcome = ("started_pending"
                                       if _user_action_result.get("pending")
                                       else "done")
                else:
                    _intent_outcome = "failed"
            _turn_action_intent = ({
                "action_type": _user_intent_specs[0].get("action_type"),
                "target": _user_intent_specs[0].get("target"),
                "outcome": _intent_outcome,
            } if _user_intent_specs else None)
            _turn_action_queue_len = max(0, len(_user_intent_specs) - 1)
            _cog_full = build_cognitive_context(
                text=text, channel=_channel, parsed=parsed,
                perception_connected=_mc_online_now,
                environment=({
                    "position": (_mc_state or {}).get("position"),
                    "health": (_mc_state or {}).get("health"),
                    "food": (_mc_state or {}).get("food"),
                    "nearby_entities": len((_mc_state or {}).get("nearbyEntities") or []),
                    "players_nearby": [p.get("name")
                                       for p in ((_mc_state or {}).get("playersNearby") or [])],
                } if _mc_online_now else {}),
                attention_context=attention_context(
                    kg, engine, _mode,
                    modulation=drive_evaluator.modulation()),
                mode=_mode, demand=_demand,
                world_state=(world_state_snapshot(kg) if _mc_online_now else {}),
                mood=persona.mood_context(),
                cognitive_events=(regulation.drain_cognitive_events(
                    llm_mode_min=1, limit=3)
                    if (not _reflex_fast and not _l1_fast) else []),
                recent_dialogue=get_chat_log().recent(n=6),
                faiss_hits=faiss_hits,
                recalled_nodes=[(h.get("hit") if isinstance(h, dict) else str(h))
                                for h in (faiss_hits or [])][:12],
                tendencies=_tendencies,
                drives=_drives,
                networks=drive_evaluator.field.networks.levels(),
                action_tendencies=drive_evaluator.field.action_tendencies(),
                hormone=(reward_system.modulation() if reward_system else None),
                exploration=_exploration,
                decision=_dd,
                speech_act_landscape=_sa_projection,
                demand_analysis=_demand_analysis,   # §15：三层结构入 OGCTX
                action_intent=_turn_action_intent,
                action_queue_len=_turn_action_queue_len,
                evidence={"mc_action": _mc_action_result,
                          "mc_session": _sess_ev,
                          "web_results": _web_results,
                          "file_result": _file_result,
                          "eye_result": _eye_result},
            )
        except Exception as _cce:
            logger.warning(f"[CognitiveContext] 回合构建失败（语言层降级为无上下文）: {_cce}")
            _cog_full = {}

        # -------------------------------------------------
        # 2b. 生成 LLM 回答（基于激活扩散后的 Top-K 认知焦点）
        # -------------------------------------------------
        # 行为竞争的胜者决定本轮输出形态：
        #   explore_ask   → 探索提问（LLM 只做目标→问句转译）
        #   explore_search→ 自己搜索后汇报
        #   respond/minimal → 正常回答
        #   silence       → 不生成回答（不回应是合法结果）
        llm_answer = None

        if _dd.get("decision") == "explore_ask":
            # ── 探索行为：向用户提问（ExploreByAskingUser）──
            # 只有行为竞争选出它才走到这里；LLM 只负责把已决定的探索目标
            # 转成自然语言问句（P7），且受预算门控。
            _ask_target = (_dd.get("exploration", {}) or {}).get("target") \
                or _exploration.get("target")
            _ask_ok = True
            if llm_budget is not None and not llm_budget.can_call("language"):
                _ask_ok = False
                logger.info("[Exploration] 语言预算不足，explore_ask 降级为不问（好奇保留）")
            if _ask_ok:
                try:
                    llm_budget.register("language", tokens=200)
                    _ask_info = dict(curiosity_info)
                    _ask_info["target"] = _ask_target
                    curiosity_question = curiosity_engine.generate_question(cog_language, _ask_info)
                except Exception as e:
                    logger.exception(f"[Exploration] 生成问题异常: {e}")
                if curiosity_question:
                    _t_node = curiosity_engine.ensure_exploration_target(kg, _ask_target or "")
                    curiosity_engine.record_pending_inquiry(
                        kg, curiosity_question, _ask_target, _t_node,
                        curiosity_info, config, engine=engine)
                    llm_answer = curiosity_question
                    logger.info(f"[Exploration] explore_ask 执行: {curiosity_question}")
                    # （同上：局部 import fas_log 制造 UnboundLocalError，已删）
                    fas_log.get_logger(fas_log.CURIOSITY).info(
                        "question_aroused", "探索提问胜出并发出",
                        target=str(_ask_target or ""),
                        question=fas_log.text(curiosity_question),
                        trigger_type=str(curiosity_info.get("trigger_type") or ""))
                else:
                    logger.info("[Exploration] 问题生成失败，本轮降级为不问")

        if llm_answer is None and _dd.get("decision") == "explore_search":
            # ── 探索行为：自己搜索（ExploreBySearch，论文 §在线知识扩展）──
            _search_target = (_dd.get("exploration", {}) or {}).get("target") \
                or _exploration.get("target")
            try:
                from actions.web_search import do_web_search as _dws
                _web_results = _dws(str(_search_target or ""), top_n=5)
                if _web_results:
                    _web_query = str(_search_target or "")
                    with kg._lock:
                        _ws_node = kg.get_node("网络搜索")
                        if _ws_node:
                            _ws_node.activation = min(5.0, _ws_node.activation + 3.0)
                            _ws_node.touch()
                            engine.mark_active([_ws_node.id])
                    _t_node = curiosity_engine.ensure_exploration_target(kg, _search_target or "")
                    curiosity_engine.note_interest(kg, _t_node, "searched", config)
                    logger.info(f"[Exploration] explore_search 执行: '{_search_target}' "
                                f"命中 {len(_web_results)} 条（结果待汇报 → 本轮转回应）")
                else:
                    logger.info("[Exploration] 搜索无结果，探索失败如实降级")
            except Exception as _ese:
                logger.warning(f"[Exploration] explore_search 失败: {_ese}")

        # ── Level 1 短回应（先行动，后回复；认知上下文重构 2026-09-20）──
        # L1 走 Context Compiler 的 L1 编译：只拿决定（应答姿态/长度约束）、
        # 证据（动作状态 executing/queued/done/failed——语言层不许把
        # 没做完的说成做完了）、语气底色（mood/最近对话）。
        # 会话状态机有结果（连接/要端口/取消）时必须回应——L1 对"51027"
        # 这类输入的解析可能拿不准 needs_reply，但"进没进去"不能沉默。
        if (_l1_fast and llm_answer is None and not _reflex_fast
                and (parsed.get("needs_reply", True) or _mc_session_action
                     or _user_action_result)
                and (not _silence_chosen or _mc_session_action)):
            _ctx_l1 = compile_for_language(_cog_full, path="L1")
            _ev1 = _ctx_l1.get("action_evidence") or {}
            if not _ev1 or _ev1.get("status") in (None, "", "none"):
                # 会话状态机的结果是本轮最该说出口的事（不经 LLM 解析也要说）
                _ev1 = _ctx_l1.get("session_evidence") or {}
            try:
                if llm_budget is not None and llm_budget.can_call("language"):
                    llm_budget.register("language", tokens=200)
                    with fas_log.llm_purpose("response_generation_L1"):
                        llm_answer = nlp.answer_short(
                            text, _ev1 or None, cognitive_context=_ctx_l1)
                    logger.info(f"[NLP] L1 短回应: {(llm_answer or '')[:80]}")
            except Exception as _ase:
                logger.warning(f"[NLP] L1 短回应失败（转正常回答）: {_ase}")
                llm_answer = None

        if llm_answer is None and _silence_chosen:
            # Reflection Evolution R4: 沉默是真实候选行为，本轮不生成回答
            pass
        elif llm_answer is None:

            logger.info("[NLP] 准备生成 LLM 回答")

            theta_action = config.get("theta_action", 0.5)

            with engine._lock:
                llm_node = engine.name_to_node.get("LLM回答")

            # 搜索/文件动作结果就绪时强制走回答生成（动作节点 -辅助→ LLM回答 通路语义）
            if ((llm_node is None or llm_node.activation < theta_action)
                    and (_web_results or _file_result or _eye_result
                         or _mc_action_result or _dd.get("decision") in ("respond", "minimal"))):
                with kg._lock:
                    if llm_node:
                        llm_node.activation = max(llm_node.activation, theta_action)
                        llm_node.touch()
                        engine.mark_active([llm_node.id])

            # 反射快路径已用"行动+游戏内短回执"回答 → 跳过 LLM 答案（不重复说话）
            if llm_node and llm_node.activation >= theta_action and not _reflex_fast:

                raw_topk, topk_edges = engine.get_topk(k=20)

                # 过滤掉基础设施节点，只保留真实语义/情景知识节点（V2: 基于 label 动态检测）
                # raw_topk ≤ k 项，逐项判 label 即可——不做全图扫描集合
                topk_nodes = [
                    n for n in raw_topk
                    if is_cognitive_visible(n)
                    and not n.id.startswith("坐标(")
                    # 非阻塞锁：对象照常参与扩散，但不进入回答区（认知输出）
                    and not regulation.locks.hides_node(n.id, "topk")
                ]
                if not topk_nodes:
                    topk_nodes = raw_topk[:5]

                # 过滤边：只保留两端都在知识节点中的边
                knowledge_ids = {n.id for n in topk_nodes}
                knowledge_edges = [
                    e for e in topk_edges
                    if (e.src in knowledge_ids and e.dst in knowledge_ids)
                ]

                logger.info(
                    f"[NLP] LLM回答节点激活度: {llm_node.activation:.4f}, "
                    f"TopK节点数: {len(raw_topk)}, 过滤后: {len(topk_nodes)}, 边: {len(knowledge_edges)}"
                )
                # 回答依据留痕：她这一轮"想到了什么"必须可查——能力节点
                # （self_capability）是否进了回答区，一眼可见。
                logger.info(
                    "[NLP] 回答区节点: "
                    + ", ".join(f"{n.id}({n.activation:.2f})" for n in topk_nodes[:15])
                )

                try:
                    _ctx_type = parsed.get("assertion_type") if parsed.get("assertion_type") == "social" else None
                    # ── Phase C（世界接入生命周期）已前移到 turn 开头（2026-09-19 修复）：
                    #   旧位置在 LLM 回答分支内——60+ 秒的解析管线 + 用户提前关服
                    #   会让"来玩Minecraft吧端口XXXXX"整轮死在半路（实测进不去世界
                    #   的直接原因）；且 L1 快轮次根本到不了这里。现在它先于 NLP
                    #   执行、纯确定性、与反射快路径同层，_mc_session_action 在
                    #   下方 _cog_ctx 照常注入。──
                    # ── MODE0 反射命令已完全前移（认知上下文重构 2026-09-20）──
                    #   旧内联块会在 L2 分支里重置 _mc_action_result 并二次派发
                    #   ——执行权/证据已在 turn 开头统一形成（Action Concept 解析
                    #   → ActionManager → _cog_full.evidence），此处不再重复。
                    # ── Cognitive Context 编译（回合级已构建 _cog_full）──
                    # L2 全编译：输出键与重构前的扁平 _cog_ctx 完全兼容
                    # （nlp_processor 消费侧零行为变化），并附带
                    # decision/action_evidence 升级契约（证据状态感知渲染）。
                    _cog_ctx = compile_for_language(_cog_full, path="L2")
                    # reflex 快路径已在 turn 开头执行过 → 这里整体跳过防重复。
                    # （旧内联的 MODE0 反射块与 turn 开头同源：若回合中途
                    #  才连上世界，也已在 session 块里走向用户，不再重复挖指令）
                    _budget_factor = 1.0
                    try:
                        _budget_factor = float(drive_evaluator.modulation().get(
                            "llm.budget_factor", 1.0))
                    except Exception as _bfe:
                        logger.warning(f"[LLMBudget] budget_factor 读取失败(回落 1.0): {_bfe}")
                    if _mode == MODE0_GRAPH_ONLY or not llm_budget.can_call(
                            _mode, budget_factor=_budget_factor):
                        # 预算拒绝/图谱自足：跳过语言实现（图谱继续，降级为不回答）
                        llm_answer = None
                        logger.info(f"[LLMBudget] mode={_mode} 预算/模式拒绝，图谱自足降级")
                    else:
                        with fas_log.llm_purpose("response_generation"):
                            llm_answer = nlp.answer_question(
                                text, topk_nodes, knowledge_edges,
                                context_type=_ctx_type, cognitive_context=_cog_ctx)
                        llm_budget.register(_mode, tokens=800)
                    # None 是"预算/模式拒绝"的合法结果（上方降级分支），不是
                    # 异常——老代码在这里无条件切片，一旦 can_call 返回 False
                    # （配额耗尽、或实验屏蔽 cognition_llm）整条回答路径就抛
                    # TypeError 被外层吞掉，图侧收尾（搜索记录/激活清零）全跳过。
                    logger.info(f"[NLP] LLM回答: {(llm_answer or '')[:200]}...")

                    with kg._lock:
                        # P0-6：不再创建 回答记录_N 节点——回答文本是认知活动的
                        # 痕迹而非认知状态，归宿是对话日志（chat_log.record，
                        # 本流程末尾统一写入）。图只保留影响未来的结构。
                        # 保留 LLM回答 节点激活清零：表示"本轮已执行、能量已消耗"。
                        llm_answer_node = kg.nodes.get("LLM回答")
                        if llm_answer_node:
                            llm_answer_node.activation = 0.0
                            llm_answer_node.touch()

                        # 网络搜索记录节点（万物皆图：动作留痕）
                        if _web_results:
                            import time as _t2
                            _sr_id = f"搜索记录_{int(_t2.time())}"
                            kg.add_node(Node(
                                id=_sr_id, weight=0.5, label="procedural",
                                extra_attrs={
                                    "query": _web_query,
                                    "results": [
                                        {"title": r.get("title", "")[:60],
                                         "snippet": r.get("snippet", "")[:120],
                                         "url": r.get("url", "")[:200]}
                                        for r in _web_results[:5]
                                    ],
                                }
                            ))
                            kg.add_edge(Edge(
                                src=_sr_id, dst="网络搜索", relation="执行",
                                weight=0.9))
                            # 连到查询中出现的已有图实体（只连已存在节点，不造新实体）
                            with kg._lock:
                                _linked = 0
                                for _nid in list(kg.nodes.keys()):
                                    if (_nid and len(_nid) >= 2
                                            and _nid not in ("用户", "Fascinator")
                                            and _nid in _web_query
                                            and _linked < 3):
                                        n2 = kg.nodes[_nid]
                                        if n2.label not in ("infrastructure", "procedural"):
                                            kg.add_edge(Edge(
                                                src=_sr_id, dst=_nid,
                                                relation="涉及", weight=0.5))
                                            _linked += 1
                            logger.info(f"[WebSearch] 搜索记录入图: {_sr_id}")

                        # 文件操作留痕节点（万物皆图：动作留痕）
                        if _file_result and _file_result.get("ok"):
                            import time as _t3
                            _fr_id = f"文件操作记录_{int(_t3.time())}"
                            kg.add_node(Node(
                                id=_fr_id, weight=0.5, label="procedural",
                                extra_attrs={
                                    "action": _file_result.get("action"),
                                    "path": _file_result.get("path", "")[:200],
                                    "size": _file_result.get("size", 0),
                                }
                            ))
                            kg.add_edge(Edge(
                                src=_fr_id, dst="文件操作", relation="执行",
                                weight=0.9))
                            logger.info(f"[FileAction] 操作记录入图: {_fr_id}")

                except Exception as e:
                    logger.exception(f"[LLM回答异常] {e}")
                    llm_answer = f"[回答生成失败: {e}]"
        # L1-CON-5:占位串(=内层/外层 catch 双路产物)不是真实话语——
        # 不得进经历时间轴,也不得作为 Say-Do 正则的驱动力
        _llm_answer_failed = bool(llm_answer) and str(llm_answer).startswith(
            "[回答生成失败")

        # 经验时间轴：她说的话是一条 ACTION（actor=self；沉默则无事件——如实）
        if llm_answer is not None and not _llm_answer_failed:
            try:
                _say_evt = make_event(
                    EVENT_ACTION, actor="self", subject="say",
                    content={"text": str(llm_answer)[:200],
                             "target": None,
                             "kind": "dialogue_answer"},
                    source="dialogue")
                timeline.append(_say_evt)
                causal_learner.record_action(_say_evt)
            except Exception as _tle:
                logger.debug(f"[Experience] 回答 ACTION 入轴跳过: {_tle}")

        # ── 说做一致（Say-Do Gap 修复 2026-09-19）────────────────
        # 病灶：她回答"正在靠近那只僵尸"但没有任何东西驱动 bot——说与做脱钩，
        # 用户看到的就是"只回话不动"。修复：回答里承诺的动作（靠近/过去/
        # 杀掉…）+ 目标实体当前可见 → **真的执行**，并把真实结果追加进回答。
        if (_mc_state and _mc_state.get("connected") and llm_answer
                and not _reflex_fast
                and re.search(r"(靠近|走过去|过去|这就|马上|试试|杀掉|干掉|打死)", llm_answer)):
            try:
                _mc_state = mc["get_state"]() or _mc_state   # 即时刷新（旧快照已过时 40~90s）
                _sd_kill = bool(re.search(r"(杀掉|干掉|打死)", llm_answer))
                _ents = _mc_state.get("nearbyEntities") or []
                _sd_target = None
                # 目标匹配：回答里点名的可见实体 → 解析节点翻译后匹配
                for e in _ents:
                    for nm in (str(e.get("displayName") or ""), str(e.get("name") or "")):
                        if nm and nm in str(llm_answer):
                            _sd_target = str(e.get("name")); break
                    if _sd_target: break
                if _sd_target is None:
                    from minecraft.reflex import translate_entity as _txe
                    for pn in (parsed.get("nodes") or []):
                        p_en = _txe(str(pn))
                        if len(p_en) < 2:
                            continue
                        for e in _ents:
                            if p_en in str(e.get("name") or ""):
                                _sd_target = str(e.get("name")); break
                        if _sd_target: break
                if _sd_target:
                    # 经 ActionManager 执行（承诺的动作与其它动作同一管路）
                    _sd_spec = {"action_type": "attack_entity" if _sd_kill
                                else "navigate_to_entity",
                                "target": _sd_target,
                                "params": {"entity": _sd_target,
                                           "keep_distance": 2.0},
                                "motivation": "user_commitment",
                                "reason": ["用户对话承诺"]}
                    _sd_prop = action_manager.propose(_sd_spec, source="user")
                    _sd_ok = bool(_sd_prop.get("started"))
                    llm_answer = (llm_answer + ("（动身了）" if _sd_ok
                                  else f"（做不到：{_sd_prop.get('reason') or 'unknown'}）"))
                    _mc_action_result = {"success": _sd_ok,
                                         "action": _sd_spec["action_type"],
                                         "target": _sd_target,
                                         "reason": _sd_prop.get("reason")}
                    logger.info(f"[SayDo] 承诺动作已提交: target={_sd_target} "
                                f"kill={_sd_kill} → {_sd_prop}")
                else:
                    # 承诺了动作但目标不在视野：如实修正，不假装
                    if _sd_kill:
                        llm_answer += "（不过我现在没看到目标）"
                        logger.info("[SayDo] 承诺攻击但目标不可见 → 已如实修正回答")
            except Exception as _sde:
                logger.debug(f"[SayDo] 说做一致钩子跳过: {_sde}")

        # -------------------------------------------------
        # 保存 + 确保自动衰减在后台持续运行
        # -------------------------------------------------

        # 认知周期结束：记录本轮结果（Phase 2+ 会在此做奖励/需求/参数结算）
        if _cycle_id:
            try:
                # 观测：周期收尾（end_cycle 会清 cycle 关联，必须先发事件）
                _fas_drives = {}
                try:
                    with kg._lock:
                        for _dn in ("CuriosityDrive", "SocialDrive",
                                    "LearningDrive", "ConsistencyDrive"):
                            _nd = kg.nodes.get(_dn)
                            if _nd is not None:
                                _fas_drives[_dn] = round(
                                    float(getattr(_nd, "activation", 0.0) or 0.0), 3)
                except Exception:
                    pass
                fas_log.get_logger(fas_log.COGNITION).info(
                    "cycle_end", "回合周期结束",
                    outcome="answered" if llm_answer else (_dd.get("decision") or "silent"),
                    answered=bool(llm_answer), behavior=_dd.get("decision"),
                    llm_calls=fas_log.trace_count("llm_calls"),
                    duration_ms=round((time.time() - _turn_t0) * 1000),
                    action=bool(_user_action_result or _mc_action_result),
                    curiosity=bool(curiosity_question), drives=_fas_drives)
            except Exception as _cle:
                logger.warning(f"[NLP] cycle_end 观测记录失败(仅观测,回合正常): {_cle}")
            try:
                internal_state.end_cycle(_cycle_id, outcome={
                    "answered": bool(llm_answer),
                    "curiosity": bool(curiosity_question),
                    "faiss_hits": len(faiss_hits or []),
                })
            except Exception as _ise:
                logger.debug(f"[InternalState] 周期结算失败: {_ise}")

        _save()

        # 回合边界：清空本轮激活来源（此后图内传播统一受注意资源约束，
        # 发出即扣除、随衰减排空——见 docs/activation_model.md）
        try:
            engine.clear_anchors()
        except Exception as _cae:
            logger.debug(f"[Diffusion] 锚点清理跳过: {_cae}")

        # 持续认知循环接管无输入期的图谱活动；请求处理完毕解除 busy 暂停
        cc.set_busy(False)

        # 确保启动时自动扩散处于停止状态，避免前端刷新后后端线程继续运行
        engine.stop_auto()

        # -------------------------------------------------
        # 返回激活节点 + 回答
        # -------------------------------------------------

        # 响应展示改走活跃前沿快照（不再锁内全图扫描）：
        # infrastructure/procedural 是处理管线自身点燃的内部节点
        # （社交互动、LLM回答等），不对外展示——过滤在快照内完成，
        # 语义与旧全图扫描一致（前沿外激活已为 0）。
        _snap_nodes, _snap_edges = engine.active_snapshot()
        active_nodes = [n.to_dict() for n in _snap_nodes]
        active_edges = [e.to_dict() for e in _snap_edges]

        logger.info(
            f"[NLP] 返回 "
            f"{len(active_nodes)} 激活节点"
        )
        # 观测：回复出口（answer 按 log_text_policy 处理，默认截断+哈希）
        fas_log.get_logger(fas_log.INPUT).info(
            "response_sent", "回复已发往来路",
            silence=bool(_silence_chosen), answered=bool(llm_answer),
            answer=fas_log.text(llm_answer) if llm_answer else None,
            answer_len=len(llm_answer or ""),
            active_nodes=len(active_nodes))

        response_data = {
            "success": True,
            "parsed": parsed,
            "graph": {
                "nodes": active_nodes,
                "edges": active_edges
            }
        }

        if _cycle_id:
            response_data["cycle_id"] = _cycle_id
        if _channel != "web":
            response_data["channel"] = _channel
        if parsed.get("assertion_type"):
            response_data["assertion_type"] = parsed["assertion_type"]
        if faiss_hits:
            response_data["faiss_hits"] = faiss_hits

        if llm_answer is not None:
            response_data["answer"] = llm_answer
            response_data["answer_generated"] = True

        # ── 好奇/探索状态信息 ──
        if curiosity_question:
            response_data["curiosity"] = {
                "active": True,
                "question": curiosity_question,
                "trigger_type": curiosity_info.get("trigger_type"),
                "state": "waiting_answer"
            }
        elif curiosity_engine.is_curiosity_active(kg):
            state = curiosity_engine.get_curiosity_state(kg)
            response_data["curiosity"] = {
                "active": True,
                "question": state.get("question"),
                "trigger_type": state.get("trigger_type"),
                "state": "waiting_answer"
            }
        if curiosity_resolved and curiosity_resolved.get("resolved"):
            response_data["curiosity_resolved"] = {
                "resolved": True,
                "question": curiosity_resolved.get("question"),
                "answer": curiosity_resolved.get("answer"),
            }
        elif curiosity_resolved and curiosity_resolved.get("dismissed"):
            # 探索提问被用户忽略/超时：如实上报（兴趣保留，不算被回答）
            response_data["curiosity_resolved"] = {
                "resolved": False,
                "dismissed": True,
                "reason": curiosity_resolved.get("reason"),
                "question": curiosity_resolved.get("question"),
                "target": curiosity_resolved.get("target"),
            }

        # ── 探索调试块（可解释性：signal → drive → 候选 → 裁决全程可查）──
        try:
            _drive_val = _exploration.get("drive")
            if _drive_val == 0.0:
                _cdn = kg.get_node("CuriosityDrive")
                _drive_val = round(float(getattr(_cdn, "activation", 0.0) or 0.0), 3)
            response_data["exploration"] = {
                "signals": curiosity_info.get("signals", []),
                "occasion_factor": curiosity_info.get("occasion_factor"),
                "drive_value": _drive_val,
                "candidates": [
                    {k: c.get(k) for k in ("action", "target", "score",
                                           "factors", "reason")}
                    for c in (_exploration.get("candidates") or [])
                ],
                "selected": (_dd.get("exploration", {}) or {}).get("selected"),
                "inhibition": inhibition_active(),
                "decision": _dd.get("decision"),
                "desire": _dd.get("desire"),
                "question": curiosity_question,
            }
        except Exception as _expe:
            logger.debug(f"[Exploration] 调试块组装失败: {_expe}")

        # ── Cognitive Context 调试视图（语言→编译→决定→状态→激活 全链路可追溯）──
        try:
            response_data["cog_ctx"] = _cog_debug_view(_cog_full)
        except Exception as _cde:
            logger.debug(f"[CognitiveContext] 调试视图组装失败: {_cde}")

        # 认知需求三层调试块（demand/gap/resource/mode/why 全量可查）
        try:
            response_data["cognitive_demand"] = {
                "demand": _demand_analysis["demand"],
                "gap": _demand_analysis["gap"],
                "resource": _demand_analysis["resource"],
                "mode": _demand_analysis["mode"],
                "why": _demand_analysis["why"],
                "legacy_score": _demand_analysis.get("legacy", {}).get("score"),
            }
        except Exception as _cdee:
            logger.debug(f"[CognitiveDemand] 调试块组装失败: {_cdee}")

        # ── 对话日志 ──
        chat_log = get_chat_log()
        chat_log.record(
            user_input=text,
            system_response=llm_answer,
            curiosity=curiosity_question is not None,
            curiosity_question=curiosity_question,
            curiosity_resolved=curiosity_resolved is not None and curiosity_resolved.get("resolved", False),
            curiosity_answer=curiosity_resolved.get("answer") if curiosity_resolved else None,
        )

        # ── 表达事件入图（ExprFB）+ buffer 兼容视图 ──
        # FAS 每轮表达（含选择沉默）成为统一图谱中的事件节点：
        # 表达本身是激活源，"关于话题/涉及行为/经历"边让它成为下一轮
        # 用户输入可以"回应到"的结构；buffer 只是内存镜像（反思兼容）。
        try:
            expression_feedback.record_expression(
                kg, engine, buffer,
                user_text=text,
                answer=None if _llm_answer_failed else llm_answer,
                behavior=classify_behavior(llm_answer),
                context=_behavior_contexts[0],
                context_ids=_behavior_contexts,
                dialogue_act=parsed.get("dialogue_act", ""),
                response_expectation=parsed.get("response_expectation", ""),
                topics=[
                    n if isinstance(n, str) else n.get("id", "")
                    for n in (parsed.get("nodes") or [])[:8]
                ],
                silence_chosen=_silence_chosen,
                cycle_id=_cycle_id,
                config=config)
        except Exception as _ee:
            logger.warning(f"[ExprFB] 表达事件记录失败: {_ee}")

        if _silence_chosen:
            response_data["behavior"] = "silence"
            response_data["response_desire"] = _dd.get("desire")
        elif _dd.get("constraints"):
            response_data["response_constraints"] = _dd["constraints"]
        elif _tendencies:
            response_data["behavior_tendencies"] = _tendencies
        if _web_results:
            response_data["web_search"] = {
                "query": _web_query,
                "count": len(_web_results),
                "results": [
                    {"title": r.get("title", ""), "url": r.get("url", "")}
                    for r in _web_results[:3]
                ],
            }
        if _file_result:
            response_data["file_action"] = _file_result
        if _mc_action_result:
            response_data["mc_action"] = _mc_action_result
        if _mc_session_action:
            # 会话状态机回执（请求端口/连接结果）——前端 Auto 面板依赖此字段
            response_data["mc_session"] = _mc_session_action
        if _eye_result:
            response_data["eye"] = {
                "count": _eye_result.get("count"),
                "salient": [x["text"][:60] for x in _eye_result.get("salient", [])],
            }

        # -------------------------------------------------
        # 5. 记忆草稿回填（提取与沉淀已在 2a-pre.5 完成，先于回答生成）
        # -------------------------------------------------
        memory_draft = _turn_memory_draft
        if (memory_draft and not memory_draft.get("error")
                and (memory_draft.get("nodes") or memory_draft.get("edges"))):
            response_data["memory_draft"] = memory_draft
            response_data["buffer_stats"] = buffer.stats()
            if _auto_added:
                if curiosity_resolved:
                    response_data.setdefault("curiosity_resolved", {})
                    response_data["curiosity_resolved"]["auto_consolidated"] = True
                    response_data["curiosity_resolved"]["added"] = _auto_added

        # ── Self Model Phase 1: 持续性主体模型更新 ──
        # 仅当有 memory_draft 时触发；临时表态通过重要性门槛过滤
        self_update_result = None
        if memory_draft and memory_draft.get("nodes"):
            try:
                with engine._lock:
                    topk_snapshot, _ = engine.get_topk(k=10)
                emotion_ctx = {
                    "active_emotions": [
                        nid for nid, node in kg.nodes.items()
                        if node.extra_attrs.get("type") == "emotion"
                        and node.activation > 0.1
                    ]
                }
                self_update_result = self_updater.process_turn(
                    user_text=text,
                    parsed_nlp=parsed,
                    memory_draft=memory_draft,
                    emotion_context=emotion_ctx,
                    topk_nodes=topk_snapshot,
                )
                if self_update_result and self_update_result.get("updates_applied"):
                    logger.info(
                        f"[SelfModel] 本轮更新: "
                        f"{len(self_update_result['updates_applied'])} applied, "
                        f"{len(self_update_result.get('updates_deferred', []))} deferred"
                    )
            except Exception as _se:
                logger.warning(f"[SelfModel] 更新跳过: {_se}")

        # ── Reflection Phase 2: 反思触发检查 ──
        if not curiosity_question:  # 好奇问题活跃时不触发反思
            try:
                emo_ctx = {
                    "active_emotions": [
                        nid for nid, node in kg.nodes.items()
                        if node.extra_attrs.get("type") == "emotion"
                        and node.activation > 0.1
                    ]
                }
                trigger_type = reflection_engine.check_triggers(emotion_context=emo_ctx)
                if trigger_type:
                    logger.info(f"[Reflection] 触发类型: {trigger_type}，开始反思循环")
                    with engine._lock:
                        topk_snapshot, _ = engine.get_topk(k=15)
                    chat_entries = get_chat_log().recent(n=15)
                    ref_result = reflection_engine.run(
                        trigger=trigger_type,
                        topk_nodes=topk_snapshot,
                        chat_log_entries=chat_entries,
                    )
                    if ref_result and not ref_result.get("skipped"):
                        response_data["reflection"] = {
                            "triggered": True,
                            "reflection_id": ref_result.get("reflection_id"),
                            "trigger": ref_result.get("trigger"),
                            "approved": len(ref_result.get("approved", [])),
                            "pending": len(ref_result.get("pending", [])),
                            "deferred": len(ref_result.get("deferred", [])),
                        }
                        _save()
            except Exception as _re:
                logger.warning(f"[Reflection] 反思循环异常: {_re}")

        # ── Drive System: 认知场状态（Tension→Drive→Network→Modulation）──
        drive_result = drive_evaluator.evaluate()
        if drive_result.get("dominant"):
            response_data["drive_state"] = {
                "dominant": drive_result["dominant"],
                "drives": drive_result["drives"],
                # 升级结构（旧字段不动；新增供前端/调试观测）
                "tensions": drive_result.get("tensions"),
                "networks": drive_result.get("networks"),
                "action_tendencies": drive_result.get("action_tendencies"),
                "modulation": drive_evaluator.field.modulation.snapshot(),
            }
        try:
            logger.info("[CognitiveField] "
                        + drive_evaluator.field.explain_compact())
        except Exception as _xfe:
            logger.warning(f"[CognitiveField] explain_compact 日志失败(纯观测): {_xfe}")
        # 行动 trace（§二十一）：一次行动怎么产生的——候选/来源/胜者
        try:
            _at_cands = {c["behavior"]: {
                "strength": c.get("activation", 0.0),
                "lifecycle": c.get("lifecycle", ""),
                "affordances": {}} for c in (_dd.get("candidates") or [])
                if c.get("kind") in ("behavior", "action_concept")}
            logger.info("[ActionTrace] " + action_space.trace_compact(
                _at_cands, winner=_dd.get("winner_behavior", ""),
                context=(_behavior_contexts or [""])[0]))
        except Exception as _ate:
            logger.warning(f"[ActionTrace] 日志失败(纯观测): {_ate}")

        # ── Action System Phase 4: 认知 Capability 评估 ──
        # Ask 不再独立触发：只有本轮行为竞争真的选出了探索行为，才产生
        # Ask intent（作为竞争结果的图上记录，不是旁路决策）。
        _exp_selected = (_dd.get("exploration", {}) or {}).get("selected")
        _exp_outcome = None
        if _exp_selected:
            _exp_outcome = {
                "action": _exp_selected,
                "target": (_dd.get("exploration", {}) or {}).get("target"),
                "question": curiosity_question,
                "executed": bool(curiosity_question) or bool(_web_results and _exploration.get("target")),
            }
        action_result = action_selector.evaluate(user_input=text,
                                                 exploration=_exp_outcome)
        if action_result.get("selected"):
            selected = action_result["selected"]
            response_data["action_state"] = {
                "selected": selected,
                "intents": action_result.get("intents", []),
            }
            # 持久化选中的 Intent
            action_selector.persist_intent(selected)

        # ── Compound Phrase: 输出帧查询和分类结果 ──
        if frame_queries or compound_result.get("relational_nodes"):
            response_data["compound_phrase"] = {
                "relational_nodes": compound_result.get("relational_nodes", []),
                "lexicalized_nodes": compound_result.get("lexicalized_nodes", []),
                "frame_queries": [
                    {
                        "frame": fq.get("frame"),
                        "attribute_type": fq.get("attribute_type"),
                        "possessor": fq.get("possessor"),
                        "query_description": fq.get("query_description"),
                    }
                    for fq in frame_queries
                ],
                "classifications": {
                    phrase: {
                        "category": c.get("category"),
                        "reason": c.get("reason"),
                    }
                    for phrase, c in compound_result.get("classifications", {}).items()
                },
            }

        if click_data:
            response_data["click"] = click_data

        return jsonify(response_data)

    except Exception as e:

        logger.exception(
            f"[NLP异常] {e}"
        )

        traceback.print_exc()

        # 异常路径也要收周期，避免悬挂的 current_cycle
        try:
            if _cycle_id:
                fas_log.get_logger(fas_log.COGNITION).error(
                    "cycle_end", f"回合周期异常结束: {str(e)[:120]}",
                    outcome="error", error=str(e)[:200])
                internal_state.end_cycle(_cycle_id, outcome={"error": str(e)[:200]})
        except Exception as _e2:
            logger.warning(f"[NLP] 异常路径周期结算失败(cycle 可能悬挂): {_e2}")

        # 异常路径同样解除 CC 的 busy 暂停，防止循环卡死
        cc.set_busy(False)
        # L1-DGR-03:异常路径同样清激活来源,防上一轮来源身份跨轮残留
        try:
            engine.clear_anchors()
        except Exception as _e3:
            logger.warning(f"[NLP] 异常路径清锚失败(激活来源可能残留): {_e3}")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/nlp/logs", methods=["GET"])
def nlp_logs():
    try:
        return jsonify({
            "logs": nlp.get_logs()
        })
    except Exception as e:
        logger.exception(f"[NLP日志异常] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/memory/approve", methods=["POST"])
def approve_memory():
    try:
        data = request.json or {}
        draft = data.get("draft", {})
        if not draft or not draft.get("nodes"):
            return jsonify({"error": "缺少知识图谱草稿"}), 400

        kg = engine.kg
        import time
        added = {"nodes": 0, "edges": 0}
        skipped = []

        assertion_type = draft.get("assertion_type", "semantic")
        default_label = "declarative-episodic" if assertion_type == "episodic" else "declarative-semantic"

        for n in draft.get("nodes", []):
            nid = str(n.get("id", "")).strip()
            if not nid:
                continue
            # 名称归并：与自动巩固路径一致，避免近似重复节点
            _merged = _fuzzy_match_node(nid)
            if _merged != nid:
                skipped.append(f"{nid}→{_merged}")
            nid = _merged
            if nid in kg.nodes:
                skipped.append(nid)
                continue
            label = str(n.get("node_type", "概念"))
            kg_label, kg_space = _node_label_space(label, default_label)
            from graph_model import Node as GNode
            kg.add_node(GNode(id=nid, weight=0.6, label=kg_label, graph_space=kg_space))
            added["nodes"] += 1

        for e in draft.get("edges", []):
            src = str(e.get("src", "")).strip()
            dst = str(e.get("dst", "")).strip()
            if not src or not dst or src == dst:
                continue
            # 边端点同样走归并，避免指向被合并掉的原名称节点
            src = _fuzzy_match_node(src)
            dst = _fuzzy_match_node(dst)
            if src == dst:
                continue
            rel = str(e.get("type", e.get("relation", "关联"))).strip()
            w = float(e.get("weight", 0.6))
            w = max(0.1, min(1.0, w))
            from graph_model import Edge as GEdge
            existing = kg.get_edge(src, dst, rel)
            if existing:
                if w > existing.weight:
                    existing.weight = w
                skipped.append(f"{src}-[{rel}]->{dst}")
                continue
            _edge_ok = kg.add_edge(GEdge(
                src=src, dst=dst, relation=rel, weight=w,
                relation_category=_edge_category(rel)))
            if not _edge_ok:
                # L1-CON-3:add_edge 端点缺失返回 False——标注生命周期
                logger.warning(
                    "[Approve] 边被丢弃(端点可能不在图内): %s-[%s]->%s",
                    src, rel, dst)
                skipped.append(f"{src}-[{rel}]->{dst}")
                continue
            added["edges"] += 1

        # 事件框架收尾：时间锚点 + 父子事件挂接（论文 §3.2/§3.4）
        _wire_event_structure(kg, draft, added, "Approve")

        _save(force=True)

        # 标记 EpisodicBuffer 中对应经历已晋升
        buffer_exp = buffer.find_by_text(data.get("original_text", "") or data.get("text", ""))
        if buffer_exp:
            buffer_exp.importance = min(1.0, buffer_exp.importance + 0.2)
            buffer.promote_to_long_term(buffer_exp)

        logger.info(f"[Memory] 批准: +{added['nodes']}节点 +{added['edges']}边, 跳过{len(skipped)}")
        return jsonify({
            "success": True,
            "added": added,
            "skipped": skipped,
            "buffer_promoted": buffer_exp is not None
        })

    except Exception as e:
        logger.exception(f"[Memory/approve异常] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/replay", methods=["POST"])
def replay_memory():
    """把历史轮次重新过一遍记忆抽取——事故恢复用。

    为什么做成端点而不是独立脚本：记忆写入有一套语义（模糊名称归并、
    事件框架挂接、缓冲区晋升、name→node 索引同步），都封装在
    _auto_consolidate_curiosity_knowledge 里。恢复脚本自己再实现一遍
    等于开第二条写入路径，两条路早晚长歪——恢复必须走在线那条。

    请求: {"turns": [{"time": "...", "source_text": "用户原话"}], "dry_run": false}
    铁律: source_text 必须是**用户**说过的话。她的回复不是用户陈述，
          拿它抽取等于伪造用户经历（这类错误用户已经抓过两次）。
    """
    try:
        data = request.get_json(silent=True) or {}
        turns = data.get("turns") or []
        dry_run = bool(data.get("dry_run"))
        if not isinstance(turns, list) or not turns:
            return jsonify({"error": "turns 为空"}), 400
        if len(turns) > 40:
            return jsonify({"error": f"单次最多 40 轮（收到 {len(turns)}），请分批"}), 400

        results = []
        tot_n = tot_e = 0
        n_ok = n_err = 0
        for t in turns:
            text = str((t or {}).get("source_text") or "").strip()
            stamp = str((t or {}).get("time") or "")
            if not text:
                n_err += 1
                results.append({"time": stamp, "error": "空来源"})
                continue
            # 用该轮的真实时间做抽取锚点：否则"今天/明天"全被解析成服务端当天，
            # 一周的历史记忆会挤在同一天上，情景记忆时间轴就废了
            _t_when = None
            if stamp:
                from datetime import datetime as _dt_replay
                for _fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                             "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                    try:
                        _t_when = _dt_replay.strptime(stamp[:19], _fmt)
                        break
                    except ValueError:
                        continue
                if _t_when is None and stamp:
                    logger.warning(f"[Replay] 时间戳格式无法解析({stamp[:30]})，按无锚点重放")
            try:
                with engine._lock:
                    topk_check, _ = engine.get_topk(k=20)
                    _focus_events = _get_focus_context(kg)
                draft = nlp.extract_assertion_graph(
                    text, topk_check, focus_events=_focus_events, now=_t_when)
            except Exception as e:
                n_err += 1
                results.append({"time": stamp, "error": f"{type(e).__name__}: {str(e)[:120]}"})
                continue
            if not draft or draft.get("error"):
                n_err += 1
                results.append({"time": stamp, "error": str((draft or {}).get("error", "空草稿"))[:120]})
                continue

            nodes = draft.get("nodes") or []
            edges = draft.get("edges") or []
            added = None
            if not dry_run and (nodes or edges):
                buffer.add_experience(
                    raw_text=text,
                    nodes=nodes,
                    edges=edges,
                    assertion_type=draft.get("assertion_type", "episodic"),
                )
                buffer.get_promotion_candidates()
                added = _auto_consolidate_curiosity_knowledge(kg, draft, buffer, text)
            n_ok += 1
            if added:
                tot_n += added.get("nodes", 0)
                tot_e += added.get("edges", 0)
            results.append({
                "time": stamp,
                "source_text": text[:120],
                "assertion_type": draft.get("assertion_type", "episodic"),
                "draft_nodes": len(nodes),
                "draft_edges": len(edges),
                "added_nodes": (added or {}).get("nodes", 0),
                "added_edges": (added or {}).get("edges", 0),
                "node_ids": [str((x.get("id") if isinstance(x, dict) else x))[:40]
                             for x in nodes[:6]],
            })

        if not dry_run and tot_n:
            _save(force=True)
        logger.info(
            f"[Memory/Replay] {'干跑' if dry_run else '写入'}: {n_ok} 轮成功, {n_err} 轮失败, "
            f"+{tot_n}节点 +{tot_e}边")
        return jsonify({
            "dry_run": dry_run,
            "processed": n_ok,
            "failed": n_err,
            "added_nodes": tot_n,
            "added_edges": tot_e,
            "graph_nodes": len(kg.nodes),
            "graph_edges": len(kg.edges),
            "results": results,
        })
    except Exception as e:
        logger.exception(f"[Memory/Replay异常] {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# EpisodicBuffer API
# =========================================================

# ── B7/§24 只读观测端点（学习链在运行时可查询；薄函数、零写入、零 LLM）──

@app.route("/api/debug/causal", methods=["GET"])
def debug_causal():
    """因果学习现状：归因窗、候选、聚合、假设、已晋升（§26 事实路径）。"""
    try:
        return jsonify({"ok": True, "report": causal_learner.debug_report(
            n=int(request.args.get("n", 12)))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/debug/curiosity", methods=["GET"])
def debug_curiosity():
    """好奇账本现状：活跃未知的兴趣值 + 经验习得的物种（§六/§十五）。"""
    try:
        from graph_model import is_live_unknown_id
        unknowns = []
        with kg._lock:
            ids = [nid for nid in kg.nodes
                   if is_live_unknown_id(kg, nid)]
            for nid in sorted(ids)[:40]:
                it = curiosity_engine.get_interest(kg, nid) or {}
                unknowns.append({"node": nid, "level": it.get("level", 0.0),
                                 "ask": it.get("ask_count", 0),
                                 "resolved": it.get("resolved_count", 0)})
            species = [{"node": nid,
                        "observed": (kg.nodes[nid].extra_attrs or {}).get(
                            "observed_count", 1)}
                       for nid in kg.nodes
                       if (kg.nodes[nid].extra_attrs or {}).get("type") == "mc_species"
                       and (kg.nodes[nid].extra_attrs or {}).get("source") == "experience"]
        return jsonify({"ok": True, "live_unknowns": unknowns,
                        "learned_species": species,
                        # 真机观察（2026-09-23）：活缺口清单 + 背包刷新节拍
                        "open_gaps": __import__(
                            "prior_knowledge").open_gaps(kg),
                        "inv_debug": {
                            "inv_count": getattr(mc_embodiment, "_inv_count", None),
                            "inv_every": getattr(mc_embodiment, "_inv_every", None),
                            "last_inv_n": len(mc_embodiment._last_inv or [])
                            if getattr(mc_embodiment, "_last_inv", None) is not None else None}})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/debug/world", methods=["GET"])
def debug_world():
    """世界事件流现状：时间轴近事 + 开着的归因窗数（§七/§九只读）。"""
    try:
        evs = [{"ts": e.get("ts_str"), "type": e.get("event_type"),
                "actor": e.get("actor"), "subject": e.get("subject"),
                "change": (e.get("content") or {}).get("change")}
               for e in timeline.recent(n=int(request.args.get("n", 25)))]
        return jsonify({"ok": True, "recent_events": evs,
                        "pending_windows": len(causal_learner._pending)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/debug/drives", methods=["GET"])
def debug_drives():
    """驱动力/内部状态现状：Drive 激活 + 张力读数 + 需求（§十八可观测）。"""
    try:
        drives = {}
        with kg._lock:
            for nid, n in kg.nodes.items():
                if (n.extra_attrs or {}).get("type") == "drive":
                    drives[nid] = {"activation": round(n.activation, 3),
                                   "status": (n.extra_attrs or {}).get("status")}
        tensions = {}
        try:
            # field.tensions.raws() = {张力名: {来源名: {raw/note/contrib}}}
            tensions = drive_evaluator.field.tensions.raws()
        except Exception as _dre:
            logger.warning(f"[Debug] 张力详情读取失败(tensions 留空): {_dre}")
        return jsonify({"ok": True, "drives": drives,
                        "needs": internal_state.snapshot().get("needs"),
                        "tensions": tensions})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/debug/capability", methods=["GET"])
def debug_capability():
    """能力经验回流现状：按概念/能力节点读经验摘要（§14 只读派生）。"""
    try:
        node = request.args.get("node", "")
        if not node:
            caps = [nid for nid, n in kg.nodes.items()
                    if (n.extra_attrs or {}).get("type") == "capability"][:30]
            return jsonify({"ok": True, "capabilities": caps,
                            "hint": "?node=能力:采集 或概念名"})
        return jsonify({"ok": True,
                        "summary": cap_index.experience_summary(node,
                                                                causal=causal_learner)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/buffer", methods=["GET"])
def get_buffer():
    try:
        keyword = request.args.get("keyword", "").strip()
        n = int(request.args.get("n", 20))
        if keyword:
            related = buffer.retrieve_related(keyword, n=n)
            return jsonify({
                "stats": buffer.stats(),
                "related": related
            })
        return jsonify(buffer.get_all())
    except Exception as e:
        logger.exception(f"[Buffer/GET异常] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/buffer/promote", methods=["POST"])
def promote_buffer():
    try:
        data = request.json or {}
        text = (data.get("original_text", "") or data.get("text", "")).strip()

        if text:
            exp = buffer.find_by_text(text)
            if exp:
                buffer.promote_to_long_term(exp)
                return jsonify({"success": True, "promoted": exp.to_dict()})
            return jsonify({"success": False, "error": "未找到匹配经历"}), 404

        candidates = buffer.get_promotion_candidates()
        for c in candidates:
            buffer.promote_to_long_term(c)
        return jsonify({
            "success": True,
            "auto_promoted": len(candidates),
            "candidates": [c.to_dict() for c in candidates]
        })
    except Exception as e:
        logger.exception(f"[Buffer/Promote异常] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/buffer/clear", methods=["POST"])
def clear_buffer():
    try:
        buffer.clear()
        return jsonify({"success": True, "stats": buffer.stats()})
    except Exception as e:
        logger.exception(f"[Buffer/Clear异常] {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# Embedding Debug API
# =========================================================

@app.route("/api/embedding/debug", methods=["POST"])
def embedding_debug():
    try:
        data = request.json or {}
        query = data.get("query", "").strip()
        top_k = int(data.get("top_k", 10))
        if not query:
            return jsonify({"error": "缺少查询文本"}), 400

        result = emb_mgr.search_debug(query, top_k=top_k)
        result["stats"] = emb_mgr.stats()
        result["ready"] = emb_mgr.ready()
        return jsonify(result)
    except Exception as e:
        logger.exception(f"[Embedding/Debug异常] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/embedding/stats", methods=["GET"])
def embedding_stats():
    try:
        return jsonify(emb_mgr.stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/embedding/rebuild", methods=["POST"])
def embedding_rebuild():
    try:
        emb_mgr.build_index(engine.kg)
        return jsonify({"success": True, "stats": emb_mgr.stats()})
    except Exception as e:
        logger.exception(f"[Embedding/Rebuild异常] {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# 扩散
# =========================================================

@app.route("/api/diffuse/step", methods=["POST"])
def diffuse_step():

    logger.info(
        "[API] diffuse_step"
    )

    try:
        engine.decay_step()
        engine.diffuse_step()
        _save()

        return jsonify({
            "success": True,
            "message": "扩散一步完成",
            "graph": _active_graph_dict(0.0)
        })

    except Exception as e:

        logger.exception(
            f"[扩散异常] {e}"
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/diffuse/round", methods=["POST"])
def diffuse_round():

    logger.info(
        "[API] diffuse_round"
    )

    try:
        info = engine.diffuse_round()
        _save()

        return jsonify({
            "success": True,
            "message": "扩散轮完成",
            "info": info,
            "graph": _active_graph_dict(0.0)
        })

    except Exception as e:

        logger.exception(
            f"[扩散轮异常] {e}"
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/diffuse/start", methods=["POST"])
def diffuse_start():
    try:
        started = engine.start_auto()
        return jsonify({
            "success": True,
            "started": started
        })
    except Exception as e:
        logger.exception(f"[AutoStart异常] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/diffuse/stop", methods=["POST"])
def diffuse_stop():
    try:
        stopped = engine.stop_auto()
        return jsonify({
            "success": True,
            "stopped": stopped
        })
    except Exception as e:
        logger.exception(f"[AutoStop异常] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/diffuse/status", methods=["GET"])
def diffuse_status():
    """Get current auto-diffusion status."""
    try:
        return jsonify({
            "running": engine._running,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# top-k
# =========================================================

@app.route("/api/topk", methods=["GET"])
def get_topk():

    nodes, edges = engine.get_topk()

    def node_to_dict(n):
        return {
            "id": n.id,
            "name": n.id,
            "activation": n.activation,
            "label": n.label
        }

    def edge_to_dict(e):
        return {
            "src": e.src,
            "dst": e.dst,
            "type": e.relation,
            "weight": e.weight,
            "activation": e.activation
        }

    raw = {
        "nodes": [node_to_dict(n) for n in nodes],
        "edges": [edge_to_dict(e) for e in edges]
    }
    filtered = _apply_temp_filter(raw)
    return jsonify(filtered)

# =========================================================
# Action Queue
# =========================================================

@app.route("/api/actions/queue", methods=["GET"])
def action_queue():
    try:
        if hasattr(engine, "_refresh_action_queue"):
            # 只在自动扩散运行中才刷新行动队列，避免无意义的日志输出
            if engine._running:
                engine._refresh_action_queue()
        else:
            logger.warning("_refresh_action_queue 不存在")

            return jsonify({
                "queue": [],
                "warning":
                    "_refresh_action_queue 不存在"
            })

        queue_data = [

            {
                "activation": round(a, 4),
                "node_id": nid
            }

            for a, nid in engine.action_queue
        ]

        logger.info(
            f"[ActionQueue] "
            f"{len(queue_data)} 个动作"
        )

        return jsonify({
            "queue": queue_data
        })

    except Exception as e:

        logger.exception(
            f"[ActionQueue异常] {e}"
        )

        return jsonify({
            "error": str(e)
        }), 500

# =========================================================
# Execute Action
# =========================================================

@app.route("/api/actions/execute", methods=["POST"])
def execute_action():

    logger.info(
        "[API] execute_action"
    )

    try:

        data = request.json or {}

        node_id = data.get("node_id")

        # 自动选择

        if not node_id:

            logger.info(
                "[Action] 自动选择动作"
            )

            if hasattr(
                    engine,
                    "_refresh_action_queue"
            ):

                engine._refresh_action_queue()

            if getattr(
                    engine,
                    "action_queue",
                    None
            ):

                node_id = (
                    engine.action_queue[0][1]
                )

            else:

                logger.warning(
                    "[Action] 队列为空"
                )

                return jsonify({
                    "error": "行动队列为空"
                }), 400

        logger.info(
            f"[Action] 执行: {node_id}"
        )

        result = engine.execute_action(
            node_id
        )

        logger.info(
            f"[Action] 结果: {result}"
        )

        # Everything is Graph: Inject emotion based on execution result
        if result.get("success"):
            inject_emotion(kg, f"程序执行: {node_id}", "开心")
        else:
            inject_emotion(kg, f"程序失败: {node_id}", "沮丧")

        _save()

        return jsonify(result)

    except Exception as e:

        logger.exception(
            f"[Action异常] {e}"
        )

        return jsonify({
            "error": str(e)
        }), 500

# =========================================================
# Action Log
# =========================================================

@app.route("/api/actions/log", methods=["GET"])
def action_log():

    logger.info(
        "[API] action_log"
    )

    return jsonify({
        "log": engine.execution_log
    })


# =========================================================
# Everything is Graph: Self API
# =========================================================

@app.route("/api/self", methods=["GET"])
def get_self():
    """Get Self node state snapshot from graph (not database query)."""
    try:
        state = get_self_state(kg)
        return jsonify({"success": True, "state": state})
    except Exception as e:
        logger.exception(f"[Self API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/diffuse", methods=["POST"])
def self_diffuse():
    """Diffuse from Self node - natural emergence of answers."""
    try:
        data = request.json or {}
        topic = data.get("topic", None)
        k = int(data.get("k", 15))
        result = self_diffusion_for_answer(kg, topic=topic, k=k)
        nodes_out = []
        if result.get("self_node"):
            sn = result["self_node"]
            nodes_out.append({"id": sn.id, "activation": sn.activation,
                "label": sn.label, "weight": sn.weight,
                "confidence": getattr(sn, "confidence", 0.5)})
        for n in result.get("top_nodes", []):
            nodes_out.append({"id": n.id, "activation": n.activation,
                "label": n.label, "weight": n.weight,
                "confidence": getattr(n, "confidence", 0.5)})
        return jsonify({"success": True, "self_diffusion": {
            "nodes": nodes_out,
            "edges": [e.to_dict() for e in result.get("edges", [])],
            "current_emotion": result.get("current_emotion"),
            "preferences": result.get("preferences", []),
            "goals": result.get("goals", []),
        }})
    except Exception as e:
        logger.exception(f"[Self Diffuse] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/preference", methods=["POST"])
def self_preference():
    """Set or adjust preference (preference = edge weight)."""
    try:
        data = request.json or {}
        target = str(data.get("target", "")).strip()
        action = data.get("action", "set")
        if not target:
            return jsonify({"error": "missing target"}), 400
        if action == "adjust":
            delta = float(data.get("delta", 0.01))
            adjust_preference(kg, target, delta)
            return jsonify({"success": True, "target": target, "delta": delta})
        else:
            weight = float(data.get("weight", 0.25))
            set_preference(kg, target, weight)
            return jsonify({"success": True, "target": target, "weight": weight})
    except Exception as e:
        logger.exception(f"[Preference API] {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# Search


@app.route("/api/self/emotion", methods=["POST"])
def self_emotion():
    """Inject emotion into graph: event -> emotion -> Self."""
    try:
        data = request.json or {}
        event = str(data.get("event", "")).strip()
        emotion_type = str(data.get("emotion", "")).strip()
        if not event or not emotion_type:
            return jsonify({"error": "missing event or emotion"}), 400
        inject_emotion(kg, event, emotion_type)
        return jsonify({"success": True, "event": event, "emotion": emotion_type})
    except Exception as e:
        logger.exception(f"[Emotion API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/goal", methods=["POST"])
def self_goal():
    """Create goal node connected to Self."""
    try:
        data = request.json or {}
        goal_desc = str(data.get("goal", "")).strip()
        parent = data.get("parent", None)
        if not goal_desc:
            return jsonify({"error": "missing goal"}), 400
        set_goal(kg, goal_desc, parent_goal=parent)
        return jsonify({"success": True, "goal": goal_desc})
    except Exception as e:
        logger.exception(f"[Goal API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/thought", methods=["POST"])
def self_thought():
    """Trigger internal thought generation."""
    try:
        thought_id = engine.generate_thought(nlp_processor=nlp)
        if thought_id:
            _save()
            return jsonify({"success": True, "thought_id": thought_id})
        return jsonify({"success": False, "message": "No active nodes"})
    except Exception as e:
        logger.exception(f"[Thought API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/mc/status", methods=["GET"])
def mc_status():
    """Minecraft 桥状态：Haru 在游戏中的实时状态。"""
    try:
        st = mc["get_state"]()
        return jsonify({"success": True, "state": st})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mc/say", methods=["POST"])
def mc_say_ep():
    """让 Haru 在游戏内聊天。"""
    try:
        d = request.json or {}
        ok = mc["say"](str(d.get("text", ""))[:240])
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/action/state", methods=["GET"])
def action_state_ep():
    """Action 节点系统状态：当前动作（承诺期）/目标队列/最近结算/统计。"""
    try:
        return jsonify({"success": True, "action": action_manager.status(),
                        "autonomy": autonomy.state()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/skills", methods=["GET"])
def skills_catalog_ep():
    """技能库目录（FAS 当前拥有的具身能力清单）。"""
    try:
        return jsonify({"success": True,
                        "skills": mc_embodiment.skill_catalog()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proactive/poll", methods=["GET"])
def proactive_poll():
    """前端轮询：取走 CC 决定表达的主动消息（最后一公里通道）。"""
    try:
        return jsonify({"success": True, "messages": cc.poll()})
    except Exception as e:
        logger.exception(f"[Proactive] {e}")
        return jsonify({"error": str(e)}), 500


# =========================================================
# 认知调节 API（锁 / 状态检测器 / 触发器）
# 约定与其它端点一致：成功 {"success": True, ...}，失败 {"error": ...} + 状态码。
# 三个模块各自原子落盘（json_store），不引入新框架/新数据库。
# =========================================================

@app.route("/api/regulation/state", methods=["GET"])
def regulation_state():
    """总览：锁 + 检测器 + 触发器 + 最近的认知事件。"""
    try:
        return jsonify({"success": True, **regulation.state()})
    except Exception as e:
        logger.exception(f"[RegulationState] {e}")
        return jsonify({"error": str(e)}), 500


# ── 锁 ────────────────────────────────────────────────────

@app.route("/api/regulation/locks", methods=["GET"])
def regulation_locks_list():
    try:
        target_id = request.args.get("target_id") or None
        target_type = request.args.get("target_type") or None
        enabled_only = request.args.get("enabled_only") in ("1", "true", "yes")
        return jsonify({"success": True,
                        "locks": regulation.locks.list(
                            target_id=target_id, target_type=target_type,
                            enabled_only=enabled_only)})
    except Exception as e:
        logger.exception(f"[LockList] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/locks", methods=["POST"])
def regulation_locks_create():
    """建锁。body：target_type, target_id, lock_type, scope, reason, expires_at..."""
    try:
        data = request.json or {}
        if not str(data.get("target_id", "")).strip():
            return jsonify({"error": "target_id 不能为空"}), 400
        result = regulation.create_lock(
            target_type=data.get("target_type", "node"),
            target_id=data.get("target_id"),
            lock_type=data.get("lock_type", "blocking"),
            scope=data.get("scope", "diffusion"),
            reason=data.get("reason", ""),
            source=data.get("source", "manual"),
            priority=int(data.get("priority", 0) or 0),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata") or {},
        )
        if not result.get("ok"):
            return jsonify({"error": result.get("error")}), 400
        return jsonify({"success": True, "lock": result["lock"]})
    except Exception as e:
        logger.exception(f"[LockCreate] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/locks/check", methods=["GET"])
def regulation_locks_check():
    """锁状态查询：某对象当前是否被阻塞/被隐藏，以及生效的锁与原因。"""
    try:
        target_type = request.args.get("target_type", "node")
        target_id = request.args.get("target_id", "")
        if not target_id:
            return jsonify({"error": "target_id 不能为空"}), 400
        return jsonify({"success": True,
                        **regulation.locks.effective_for(target_type, target_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/locks/<lock_id>", methods=["GET"])
def regulation_locks_get(lock_id):
    try:
        lock = regulation.locks.get(lock_id)
        if lock is None:
            return jsonify({"error": f"锁不存在: {lock_id}"}), 404
        return jsonify({"success": True, "lock": lock})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/locks/<lock_id>", methods=["PUT"])
def regulation_locks_update(lock_id):
    """更新锁（含启用/禁用：body.enabled）。删除锁用 DELETE。"""
    try:
        data = request.json or {}
        fields = {k: data[k] for k in
                  ("lock_type", "scope", "reason", "source", "priority",
                   "expires_at", "metadata", "enabled") if k in data}
        result = regulation.update_lock(lock_id, **fields)
        if not result.get("ok"):
            code = 404 if "不存在" in str(result.get("error")) else 400
            return jsonify({"error": result.get("error")}), code
        return jsonify({"success": True, "lock": result["lock"]})
    except Exception as e:
        logger.exception(f"[LockUpdate] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/locks/<lock_id>", methods=["DELETE"])
def regulation_locks_delete(lock_id):
    """只删锁，不删被锁对象。"""
    try:
        result = regulation.delete_lock(lock_id)
        if not result.get("ok"):
            return jsonify({"error": result.get("error")}), 404
        return jsonify({"success": True, "deleted": result["deleted"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 话题终止抑制期（§18.2 #10：锁承载 + 显式解除 API）──────

@app.route("/api/regulation/termination", methods=["GET"])
def regulation_termination_get():
    """当前抑制状态：active / lock_id / expires_at / remaining_s。"""
    try:
        return jsonify({"success": True, **termination_state()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/termination/release", methods=["POST"])
def regulation_termination_release():
    """显式解除抑制：浮点通道清零 + 锁置 disabled 并进审计。"""
    try:
        reason = ""
        if request.is_json and isinstance(request.get_json(silent=True), dict):
            reason = str(request.get_json(silent=True).get("reason") or "")
        out = release_termination(reason or "api")
        return jsonify({"success": bool(out.get("ok")),
                        "result": out, "state": termination_state()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 状态检测器 ────────────────────────────────────────────

@app.route("/api/regulation/monitors", methods=["GET"])
def regulation_monitors_list():
    try:
        return jsonify({"success": True, "monitors": regulation.monitors.list(),
                        "recent_events": regulation.monitors.recent_events(20)})
    except Exception as e:
        logger.exception(f"[MonitorList] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/monitors", methods=["POST"])
def regulation_monitors_create():
    try:
        data = request.json or {}
        result = regulation.monitors.create(
            monitor_id=data.get("id") or None,
            name=data.get("name", ""),
            description=data.get("description", ""),
            target_type=data.get("target_type", "external"),
            target_id=data.get("target_id", ""),
            state_type=data.get("state_type", "number"),
            unit=data.get("unit", ""),
            mode=data.get("mode", "event"),
            poll_interval_s=float(data.get("poll_interval_s", 0) or 0),
            enabled=bool(data.get("enabled", True)),
            schema=data.get("schema") or {},
            epsilon=data.get("epsilon"),
            metadata=data.get("metadata") or {},
        )
        if not result.get("ok"):
            return jsonify({"error": result.get("error")}), 400
        return jsonify({"success": True, "monitor": result["monitor"]})
    except Exception as e:
        logger.exception(f"[MonitorCreate] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/monitors/<monitor_id>", methods=["GET"])
def regulation_monitors_get(monitor_id):
    try:
        mon = regulation.monitors.get(monitor_id)
        if mon is None:
            return jsonify({"error": f"检测器不存在: {monitor_id}"}), 404
        return jsonify({"success": True, "monitor": mon})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/monitors/<monitor_id>", methods=["PUT"])
def regulation_monitors_update(monitor_id):
    try:
        data = request.json or {}
        fields = {k: data[k] for k in
                  ("name", "description", "target_type", "target_id", "state_type",
                   "unit", "schema", "epsilon", "mode", "poll_interval_s",
                   "enabled", "metadata") if k in data}
        result = regulation.monitors.update(monitor_id, **fields)
        if not result.get("ok"):
            code = 404 if "不存在" in str(result.get("error")) else 400
            return jsonify({"error": result.get("error")}), code
        return jsonify({"success": True, "monitor": result["monitor"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/monitors/<monitor_id>", methods=["DELETE"])
def regulation_monitors_delete(monitor_id):
    try:
        result = regulation.monitors.delete(monitor_id)
        if not result.get("ok"):
            return jsonify({"error": result.get("error")}), 404
        return jsonify({"success": True, "deleted": result["deleted"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/monitors/<monitor_id>/observe", methods=["POST"])
def regulation_monitors_observe(monitor_id):
    """提交一次状态观测（事件驱动入口）：body {value, source?, confidence?}。

    状态未变化时不产生事件（去重）；变化时自动进入触发器条件评估。
    """
    try:
        data = request.json or {}
        if "value" not in data:
            return jsonify({"error": "缺少 value"}), 400
        result = regulation.submit_state(
            monitor_id, data["value"],
            source=data.get("source", "event"),
            confidence=data.get("confidence"))
        if not result.get("ok"):
            code = 404 if "不存在" in str(result.get("error", "")) else 400
            return jsonify({"error": result.get("error")}), code
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.exception(f"[MonitorObserve] {e}")
        return jsonify({"error": str(e)}), 500


# ── 触发器 ────────────────────────────────────────────────

@app.route("/api/regulation/triggers", methods=["GET"])
def regulation_triggers_list():
    try:
        return jsonify({"success": True, "triggers": regulation.triggers.list(),
                        "recent_log": regulation.triggers.recent_log(20)})
    except Exception as e:
        logger.exception(f"[TriggerList] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/triggers", methods=["POST"])
def regulation_triggers_create():
    """建触发器。condition 必须是结构化条件（见 cognitive_triggers.py 顶部）。"""
    try:
        data = request.json or {}
        result = regulation.triggers.create(
            trigger_id=data.get("id") or None,
            name=data.get("name", ""),
            description=data.get("description", ""),
            monitor_id=data.get("monitor_id", "*"),
            event_types=data.get("event_types") or ["state_changed"],
            condition=data.get("condition") or {},
            action=data.get("action") or {"type": "record"},
            fire_mode=data.get("fire_mode", "edge"),
            cooldown_s=float(data.get("cooldown_s", 0) or 0),
            priority=int(data.get("priority", 0) or 0),
            once=bool(data.get("once", False)),
            llm_mode=int(data.get("llm_mode", 0) or 0),
            enabled=bool(data.get("enabled", True)),
            metadata=data.get("metadata") or {},
        )
        if not result.get("ok"):
            return jsonify({"error": result.get("error")}), 400
        return jsonify({"success": True, "trigger": result["trigger"]})
    except Exception as e:
        logger.exception(f"[TriggerCreate] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/triggers/log", methods=["GET"])
def regulation_triggers_log():
    """触发记录（含被冷却/once 拒绝的条目——拒绝也要可见，不静默）。"""
    try:
        n = int(request.args.get("n", 50) or 50)
        return jsonify({"success": True,
                        "log": regulation.triggers.recent_log(max(1, min(200, n)))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/triggers/parse", methods=["POST"])
def regulation_triggers_parse():
    """Mode 1：自然语言 → 候选触发器定义（不落盘，人工确认后再 POST 创建）。

    这是 LLM 在认知调节里唯一被允许出现的位置（创建期一次性解析）；
    运行时的条件评估永远是确定性的。
    """
    try:
        data = request.json or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "缺少 text"}), 400

        def _llm(prompt):
            return nlp.llm.invoke(prompt).content

        result = regulation.parse_trigger_text(text, _llm)
        if not result.get("ok"):
            return jsonify({"error": result.get("error"),
                            "raw": result.get("raw")}), 400
        return jsonify({"success": True, "candidate": result["candidate"]})
    except Exception as e:
        logger.exception(f"[TriggerParse] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/triggers/<trigger_id>", methods=["GET"])
def regulation_triggers_get(trigger_id):
    try:
        trig = regulation.triggers.get(trigger_id)
        if trig is None:
            return jsonify({"error": f"触发器不存在: {trigger_id}"}), 404
        return jsonify({"success": True, "trigger": trig})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/triggers/<trigger_id>", methods=["PUT"])
def regulation_triggers_update(trigger_id):
    try:
        data = request.json or {}
        fields = {k: data[k] for k in
                  ("name", "description", "monitor_id", "event_types", "condition",
                   "fire_mode", "cooldown_s", "priority", "once", "action",
                   "llm_mode", "enabled", "metadata") if k in data}
        result = regulation.triggers.update(trigger_id, **fields)
        if not result.get("ok"):
            code = 404 if "不存在" in str(result.get("error")) else 400
            return jsonify({"error": result.get("error")}), code
        return jsonify({"success": True, "trigger": result["trigger"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/triggers/<trigger_id>", methods=["DELETE"])
def regulation_triggers_delete(trigger_id):
    try:
        result = regulation.triggers.delete(trigger_id)
        if not result.get("ok"):
            return jsonify({"error": result.get("error")}), 404
        return jsonify({"success": True, "deleted": result["deleted"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulation/events", methods=["GET"])
def regulation_events():
    """最近的认知事件（触发器产物）。llm_mode_min 过滤 LLM 介入等级。"""
    try:
        n = int(request.args.get("n", 10) or 10)
        mode_min = int(request.args.get("llm_mode_min", 0) or 0)
        return jsonify({"success": True,
                        "events": regulation.recent_cognitive_events(n, mode_min)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# 自主认知与行动 API（Phase D）
# 约定与其它端点一致：成功 {"success": True, ...}，失败 {"error": ...} + 状态码。
# 前端只读"真实后端状态"：意图/得分/依据/结果全部来自 AutonomousLoop.state()，
# 不在界面上编造"她正在思考"。
# =========================================================

@app.route("/api/autonomy/state", methods=["GET"])
def autonomy_state():
    """自主模式、当前意图/目标/行动、候选与评分、动机摘要、最近日志。"""
    try:
        return jsonify({"success": True, **autonomy.state(),
                        "candidates": autonomy.state().get("candidates", []),
                        "log": autonomy.recent_log(20),
                        "history": autonomy.candidate_history(5)})
    except Exception as e:
        logger.exception(f"[AutonomyState] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/autonomy/mode", methods=["POST"])
def autonomy_mode():
    """开启/关闭自主行动：body {"mode": "on"|"off"}。"""
    try:
        data = request.json or {}
        result = autonomy.set_mode(data.get("mode", "off"))
        if not result.get("ok"):
            return jsonify({"error": result.get("error")}), 400
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/autonomy/pause", methods=["POST"])
def autonomy_pause():
    """暂停自主循环（保留模式，不产生新动作）。"""
    try:
        data = request.json or {}
        return jsonify({"success": True, **autonomy.pause(data.get("reason") or "手动暂停")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/autonomy/resume", methods=["POST"])
def autonomy_resume():
    try:
        return jsonify({"success": True, **autonomy.resume()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/autonomy/stop", methods=["POST"])
def autonomy_stop():
    """停止自主行动 = 关闭模式 + 取消在飞动作（最彻底的"停手"）。"""
    try:
        result = autonomy.set_mode("off")
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/autonomy/step", methods=["POST"])
def autonomy_step():
    """手动跑一次自主决策（观察/调试用；与 tick 同一条确定性路径）。"""
    try:
        return jsonify({"success": True, "result": autonomy.tick()})
    except Exception as e:
        logger.exception(f"[AutonomyStep] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/autonomy/goals", methods=["GET"])
def autonomy_goals():
    try:
        return jsonify({"success": True, "goals": autonomy.goals()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/autonomy/goals", methods=["POST"])
def autonomy_goals_create():
    """加一个待办目标。两种方式：
      结构化：{"type": "collect", "target": "木头", "params": {...}}
      Mode 1：{"text": "去砍点木头"} → 由 LLM 转成结构化目标（预算门控）
    """
    try:
        data = request.json or {}
        if data.get("type"):
            result = autonomy.add_goal({"type": data["type"],
                                        "target": data.get("target", ""),
                                        "params": data.get("params") or {},
                                        "text": data.get("text", "")})
            if not result.get("ok"):
                return jsonify({"error": result.get("error")}), 400
            return jsonify({"success": True, **result})
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "需要 type 或 text"}), 400

        def _llm(prompt):
            return nlp.llm.invoke(prompt).content

        parsed = autonomy.goal_from_text(text, _llm)
        if not parsed.get("ok"):
            return jsonify({"error": parsed.get("error"),
                            "raw": parsed.get("raw")}), 400
        cand = parsed["candidate"]
        result = autonomy.add_goal({"type": cand.get("type"),
                                    "target": cand.get("target", ""),
                                    "params": cand.get("params") or {},
                                    "text": cand.get("text", "") or text})
        return jsonify({"success": True, "parsed": cand, **result})
    except Exception as e:
        logger.exception(f"[AutonomyGoals] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/autonomy/goals/<int:index>", methods=["DELETE"])
def autonomy_goals_delete(index):
    try:
        goals = autonomy.goals()
        if index < 0 or index >= len(goals):
            return jsonify({"error": f"目标索引越界: {index}"}), 404
        goals.pop(index)
        autonomy.clear_goals()
        for g in goals:
            autonomy.add_goal({k: v for k, v in g.items() if k != "created"})
        return jsonify({"success": True, "goals": autonomy.goals()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/autonomy/log", methods=["GET"])
def autonomy_log():
    try:
        n = int(request.args.get("n", 30) or 30)
        return jsonify({"success": True, "log": autonomy.recent_log(max(1, min(100, n))),
                        "history": autonomy.candidate_history(10)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# 内部状态 API（需求 / 神经调制 / 性格）
# 约定与其它端点一致：成功 {"success": True, ...}，失败 {"error": ...} + 状态码。
# 写操作一律经 InternalState 的唯一入口（apply_delta/set_value），并留痕。
# =========================================================

# =========================================================
# 对话通道 API（世界里的文字聊天 / 外部桥接）
# 通用：任何注册进来的通道都能查询状态、启停、手工发一句测试。
# 认知逻辑不在这一层——通道只负责收发，回合走同一条 /api/nlp 管线。
# =========================================================

@app.route("/api/channels/state", methods=["GET"])
def channels_state():
    """通道清单 + 收发统计 + 最近事件 + 世界里他人说过的话。"""
    try:
        st = channel_hub.state()
        for ch in st.get("channels", []):
            adapter = channel_hub._channels.get(ch["id"])
            if adapter is not None and hasattr(adapter, "state"):
                try:
                    ch["adapter"] = adapter.state()
                except Exception:
                    ch["adapter"] = {}
        return jsonify({"success": True, **st})
    except Exception as e:
        logger.exception(f"[ChannelsState] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/channels/<channel_id>", methods=["POST"])
def channels_toggle(channel_id):
    """启用/停用某个通道：body {"enabled": true|false}。"""
    try:
        data = request.json or {}
        if "enabled" not in data:
            return jsonify({"error": "缺少 enabled"}), 400
        result = channel_hub.set_enabled(channel_id, bool(data["enabled"]))
        if not result.get("ok"):
            return jsonify({"error": result.get("error")}), 404
        config[f"channel_{channel_id}"] = bool(data["enabled"])
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/channels/<channel_id>/send", methods=["POST"])
def channels_send(channel_id):
    """往通道里发一句（测试用/人工代发）：body {"text": "..."}。"""
    try:
        data = request.json or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "缺少 text"}), 400
        adapter = channel_hub._channels.get(channel_id)
        if adapter is None:
            return jsonify({"error": f"通道不存在: {channel_id}"}), 404
        result = adapter.send(text, {"sender": "manual", "channel": channel_id})
        if not result.get("ok"):
            return jsonify({"error": result.get("detail"), "sent": False}), 400
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.exception(f"[ChannelsSend] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/channels/recent", methods=["GET"])
def channels_recent():
    """最近通道事件（收到/回复/失败）+ 他人消息。"""
    try:
        n = int(request.args.get("n", 30) or 30)
        return jsonify({"success": True,
                        "recent": channel_hub.recent(max(1, min(100, n))),
                        "other_messages": channel_hub.other_messages(20)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/state", methods=["GET"])
def internal_state_get():
    """需求 / 调制变量 / 性格 / 期望表 / 最近奖励与轮次。"""
    try:
        return jsonify({"success": True, **internal_state.state()})
    except Exception as e:
        logger.exception(f"[InternalStateGet] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/explain", methods=["GET"])
def internal_explain():
    """解释某一轮：该轮的状态变化 + 奖励 + 人话摘要（可回答"为什么她想探索"）。"""
    try:
        cycle_id = request.args.get("cycle_id")
        if not cycle_id:
            cycles = internal_state.recent_cycles(1)
            if not cycles:
                return jsonify({"success": True, "cycle_id": None,
                                "summary": "还没有任何认知周期记录"})
            cycle_id = cycles[0]["id"]
        return jsonify({"success": True, **internal_state.explain(cycle_id)})
    except Exception as e:
        logger.exception(f"[InternalExplain] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/cycles", methods=["GET"])
def internal_cycles():
    try:
        n = int(request.args.get("n", 20) or 20)
        return jsonify({"success": True,
                        "cycles": internal_state.recent_cycles(max(1, min(200, n)))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/rewards", methods=["GET"])
def internal_rewards():
    try:
        n = int(request.args.get("n", 20) or 20)
        return jsonify({"success": True,
                        "rewards": internal_state.recent_rewards(max(1, min(200, n)))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/history", methods=["GET"])
def internal_history():
    """某变量的变化历史（含 cycle_id / 来源 / 原因）——可解释性的原始材料。"""
    try:
        kind = request.args.get("kind", "modulator")
        name = request.args.get("name", "")
        n = int(request.args.get("n", 30) or 30)
        if kind not in ("modulator", "need", "trait"):
            return jsonify({"error": "kind 必须是 modulator/need/trait"}), 400
        if not name:
            return jsonify({"error": "name 不能为空"}), 400
        return jsonify({"success": True, "kind": kind, "name": name,
                        "history": internal_state.history(kind, name, max(1, min(200, n)))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/modulator/<name>", methods=["POST"])
def internal_set_modulator(name):
    """人工调节激素水平（前端滑块）：body {"level": 0-1, "reason": "..."}。

    注意键名与语义：`level` 是历史契约名，R2 之后 `set_value("modulator", …)` 写的是
    **浓度真值（tonic）**，不是响应视图；前端因此把滑杆绑 conc、并同时显示
    "响应 / 浓度"两个数（index.html renderInternalModulators）。
    """
    try:
        data = request.json or {}
        if "level" not in data:
            return jsonify({"error": "缺少 level"}), 400
        result = internal_state.set_value("modulator", name, data["level"],
                                          source="manual",
                                          reason=data.get("reason") or "前端手动调节")
        if not result.get("ok"):
            code = 404 if "未知" in str(result.get("error")) else 400
            return jsonify({"error": result.get("error")}), code
        internal_state.sync_graph()
        internal_state.save()          # 人工调节立即落盘（否则重启丢失）
        return jsonify({"success": True, **result,
                        "state": internal_state.state()["modulators"].get(name)})
    except Exception as e:
        logger.exception(f"[InternalSetModulator] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/need/<name>", methods=["POST"])
def internal_set_need(name):
    """人工设定需求水平（调试/干预用；正常由经历驱动）。"""
    try:
        data = request.json or {}
        if "level" not in data:
            return jsonify({"error": "缺少 level"}), 400
        result = internal_state.set_value("need", name, data["level"],
                                          source="manual",
                                          reason=data.get("reason") or "前端手动调节")
        if not result.get("ok"):
            code = 404 if "未知" in str(result.get("error")) else 400
            return jsonify({"error": result.get("error")}), code
        internal_state.sync_graph()
        internal_state.save()
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/traits", methods=["POST"])
def internal_set_traits():
    """设定性格维度（用户可自定义）。body：{"traits": {"caution": 0.7, ...}}。

    这是**显式人工设定**（出厂值可改，属用户要求），与 Phase 5 的慢学习不同：
    慢学习走证据门槛，人工设定直接生效但一律留痕（source=manual）。
    """
    try:
        data = request.json or {}
        traits = data.get("traits") or {}
        if not isinstance(traits, dict) or not traits:
            return jsonify({"error": "需要 traits 对象"}), 400
        results = {}
        for name, value in traits.items():
            results[name] = internal_state.set_value(
                "trait", name, value, source="manual",
                reason=data.get("reason") or "前端设定")
        bad = {k: v for k, v in results.items() if not v.get("ok")}
        if bad and len(bad) == len(results):
            return jsonify({"error": "; ".join(str(v.get("error")) for v in bad.values())}), 400
        internal_state.sync_graph()
        internal_state.save()
        return jsonify({"success": True, "applied": results,
                        "traits": internal_state.state()["traits"]})
    except Exception as e:
        logger.exception(f"[InternalSetTraits] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/reset", methods=["POST"])
def internal_reset():
    """一键恢复出厂：body {"target": "traits"|"modulators"|"all"}。"""
    try:
        data = request.json or {}
        target = str(data.get("target", "all"))
        out = {}
        if target in ("traits", "all"):
            out["traits"] = internal_state.reset_traits()
        if target in ("modulators", "all"):
            out["modulators"] = internal_state.reset_modulators()
        if not out:
            return jsonify({"error": "target 必须是 traits/modulators/all"}), 400
        internal_state.sync_graph()
        internal_state.save()
        return jsonify({"success": True, "reset": target})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/snapshot", methods=["GET", "POST"])
def internal_snapshot():
    """GET 取快照（可保存）；POST 用 body.snapshot 恢复（人工回滚）。"""
    try:
        if request.method == "GET":
            return jsonify({"success": True, "snapshot": internal_state.snapshot()})
        data = request.json or {}
        snap = data.get("snapshot")
        if not isinstance(snap, dict) or not snap:
            return jsonify({"error": "需要 snapshot 对象"}), 400
        result = internal_state.restore(snap)
        internal_state.sync_graph()
        internal_state.save()
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.exception(f"[InternalSnapshot] {e}")
        return jsonify({"error": str(e)}), 500


# ── R2 实验台（P12，§26）：驱动的是真实管线，不是为调试另写的一份 ──
# 六个端点都只是 modulator_lab.ModulatorLab 的**转发**——扇出来自图上的边、
# 衰减走统一时基、写入走唯一入口。`advance` 会改**活体**状态（快进即真衰减），
# 这正是实验台的存在意义；想回滚先用 /api/internal/snapshot。
_lab = None


def _get_lab():
    global _lab
    if _lab is None:
        from modulator_lab import ModulatorLab
        _lab = ModulatorLab(kg, internal_state, config,
                            engine=reward_system.mod_engine(),
                            reward=reward_system, persona=persona,
                            field=drive_evaluator.field)
    return _lab


@app.route("/api/internal/lab/observe", methods=["GET"])
def lab_observe():
    """一次拿全观测面：时钟会话 + 全部通道数值 + 14 参数 + 最近动作。"""
    try:
        return jsonify({"success": True, **_get_lab().observe()})
    except Exception as e:
        logger.exception(f"[Lab] observe: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/lab/chain", methods=["GET"])
def lab_chain():
    """一条链的全貌（事件→边→调制器→参数/偏置/心情）；?event_type= 聚焦一类。"""
    try:
        return jsonify({"success": True,
                        **_get_lab().chain(request.args.get("event_type"))})
    except Exception as e:
        logger.exception(f"[Lab] chain: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/lab/event", methods=["POST"])
def lab_event():
    """注入一次结构化调制事件（与 reward/CC 同一个 emit）。
    body：{"event_type": "reward_rpe", ...ModulationEvent 的字段}。"""
    try:
        data = dict(request.json or {})
        et = str(data.pop("event_type", "") or "").strip()
        if not et:
            return jsonify({"error": "需要 event_type"}), 400
        out = _get_lab().inject(et, **data)
        return jsonify({"success": bool(out.get("ok")), **out})
    except Exception as e:
        logger.exception(f"[Lab] event: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/lab/outcome", methods=["POST"])
def lab_outcome():
    """结果结算的完整一圈（evaluate → release → modulation）。
    body：{"social_outcome": "rejected", "self_outcome": null, "behavior": ..., "context": ...}。"""
    try:
        data = request.json or {}
        out = _get_lab().outcome(
            social_outcome=data.get("social_outcome"),
            self_outcome=data.get("self_outcome"),
            behavior=str(data.get("behavior") or "experiment"),
            context=str(data.get("context") or "lab"),
            ref=str(data.get("ref") or ""))
        if out.get("ok"):
            internal_state.save()
        return jsonify({"success": bool(out.get("ok")), **out})
    except Exception as e:
        logger.exception(f"[Lab] outcome: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/lab/dial", methods=["POST"])
def lab_dial():
    """手动拧浓度真值（kind: modulator|need|trait）——与滑杆同一个写入口。"""
    try:
        data = request.json or {}
        if "name" not in data or "value" not in data:
            return jsonify({"error": "需要 name 与 value"}), 400
        out = _get_lab().dial(str(data["name"]), float(data["value"]),
                              kind=str(data.get("kind") or "modulator"),
                              reason=str(data.get("reason") or "实验台调节"))
        if out.get("ok"):
            internal_state.save()
        return jsonify({"success": bool(out.get("ok")), **out})
    except Exception as e:
        logger.exception(f"[Lab] dial: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/internal/lab/time", methods=["POST"])
def lab_time():
    """虚拟时基：{"mode": "open"|"advance"|"close", "minutes": n}。
    advance 逐块重放生产的状态拍（decay + 昼夜/需求漂移）；close 原样交还时钟。"""
    try:
        data = request.json or {}
        mode = str(data.get("mode") or "").strip().lower()
        lab = _get_lab()
        if mode == "open":
            out = lab.open()
        elif mode == "close":
            out = lab.close()
        elif mode == "advance":
            out = lab.advance(float(data.get("minutes") or 0.0))
        else:
            return jsonify({"error": "mode 必须是 open/advance/close"}), 400
        if mode != "open" and out.get("ok"):
            internal_state.save()
        return jsonify({"success": bool(out.get("ok")), **out})
    except Exception as e:
        logger.exception(f"[Lab] time: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/cognition/state", methods=["GET"])
def cognition_state():
    """持续认知可解释性：当前 CI 列表与全部决策依据。"""
    try:
        return jsonify({"success": True, **cc.state()})
    except Exception as e:
        logger.exception(f"[CognitionState] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/dispositions", methods=["GET"])
def self_dispositions():
    """Reflection Evolution: 查看行为倾向及其证据（表达方式自然形成过程可观测）。"""
    try:
        return jsonify({
            "success": True,
            "dispositions": disposition_store.list_dispositions(),
            "recent_expressions": buffer.get_expressions(n=10),
        })
    except Exception as e:
        logger.exception(f"[Dispositions] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/reflection", methods=["POST"])
def self_reflection():
    """Trigger reflection cycle (Phase 2: 用户主动触发)。"""
    try:
        with engine._lock:
            topk_snapshot, _ = engine.get_topk(k=15)
        chat_entries = get_chat_log().recent(n=15)
        ref_result = reflection_engine.run(
            trigger="user",
            topk_nodes=topk_snapshot,
            chat_log_entries=chat_entries,
        )
        _save()
        if ref_result.get("skipped"):
            return jsonify({
                "success": True,
                "reflection": ref_result,
                "message": ref_result.get("reason", "skipped")
            })
        return jsonify({"success": True, "reflection": ref_result})
    except Exception as e:
        logger.exception(f"[Reflection API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/reflection/candidates", methods=["GET"])
def reflection_candidates():
    """Get pending reflection candidates awaiting approval."""
    try:
        pending = reflection_engine.get_pending_candidates()
        return jsonify({"success": True, "pending": pending, "count": len(pending)})
    except Exception as e:
        logger.exception(f"[Reflection API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/reflection/approve", methods=["POST"])
def reflection_approve():
    """Approve a pending reflection candidate (pending → approved)。

    请求体: {"reflection_id": "反思_xxx", "candidate_index": 0}
    """
    try:
        data = request.json or {}
        ref_id = str(data.get("reflection_id", "")).strip()
        cand_idx = int(data.get("candidate_index", -1))

        if not ref_id or cand_idx < 0:
            return jsonify({"error": "需要 reflection_id 和 candidate_index"}), 400

        ref_node = kg.get_node(ref_id)
        if not ref_node:
            return jsonify({"error": f"反思节点不存在: {ref_id}"}), 404

        candidates = ref_node.extra_attrs.get("candidates", [])
        if cand_idx >= len(candidates):
            return jsonify({"error": f"candidate_index 越界 (共 {len(candidates)} 条)"}), 400

        cand = candidates[cand_idx]

        # 写入 Self Model
        from self_model import SelfGraphManager
        mgr = SelfGraphManager(kg)
        now = now_str()

        ctype = cand.get("type", "")
        if ctype == "belief_update":
            mgr.upsert_belief(cand.get("content", ""), {
                "confidence": cand.get("confidence", 0.6),
                "source": "llm_inference",
                "evidence_count": len(cand.get("evidence", [])),
                "first_observed": now,
                "last_reinforced": now,
            })
        elif ctype == "preference_update":
            mgr.upsert_preference(cand.get("target", ""), {
                "confidence": cand.get("confidence", 0.6),
                "source": "llm_inference",
                "evidence_count": len(cand.get("evidence", [])),
                "first_observed": now,
                "last_reinforced": now,
            })

        # 更新反思节点的计数
        ref_node.extra_attrs["approved_count"] = ref_node.extra_attrs.get("approved_count", 0) + 1
        ref_node.extra_attrs["pending_count"] = max(0, ref_node.extra_attrs.get("pending_count", 0) - 1)
        ref_node.touch()

        _save()
        return jsonify({"success": True, "approved": cand, "reflection_id": ref_id})
    except Exception as e:
        logger.exception(f"[Reflection API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/experience", methods=["POST"])
def self_experience():
    """Add episodic memory connected to Self."""
    try:
        data = request.json or {}
        raw_text = str(data.get("text", "")).strip()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        if not raw_text:
            return jsonify({"error": "missing text"}), 400
        exp_id = add_experience(kg, raw_text, nodes, edges)
        _save()
        return jsonify({"success": True, "experience_id": exp_id})
    except Exception as e:
        logger.exception(f"[Experience API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/emotion/current", methods=["GET"])
def self_emotion_current():
    """Get current dominant emotion from Self diffusion."""
    try:
        emo = get_current_emotion(kg)
        return jsonify({"success": True, "current_emotion": emo})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/preferences", methods=["GET"])
def self_preferences_list():
    """List all preferences (Self -> like -> target edges)."""
    try:
        prefs = get_preferences(kg)
        return jsonify({"success": True, "preferences": prefs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/goals", methods=["GET"])
def self_goals_list():
    """List all goals."""
    try:
        goals = get_goals(kg)
        return jsonify({"success": True, "goals": goals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# Self Model API (Phase 1)
# ═══════════════════════════════════════════════════════════════

@app.route("/api/self/model", methods=["GET"])
def self_model_summary():
    """Get Self Model structured summary with confidence annotations."""
    try:
        summary = get_self_model_summary(kg)
        return jsonify({"success": True, "self_model": summary})
    except Exception as e:
        logger.exception(f"[SelfModel API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/model/beliefs", methods=["GET"])
def self_model_beliefs():
    """List all Beliefs with full provenance."""
    try:
        from self_model import SelfGraphManager
        mgr = SelfGraphManager(kg)
        beliefs = mgr.get_beliefs()
        return jsonify({"success": True, "beliefs": beliefs, "count": len(beliefs)})
    except Exception as e:
        logger.exception(f"[SelfModel API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/model/preferences", methods=["GET"])
def self_model_preferences():
    """List all Preferences with full provenance."""
    try:
        from self_model import SelfGraphManager
        mgr = SelfGraphManager(kg)
        prefs = mgr.get_preferences()
        return jsonify({"success": True, "preferences": prefs, "count": len(prefs)})
    except Exception as e:
        logger.exception(f"[SelfModel API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/self/model/audit", methods=["GET"])
def self_model_audit():
    """Audit view: full provenance chain for all Beliefs and Preferences."""
    try:
        audit = get_self_model_audit(kg)
        return jsonify({"success": True, "audit": audit, "count": len(audit)})
    except Exception as e:
        logger.exception(f"[SelfModel API] {e}")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# Drive System API (Phase 3 MVP)
# ═══════════════════════════════════════════════════════════════

@app.route("/api/drive", methods=["GET"])
def drive_state():
    """Get current drive system state — dominant drive and all drive activations."""
    try:
        state = drive_evaluator.get_drive_state()
        return jsonify({"success": True, "drive_state": state})
    except Exception as e:
        logger.exception(f"[Drive API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/drive/evaluate", methods=["POST"])
def drive_evaluate():
    """Force re-evaluation of all drives."""
    try:
        result = drive_evaluator.evaluate(force=True)
        return jsonify({"success": True, "evaluation": result})
    except Exception as e:
        logger.exception(f"[Drive API] {e}")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# Action System API (Phase 4 MVP)
# ═══════════════════════════════════════════════════════════════

@app.route("/api/action/capabilities", methods=["GET"])
def action_capabilities():
    """List all cognitive capabilities and their status."""
    try:
        result = action_selector.evaluate()
        return jsonify({
            "success": True,
            "capabilities": result.get("capabilities", []),
            "count": len(result.get("capabilities", [])),
        })
    except Exception as e:
        logger.exception(f"[Action API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/action/intents", methods=["GET"])
def action_intents():
    """Get currently active ActionIntents."""
    try:
        intents = action_selector.get_active_intents()
        return jsonify({"success": True, "intents": intents, "count": len(intents)})
    except Exception as e:
        logger.exception(f"[Action API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/action/state", methods=["GET"])
def action_state():
    """Get full Action System state: capabilities + intents + recent history."""
    try:
        state = action_selector.get_state()
        return jsonify({"success": True, "action_state": state})
    except Exception as e:
        logger.exception(f"[Action API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/evolution", methods=["GET"])
def evolution_log():
    """Get graph evolution log. Growth = graph structure changes."""
    try:
        evo_log = get_evolution_log()
        n = int(request.args.get("n", 50))
        change_type = request.args.get("type", None)
        if change_type:
            from graph_evolution_log import ChangeType
            try:
                ct = ChangeType(change_type)
                entries = evo_log.by_type(ct, n=n)
            except ValueError:
                entries = evo_log.recent(n=n)
        else:
            entries = evo_log.recent(n=n)
        return jsonify({"success": True, "entries": entries, "stats": evo_log.stats()})
    except Exception as e:
        logger.exception(f"[Evolution API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/evolution/stats", methods=["GET"])
def evolution_stats():
    """Graph evolution statistics."""
    try:
        evo_log = get_evolution_log()
        return jsonify({"success": True, "stats": evo_log.stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =========================================================
# LLM Provider Switching
# =========================================================

@app.route("/api/llm/switch", methods=["POST"])
def llm_switch():
    """Switch LLM provider between ollama and deepseek."""
    try:
        data = request.json or {}
        provider = str(data.get("provider", "mimo")).strip()
        if provider == "ollama" and not ollama_enabled():
            return jsonify({"error": "Ollama 未启用（config.py enable_ollama=false）"}), 400
        api_key = data.get("api_key", None)
        model = data.get("model", None)
        nlp.switch_provider(provider, api_key=api_key, model=model)
        _save_llm_config()
        saved = _load_llm_config()
        # 密钥不进日志/响应
        if saved.get("api_key"):
            saved["api_key"] = "***"
        logger.info(f"[LLM Switch] Saved: {saved}")
        return jsonify({
            "success": True,
            "provider": getattr(nlp, '_provider_name', provider),
            "model": nlp.model_name(),
            "saved_file": saved
        })
    except Exception as e:
        logger.exception(f"[LLM Switch] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/llm/status", methods=["GET"])
def llm_status():
    """Get current LLM provider and model info."""
    cfg = _load_llm_config()
    has_key = bool(cfg.get("api_key"))
    if has_key:
        cfg["api_key"] = "***"
    return jsonify({
        "model": nlp.model_name(),
        "provider": getattr(nlp, '_provider_name', 'unknown'),
        "has_api_key": has_key,
        "ollama_enabled": ollama_enabled(),
        "saved": cfg,
    })


# =========================================================
# Conversation Gap Detection
# =========================================================

@app.route("/api/conversation/gap/record", methods=["POST"])
def gap_record():
    """Record a conversation turn for later gap analysis."""
    try:
        data = request.json or {}
        user_input = data.get("user_input", "")
        assistant_response = data.get("assistant_response", "")
        topk_nodes = data.get("topk_nodes", [])
        topk_edges = data.get("topk_edges", [])
        fab_score = float(data.get("fabrication_score", 0.0))
        gap_detector.record_turn(user_input, assistant_response,
                                  topk_nodes, topk_edges, fab_score)
        return jsonify({"success": True, "turns_recorded": len(gap_detector._conversation_log)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversation/gap/analyze", methods=["POST"])
def gap_analyze():
    """Analyze recorded conversation for cognitive gaps."""
    try:
        gaps = gap_detector.analyze_session(kg, nlp_processor=nlp)
        return jsonify({"success": True, "gaps": gaps, "stats": gap_detector.stats()})
    except Exception as e:
        logger.exception(f"[Gap Analyze] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversation/gap/pending", methods=["GET"])
def gap_pending():
    """Get pending (unreviewed) gaps."""
    return jsonify({"success": True, "pending": gap_detector.get_pending()})


@app.route("/api/conversation/gap/review", methods=["POST"])
def gap_review():
    """Mark a gap as reviewed."""
    try:
        data = request.json or {}
        index = int(data.get("index", -1))
        gap_detector.mark_reviewed(index)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversation/gap/clear", methods=["POST"])
def gap_clear():
    """Clear all gaps."""
    gap_detector.clear()
    return jsonify({"success": True})

# =========================================================
# Chat Log API
# =========================================================

@app.route("/api/chat/log", methods=["GET"])
def chat_log_recent():
    """Get recent chat log entries."""
    try:
        n = int(request.args.get("n", 50))
        log = get_chat_log()
        return jsonify({"success": True, "entries": log.recent(n), "stats": log.stats()})
    except Exception as e:
        logger.exception(f"[ChatLog API] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/log/search", methods=["GET"])
def chat_log_search():
    """Search chat log by keyword."""
    try:
        q = request.args.get("q", "").strip()
        n = int(request.args.get("n", 50))
        if not q:
            return jsonify({"error": "缺少搜索关键词"}), 400
        log = get_chat_log()
        return jsonify({"success": True, "query": q, "entries": log.search(q, n)})
    except Exception as e:
        logger.exception(f"[ChatLog Search] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/log/stats", methods=["GET"])
def chat_log_stats():
    """Get chat log statistics."""
    try:
        return jsonify({"success": True, "stats": get_chat_log().stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/log/clear", methods=["POST"])
def chat_log_clear():
    """Clear all chat log entries."""
    try:
        get_chat_log().clear()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================

@app.route("/api/search", methods=["GET"])
def search_graph():

    import re

    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"nodes": [], "edges": []})

    logger.info(f"[Search] q={q}")

    def _match_node_id(nid: str, query: str) -> bool:
         if not nid or not query:
             return False
         nlow = nid.lower()
         qlow = query.lower()
         # 完全匹配
         if nlow == qlow:
             return True
         # 后缀边界匹配：查询词紧邻的右侧字符必须是边界(非字母数字/末尾)
         idx = nlow.find(qlow)
         while idx != -1:
             next_char = nlow[idx + len(qlow)] if idx + len(qlow) < len(nlow) else ' '
             if not next_char.isalnum():
                 return True
             idx = nlow.find(qlow, idx + 1)
         return False

    with kg._lock:
        matched_nodes = [
            n.to_dict()
            for nid, n in kg.nodes.items()
            if _match_node_id(nid, q)
        ]

        matched_edges = [
            e.to_dict() for e in kg.edges
            if (
                _match_node_id(e.src, q)
                or _match_node_id(e.dst, q)
                or q.lower() in e.relation.lower()
            )
        ]

    logger.info(f"[Search] nodes={len(matched_nodes)} edges={len(matched_edges)}")
    return jsonify({"nodes": matched_nodes, "edges": matched_edges})


@app.route("/api/search/<node_id>", methods=["PUT"])
def update_searched_node(node_id):

    nid = str(node_id).strip()
    data = request.json or {}

    _deny = _execution_write_denied(data)
    if _deny:
        logger.warning(f"[SECURITY] 拒绝 PUT /api/search/{node_id} 写入 execution: {_deny}")
        return jsonify({"error": _deny}), 403

    with kg._lock:
        node = kg.nodes.get(nid)
        if not node:
            return jsonify({"error": f"节点不存在: {nid}"}), 404

        _label_changed = False
        if "weight" in data:
            node.weight = float(data["weight"])
        if "label" in data and data["label"] in [
            "declarative-semantic", "declarative-episodic", "procedural"
        ]:
            if node.label != data["label"]:
                _label_changed = True
            node.label = data["label"]
        if "execution" in data:
            node.execution = data["execution"]
        if "activation" in data:
            node.activation = max(0.0, min(
                config.get("activation_max", 5.0),
                float(data["activation"])
            ))
        node.touch()

    with engine._lock:
        engine.name_to_node[nid] = node
    # activation 直写 → 前沿同步；label 直改 → 行动队列资格重估
    if "activation" in data:
        engine.mark_active([nid])
    if _label_changed:
        engine.note_action_dirty(nid)

    _save()
    return jsonify({"success": True})


@app.route("/api/search", methods=["PUT"])
def update_searched_edge():

    data = request.json or {}
    src = str(data.get("src", "")).strip()
    dst = str(data.get("dst", "")).strip()
    rel = str(data.get("relation", data.get("type", ""))).strip()
    if not src or not dst or not rel:
        return jsonify({"error": "src/dst/relation 不能为空"}), 400

    with kg._lock:
        edge = kg.get_edge(src, dst, rel)
        if not edge:
            return jsonify({"error": "边不存在"}), 404

        if "weight" in data:
            edge.weight = float(max(-2.0, min(2.0, data["weight"])))
        if "activation" in data:
            edge.activation = max(0.0, min(3.0, float(data["activation"])))
        edge.touch()

    if "activation" in data:
        engine.mark_edges_active([edge])

    _save()
    return jsonify({"success": True})

# =========================================================
# Config
# =========================================================

@app.route("/api/config", methods=["GET"])
def get_config():

    return jsonify(config)


@app.route("/api/config", methods=["PUT"])
def update_config():

    logger.info(
        "[API] update_config"
    )

    try:

        data = request.json or {}

        for k, v in data.items():

            if k in config:

                try:

                    config[k] = type(
                        config[k]
                    )(v)

                except Exception:

                    logger.warning(
                        f"[Config] "
                        f"无法更新 {k}"
                    )

        engine.config = config

        logger.info(
            "[Config] 更新完成"
        )

        return jsonify(config)

    except Exception as e:

        logger.exception(
            f"[Config异常] {e}"
        )

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/learning/explore", methods=["GET"])
def learning_explore():
    """
    学习模块步骤 1：在大图谱中检索特定词汇及现有邻居，同时融合 LLM 实时生成的扩充推荐，标记混入状态供前端可视化审核
    """
    word = request.args.get("word", "").strip()
    if not word:
        return jsonify({"error": "请输入有效的核心概念词汇"}), 400
    # 可选的附加说明，帮助 LLM 消歧但不影响节点名
    context = request.args.get("context", "").strip()

    logger.info(f"[学习模块] 正在检索并生成扩充子图: {word}" +
                (f" (context: {context})" if context else ""))

    existing_nodes = []
    existing_edges = []

    # 1. 查询大图谱，搜集该词当前已存在的节点与直接邻居
    with kg._lock:
        word_node = kg.nodes.get(word)
        if word_node:
            existing_nodes.append({**word_node.to_dict(), "status": "existing"})
            # 搜集所有直接相连的边
            for edge in kg.edges:
                if edge.src == word or edge.dst == word:
                    existing_edges.append({**edge.to_dict(), "status": "existing"})
                    # 将邻居节点一并取出
                    neighbor_id = edge.dst if edge.src == word else edge.src
                    if neighbor_id in kg.nodes and not any(n["id"] == neighbor_id for n in existing_nodes):
                        existing_nodes.append({**kg.nodes[neighbor_id].to_dict(), "status": "existing"})

    # 2. 调用 LLM 扩充引擎生成高质关联推荐
    llm_res = nlp.expand_concept(word, context=context)
    proposed_nodes = llm_res.get("nodes", [])
    proposed_edges = llm_res.get("edges", [])
    node_meta_list = llm_res.get("node_meta", llm_res.get("nodes_meta", []))
    node_meta = {}
    if isinstance(node_meta_list, list):
        for it in node_meta_list:
            if not isinstance(it, dict):
                continue
            nid = str(it.get("id", it.get("node", ""))).strip()
            if not nid:
                continue
            node_meta[nid] = {
                "confidence": it.get("confidence", None),
                "reason": it.get("reason", None)
            }

    # 3. 混合去重新旧节点，进行关系差异化补齐
    if word not in proposed_nodes:
        proposed_nodes.insert(0, word)

    if not proposed_edges and len(proposed_nodes) > 1:
        fallback_edges = []
        for nid in proposed_nodes[1: min(len(proposed_nodes), 9)]:
            nid = str(nid).strip()
            if not nid:
                continue
            fallback_edges.append({
                "src": word,
                "dst": nid,
                "type": "关联",
                "relation": "关联",
                "weight": 0.4,
                "confidence": 0.2,
                "reason": "兜底关联"
            })
        proposed_edges = fallback_edges

    response_nodes = list(existing_nodes)
    response_edges = list(existing_edges)
    existing_node_ids = {n["id"] for n in existing_nodes}

    # 混入 AI 新推荐的缺少的节点
    for node_id in proposed_nodes:
        node_id = str(node_id).strip()
        if node_id not in existing_node_ids:
            response_nodes.append({
                "id": node_id,
                "weight": 0.5,
                "activation": 0.0,
                "label": "declarative-semantic",
                "execution": None,
                "extra_attrs": {
                    "llm_confidence": node_meta.get(node_id, {}).get("confidence", None),
                    "llm_reason": node_meta.get(node_id, {}).get("reason", None),
                },
                "status": "proposed"  # 标识为待审核的 AI 新推荐项
            })
            existing_node_ids.add(node_id)

    # 混入 AI 新推荐的缺少的邻居连边（🔍 修复 Bug 2: 解决 type 和 relation 键名不一致）
    for pe in proposed_edges:
        src = str(pe.get("src", pe.get("from", pe.get("source", "")))).strip()
        dst = str(pe.get("dst", pe.get("to", pe.get("target", "")))).strip()
        rel = str(pe.get("type", pe.get("relation", "关联"))).strip()
        w = float(pe.get("weight", 0.5))
        conf = pe.get("confidence", None)
        reason = pe.get("reason", None)

        # 检查大图谱已有和当前的待审核队列中是否重复
        edge_exists = False
        for ee in response_edges:
            ee_rel = ee.get("relation", ee.get("type", ""))
            if ee["src"] == src and ee["dst"] == dst and ee_rel == rel:
                edge_exists = True
                break

        if not edge_exists:
            response_edges.append({
                "src": src,
                "dst": dst,
                "relation": rel,
                "weight": w,
                "activation": 0.0,
                "confidence": conf,
                "reason": reason,
                "status": "proposed"  # 标识为待审核的 AI 新推荐边
            })

    return jsonify({
        "word": word,
        "exists_in_graph": word_node is not None,
        "nodes": response_nodes,
        "edges": response_edges
    })


@app.route("/api/learning/import", methods=["POST"])
def learning_import():
    """
    学习模块步骤 2：原子化写入经过前端人为审核、编辑、剔除后的最终干净子图数据并落盘
    """
    try:
        data = request.json or {}
        nodes_to_import = data.get("nodes", [])
        edges_to_import = data.get("edges", [])

        logger.info(f"[学习模块] 准备并入大图谱，节点数: {len(nodes_to_import)}, 边数: {len(edges_to_import)}")

        with kg._lock:
            # 1. 原子批量导入并更新节点信息
            _label_changed_ids = []
            for n_data in nodes_to_import:
                nid = str(n_data["id"]).strip()
                if nid in kg.nodes:
                    # 如果原图谱已有，则同步用户在审核面板里修改过的核心字段
                    node = kg.nodes[nid]
                    node.weight = float(n_data.get("weight", node.weight))
                    if node.label != n_data.get("label", node.label):
                        node.label = n_data.get("label", node.label)
                        _label_changed_ids.append(nid)
                    node.touch()
                else:
                    # 如果没有则全新实例化创建（适配 VALID_LABELS 约束）
                    kg.add_node(Node(
                        id=nid,
                        weight=float(n_data.get("weight", 0.5)),
                        label=n_data.get("label", "declarative-semantic")
                    ))
                # 动态刷新扩散引擎的反向索引映射
                engine.name_to_node[nid] = kg.nodes[nid]
            # label 变更影响行动队列资格（批量收集，一次标记）
            if _label_changed_ids:
                engine.note_action_dirty(_label_changed_ids)

            # 2. 原子批量导入并补齐连边
            for e_data in edges_to_import:
                src = str(e_data["src"]).strip()
                dst = str(e_data["dst"]).strip()
                relation = str(e_data.get("relation", e_data.get("type", "关联"))).strip()
                weight = float(e_data.get("weight", 0.5))

                # 判定排重逻辑
                duplicate = False
                for edge in kg.edges:
                    if edge.src == src and edge.dst == dst and edge.relation == relation:
                        edge.weight = weight  # 覆盖为审核调优后的权重
                        duplicate = True
                        break

                if not duplicate:
                    kg.add_edge(Edge(
                        src=src,
                        dst=dst,
                        relation=relation,
                        weight=weight
                    ))

        # 强制将内存中的修改刷新写入底层 JSON 持久化存储
        _save(force=True)
        return jsonify({"status": "success", "message": "子图审核完毕，已安全并入主知识图谱！"})

    except Exception as e:
        logger.exception(f"[学习模块写入异常]: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# [V4 Phase 1] Knowledge Pack Management APIs
# =========================================================

@app.route("/api/packs", methods=["GET"])
def get_packs():
    """返回所有 Knowledge Pack"""
    try:
        return jsonify({"success": True, "packs": pack_mgr.list_packs()})
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/packs/enabled", methods=["GET"])
def get_enabled_packs():
    """返回当前启用的 Pack"""
    try:
        return jsonify({"success": True, "packs": pack_mgr.get_enabled_pack_list()})
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs/enabled: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/packs/enable", methods=["POST"])
def enable_pack():
    """启用指定 Pack"""
    try:
        data = request.json or {}
        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"error": "缺少 pack name"}), 400
        ok = pack_mgr.enable_pack(name)
        if not ok:
            return jsonify({"error": f"Pack 不存在: {name}"}), 404
        global kg
        kg = pack_mgr.merge_runtime_graph()
        engine.kg = kg
        engine.name_to_node = {str(n.id).strip(): n for nid, n in kg.nodes.items()}
        kg.rebuild_indexes()   # merge 不保证走索引维护路径——统一重建派生缓存
        engine.reattach_graph(kg)  # 前沿/回调/行动队列重绑新图
        logger.info(f"[Pack] 启用 {name} 后重新合并 Runtime Graph")
        return jsonify({"success": True, "message": f"Pack {name} 已启用",
                        "stats": pack_mgr.stats()})
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs/enable: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/packs/disable", methods=["POST"])
def disable_pack():
    """禁用指定 Pack"""
    try:
        data = request.json or {}
        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"error": "缺少 pack name"}), 400
        ok = pack_mgr.disable_pack(name)
        if not ok:
            return jsonify({"error": f"Pack 不存在: {name}"}), 404
        global kg
        kg = pack_mgr.merge_runtime_graph()
        engine.kg = kg
        engine.name_to_node = {str(n.id).strip(): n for nid, n in kg.nodes.items()}
        kg.rebuild_indexes()   # merge 不保证走索引维护路径——统一重建派生缓存
        engine.reattach_graph(kg)  # 前沿/回调/行动队列重绑新图
        logger.info(f"[Pack] 禁用 {name} 后重新合并 Runtime Graph")
        return jsonify({"success": True, "message": f"Pack {name} 已禁用",
                        "stats": pack_mgr.stats()})
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs/disable: {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/api/packs/create", methods=["POST"])
def create_pack():
    """创建新的 Knowledge Pack"""
    try:
        data = request.json or {}
        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"error": "缺少 pack name"}), 400
        pack_type = data.get("type", "semantic")
        description = data.get("description", "")
        author = data.get("author", "")
        priority = int(data.get("priority", 10))
        pack = pack_mgr.create_pack(name, pack_type=pack_type,
                                     description=description,
                                     author=author, priority=priority)
        return jsonify({"success": True, "pack": pack.to_dict()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs/create: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/packs/delete", methods=["POST"])
def delete_pack():
    """删除指定 Pack"""
    try:
        data = request.json or {}
        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"error": "缺少 pack name"}), 400
        ok = pack_mgr.delete_pack(name)
        if not ok:
            return jsonify({"error": f"Pack 不存在: {name}"}), 404
        global kg
        kg = pack_mgr.merge_runtime_graph()
        engine.kg = kg
        engine.name_to_node = {str(n.id).strip(): n for nid, n in kg.nodes.items()}
        kg.rebuild_indexes()   # merge 不保证走索引维护路径——统一重建派生缓存
        engine.reattach_graph(kg)  # 前沿/回调/行动队列重绑新图
        logger.info(f"[Pack] 删除 {name} 后重新合并 Runtime Graph")
        return jsonify({"success": True, "message": f"Pack {name} 已删除",
                        "stats": pack_mgr.stats()})
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs/delete: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/packs/rename", methods=["POST"])
def rename_pack():
    """重命名 Pack"""
    try:
        data = request.json or {}
        old_name = str(data.get("old_name", "")).strip()
        new_name = str(data.get("new_name", "")).strip()
        if not old_name or not new_name:
            return jsonify({"error": "缺少 old_name 或 new_name"}), 400
        ok = pack_mgr.rename_pack(old_name, new_name)
        if not ok:
            return jsonify({"error": f"重命名失败: {old_name}"}), 400
        return jsonify({"success": True, "message": f"{old_name} -> {new_name}"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs/rename: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/packs/import", methods=["POST"])
def import_pack():
    """导入外部 Pack"""
    try:
        data = request.json or {}
        source_path = str(data.get("source_path", "")).strip()
        pack_name = data.get("pack_name", None)
        if not source_path:
            return jsonify({"error": "缺少 source_path"}), 400
        pack = pack_mgr.import_pack(source_path, pack_name=pack_name)
        return jsonify({"success": True, "pack": pack.to_dict()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs/import: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/packs/export", methods=["GET"])
def export_pack():
    """导出 Pack"""
    try:
        name = request.args.get("name", "").strip()
        dest_path = request.args.get("dest_path", "").strip()
        if not name or not dest_path:
            return jsonify({"error": "缺少 name 或 dest_path"}), 400
        result = pack_mgr.export_pack(name, dest_path)
        return jsonify({"success": True, "exported_to": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs/export: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/packs/reload", methods=["POST"])
def reload_packs():
    """重新扫描并合并所有 Pack"""
    try:
        global kg
        kg = pack_mgr.merge_runtime_graph()
        engine.kg = kg
        engine.name_to_node = {str(n.id).strip(): n for nid, n in kg.nodes.items()}
        kg.rebuild_indexes()   # merge 不保证走索引维护路径——统一重建派生缓存
        engine.reattach_graph(kg)  # 前沿/回调/行动队列重绑新图
        logger.info("[Pack] 手动重新加载完成")
        try:
            if not emb_mgr.ready():
                emb_mgr.build_index(kg)
        except Exception as e:
            # L1-CON-6/PIN-04 留痕:索引未重建时 faiss 召回静默失效——
            # "索引陈旧但 ready() 仍 True"会让检索悄悄哑火,必须可见。
            logger.warning(f"[Pack] 向量索引重建失败(重新加载后 faiss 仍指向旧索引): {e}")
        return jsonify({"success": True, "message": "已重新加载",
                        "stats": pack_mgr.stats()})
    except Exception as e:
        logger.exception(f"[Pack API异常] /api/packs/reload: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# Ear 听觉感知 API
# =========================================================

@app.route("/api/ear/status", methods=["GET"])
def ear_status():
    """获取 Ear 听觉感知模块状态"""
    try:
        status = ear_processor.get_status()
        return jsonify({"success": True, **status})
    except Exception as e:
        logger.exception(f"[Ear API] status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ear/enable", methods=["POST"])
def ear_enable():
    """启用 Ear 听觉感知"""
    try:
        ear_processor.enable()
        return jsonify({"success": True, "enabled": True})
    except Exception as e:
        logger.exception(f"[Ear API] enable: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ear/disable", methods=["POST"])
def ear_disable():
    """禁用 Ear 听觉感知"""
    try:
        ear_processor.disable()
        return jsonify({"success": True, "enabled": False})
    except Exception as e:
        logger.exception(f"[Ear API] disable: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ear/process", methods=["POST"])
def ear_process():
    """处理音频文件，注入图谱"""
    try:
        data = request.json or {}
        audio_path = data.get("audio_path", "").strip()

        if not audio_path:
            return jsonify({"error": "audio_path 不能为空"}), 400

        if not ear_processor.enabled:
            return jsonify({"error": "Ear 听觉感知未启用，请先在设置中开启"}), 400

        result = ear_processor.process_file(audio_path)

        # 将感知结果注入知识图谱
        if "timeline" in result:
            _inject_ear_result_to_graph(kg, result)

        _save()
        return jsonify(result)
    except Exception as e:
        logger.exception(f"[Ear API] process: {e}")
        return jsonify({"error": str(e)}), 500


# =========================================================
# Vision 视觉感知 API
# =========================================================

@app.route("/api/eye/status", methods=["GET"])
def eye_status():
    """Eye 屏幕感知器状态（OCR 库可用性）。"""
    try:
        from eye.screen_ocr import _get_ocr
        _get_ocr()
        return jsonify({"success": True, "available": True,
                        "engine": "rapidocr-onnxruntime (local)"})
    except Exception as e:
        return jsonify({"success": True, "available": False, "error": str(e)})


@app.route("/api/eye/capture", methods=["POST"])
def eye_capture():
    """截屏 + OCR：识别屏幕文字及方位，并注入图谱。"""
    try:
        data = request.json or {}
        region = data.get("region")  # [x,y,w,h] 可选
        from eye.screen_ocr import recognize_text, salient_texts
        result = recognize_text(region=region)
        result["salient"] = salient_texts(
            result, kg=kg, embedder=emb_mgr if emb_mgr.ready() else None)
        node_id = _inject_eye_result_to_graph(kg, result)
        result["graph_node"] = node_id
        _save()
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.exception(f"[Eye API] capture: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/status", methods=["GET"])
def vision_status():
    """获取 Vision 视觉感知模块状态"""
    try:
        stats = vision_processor.get_status()
        return jsonify({"success": True, **stats})
    except Exception as e:
        logger.exception(f"[Vision API] status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/enable", methods=["POST"])
def vision_enable():
    """启用 Vision 视觉感知"""
    try:
        vision_processor.enable()
        return jsonify({"success": True, "enabled": True})
    except Exception as e:
        logger.exception(f"[Vision API] enable: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/disable", methods=["POST"])
def vision_disable():
    """禁用 Vision 视觉感知"""
    try:
        vision_processor.disable()
        return jsonify({"success": True, "enabled": False})
    except Exception as e:
        logger.exception(f"[Vision API] disable: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/process", methods=["POST"])
def vision_process():
    try:
        data = request.json or {}
        image_path = data.get("image_path", "").strip()

        logger.info(f"[Vision API] 收到请求, raw_path={repr(data.get('image_path', ''))}")
        logger.info(f"[Vision API] 处理后路径: {repr(image_path)}")

        if not image_path:
            return jsonify({"error": "image_path 不能为空"}), 400

        if not vision_processor.enabled:
            return jsonify({"error": "Vision 视觉感知未启用，请先在设置中开启"}), 400

        # 去掉可能的多余引号
        image_path = image_path.strip('"').strip("'").strip()

        import os as _os
        logger.info(f"[Vision API] 文件是否存在: {_os.path.exists(image_path)}")
        if not _os.path.exists(image_path):
            # 尝试修复路径分隔符
            alt_path = image_path.replace('\\', '/').replace('//', '/')
            logger.info(f"[Vision API] 尝试路径: {repr(alt_path)}, exists={_os.path.exists(alt_path)}")

        activate = data.get("activate_graph", True)
        result = vision_processor.process_file(image_path, activate_graph=activate)

        _save()
        return jsonify(result)
    except Exception as e:
        logger.exception(f"[Vision API] process: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/objects", methods=["GET"])
def vision_objects():
    """查看当前视觉对象"""
    try:
        status_filter = request.args.get("status", None)
        objects = vision_processor.get_objects(status=status_filter if status_filter else None)
        stats = vision_processor.memory.get_stats()
        return jsonify({"success": True, "objects": objects, "stats": stats})
    except Exception as e:
        logger.exception(f"[Vision API] objects: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/confirm", methods=["POST"])
def vision_confirm():
    """
    用户确认视觉对象

    POST body:
        {
            "object_id": "UnknownObject0001",
            "name": "狗"
        }
    """
    try:
        data = request.json or {}
        object_id = str(data.get("object_id", "")).strip()
        name = str(data.get("name", "")).strip()

        if not object_id or not name:
            return jsonify({"error": "object_id 和 name 不能为空"}), 400

        result = vision_processor.confirm_object(object_id, name)
        _save()
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.exception(f"[Vision API] confirm: {e}")
        return jsonify({"error": str(e)}), 500


# =========================================================
# Action 动作注册表 API
# =========================================================

@app.route("/api/actions/registered", methods=["GET"])
def action_registered():
    """列出所有已注册的 Action + 图谱中所有 procedural 节点"""
    try:
        # Action 注册表中的
        reg_actions = list_actions()

        # 图谱中所有 procedural 节点
        graph_procedural = []
        with kg._lock:
            for nid, node in kg.nodes.items():
                if node.label == "procedural":
                    graph_procedural.append({
                        "node_id": nid,
                        "activation": round(node.activation, 4),
                        "has_execution": bool(node.execution),
                        "registered": nid in reg_actions,
                    })

        return jsonify({
            "success": True,
            "registered": reg_actions,
            "graph_procedural": graph_procedural,
            "count": len(reg_actions),
        })
    except Exception as e:
        logger.exception(f"[Action API] registered: {e}")
        return jsonify({"error": str(e)}), 500


# =========================================================
# 经验时间轴 / 因果发现 — debug API（§十四 可观测性）
# =========================================================

@app.route("/api/experience/timeline", methods=["GET"])
def experience_timeline_api():
    """最近的事件级经验（可按类型/来源过滤）。Timeline 是经历层，非 debug log。"""
    try:
        n = int(request.args.get("n", 40))
        etype = request.args.get("type") or None
        actor = request.args.get("actor") or None
        events = timeline.recent(n=n, event_type=etype, actor=actor)
        return jsonify({"success": True, "count": len(timeline), "events": events})
    except Exception as e:
        logger.exception(f"[Experience API] timeline: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/experience/causal", methods=["GET"])
def experience_causal_api():
    """因果学习报告：候选关联 / 聚合统计 / 假设 / 已晋升 KG 的泛化关系。"""
    try:
        return jsonify({"success": True, **causal_learner.debug_report()})
    except Exception as e:
        logger.exception(f"[Experience API] causal: {e}")
        return jsonify({"error": str(e)}), 500


# =========================================================
# 启动
# =========================================================

# 通道中枢在此启动：全部路由已注册完毕（修复启动竞态，见 ~812 行注释）
channel_hub.start()
logger.info("[Channel] 通道中枢启动（路由注册完成后）")

# ── 观测：SYSTEM 启动事件 + 周期摘要的图谱规模提供者 ──
try:
    with kg._lock:
        _fas_n, _fas_e = len(kg.nodes), len(kg.edges)
    fas_log.get_logger(fas_log.SYSTEM).info(
        "startup", f"FAS 装配完成（{_fas_n} 节点 / {_fas_e} 边）",
        nodes=_fas_n, edges=_fas_e, pid=os.getpid(),
        logging_level=fas_log.get_level(),
        text_policy=str(config.get("log_text_policy")))
    fas_log.register_summary_provider(lambda: {
        "graph_nodes": len(kg.nodes), "graph_edges": len(kg.edges)})
except Exception as _sue:
    logger.warning(f"[Boot] 启动观测/摘要提供者注册失败(仅观测): {_sue}")

# ── 学习实验：横幅 + 结构化目标注入（§20/§21；零 LLM，不走对话解析）──
try:
    if _xm.enabled():
        _xm.banner()
        if _xm.mode() == "learning_closed_loop":
            _xgoal = _xm.goal_obtain()
            if _xgoal and not any(
                    str(g.get("source") or "") == "experiment_obtain"
                    and str(g.get("type") or "") == "obtain"
                    and str(g.get("target") or "") == _xgoal
                    for g in autonomy.goals()):
                autonomy.add_goal({"type": "obtain", "target": _xgoal,
                                   "source": "experiment_obtain",
                                   "text": f"实验目标：自主获得 {_xgoal}"})
                _xm.xlog("GOAL", f"注入结构化目标 obtain:{_xgoal}")
except Exception as _xe:
    logger.warning(f"[Experiment] 目标注入失败: {_xe}")

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    # ── 监听地址 ──
    # 默认只监听回环（127.0.0.1）。原因：全部 145 个路由无鉴权，且 /api/nodes
    # 可写 procedural 节点的 execution（会被 exec 执行）——绑定 0.0.0.0 等于把
    # 代码注入面暴露到整个局域网。确需局域网访问（另一台机器开前端）才显式
    # 指定：FAS_HOST=0.0.0.0 python app.py，并自行确保网络环境可信。
    _host = (os.environ.get("FAS_HOST", "127.0.0.1") or "").strip() or "127.0.0.1"
    _host_is_local = _host in ("127.0.0.1", "localhost", "::1")

    logger.info("=" * 60)
    logger.info(
        f"Fascinator 后端启动 "
        f"(监听: {_host}:{port})"
    )
    logger.info("=" * 60)
    fas_log.get_logger(fas_log.SYSTEM).info(
        "server_listening", f"HTTP 服务监听 {_host}:{port}",
        host=_host, port=port, local=_host_is_local)

    # ── API 访问守卫引导（§18.2 #19，2026-09-22）──
    # Origin 核验常开（本机也挡跨源 CSRF/DNS rebinding）；token 只在
    # 非回环来源强制（回环零摩擦不变，api_token_require_loopback 收紧）。
    # 非回环绑定且未配 token → 生成一次并持久化 data/api_token.json。
    _token = _ensure_api_token_for_bind(_host_is_local)
    if not _host_is_local:
        logger.warning(
            f"[SECURITY] 正在监听非回环地址 {_host}：局域网来源需 token，"
            f"Origin 核验已常开。首次请在浏览器打开 "
            f"http://{_host}:{port}/?token={_token} 换取会话 cookie"
            f"（token 已存 data/api_token.json，删文件重启=轮换）。"
        )
    elif config.get("api_token_require_loopback", False):
        logger.info("[SECURITY] 回环 token 已启用（api_token_require_loopback）")

    # ⚠️ 禁止 Flask debug reloader
    # 否则会启动两个扩散线程

    # ── 构建 Embedding 索引 ──
    # 索引是派生缓存，图谱是唯一真源：签名一致复用缓存，不一致全量重建。
    try:
        emb_mgr.refresh(kg)
    except Exception as e:
        logger.warning(f"[Embedding] 索引构建跳过 (可能缺少 sentence-transformers/faiss): {e}")

    # ── 前端地址：明确打印 + 可选自动打开浏览器 ──
    # Flask 自己的横幅只列监听地址（0.0.0.0:5000），不说"在浏览器里打开这个"，
    # 而服务常被后台启动，控制台看不到——所以这里给出可直接点的前端链接。
    _local_url = f"http://127.0.0.1:{port}/"
    try:
        import socket as _socket
        _lan_ip = _socket.gethostbyname(_socket.gethostname())
    except Exception:
        _lan_ip = None
    logger.info("=" * 60)
    logger.info(f"前端地址：{_local_url}")
    if _host_is_local:
        logger.info("局域网内其它设备不可访问（如需开放：FAS_HOST=0.0.0.0 并自担风险）")
    elif _lan_ip:
        _entry = f"/?token={_token}" if _token else "/"
        logger.info(f"局域网内其它设备：http://{_lan_ip}:{port}{_entry}")
    logger.info("=" * 60)

    # 自动打开浏览器（config.auto_open_browser，默认开）。调试重启时嫌烦可：
    # （浏览器自动打开挪到端口预检之后——拒启时不该白开页面。）

    # ── 单实例端口预检（2026-09-22 收尾）────────────────────────
    # 双进程抢端口曾造成"杀错进程丢内存态词表"事故（见 docs 服务器进程坑）。
    # Werkzeug 在 Windows 上带 SO_REUSEADDR 能"劫持"已监听端口——预检用不带
    # 该选项的普通 bind 探测，被占则直接拒绝启动，而不是悄悄起第二个大脑。
    try:
        import socket as _probe_sock
        _probe = _probe_sock.socket(
            _probe_sock.AF_INET, _probe_sock.SOCK_STREAM)
        try:
            _probe.bind((_host, port))
        finally:
            _probe.close()
    except OSError as _pe:
        logger.error(
            f"[启动] 端口 {_host}:{port} 已被占用（{_pe}）——疑似另一个 FAS "
            f"实例在运行。请先确认/停止旧实例，或换 PORT 再启动；"
            f"绝不要盲杀 python/node 进程（内存态图谱/词表会丢）。")
        _BOOT_REJECTED = True   # 拒启实例禁止落盘（防旧副本覆盖在跑实例）
        raise SystemExit(2)

    # 预检通过才挂浏览器定时器（否则拒启时会白开一个连不上的页面）
    #   设 config 为 false，或起服务时带环境变量 FAS_NO_BROWSER=1。
    if config.get("auto_open_browser", True) and not os.environ.get("FAS_NO_BROWSER"):
        def _open_browser():
            import webbrowser
            try:
                webbrowser.open(_local_url)
            except Exception as _we:
                logger.warning(f"[启动] 自动打开浏览器失败（请手动访问 {_local_url}）: {_we}")
        # 等 Flask 真正开始监听再开，避免浏览器抢在前面拿到连接失败
        threading.Timer(1.5, _open_browser).start()
        logger.info("[启动] 1.5 秒后自动打开浏览器（FAS_NO_BROWSER=1 可关闭）")

    try:
        app.run(
            host=_host,
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("[Shutdown] Ctrl-C：优雅停机（循环/通道/日志收口）")
    finally:
        # ── 优雅停机（2026-09-22 收尾）──
        # 图谱落盘与 fas_log 收口已有 atexit；这里补停两条自持循环线程，
        # 并给在飞动作一次"如实取消"的机会（不假成功，不静默丢结算）。
        try:
            action_manager.cancel("FAS 停机")
        except Exception as _sce:
            logger.warning(f"[Shutdown] 行动取消失败(在飞动作未结算): {_sce}")
        try:
            channel_hub.stop()
        except Exception as _che:
            logger.warning(f"[Shutdown] 通道中枢停止失败: {_che}")
        try:
            cc.stop()
        except Exception as _cce:
            logger.warning(f"[Shutdown] CC 循环停止失败: {_cce}")
        try:
            import minecraft.bridge as _mb
            if _mb.health():
                _mb.call("/quit", {}, timeout=2)   # 不留孤儿 node/bot 进程
        except Exception as _mbe:
            logger.warning(f"[Shutdown] MC 桥退出请求失败(可能留孤儿进程): {_mbe}")
        fas_log.get_logger(fas_log.SYSTEM).info(
            "server_stopped", "HTTP 服务退出，循环/通道已停")



