# eye/screen_ocr.py — FAS 屏幕感知器（截图 + OCR 文字识别 + 方位 + 图谱显著性）
# ============================================================================
# 万物皆图：本模块是"眼睛"感知器的执行器（与 ear/vision 同模式）。
#
# 分工（2026-09-22 重构，用户设计准则）：
#   **OCR 负责"看见什么"，图谱负责"什么值得注意"。**
#   旧显著性 = len(text) × confidence + 硬长度过滤——一套与认知体系平行的
#   视觉打分器，已废弃。现在每条 OCR 文本只是 VisualTextCandidate，
#   显著性由它与激活的认知图谱的关联决定：
#     图谱关联（名字精确匹配 → 已有 FAISS 语义召回，复用现成索引，
#       不做"OCR文本×全图节点"扫描）
#     ×当前激活（节点能量=当下认知状态；近期性/相关性都活在这条信号里）
#     ∨视觉新颖性（与近期 eye_text_* 情景记忆对比：重复=低，
#       数值/状态变化=最高——"Health: 20→Health: 4""Minecraft→You Died"）
#     ×OCR 质量（置信度乘性）
#     ±长度软因子（仅作 tie-break，不再硬删——"死""HP"一个字也允许入围）
#   关联不上的未知文本不因"不认识"被丢弃：未知本身就是视觉新颖性来源。
#
# 触发与注入：对话路径（app.py 概念层 SCREEN_OBSERVE）与自主路径
# （eye/observer.py → ActionManager）共用本文件的同一条流水线。
#
# 隐私：图像仅在本机内存中处理，不落盘、不上传。
# ============================================================================

import logging
import re
import threading

logger = logging.getLogger(__name__)

_ocr = None
_ocr_lock = threading.Lock()

# 文本→图谱关联的进程内缓存（纯计算缓存：同样的词不必二次 embedding）
_ASSOC_CACHE: "dict[str, tuple]" = {}
_ASSOC_CACHE_CAP = 512

# 数字归一（"Health: 20"→"Health: #"）——同模板不同数值 = 状态变化
_DIGITS_RE = re.compile(r"\d+(?:\.\d+)?")

# 词级探测用切分（词表级 O(1) 查点，不是全图扫描）
_TOKEN_SPLIT_RE = re.compile(r"[\s，。、,:;:|/\\\-()（）\[\]{}【】\"'“”]+")


def _template(text: str) -> str:
    return _DIGITS_RE.sub("#", str(text or "").strip())


def _get_ocr():
    """懒加载 RapidOCR（模型首次加载约 1-2 秒）。"""
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
            logger.info("[Eye] RapidOCR 初始化完成")
        return _ocr


def capture_screen(region=None) -> object:
    """截取屏幕。region=(x, y, w, h) 可选，默认主屏全屏。返回 PIL Image。"""
    import mss
    from PIL import Image
    with mss.mss() as sct:
        if region:
            x, y, w, h = region
            monitor = {"left": x, "top": y, "width": w, "height": h}
        else:
            monitor = sct.monitors[1]  # 主屏
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    return img


def recognize_text(region=None, min_score: float = 0.5) -> dict:
    """截图 + OCR。返回 {items: [{text, box, score, center}], full_text, count}。

    box 为四点坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]，
    center 为文字中心点 (cx, cy)——即"文字的方位"。
    每条 item 就是一个 VisualTextCandidate（结构已够，不另立 class）。
    """
    img = capture_screen(region)
    ocr = _get_ocr()
    result, _ = ocr(img)
    items = []
    for box, text, score in (result or []):
        score = float(score)
        if score < min_score or not str(text).strip():
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({
            "text": str(text).strip(),
            "box": box,
            "center": (round(sum(xs) / 4), round(sum(ys) / 4)),
            "score": round(score, 3),
        })
    # 按阅读顺序排序：先上后下、同高先左
    items.sort(key=lambda it: (it["center"][1] // 20, it["center"][0]))
    return {
        "items": items,
        "full_text": "\n".join(it["text"] for it in items),
        "count": len(items),
        "region": list(region) if region else "primary_full",
    }


# ── 图谱关联（复用现有索引，不建第二套知识库）────────────────

def _associate(text: str, kg, embedder, use_embed: bool = True) -> list:
    """一条屏幕文字 → [(node_id, relevance 0~1)]。

    两级，全部走现成设施：
      1. 名字精确命中（kg.get_node O(1)）→ relevance 1.0；
      2. 现有 FAISS EmbeddingManager.search（全图节点向量索引）→
         similarity 0.58 起，坡道映射到 relevance。
    绝不遍历全图节点做子串匹配；embedding 调用有预算（见 salient_texts）。
    """
    key = text.strip()
    if not key:
        return []
    cached = _ASSOC_CACHE.get(key + ("\x00e" if use_embed else "\x00n"))
    if cached is not None:
        return cached
    hits = []
    if kg is not None:
        try:
            if kg.get_node(key) is not None:
                hits.append((key, 1.0))
            else:
                # 词级探测（仍是 O(1) dict 查询）："Minecraft 正在运行中"
                # 经 token "Minecraft" 关联到图上节点；不做全图子串扫描
                for tok in _TOKEN_SPLIT_RE.split(key):
                    tok = tok.strip()
                    if len(tok) >= 2 and tok != key and kg.get_node(tok) is not None:
                        hits.append((tok, 0.9))
                        break
        except Exception:
            pass
    if not hits and use_embed and embedder is not None:
        try:
            for h in embedder.search(key, top_k=3, min_similarity=0.58):
                sim = float(h.get("similarity") or 0.0)
                # 0.58→0.45，0.93+→1.0（低于 0.58 的弱相似不值得当关联）
                rel = max(0.0, min(1.0, (sim - 0.58) / 0.35 * 0.55 + 0.45))
                if rel > 0:
                    hits.append((h.get("node_id"), round(rel, 3)))
        except Exception as e:
            logger.debug(f"[Eye] 语义关联失败（跳过）: {e}")
    out = hits[:3]
    if len(_ASSOC_CACHE) >= _ASSOC_CACHE_CAP:
        _ASSOC_CACHE.clear()
    _ASSOC_CACHE[key + ("\x00e" if use_embed else "\x00n")] = out
    return out


def _recent_visual_state(kg, recent_n: int = 6):
    """从既有 episodic eye_text_* 节点读"最近看过什么"（新颖性参照系）。

    不新建视觉状态库——情景记忆本身就是记忆的真相源。
    返回 (seen_raw: set, seen_tpl: {template: set(raw)})。
    """
    if kg is None:
        return set(), {}
    try:
        def _ts(nid):
            try:
                return int(str(nid).split("_")[2])
            except (IndexError, ValueError):
                return 0
        with kg._lock:
            ids = sorted((nid for nid in kg.nodes
                          if str(nid).startswith("eye_text_")),
                         key=_ts)[-recent_n:]
            nodes = [kg.nodes[i] for i in ids]
    except Exception:
        return set(), {}
    seen_raw, tpl = set(), {}
    for n in nodes:
        ea = n.extra_attrs or {}
        for t in (ea.get("texts") or []):
            raw = str(t.get("text") or "").strip()
            if not raw:
                continue
            seen_raw.add(raw)
            tpl.setdefault(_template(raw), set()).add(raw)
        # 全量文本快照（未入选显著性的也在）：否则重复的 junk 永远"新颖"
        for raw in (ea.get("seen_all") or []):
            raw = str(raw or "").strip()
            if not raw:
                continue
            seen_raw.add(raw)
            tpl.setdefault(_template(raw), set()).add(raw)
    return seen_raw, tpl


def _novelty(text: str, seen_raw: set, seen_tpl: dict) -> float:
    """视觉新颖性：与近期屏幕记忆对比（§七：状态变化>新出现>重复）。"""
    if not seen_raw and not seen_tpl:
        return 0.8          # 没有近期记忆：一切皆初见（封顶低于相关命中）
    if text in seen_raw:
        return 0.05         # 完全重复：不该再当成新事件
    tpl = _template(text)
    others = seen_tpl.get(tpl)
    if others is not None:
        # 同模板不同数字/值：Health: 20 → Health: 4 = 状态变化，最高
        return 1.0 if text not in others else 0.05
    return 0.8              # 全新出现的文本


def salient_texts(eye_result: dict, top_n: int = 5, kg=None,
                  embedder=None, recent_n: int = 6,
                  embed_budget: int = 14) -> list:
    """图谱感知的显著性排序（替代旧 len×conf 死公式）。

    证据结构（每项随结果返回，进 episodic 节点，供认知层/审计读取）：
      graph_relevance      最强图谱关联（名字精确=1.0 / FAISS 语义映射）
      activation_relevance 命中节点的当前激活（/3 归一）——"现在在想什么"
      novelty              与近期屏幕记忆的新颖度（见 _novelty）
      ocr_confidence       感知质量（乘性，不做主排序依据）
      text_length          长度软因子（仅 tie-break；1 字轻罚，不硬删）

    合成是结构化的，不是九项线性加权和：
      相关通道 g = rel × (0.55 + 0.45·act)     图谱亮且被想到才算"相关"
      新颖通道 f = 0.60 × novelty              未知/变化的保底（封顶低于
                                               强相关项，重复项自动沉底）
      score = max(g, f) × (0.78 + 0.22·conf) + 长度tie − 短噪声软罚
    kg=None（离线单测/无图降级）：无语义参照系可用，只按感知质量排序，
    这是诚实的退化路径而不是第二套打分系统。

    性能：embedding 查询有预算（embed_budget，按文本长度降序优先），
    名字精确命中零成本；关联结果进程内缓存。
    """
    items = list(eye_result.get("items") or [])
    if not items:
        return []
    with_embed = set()
    if kg is not None and embedder is not None:
        # 预算给"最可能承载信息"的长文本；短文本靠精确名匹配兜底
        budget_order = sorted(items, key=lambda it: -len(it["text"]))
        for it in budget_order[:max(0, int(embed_budget))]:
            with_embed.add(it["text"].strip())
    seen_raw, seen_tpl = _recent_visual_state(kg, recent_n)

    scored = []
    for it in items:
        text = str(it.get("text") or "").strip()
        if not text:
            continue
        conf = float(it.get("score") or 0.0)
        rel = 0.0
        act = 0.0
        matches = []
        if kg is not None:
            hits = _associate(text, kg,
                              embedder if text in with_embed else None)
            matches = [{"node": nid, "rel": r} for nid, r in hits]
            rel = max((r for _n, r in hits), default=0.0)
            try:
                with kg._lock:
                    act = max((min(1.0, float(kg.nodes[nid].activation or 0) / 3.0)
                               for nid, _r in hits if nid in kg.nodes),
                              default=0.0)
            except Exception:
                pass
            nov = _novelty(text, seen_raw, seen_tpl)
        else:
            nov = 0.5      # 无图：无参照系可用，中性；退化为按质量排
        g = rel * (0.55 + 0.45 * act)
        f = 0.60 * nov
        base = max(g, f) if kg is not None else (0.3 + 0.5 * conf)
        s = base * (0.78 + 0.22 * conf)
        s += 0.04 * min(1.0, len(text) / 24.0)   # 长度只做微弱 tie-break
        if len(text) <= 1:
            s -= 0.05                            # 单字符：软罚，不是删除
        if kg is not None and rel == 0.0 and nov <= 0.06:
            s -= 0.10                            # 重复且无关：沉底
        if s < 0.04:
            continue                              # 噪声地板（非硬长度过滤）
        scored.append({
            "text": text,
            "center": it.get("center"),
            "box": it.get("box"),
            "confidence": round(conf, 3),
            "score": round(s, 3),
            "evidence": {
                "graph_relevance": round(rel, 3),
                "activation_relevance": round(act, 3),
                "novelty": round(nov, 3),
                "ocr_confidence": round(conf, 3),
                "text_length": len(text),
            },
            "graph_matches": matches[:3],
        })
    scored.sort(key=lambda e: -e["score"])
    seen, out = set(), []
    for e in scored:
        if e["text"] in seen:
            continue
        seen.add(e["text"])
        out.append(e)
        if len(out) >= int(top_n):
            break
    return out


def inject_observation(kg, eye_result: dict, engine=None):
    """一次观察 = 一条视觉情景记忆（eye_text_* episodic 节点）。

    文字不逐条建永久知识节点（防图谱爆炸）；每条显著文本携带
    方位 + 置信度 + 显著性证据 + 图谱关联，让"当时注意到什么、
    为什么注意到"可以被回忆和审计。engine 给定时点亮"看屏幕"能力
    节点并 mark_active（激活直写点必须进前沿——不变量）。
    """
    import time as _t
    salient = eye_result.get("salient") or []
    if not salient:
        return None
    node_id = f"eye_text_{int(_t.time())}"
    if kg.get_node(node_id) is not None:   # 同秒二次观察
        node_id = f"eye_text_{int(_t.time())}_{salient[0]['score']}"
    from graph_model import Node
    kg.add_node(Node(
        id=node_id,
        weight=0.4,
        label="declarative-episodic",
        graph_space="episodic",
        extra_attrs={
            "type": "eye_perception",
            "count": eye_result.get("count", 0),
            "texts": [{
                "text": str(s.get("text"))[:120],
                "center": list(s.get("center") or []),
                "box": s.get("box"),
                "confidence": s.get("confidence"),
                "salience": s.get("score"),
                "evidence": s.get("evidence"),
                "graph_matches": (s.get("graph_matches") or [])[:2],
            } for s in salient],
            # 全量视图快照（纯文本，限量）：新颖性参照系。没有它，
            # 落选噪声会在每次观察里永远"崭新"。
            "seen_all": [str(it.get("text") or "").strip()[:80]
                         for it in (eye_result.get("items") or [])][:60],
        }))
    if engine is not None:
        try:
            node = kg.get_node("看屏幕")
            if node is not None:
                node.activation = min(5.0, float(node.activation or 0.0) + 3.0)
                node.touch()
                engine.mark_active([node.id])
        except Exception as e:
            logger.debug(f"[Eye] 看屏幕激活跳过: {e}")
    logger.info(f"[Eye] 屏幕记录入图: {node_id} "
                f"({eye_result.get('count')} 条文字, {len(salient)} 条显著)")
    return node_id
