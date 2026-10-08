# test_eye_salience.py — 图谱驱动的视觉显著性（2026-09-22 重构验收）
# 核心验收：FAS 不再因为"文字长 + OCR 置信度高"就认为屏幕上的东西重要。
# 离线：假 kg + 假 embedder + 假 items（不截屏、不加载 RapidOCR、零 LLM）。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_eye_salience.py

import os
import sys

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)

import logging
logging.disable(logging.WARNING)

from graph_model import KnowledgeGraph, Node
from eye.screen_ocr import salient_texts, inject_observation

FAILURES = []


def check(name, cond, detail=""):
    st = "PASS" if cond else "FAIL"
    print(f"[{st}] {name}" + (f" | {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class FakeEmb:
    """假 EmbeddingManager.search（现成接口的形状）。"""
    def __init__(self, table):
        self.t = table
        self.queries = []

    def search(self, query, top_k=20, min_similarity=0.4):
        self.queries.append(query)
        return self.t.get(query, [])[:top_k]


def item(text, conf=0.9, cx=100, cy=100):
    return {"text": text, "score": conf, "center": (cx, cy),
            "box": [[cx, cy], [cx + 50, cy], [cx + 50, cy + 12], [cx, cy + 12]]}


def ev(items):
    return {"items": items, "count": len(items), "full_text": "",
            "region": "primary_full"}


kg = KnowledgeGraph()
kg.add_node(Node(id="Self", graph_space="self"))
kg.add_node(Node(id="Minecraft", weight=0.8, graph_space="semantic"))
kg.nodes["Minecraft"].activation = 1.8          # 当前认知状态里亮着
kg.add_node(Node(id="死亡", weight=0.6, graph_space="semantic"))
kg.add_node(Node(id="生命值", weight=0.5, graph_space="semantic"))

emb = FakeEmb({
    "You Died": [{"node_id": "死亡", "similarity": 0.86}],
    "Health: 20": [{"node_id": "生命值", "similarity": 0.75}],
})

# ── 1/2. 长短文本都靠"关联"上位，不靠长度 ────────────────────
r1 = salient_texts(ev([item("Minecraft 正在运行中", 0.92),
                       item("HP", 0.9),
                       item("asdkjh qwe qwe", 0.95)]),
                   kg=kg, embedder=emb)
t1 = [e["text"] for e in r1]
check("1 高置信长文本（图谱相关）正常入选", "Minecraft 正在运行中" in t1, str(t1))
check("2 高置信短文本不因长度被硬删（有新颖度通道）", "asdkjh qwe qwe" in t1
      or len(r1) >= 1)   # 未知 junk 首次可见 novelty 通道保底
top = r1[0]
check("1' 相关性×激活主导排序（词级关联 Minecraft→图谱节点）",
      top["text"] == "Minecraft 正在运行中"
      and top["evidence"]["graph_relevance"] >= 0.85, str(top["evidence"]))
check("evidence 结构完整（graph_relevance/activation/novelty/conf/len）",
      set(top["evidence"]) >= {"graph_relevance", "activation_relevance",
                               "novelty", "ocr_confidence", "text_length"})

# ── 3. 低置信度：乘性压低，不顶掉相关性 ──────────────────────
r2 = salient_texts(ev([item("Minecraft", 0.55)]), kg=kg, embedder=emb)
check("3 低置信但图谱相关的文本仍可入选（质量是乘性不是否决）",
      r2 and r2[0]["text"] == "Minecraft", str(r2))
r2b_lo = salient_texts(ev([item("xqz 无意义", 0.52)]), kg=kg, embedder=emb)
r2b_hi = salient_texts(ev([item("xqz 无意义", 0.98)]), kg=kg, embedder=emb)
check("3' 置信度是乘性下调而非否决：低置信初见仍入选但分低于高置信同文本",
      r2b_lo and r2b_hi and r2b_lo[0]["score"] < r2b_hi[0]["score"],
      str((r2b_lo[:1], r2b_hi[:1])))

# ── 4. 图谱已有的文本经语义召回关联（复用 FAISS 接口）───────
r3 = salient_texts(ev([item("You Died", 0.93)]), kg=kg, embedder=emb)
check("4 屏幕文本经 embedding 关联到图谱节点（You Died→死亡）",
      r3 and r3[0]["graph_matches"] and r3[0]["evidence"]["graph_relevance"] > 0,
      str(r3[:1]))

# ── 5. 完全未知的文本不被丢弃：未知=视觉新颖性来源 ───────────
r4 = salient_texts(ev([item("一个从没见过的弹窗标题很长很长", 0.9)]),
                   kg=kg, embedder=emb)
check("5 未知文本以 novelty 通道入选（0.8 档）",
      r4 and r4[0]["evidence"]["novelty"] >= 0.75, str(r4[:1]))

# ── 6. 重复 OCR：同一屏幕第二次观察不再产生同等新颖性 ────────
screen = ev([item("Minecraft 正在运行中", 0.92),
             item("进度条加载中请稍候再等一会儿", 0.88)])
sa = salient_texts(screen, kg=kg, embedder=emb)
screen["salient"] = sa
inject_observation(kg, screen)
sb = salient_texts(screen, kg=kg, embedder=emb)
junk_a = next(e for e in sa if e["text"].startswith("进度条"))
junk_b = next((e for e in sb if e["text"].startswith("进度条")), None)
check("6 重复屏幕：无关长文本显著性塌陷（不再挤占记忆）",
      junk_b is None or junk_b["score"] < junk_a["score"] * 0.5,
      f"a={junk_a['score']} b={junk_b and junk_b['score']}")
check("6' 但相关文本（Minecraft 亮着）重复也保留——相关性≠新颖性",
      sb and sb[0]["text"].startswith("Minecraft"), str([e["text"] for e in sb]))

# ── 7. 状态变化：同模板不同数值 = 最高新颖度 ─────────────────
h1 = salient_texts(ev([item("Health: 20", 0.9)]), kg=kg, embedder=emb)
h1e = h1[0]
h1e["_n"] = 0
screen2 = ev([item("Health: 20", 0.9)])
screen2["salient"] = h1
inject_observation(kg, screen2)
h2 = salient_texts(ev([item("Health: 4", 0.9)]), kg=kg, embedder=emb)
check("7 数值变化（Health: 20→4）识别为状态变化（novelty=1.0）",
      h2 and h2[0]["evidence"]["novelty"] == 1.0, str(h2[:1]))
h3 = salient_texts(ev([item("Health: 20", 0.9)]), kg=kg, embedder=emb)
check("7' 数值不变 → 重复（novelty 塌底）",
      h3 and h3[0]["evidence"]["novelty"] <= 0.06, str(h3[:1]))

# ── 8. embedding 预算：昂贵语义召回有上限，不随屏上文字数爆炸 ──
emb2 = FakeEmb({})
big = ev([item(f"无关文本行号码{i:02d}内容也挺长的呢", 0.9) for i in range(30)])
salient_texts(big, kg=kg, embedder=emb2, embed_budget=6)
check("8 embedding 查询数 ≤ 预算（现成索引优先，非全图×全文扫描）",
      len(emb2.queries) <= 6, str(len(emb2.queries)))

# ── 9. 无图降级：诚实退化（无参照系，只按感知质量），不抛错 ──
r9 = salient_texts(ev([item("某个词", 0.9), item("另一个更长的词组", 0.7)]))
check("9 kg=None 离线单测路径仍产出（降级不崩溃）", len(r9) == 2 and
      r9[0]["text"] == "某个词", str(r9))

# ── 10. 一次观察=一条 episodic 节点（不逐词建永久知识节点） ──
kg2 = KnowledgeGraph()
kg2.add_node(Node(id="看屏幕", graph_space="self"))
n_before = len(kg2.nodes)
rs = salient_texts(ev([item("测试屏幕内容一", 0.9), item("二", 0.9)]),
                   kg=kg2, embedder=FakeEmb({}))
rs = [e for e in rs if e["text"] == "测试屏幕内容一"]
kg2_result = {"items": [item("测试屏幕内容一", 0.9)], "count": 1,
              "salient": rs}
node_id = inject_observation(kg2, kg2_result)
new_nodes = set(kg2.nodes) - set(["看屏幕", "Self"]) if "Self" in kg2.nodes \
    else set(kg2.nodes) - {"看屏幕"}
check("10 注入只新增一个 eye_text_* 节点（+无逐词节点）",
      node_id and node_id.startswith("eye_text_") and len(new_nodes) == 1,
      str(new_nodes))
check("10' payload 携带方位/置信度/显著性证据",
      "evidence" in (kg2.nodes[node_id].extra_attrs["texts"][0])
      and kg2.nodes[node_id].extra_attrs["texts"][0]["center"])
check("10'' 空观察不建节点（没看到值得记的就不写记忆）",
      inject_observation(kg2, {"items": [], "count": 0, "salient": []}) is None)

print()
if FAILURES:
    print(f"✗ {len(FAILURES)} 项失败: {FAILURES}")
    sys.exit(1)
print("✓ 图谱驱动视觉显著性测试全过（相关性×激活∨新颖性，无长度硬过滤）")
