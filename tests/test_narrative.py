# test_narrative.py — 叙事分段抽取测试（切分+抽取合并逻辑；图构建由实机验证）
import sys, os, json
_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)
from nlp_processor import NLPProcessor

fail = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail and not cond else ""))
    if not cond:
        fail.append(name)

# ── 分段函数 ──
chunks = NLPProcessor.split_narrative( "第一句话。第二句话。第三句话。" * 200)
check("长文本切分为多段", len(chunks) >= 2, str(len(chunks)))
check("每段不超限", all(len(c) <= 620 for c in chunks))
check("句子不被切断", all(c.rstrip().endswith(("。", "！", "？", "!", "?")) for c in chunks))

# ── 抽取合并（桩 LLM 返回固定 JSON）──
import json as _json
CANNED = [
    {"story": "星际迷航", "characters": ["船长", "AI官"],
     "events": [{"title": "飞船出发", "actors": ["船长"], "summary": "飞船离开空间站", "entities": ["飞船"]}]},
    {"story": None, "characters": [],
     "events": [{"title": "遭遇陨石", "actors": ["船长", "AI官"], "summary": "遇到陨石带", "entities": []},
                {"title": "紧急规避", "actors": ["AI官"], "summary": "AI官规避成功", "entities": []}]},
]
from langchain_core.runnables import RunnableLambda
class _Resp:
    def __init__(self, d): self.content = _json.dumps(d, ensure_ascii=False)
def _make_chat(outs, counter):
    def _gen(*a, **k):
        r = _Resp(outs[min(counter[0], len(outs)-1)]); counter[0] += 1; return r
    return RunnableLambda(_gen)

class NlpStub:
    split_narrative = staticmethod(NLPProcessor.split_narrative)
    def __init__(self, outs):
        counter = [0]
        self.chat_llm = _make_chat(outs, counter)

stub = NlpStub([CANNED[0], CANNED[1]])
nd = NLPProcessor.extract_narrative(stub, "剧情文本。" * 160)
check("story 提取", nd["story"] == "星际迷航")
check("characters 合并去重", nd["characters"] == ["船长", "AI官"])
check("事件跨段按序合并", len(nd["events"]) == 3
      and nd["events"][0]["title"] == "飞船出发"
      and nd["events"][2]["title"] == "紧急规避")
check("事件带段号", nd["events"][1]["_chunk"] == 1)

# 畸形输出健壮性：一段坏 JSON + 一段好 JSON
stub2 = NlpStub(["不是JSON{{{", CANNED[0]])
nd2 = NLPProcessor.extract_narrative(stub2, "文本。" * 220)
check("坏JSON段被跳过不崩溃", nd2["story"] == "星际迷航" and len(nd2["events"]) == 1)

print()
if fail:
    print(f"✗ {len(fail)} 项失败: {fail}"); sys.exit(1)
print("✓ 叙事抽取测试全过（图构建由实机验证）")
