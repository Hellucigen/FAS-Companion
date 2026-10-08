# tests/test_fas_log.py — 统一运行日志系统六项验证（任务书 §27）
# ============================================================================
# 用合成数据驱动真实模块（internal_state / continuous_cognition /
# action_system / llm_provider / minecraft.bridge / eye.observer / fas_log），
# 断言日志能重建因果链。
# 严禁编造用户真实生活事件——输入均为显式假设的通用语料。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_fas_log.py
# ============================================================================

import os
import sys
import json
import time
import threading

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}"
          + ("" if cond or not detail else f" | {detail}"))
    if not cond:
        FAILURES.append(name)


import tempfile
import fas_log

LOGDIR = tempfile.mkdtemp(prefix="faslog_test_")
assert fas_log.setup(log_dir=LOGDIR, level="INFO",
                     summary_interval_s=0, console=False), "setup 失败"
fas_log.install_excepthooks()
fas_log.set_level("DEBUG", "system")   # 放开 trace_start 便于验证链路端点


def read_all():
    p = os.path.join(LOGDIR, "fas_all.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _qsize():
    try:
        return fas_log._ST.q.qsize()
    except Exception:
        return 0


def drain(sec=0.8):
    """等异步落盘队列清空。"""
    deadline = time.time() + sec
    while time.time() < deadline:
        if _qsize() == 0:
            break
        time.sleep(0.02)
    time.sleep(0.05)


# ═══ Test 1: 对话链路 —— input→cognition→decision→llm→response 同 trace ═══
from internal_state import InternalState

ist = InternalState(data_dir=tempfile.mkdtemp(prefix="ist_t1_"))
tid1 = fas_log.new_trace("dialogue")
cid1 = ist.begin_cycle("turn", {"text_len": 9, "channel": "web"})

lg = fas_log.get_logger
_long = "今天天气不错，我们聊聊 Minecraft 吧。" * 20
lg(fas_log.INPUT).info("input_received", "用户消息", source="web",
                       text=fas_log.text(_long), text_len=len(_long))
lg(fas_log.COGNITION).info("cycle_start", "回合开始", trigger="user_message")
lg(fas_log.COGNITION).info("context_built", "解析完成", parsed_nodes=4)
lg(fas_log.ACTIVATION).info("diffusion_summary", "扩散", seeds=["Minecraft", "天气"],
                            top=["Minecraft", "天气"], max_depth=6)


# 用桩 LLM 客户端走真实 _create 观测路径（类结构完整性也顺带验证）
class _Msg:
    def __init__(self, c):
        self.content = c


class _Choice:
    def __init__(self, c, fr="stop"):
        self.message = _Msg(c)
        self.finish_reason = fr


class _Usage:
    prompt_tokens = 120
    completion_tokens = 34
    total_tokens = 154


class _Resp:
    def __init__(self, c="好呀"):
        self.choices = [_Choice(c)]
        self.usage = _Usage()


class _Completions:
    def __init__(self, mode="ok"):
        self.mode = mode
        self.n = 0

    def create(self, **kw):
        self.n += 1
        if self.mode == "json_fail_first" and self.n == 1 \
                and "response_format" in kw:
            raise RuntimeError("400 response_format not supported")
        if self.mode == "always_fail":
            raise ConnectionError("provider down")
        return _Resp()


class _Chat:
    def __init__(self, mode):
        self.completions = _Completions(mode)


class _Client:
    def __init__(self, mode="ok"):
        self.chat = _Chat(mode)


def _backend(mode="ok"):
    import llm_provider as lp
    b = object.__new__(lp.MiMoBackend)
    b._model = "mimo-stub"
    b._temperature = 0.1
    b._max_tokens = 64
    b._top_p = 0.9
    b._json_mode = True
    b._json_ok = True
    b._llm = None
    b._raw_client = _Client(mode)
    return b


b1 = _backend("ok")
with fas_log.llm_purpose("dialogue_decomposition"):
    ans = b1.invoke("解析这段输入：……")
check("T1 真实 invoke 路径返回语义不变", ans == "好呀")
lg(fas_log.DECISION).info("decision_finished", "裁决", decision="respond",
                          desire=0.72, rejected=["explore_ask"])
lg(fas_log.INPUT).info("response_sent", "回复", answered=True,
                       answer=fas_log.text("是呀，今天很适合进世界看看"))
lg(fas_log.COGNITION).info("cycle_end", "回合结束", answered=True,
                           llm_calls=fas_log.trace_count("llm_calls"))
ist.end_cycle(cid1, outcome={"answered": True}, persist=False)
drain()

r1 = [r for r in read_all() if r.get("trace_id") == tid1]
chain = [r["event"] for r in r1]
check("T1 trace 串起完整对话链",
      all(x in chain for x in ("input_received", "cycle_start",
                               "context_built", "diffusion_summary",
                               "llm_call_finished", "decision_finished",
                               "response_sent", "cycle_end")), str(chain))
order = {e: chain.index(e) for e in set(chain)}
check("T1 事件顺序=因果顺序",
      order["trace_start"] < order["input_received"] < order["cycle_start"]
      < order["context_built"] < order["diffusion_summary"]
      < order["llm_call_finished"] < order["decision_finished"]
      < order["response_sent"] < order["cycle_end"], str(chain))
check("T1 begin_cycle 后全链共享同一 cycle_id",
      all(r.get("cycle_id") == cid1 for r in r1
          if r["event"] != "trace_start"),
      str([(r["event"], r.get("cycle_id")) for r in r1
           if r.get("cycle_id") != cid1 and r["event"] != "trace_start"]))
llm_r = next(r for r in r1 if r["event"] == "llm_call_finished")
check("T1 LLM 记录 purpose/tokens/latency",
      llm_r["data"]["purpose"] == "dialogue_decomposition"
      and llm_r["data"]["total_tokens"] == 154
      and llm_r["data"]["latency_ms"] >= 0, str(llm_r.get("data")))
ir = next(r for r in r1 if r["event"] == "input_received")
check("T1 长文本按策略截断并附哈希", "…[sha1:" in ir["data"]["text"]
      and ir["data"]["text_len"] == len(_long), ir["data"]["text"][:60])
check("T1 周期结束后 cycle contextvar 清空", fas_log.get_cycle() is None)
check("T1 trace 计数=本轮 LLM 调用数",
      fas_log.trace_count("llm_calls") >= 1)

# ═══ Test 2: 自主链路 —— CC 脉冲周期 + 动作候选→开始→结算 同 trace ═══
from graph_model import KnowledgeGraph, Node
from continuous_cognition import ContinuousCognition


class FN:
    def __init__(self, i, a, s="semantic"):
        self.id, self.activation, self.graph_space = i, a, s


class FakeEngine:
    def __init__(self):
        self.topk = []
        self._running = False
        self._lock = threading.RLock()

    def register_activation_source(self, ids, source_type="external_input"):
        pass

    def get_topk(self, k=15):
        return self.topk[:k], []

    def decay_step(self):
        pass

    def diffuse_step(self):
        pass

    def mark_active(self, ids):
        pass

    def mark_edges_active(self, edges):
        pass

    def clear_anchors(self):
        pass


class _R:
    content = "（随口说一句）"


class FakeNLP:
    def __init__(self):
        self.calls = 0
        from langchain_core.runnables import RunnableLambda
        self.chat_llm = RunnableLambda(self._gen)

    def _gen(self, *a, **k):
        self.calls += 1
        return _R()


class FakeBuffer:
    def add_expression(self, e):
        pass

    def unreflected_expressions(self, n=30):
        return []

    def mark_expressions_reflected(self, exprs):
        pass


kg2 = KnowledgeGraph()
kg2.add_node(Node(id="Self", weight=1.0, graph_space="self"))
kg2.add_node(Node(id="用户", weight=1.0))
kg2.add_node(Node(id="钻石", weight=0.7, activation=2.5))
kg2.add_node(Node(id="Minecraft", weight=0.8, activation=1.8))
eng2 = FakeEngine()
eng2.topk = [FN("钻石", 2.5), FN("Minecraft", 1.8)]
cfg2 = {"continuous_cognition": {
    "tick_seconds": 0.01, "pulse_every_ticks": 1,
    "form_threshold": 0.50, "express_threshold": 1.01,
    "inhibition_cooldown_s": 0.1}}
cc2 = ContinuousCognition(kg2, eng2, FakeNLP(), FakeBuffer(), cfg2)

tid2 = fas_log.new_trace("cc_auton")
cc2._last_pulse_ts = 0.0     # 强制 overdue → 脉冲必发（同 test_continuous_cognition）
before = len([r for r in read_all() if r["event"] == "cycle_start"
              and r.get("subsystem") == "cognition"
              and (r.get("data") or {}).get("trigger") == "cc_pulse"])
cc2._pulse_gate()
drain()

# 动作链：真实 ActionManager + 桩具身（pending → tick 轮询 → 结算）
from action_system import ActionManager


class StubEmb:
    name = "stub"

    def execute(self, action):
        return {"success": True, "describe": "开始移动", "pending": True,
                "reason": ""}

    def poll_action(self):
        return {"status": "done", "describe": "到达目标点",
                "detail": {"arrived": True}}

    def cancel(self):
        return {"success": True}


am = ActionManager(embodiment=StubEmb(), kg=kg2, engine=eng2, config={})
am._apply_reward = lambda *a, **k: None       # 隔离奖赏/记忆副作用（测试内）
am._write_action_memory = lambda *a, **k: None
out = am.propose({"action_type": "goto", "target": "钻石", "priority": 0.6,
                  "motivation": "合成测试动机"}, source="autonomy")
tick_out = am.tick()
drain()
check("T2 动作真实闭环", out.get("started") is True
      and tick_out.get("acted") is True, f"{out} {tick_out}")

r2 = [r for r in read_all() if r.get("trace_id") == tid2]
evs2 = [r["event"] for r in r2]
pulses = [r for r in read_all() if r["event"] == "cycle_start"
          and (r.get("data") or {}).get("trigger") == "cc_pulse"]
pulses_end = [r for r in read_all() if r["event"] == "cycle_end"
              and (r.get("data") or {}).get("kind") == "cc_pulse"]
check("T2 CC 脉冲产生独立周期（x_ 前缀，不入 c_ 账本）",
      len(pulses) > before and pulses[-1]["cycle_id"].startswith("x_"),
      f"{before}->{len(pulses)}")
check("T2 脉冲周期成对（start/end 且同 cycle_id）",
      len(pulses_end) == len(pulses)
      and pulses[-1] and pulses_end[-1]["cycle_id"] == pulses[-1]["cycle_id"])
check("T2 自主动作链 proposed→started→settled",
      "action_proposed" in evs2 and "action_started" in evs2
      and "action_settled" in evs2, str(evs2))
aid = next(r for r in r2 if r["event"] == "action_proposed")["data"]["action_id"]
check("T2 action_id 贯穿动作全链",
      all(r["data"].get("action_id") == aid
          for r in r2 if r["event"].startswith("action_")), str(evs2))
st = next(r for r in r2 if r["event"] == "action_settled")
check("T2 结算记录成功与回执", st["data"]["success"] is True
      and st["data"]["duration_s"] >= 0, str(st["data"]))
check("T2 周期结束后 x_ 关联清空", fas_log.get_cycle() is None)

# ═══ Test 3: MC 链路 —— 会话状态迁移 + 桥命令（观测→行动→回执）═══
from minecraft.session import MinecraftSession
import minecraft.bridge

tid3 = fas_log.new_trace("mc")
sess = MinecraftSession(kg2, bridge_get_state=lambda: None)
sess._set("awaiting_port", request_message="端口 25565")
sess._set("connecting", port=25565)
sess._set("connected", port=25565)
sess._set("connected", port=25565)   # 无变化不应再发
# 桥命令（打桩 urlopen）
import urllib.request as _ur


class _FakeHTTP:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({"ok": True, "describe": "开始移动"}).encode("utf-8")


_orig_urlopen = _ur.urlopen
_ur.urlopen = lambda *a, **k: _FakeHTTP()
try:
    ok_res = minecraft.bridge.call("/goto", {"x": 1})
finally:
    _ur.urlopen = _orig_urlopen
drain()
r3 = [r for r in read_all() if r.get("trace_id") == tid3]
evs3 = [(r["event"], (r.get("data") or {})) for r in r3]
trans = [d for e, d in evs3 if e == "session_state"]
check("T3 MC 会话状态迁移逐条可查（old→new）",
      len(trans) == 3 and trans[0]["old"] == "disconnected"
      and trans[-1]["new"] == "connected", str(trans))
check("T3 无变化不发事件",
      sum(1 for e, d in evs3 if e == "session_state"
          and d.get("new") == "connected") == 1, str(evs3))
bcmd = [d for e, d in evs3 if e == "bridge_command"]
check("T3 桥命令+延迟记录", len(bcmd) == 1 and bcmd[0]["ok"] is True
      and bcmd[0]["latency_ms"] >= 0 and bcmd[0]["path"] == "/goto",
      str(bcmd))
check("T3 命令返回值语义不变", ok_res.get("ok") is True)

# 感知链（合成状态）：update_perception 只在槽位真变化时发事件；
# GRAPH 变更事件能回答"这个节点为什么在图里"（source 布线）。
import minecraft.perception as mp
_st = {"connected": True, "position": {"x": 10, "y": 64, "z": -5},
       "health": 20, "food": 18, "heldItem": "石镐",
       "playersNearby": [], "nearbyBlocks": [{"name": "圆石"}],
       "nearbyEntities": []}
mp.update_perception(kg2, eng2, _st)          # 首见：全部槽位建立
mp.update_perception(kg2, eng2, dict(_st, health=15))   # 合成：血量变化
mp.update_perception(kg2, eng2, dict(_st, health=15))   # 无变化 → 应静默
drain()
r3b = [r for r in read_all() if r.get("trace_id") == tid3
       and r["event"] == "mc_perception_updated"]
check("T3 感知只在状态变化时发事件（3 次调用 → 2 条）",
      len(r3b) == 2, f"{len(r3b)} 条")
if r3b:
    _last_ch = r3b[-1]["data"]["changed"]
    check("T3 变化事件带 old→new",
          _last_ch.get("Haru的血量") == {"old": 20, "new": 15},
          str(_last_ch))
_na = [r for r in read_all()
       if r["event"] == "node_added"
       and (r.get("data") or {}).get("target") == "Haru的血量"]
check("T3 GRAPH 节点来历带 source",
      len(_na) == 1 and _na[0]["data"]["source"] == "minecraft.perception",
      str(_na))

# ═══ Test 4: 模拟 LLM 失败 —— 降级重试 / 失败续跑（§27-4）═══
tid4 = fas_log.new_trace("llmfail")
b4 = _backend("json_fail_first")
with fas_log.llm_purpose("graph_reasoning"):
    got = b4._create([{"role": "user", "content": "x"}])
check("T4 response_format 失败→降级重试→成功",
      got.choices[0].message.content == "好呀")
b4b = _backend("always_fail")
raised = False
try:
    with fas_log.llm_purpose("graph_reasoning"):
        b4b._create([{"role": "user", "content": "x"}])
except ConnectionError:
    raised = True
check("T4 全失败→异常如实上抛（日志不吞异常）", raised)
drain()
r4 = [r for r in read_all() if r.get("trace_id") == tid4]
ev4 = [(r["event"], r["level"]) for r in r4]
check("T4 降级事件可见", ("llm_call_downgrade", "WARNING") in ev4, str(ev4))
check("T4 失败事件可见且带 purpose",
      any(e == "llm_call_failed"
          and (r.get("data") or {}).get("purpose") == "graph_reasoning"
          for (e, _), r in zip(ev4, r4)), str(ev4))
fin4 = [r for r in r4 if r["event"] == "llm_call_finished"]
check("T4 重试成功后仍记完成事件(retried=True)",
      len(fin4) == 1 and fin4[0]["data"]["retried"] is True, str(fin4))
check("T4 失败不中断日志系统",
      len([r for r in r4 if r["event"] == "trace_start"]) == 1)

# ═══ Test 5: 注入异常 —— traceback 入日志且日志系统不崩（§27-5）═══
tid5 = fas_log.new_trace("exc")
_log5 = fas_log.get_logger(fas_log.ERROR)
for _ in range(50):
    try:
        raise RuntimeError("注入的合成故障 xyz")
    except RuntimeError:
        _log5.exception("injected_failure", "同类异常注入", kind="synthetic")
drain(1.2)
r5 = [r for r in read_all() if r.get("trace_id") == tid5
      and r["event"] == "injected_failure"]
check("T5 同类异常 50 次被去重折叠", len(r5) <= 2, f"got {len(r5)}")
if r5:
    check("T5 首条含真实 traceback", "RuntimeError" in (r5[0].get("exc") or ""))
cnt = fas_log._snapshot_counters().get("dedup_suppressed:injected_failure", 0)
check("T5 折叠计数如实累计≈49", cnt >= 48, f"count={cnt}")
# 线程未捕获异常钩子


def _bad_thread():
    raise ValueError("线程合成故障")


_t = threading.Thread(target=_bad_thread, daemon=True)
_t.start()
_t.join(2)
drain(0.8)
un = [r for r in read_all() if r["event"] == "uncaught_thread_exception"]
check("T5 线程未捕获异常被钩子记录（CRITICAL+traceback）",
      len(un) >= 1 and "ValueError" in (un[-1].get("exc") or ""),
      str(len(un)))
lg(fas_log.SYSTEM).info("still_alive", "系统未崩")
drain()
check("T5 异常风暴后日志仍可用",
      any(r["event"] == "still_alive" for r in read_all()))

# ═══ Test 6: 高频源不刷屏（感知扫描 / 扩散 tick / 桥轮询，§27-6）═══
base_n = len(read_all())
from eye.observer import run_observation as _robs


def _empty_capture(region=None):
    return {"items": [], "count": 0}


for _ in range(500):
    _robs(kg2, eng2, None, capture=_empty_capture)
drain()
none_ev = [r for r in read_all() if r["event"] == "screen_observed_none"]
check("T6 感知 500 次空扫描 → 聚合数条而非 500 条",
      0 < len(none_ev) <= 5, f"{len(none_ev)} 条")
agg = fas_log.aggregator("cc_diffusion_ticks_bench", fas_log.ACTIVATION,
                         "diffusion_ticks", flush_every_s=600, flush_every_n=100)
for _ in range(1000):
    agg.add()
agg.flush()
drain()
dt = [r for r in read_all() if r["event"] == "diffusion_ticks"
      and (r.get("data") or {}).get("count") == 100]
check("T6 扩散 1000 拍 → 按 100 聚合（≈10 条）", 9 <= len(dt) <= 11,
      f"{len(dt)}")
_ur.urlopen = lambda *a, **k: _FakeHTTP()
try:
    for _ in range(200):
        minecraft.bridge.get_state()
finally:
    _ur.urlopen = _orig_urlopen
drain()
mc_poll = [r for r in read_all() if r["event"] == "bridge_command"
           and (r.get("data") or {}).get("path") == "/state"]
check("T6 桥状态轮询 200 次零事件", len(mc_poll) == 0)
after = len(read_all())
check("T6 高频总受控（<100 事件 / 1700 次调用）",
      after - base_n < 100, f"Δ{after - base_n}")
# §17 成本护栏：异步队列下单条 emit 均值应远小于 1ms
_t0 = time.perf_counter()
for _ in range(5000):
    fas_log.emit(fas_log.SYSTEM, "INFO", "perf_probe", "x", k=1)
_cost_ms = (time.perf_counter() - _t0) * 1000 / 5000
check("T6 热路径 emit 均值 < 1ms/条", _cost_ms < 1.0, f"{_cost_ms:.4f}ms")
drain(2.0)

# ═══ 级别与旁路（§18 / 约束6：DEBUG 关闭时行为不变）═══
lg(fas_log.GRAPH).debug("should_be_dropped", "DEBUG 未放开")
drain()
check("默认策略下 graph DEBUG 不落盘",
      len([r for r in read_all()
           if r["event"] == "should_be_dropped"]) == 0)
fas_log.set_level("DEBUG", "graph")
lg(fas_log.GRAPH).debug("graph_debug_on", "临时放开 graph=DEBUG")
drain()
check("子系统级 DEBUG 生效（全局仍 INFO）",
      any(r["event"] == "graph_debug_on" for r in read_all())
      and fas_log.get_level() == "INFO")
fas_log.set_level("INFO", "graph")
try:
    fas_log.emit("bogus-sub", "NOT-A-LEVEL", "weird",
                 {"unserializable": object()}, bad_kw=object())
    check("坏参数 emit 不抛异常（观测层永不干扰认知）", True)
except Exception as e:
    check("坏参数 emit 不抛异常（观测层永不干扰认知）", False, repr(e))
drain()

# ═══ 会话摘要与计数器（§22）═══
ctr = fas_log._snapshot_counters()
check("计数器累计了 LLM 调用", ctr.get("llm_calls", 0) >= 2, str(ctr))
fas_log.shutdown("test_exit")
time.sleep(0.4)
alls = read_all()
summ = [r for r in alls if r["event"] == "session_summary"]
check("退出写 session_summary", len(summ) == 1
      and summ[0]["data"]["reason"] == "test_exit", str(len(summ)))
check("session_summary 含子系统事件计数与丢弃数",
      "cognition" in (summ[0]["data"].get("events") or {})
      and "dropped" in summ[0]["data"], str(list(summ[0]["data"].keys())))
files = sorted(os.listdir(LOGDIR))
check("文件布局: runtime.log + fas_all + 子系统分文件",
      "runtime.log" in files and "fas_all.jsonl" in files
      and "cognition.jsonl" in files and "error.jsonl" in files
      and "minecraft.jsonl" in files, str(files))
with open(os.path.join(LOGDIR, "runtime.log"), encoding="utf-8") as f:
    rt = f.read()
check("runtime.log 人读行含 级别/子系统/事件",
      "COGNITION cycle_start" in rt and "[INFO]" in rt, rt[:160])
check("shutdown 后 emit 静默无害",
      (fas_log.emit("system", "INFO", "post", "x"), True)[1])

# ═══ 查询工具冒烟（§24）═══
import subprocess
TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools", "analyze_logs.py")
r = subprocess.run([sys.executable, "-X", "utf8", TOOL,
                    "--file", os.path.join(LOGDIR, "fas_all.jsonl"),
                    "--trace", tid1],
                   capture_output=True, text=True, encoding="utf-8")
check("analyze_logs --trace 重建对话链",
      r.returncode == 0 and "input_received" in r.stdout
      and "response_sent" in r.stdout, r.stderr[:200])
r = subprocess.run([sys.executable, "-X", "utf8", TOOL,
                    "--file", os.path.join(LOGDIR, "fas_all.jsonl"),
                    "--cycle", cid1],
                   capture_output=True, text=True, encoding="utf-8")
check("analyze_logs --cycle 重建回合",
      r.returncode == 0 and "cycle_start" in r.stdout
      and "cycle_end" in r.stdout, r.stderr[:200])
r = subprocess.run([sys.executable, "-X", "utf8", TOOL,
                    "--file", LOGDIR, "--llm"],
                   capture_output=True, text=True, encoding="utf-8")
check("analyze_logs --llm 按 purpose 汇总",
      r.returncode == 0 and "dialogue_decomposition" in r.stdout,
      (r.stdout + r.stderr)[:200])
r = subprocess.run([sys.executable, "-X", "utf8", TOOL,
                    "--file", LOGDIR, "--errors"],
                   capture_output=True, text=True, encoding="utf-8")
check("analyze_logs --errors 收口异常",
      r.returncode == 0 and "injected_failure" in r.stdout,
      (r.stdout + r.stderr)[:200])

print()
if FAILURES:
    print(f"[FAIL] {len(FAILURES)} 项未过: {FAILURES}")
    sys.exit(1)
print("[PASS] test_fas_log 全部通过（§27 六场景 + 级别/摘要/工具）")
