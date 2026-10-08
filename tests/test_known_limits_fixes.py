# test_known_limits_fixes.py — §18.2 B 批次五项修复的离线验收
#   #10  话题终止抑制期 → system 认知锁承载（expires_at/审计/显式解除）
#   #7   桥失联 session 回写（on_bridge_lost，仅 connected 翻转）
#   #11  mode=poll 检测器数据源（register_poller + poll_due 驱动）
#   #19  API 守卫（Origin 常开 + token 按需；纯函数矩阵）
#   P1   图谱周期快照 + 自动轮转（graph_rotation）
# 离线：tmp 目录隔离持久化文件、假 health、零 LLM、不 import app.py。
# #7+#11 联合场景复刻 app._wire_bridge_liveness 的接线语义（订阅者判定
# 与 app 内一致：仅 current is False 且 on_bridge_lost 返回真才动作）。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_known_limits_fixes.py

import logging
import os
import shutil
import sys
import tempfile
import time

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)
logging.disable(logging.WARNING)

import api_guard
import graph_rotation
import dialogue_decision as dd
from cognitive_locks import LockRegistry
from state_monitors import MonitorRegistry
from graph_model import KnowledgeGraph
from minecraft.session import MinecraftSession

FAILURES = []


def check(name, cond, detail=""):
    st = "PASS" if cond else "FAIL"
    print(f"[{st}] {name}" + (f" | {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


TMP = tempfile.mkdtemp(prefix="fas_bfix_")

# ════════════════════════════════════════════════════════════════
# 1. §18.2 #10：抑制期改由 system 锁承载
# ════════════════════════════════════════════════════════════════
print("--- #10 抑制锁承载 ---")

# 1a. 未绑定注册表：浮点回退，原语义不变
dd.bind_lock_registry(None)
dd._global_clear()
dd.mark_termination(30)
check("#10 未绑定时浮点通道生效", dd.inhibition_active() is True)
dd._inhibit_until = time.time() - 1
check("#10 浮点到期即失活", dd.inhibition_active() is False)

# 1b. 绑定注册表：mark → 锁出现，expires_at/lock_type/审计齐全
reg = LockRegistry(path=os.path.join(TMP, "locks.json"))
dd.bind_lock_registry(reg)
dd._global_clear()
t0 = time.time()
dd.mark_termination(120)
sys_locks = reg.list(target_type="system",
                     target_id="dialogue_topic_inhibit")
check("#10 锁已建立（system 目标）", len(sys_locks) == 1,
      f"n={len(sys_locks)}")
lk = sys_locks[0] if sys_locks else {}
check("#10 锁携 expires_at≈now+120",
      lk.get("expires_at") is not None
      and t0 + 118 < float(lk.get("expires_at") or 0) <= t0 + 125,
      str(lk.get("expires_at")))
check("#10 non_blocking（只作状态承载，不拦扩散）",
      lk.get("lock_type") == "non_blocking")
check("#10 抑制期生效", dd.inhibition_active() is True)
check("#10 浮点通道同步（回退不失配）", dd._inhibit_until > time.time() + 100)
check("#10 system 目标恒视为存在",
      reg.target_status(lk).get("target_missing") is False,
      str(reg.target_status(lk)))

# 1c. 到期由锁决定（把锁的 expires_at 拨到过去 → 立即失活）
reg.update(lk["id"], expires_at=time.time() - 1)
check("#10 锁到期 → 失活", dd.inhibition_active() is False)

# 1d. 重新 mark：复用同一把锁（不堆重复条目），到期时间被刷新
dd.mark_termination(60)
sys_locks2 = reg.list(target_type="system",
                      target_id="dialogue_topic_inhibit")
check("#10 再 mark 复用同一锁", len(sys_locks2) == 1
      and sys_locks2[0]["id"] == lk["id"], f"n={len(sys_locks2)}")
check("#10 再 mark 重新生效且刷新 enabled",
      dd.inhibition_active() is True
      and sys_locks2[0].get("enabled") is True)

# 1e. 显式解除 API + 审计历史
out = dd.release_termination("unit-test")
check("#10 解除成功", out.get("ok") is True and dd.inhibition_active() is False)
check("#10 锁留档但禁用（审计可见）",
      reg.get(lk["id"]) is not None
      and reg.get(lk["id"]).get("enabled") is False)
hist = [h for h in reg.state()["history"]
        if h.get("lock_id") == lk["id"]]
check("#10 审计历史覆盖创建/续期/解除", len(hist) >= 3, f"n={len(hist)}")
check("#10 解除原因入审计",
      any("unit-test" in str(h.get("detail", "")) + str(h.get("reason", ""))
          for h in reg.state()["history"][-4:]),
      str(reg.state()["history"][-3:]))

# 1f. 零扩散介入：system 锁不拦任何节点/边（gate 派生集合为空）
check("#10 不拦节点（diffusion）",
      reg.blocks_node("任意事件节点", "diffusion") is False)
check("#10 不藏节点（topk）", reg.hides_node("任意事件节点", "topk") is False)
check("#10 不藏节点（all 全开）",
      reg.hides_node("任意事件节点", "all") is False)
check("#10 blocked_node_ids 不含 system 目标",
      "dialogue_topic_inhibit" not in reg.state()["blocked_nodes"])

# 1g. termination_state 可读（lock 字段暴露给前端/审计）
dd.mark_termination(90)
st = dd.termination_state()
check("#10 termination_state 携锁信息",
      st.get("active") is True and st.get("lock_id")
      and float(st.get("remaining_s") or 0) > 80, str(st))
dd.release_termination()
dd.bind_lock_registry(None)

# ════════════════════════════════════════════════════════════════
# 2. §18.2 #7 + #11：poll 检测器数据源 → 桥失联 → session 回写
# ════════════════════════════════════════════════════════════════
print("--- #7/#11 桥活性 poll ---")

kg = KnowledgeGraph()
session = MinecraftSession(kg)          # 节点初始化 disconnected
mons = MonitorRegistry(path=os.path.join(TMP, "monitors.json"))

health = {"v": True}
mid = mons.create(name="MC 桥活性", target_type="external",
                  target_id="minecraft_bridge", state_type="bool",
                  mode="poll", poll_interval_s=0.01)["monitor"]["id"]
mons.register_poller(mid, lambda: bool(health["v"]))

flips = []


def _handler(ev):                          # 与 app._wire_bridge_liveness 同语义
    if ev.get("monitor_id") != mid:
        return
    if ev.get("current") is not False:
        return
    if session.on_bridge_lost(reason="monitor_poll"):
        flips.append(ev)


mons.subscribe(_handler)

# 2a. 首轮 poll：None→True 的"出现"事件——不回写（app 语义：只处理消失）
session._set("connected", port=25565)
mons.poll_due(now=time.time() + 1)
check("#11 poll_due 调用了注册的数据源",
      mons.list()[0].get("current_value") is True)
check("#7 首轮 None→True 不动作", session.get_state() == "connected"
      and not flips)

# 2b. 值不变 → observe 去重 → 无事件（不刷屏）
mons.poll_due(now=time.time() + 2)
mons.poll_due(now=time.time() + 3)
check("#11 相同值去重（无事件）", len(flips) == 0
      and len(mons.list()[0]["changes"]) == 1)

# 2c. True→False 翻转：边沿触发，connected → disconnected 回写
health["v"] = False
mons.poll_due(now=time.time() + 4)
check("#7 桥失联翻转回写 disconnected",
      session.get_state() == "disconnected" and len(flips) == 1)
n = kg.nodes["Minecraft会话"]
check("#7 回写携失联原因", n.extra_attrs.get("lost_reason") == "monitor_poll",
      str(n.extra_attrs.get("lost_reason")))

# 2d. 持续 False：去重后不再重复事件；恢复 True 也不自动回 connected
mons.poll_due(now=time.time() + 5)
check("#7 重复 False 不刷屏", len(flips) == 1)
health["v"] = True
mons.poll_due(now=time.time() + 6)
check("#7 恢复 True 不自动重连（回写是单向的）",
      session.get_state() == "disconnected")

# 2e. 状态机护栏：非 connected 一律不动
session._set("awaiting_port", last_port=None)
check("#7 awaiting_port 不被失联回写波及",
      session.on_bridge_lost("x") is False
      and session.get_state() == "awaiting_port")
session._set("connecting", port=1)
check("#7 connecting 不被波及",
      session.on_bridge_lost("x") is False
      and session.get_state() == "connecting")
session._set("connected")
check("#7 connected 可翻转", session.on_bridge_lost("x") is True
      and session.get_state() == "disconnected")

# 2f. 未注册数据源的 poll 检测器被静默跳过（#11 原缺陷的另一半语义）
mid2 = mons.create(name="无源轮询", target_type="external",
                   target_id="no_source", state_type="bool",
                   mode="poll", poll_interval_s=0.01)["monitor"]["id"]
res = mons.poll_due(now=time.time() + 9)
check("#11 无 poller 的检测器跳过且不入结果",
      all(r.get("monitor_id") != mid2 for r in res))

# ════════════════════════════════════════════════════════════════
# 3. §18.2 #19：API 守卫判定矩阵（纯函数）
# ════════════════════════════════════════════════════════════════
print("--- #19 API 守卫 ---")
EV = api_guard.evaluate_request

# 回环无 token：现状零摩擦
r = EV(remote_addr="127.0.0.1", path="/api/graph", request_host="127.0.0.1:5000")
check("#19 回环默认放行（零摩擦不变）", r["allow"] and r["mode"] == "allow")

# 恶意网页打回环：Origin 不符 → 403（这就是常开的 CSRF/DNS-rebinding 防线）
r = EV(remote_addr="127.0.0.1", path="/api/graph",
       origin="http://evil.example", request_host="127.0.0.1:5000")
check("#19 回环上跨源 Origin → 403",
      not r["allow"] and r["status"] == 403 and r["mode"] == "deny_origin")
r = EV(remote_addr="127.0.0.1", path="/api/graph",
       referer="http://evil.example/x", request_host="127.0.0.1:5000")
check("#19 Referer 兜底同判", not r["allow"] and r["status"] == 403)

# 本机浏览器 Origin=http://localhost:5000 vs host=127.0.0.1:5000：回环名同族
r = EV(remote_addr="127.0.0.1", path="/api/graph",
       origin="http://localhost:5000", request_host="127.0.0.1:5000")
check("#19 回环名互认（localhost↔127.0.0.1）", r["allow"])

# 局域网来源 + token 生效
LAN = dict(remote_addr="192.168.1.20", request_host="192.168.1.10:5000")
r = EV(path="/api/graph", token_expected="sekrit", **LAN)
check("#19 局域网无 token → 401",
      not r["allow"] and r["status"] == 401 and r["mode"] == "deny_token")
r = EV(path="/api/graph", token_expected="sekrit", token_supplied="wrong", **LAN)
check("#19 token 错误 → 401（不回显期望值）",
      not r["allow"] and r["status"] == 401 and "sekrit" not in r["reason"])
r = EV(path="/api/graph", token_expected="sekrit", token_supplied="sekrit", **LAN)
check("#19 token 头/Bearer 通道通过", r["allow"] and r["mode"] == "allow")
r = EV(path="/api/graph", token_expected="sekrit", token_supplied="sekrit",
       token_via_query=True, **LAN)
check("#19 ?token= 引导 → bootstrap 模式（应下发 cookie）",
      r["allow"] and r["mode"] == "bootstrap")
r = EV(path="/", token_expected="sekrit", **LAN)
check("#19 静态外壳豁免（先拿外壳才能带 ?token=）",
      r["allow"] and "外壳" in r["reason"])
r = EV(path="/api/graph", token_expected="", **LAN)
check("#19 token 未配置时局域网仍只靠 Origin 防线（放行）", r["allow"])

# 收紧开关：回环也强制 token
r = EV(remote_addr="127.0.0.1", path="/api/graph",
       request_host="127.0.0.1:5000", token_expected="sekrit")
check("#19 require_token_on_loopback=false 时回环免 token", r["allow"])
r = EV(remote_addr="127.0.0.1", path="/api/graph",
       request_host="127.0.0.1:5000", token_expected="sekrit",
       require_token_on_loopback=True)
check("#19 开关打开后回环也要 token",
      not r["allow"] and r["status"] == 401)
r = EV(remote_addr="127.0.0.1", path="/api/graph",
       request_host="127.0.0.1:5000", token_expected="sekrit",
       token_supplied="sekrit", require_token_on_loopback=True)
check("#19 开关打开且携 token → 通过", r["allow"])

# 回环地址段与 v4-mapped
check("#19 is_loopback_addr 覆盖 127.x/::1/::ffff:127.x",
      api_guard.is_loopback_addr("127.0.0.53")
      and api_guard.is_loopback_addr("::1")
      and api_guard.is_loopback_addr("::ffff:127.0.0.1")
      and not api_guard.is_loopback_addr("192.168.1.5")
      and not api_guard.is_loopback_addr(""))
check("#19 host_of_url 解析裸 host:port 与 IPv6 URL",
      api_guard.host_of_url("192.168.1.10:5000") == "192.168.1.10"
      and api_guard.host_of_url("http://[::1]:5000/api") == "::1"
      and api_guard.host_of_url("") == "")

# ════════════════════════════════════════════════════════════════
# 4. §18.3 P1：图谱周期快照 + 轮转
# ════════════════════════════════════════════════════════════════
print("--- P1 快照轮转 ---")
RT = os.path.join(TMP, "runtime_graph.json")
SD = os.path.join(TMP, "graph_snapshots")
with open(RT, "w", encoding="utf-8") as f:
    f.write('{"nodes": [1]}')

# 首次落盘即出基线快照（last_ts=0 → 必过闸）
dst, ts = graph_rotation.maybe_rotate(RT, SD, interval_s=3600, keep=12,
                                      last_ts=0.0, now=time.time())
check("P1 首存出基线快照", dst and os.path.exists(dst))
check("P1 快照内容与源一致",
      open(dst, encoding="utf-8").read() == '{"nodes": [1]}')

# 时间闸：间隔内不再复制
dst2, ts2 = graph_rotation.maybe_rotate(RT, SD, interval_s=3600, keep=12,
                                        last_ts=ts, now=ts + 60)
check("P1 间隔内不动作且 last_ts 不推进", dst2 is None and ts2 == ts)

# 过闸后新快照；间隔 0 = 关闭
dst3, ts3 = graph_rotation.maybe_rotate(RT, SD, interval_s=3600, keep=12,
                                        last_ts=ts, now=ts + 3601)
check("P1 过闸出新快照", dst3 and os.path.exists(dst3) and ts3 == ts + 3601)
dst4, ts4 = graph_rotation.maybe_rotate(RT, SD, interval_s=0, keep=12,
                                        last_ts=0.0, now=time.time())
check("P1 interval_s=0 关闭轮转", dst4 is None and ts4 == 0.0)

# 轮转只保留最近 keep 份（名内嵌时间戳，字典序=时间序）
os.makedirs(SD, exist_ok=True)
for i in range(5):
    with open(os.path.join(SD, f"runtime_graph_2026010{i}_00000{i}.json"),
              "w") as f:
        f.write("{}")
removed = graph_rotation.prune(SD, keep=3)
left = graph_rotation.list_snapshots(SD)
check("P1 prune 保新删旧", len(removed) > 0 and len(left) == 3
      and left == sorted(left) and left[-1] not in removed,
      f"left={left}")

# 源文件缺失 / 快照目录不可写：绝不影响主落盘（返回 None，不抛异常）
dst5, ts5 = graph_rotation.maybe_rotate(os.path.join(TMP, "nope.json"), SD,
                                        interval_s=1, keep=3, last_ts=0.0)
check("P1 缺源文件安全跳过", dst5 is None and ts5 == 0.0)
bad_dir = os.path.join(RT, "impossible_subdir")   # src 是文件，其下不可建目录
dst6, ts6 = graph_rotation.maybe_rotate(RT, bad_dir, interval_s=1, keep=3,
                                        last_ts=0.0)
check("P1 快照目录异常不影响落盘（吞错返回 None）", dst6 is None and ts6 == 0.0)

shutil.rmtree(TMP, ignore_errors=True)

# ════════════════════════════════════════════════════════════════
print("=" * 56)
if FAILURES:
    print(f"[FAIL] {len(FAILURES)} 项未通过: {FAILURES}")
    sys.exit(1)
print("[DONE] §18.2 B 批次五项修复验收全部通过")
