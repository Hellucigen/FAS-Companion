# run_baselines.py — §8 基线执行器（sandbox；Reactive / Memory-only /
# Graph-retrieval 零 LLM；LLM-heavy 一次短计划，记录调用/令牌/延迟）
# 用法:
#   E:/Miniforge.envs/Fascinator/python.exe -X utf8 scripts/run_baselines.py \
#       [--seeds 1,2,3] [--max-ticks 120]
# 共同条件（与消融矩阵同构）：标准世界 + 目标 oak_planks + 同 seed。
#
# 基线语义（2026-09-28 定版，诚实边界）：
#   * reactive       感官直连执行：无扩散/无因果/无先验/无缺口/无自模型，
#                     意图→技能层直接找最近方块采集（这就是"反应式"的
#                     真实上限：不规划多跳，只回应当前刺激）。
#   * memory_only    前 run 图谱继承（记忆在）+ 扩散关闭：只用直连经验边，
#                     不向未见对象泛化。测"记忆复用 vs 扩散泛化"的差分。
#   * graph_retrieval 前 run 图谱继承 + 扩散开 + 因果关：图结构检索驱动，
#                     无动作经验聚合。
#   * llm_heavy      同一简单任务，用 MiMo 调度下一步动作（≤5 次调用），
#                     记录 provider/model/calls/tokens/latency/success。
#                     仅长跑简单任务，对照的公平性在报告里如实标注。
# ============================================================================

import argparse
import json
import os
import sys
import time

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
if os.getcwd() != os.path.dirname(os.path.dirname(os.path.abspath(__file__))):
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox_lab import build_stack, tick, have_item, Clock, flush_causal  # noqa
import experiment_recorder as er                                         # noqa
from run_sandbox_scenarios import _setup_standard_world, goal_done        # noqa

GOAL = "oak_planks"
MAX_TICKS = 120

BASELINES = [
    # (name, switches, inherit_prev, drip_note)
    ("reactive", {"diffusion_on": False, "causal_on": False, "prior_on": False,
                  "gap_on": False, "self_goal_on": False, "writeback_on": False},
     False),
    ("memory_only", {"diffusion_on": False, "causal_on": False, "prior_on": False,
                     "gap_on": False},
     True),
    ("graph_retrieval", {"diffusion_on": True, "causal_on": False,
                         "prior_on": True, "gap_on": True},
     True),
]


def run_zero_llm(name, seed, switches, max_ticks, inherit, graph_paths):
    kw = {"goal": GOAL, "label": f"{name}-s{seed}", "seed": seed}
    kw.update(switches)
    if inherit and graph_paths.get("full"):
        kw["inherit_graph"] = graph_paths["full"]
    st = build_stack(**kw)
    _setup_standard_world(st)
    clk = Clock()
    rec = er.RunRecorder(family="baselines", mode="sandbox",
                         seed=seed, label=name)
    rec.start({"baseline": name, "goal": GOAL, "switches": switches,
               "inherit": bool(inherit)})
    rec.snapshot("graph_start", graph_dict=st.kg.to_dict())
    done_at = None
    for i in range(max_ticks):
        tick(st, clk)
        if goal_done(st, GOAL):
            done_at = i
            rec.log("RESULT", f"达成 @tick{i}", tick=i)
            break
    flush_causal(st)
    n_actions = 0
    try:
        n_actions = sum(1 for e in st.tl._raw
                        if e.get("event_type") == "ACTION")
    except Exception:
        pass
    rec.snapshot("graph_end", graph_dict=st.kg.to_dict())
    m = {"baseline": name, "seed": seed, "success": done_at is not None,
         "success_tick": done_at, "action_count": n_actions,
         "graph": {"nodes": len(st.kg.nodes), "edges": len(st.kg.edges)},
         "inv": dict(st.world.inv_map())}
    rec.finalize(ok=True, extra=m)
    print(f"[BASE] {name:18s} seed={seed} 达成@{done_at if done_at is not None else '—'}"
          f" 动作={n_actions}", flush=True)
    return m, st


def run_llm_heavy(seed, max_steps=5):
    """LLM-heavy 基线：同一短任务（oak_planks），MiMo 调度下一步。
    动作施加：解析 LLM 输出首 token → world 命令序列（find→dig→collect /
    craft / equip）。记录 telemetry（model/calls/latency；token 计数经
    fas_log._create 旁路自动落盘，可在 logs/cognition.jsonl 归档）。"""
    from llm_provider import create_backend
    import json as _json
    # 密钥只读自 gitignored 密钥文件/环境变量（同 app.py:693-703 装配语义，
    # 绝不写入任何实验产物）
    _sk = {}
    try:
        with open(os.path.join("data", "llm_secret.json"), encoding="utf-8") as _f:
            _sk = _json.load(_f)
    except Exception:
        pass
    _cfg = {}
    try:
        with open(os.path.join("data", "llm_config.json"), encoding="utf-8") as _f:
            _cfg = _json.load(_f)
    except Exception:
        pass
    _key = (_sk.get("api_key") or _cfg.get("api_key")
            or os.environ.get("MIMO_API_KEY") or "")
    kw = {"goal": GOAL, "label": f"llm_heavy-s{seed}", "seed": seed}
    st = build_stack(**kw)
    _setup_standard_world(st)
    clk = Clock()
    rec = er.RunRecorder(family="baselines", mode="sandbox",
                         seed=seed, label="llm_heavy")
    rec.start({"baseline": "llm_heavy", "goal": GOAL,
               "llm": "MiMo 调度下一步（每步一问，≤5 问）",
               "model": _cfg.get("model", "?")})
    provider = create_backend("mimo", model=_cfg.get("model") or None,
                              api_key=_key,
                              base_url=_cfg.get("base_url") or None)
    # LLM-heavy 是唯一允许真调用的基线：临时恢复 MiMo 原 _create（sndbox
    # 零 LLM 守卫只对认知路径生效；本基线的调用就是被测对象本身）。
    import llm_provider as _lp
    _prev = _lp.MiMoBackend._create
    _lp.MiMoBackend._create = _lp.OpenAICompatBackend._create

    def _apply_action(cmd: str):
        """把 LLM 的命令词施加到世界。返回 (ok, note)。"""
        bits = (cmd or "").strip().split()
        verb = (bits[0] or "").lower()
        obj = (bits[-1] if len(bits) > 1 else "").lower()
        if verb in ("gather", "chop", "dig", "mine", "break"):
            found = st.world.call("/find_blocks", {"block": obj,
                                                   "radius": 32})
            poss = (found or {}).get("positions") or []
            if not poss:
                return False, f"{obj}_not_found"
            p = poss[0]
            r1 = st.world.call("/dig_pos", {"x": p["x"], "y": p["y"],
                                            "z": p["z"]})
            st.world.poll_result()
            if not r1.get("ok"):
                return False, str(r1.get("reason") or "dig_failed")
            st.world.call("/collect_item")
            st.world.poll_result()
            return True, "gathered"
        if verb == "craft":
            r = st.world.call("/craft", {"item": obj})
            st.world.poll_result()
            return bool(r.get("ok")), str(r.get("reason") or "crafted")
        if verb in ("equip", "hold"):
            st.world.call("/equip", {"item": obj})
            return True, "equipped"
        return False, f"unknown_verb:{verb or 'empty'}"

    calls = []
    done_at = None
    for step in range(max_steps):
        if goal_done(st, GOAL):
            done_at = step
            rec.log("RESULT", f"达成 @step{step}", step=step)
            break
        inv = dict(st.world.inv_map())
        prompt = (
            "你在 Minecraft 沙箱里，目标：获得 oak_planks（橡木木板）。\n"
            f"背包: {inv if inv else '空'}\n"
            "可执行动作（括号里填物品名）：gather <方块>（砍树采原木）/"
            " craft <物品>（合成）/ equip <物品>（手持）。\n"
            "只输出一个动词+一个物品，例如：gather oak_log"
        )
        t0 = time.time()
        try:
            resp_text = provider.invoke(prompt)
        except Exception as e:
            resp_text = f"__ERROR__ {e}"[:120]
        lat = round(time.time() - t0, 2)
        text = str(resp_text or "").strip()
        line = [ln for ln in text.splitlines() if ln.strip()]
        cmd = (line[0] if line else text)[:120]
        ok, note = _apply_action(cmd)
        calls.append({"step": step, "cmd": cmd, "ok": ok, "note": note,
                      "lat_s": lat})
        rec.log("LLM", f"step{step} {'成功' if ok else '失败:'+note}",
                latency_s=lat, cmd=cmd, model=provider.model_name)
    if done_at is None:
        rec.log("RESULT", f"未达成（{max_steps} 步上限）", step=max_steps)
    _lp.MiMoBackend._create = _prev   # 还原守卫契约
    rec.finalize(ok=True, extra={
        "baseline": "llm_heavy", "seed": seed,
        "success": done_at is not None, "success_step": done_at,
        "llm_calls": calls, "calls_count": len(calls),
        "inv": dict(st.world.inv_map())})
    print(f"[BASE] llm_heavy seed={seed} 成功{'@'+str(done_at) if done_at is not None else '否'}"
          f" 调用={len(calls)} 延迟={sum(c['lat_s'] for c in calls):.1f}s", flush=True)
    return {"baseline": "llm_heavy", "seed": seed,
            "success": done_at is not None, "success_step": done_at,
            "llm_calls": len(calls)}, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    ap.add_argument("--llm-heavy", action="store_true",
                    help="额外跑 LLM-heavy（MiMo，≤5 步）")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    # 先跑 full（继承源）
    graph_paths = {}
    kw = {"goal": GOAL, "label": "full-baseline", "seed": seeds[0]}
    st = build_stack(**kw)
    _setup_standard_world(st)
    clk = Clock()
    for i in range(args.max_ticks):
        tick(st, clk)
        if goal_done(st, GOAL):
            break
    flush_causal(st)
    src = os.path.join(er.root_dir(), "baselines")
    os.makedirs(src, exist_ok=True)
    gp = os.path.join(src, "full_graph.json")
    with open(gp, "w", encoding="utf-8") as f:
        json.dump(st.kg.to_dict(), f, ensure_ascii=False)
    graph_paths["full"] = gp
    results = {}
    for name, switches, inherit in BASELINES:
        if args.only and args.only not in name:
            continue
        rs = []
        for s in seeds:
            m, _ = run_zero_llm(name, s, dict(switches), args.max_ticks,
                                inherit, graph_paths)
            rs.append(m)
        results[name] = rs
    if args.llm_heavy:
        rs = []
        for s in seeds:
            m, _ = run_llm_heavy(s)
            rs.append(m)
        results["llm_heavy"] = rs
    out = {"goal": GOAL, "seeds": seeds, "max_ticks": args.max_ticks,
           "elapsed_s": round(time.time() - t0, 1), "results": results,
           "full_graph": gp}
    fn = os.path.join(er.root_dir(), "baselines", "baselines_summary.json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[Baselines] 完成 → {fn}（{out['elapsed_s']}s）")
    for name, rs in results.items():
        print(f"  {name:18s} "
              + ", ".join(f"s{r['seed']}@{r.get('success_tick') or r.get('success_step')}"
                          if r["success"] else f"s{r['seed']}✗" for r in rs))


if __name__ == "__main__":
    sys.exit(main())