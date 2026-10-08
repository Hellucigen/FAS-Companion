# tools/analyze_logs.py — 运行日志查询工具（任务书 §24：够用的 CLI，不做网页平台）
# ============================================================================
# 用法：
#   python tools/analyze_logs.py --trace tr000042       按 trace 重放因果链
#   python tools/analyze_logs.py --cycle c_000123       按周期（c_/x_ 前缀均可）
#   python tools/analyze_logs.py --session ses_xxx      整场会话
#   python tools/analyze_logs.py --errors               只看 ERROR/CRITICAL
#   python tools/analyze_logs.py --llm                  LLM 调用汇总（按 purpose）
#   python tools/analyze_logs.py --mc                   Minecraft 事件（桥+会话+感知）
#   python tools/analyze_logs.py --subsys cognition --level DEBUG
#   python tools/analyze_logs.py --grep "预算"           全文过滤
#   组合：--session xxx --errors        管道：... | --json
# 默认读 logs/fas_all.jsonl（--file 可指别的 jsonl / 目录多文件）。
# ============================================================================

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT = os.path.join(ROOT, "logs", "fas_all.jsonl")


def load_records(path: str):
    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    elif path.endswith("*") or "*" in os.path.basename(path):
        files = sorted(glob.glob(path))
    else:
        files = [path]
    recs = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(r, dict) and r.get("event"):
                        recs.append(r)
        except OSError as e:
            print(f"读取失败 {fp}: {e}", file=sys.stderr)
    recs.sort(key=lambda r: r.get("ts", ""))
    return recs


def fmt_line(r: dict) -> str:
    parts = [r.get("ts", ""), f"[{r.get('level','')}]",
             (r.get("subsystem") or "").upper(), r.get("event", "")]
    for k in ("cycle_id", "trace_id"):
        if r.get(k):
            parts.append(r[k])
    line = " ".join(parts)
    if r.get("msg"):
        line += " — " + str(r["msg"]).replace("\n", " ⏎ ")
    d = r.get("data") or {}
    drop = {"cycle_id", "trace_id", "session_id"}
    show = {k: v for k, v in d.items() if k not in drop}
    if show:
        s = json.dumps(show, ensure_ascii=False, default=str)
        line += " " + (s[:400] + "…" if len(s) > 400 else s)
    if r.get("exc"):
        line += "  ⚠EXC:" + r["exc"].splitlines()[-1][:160]
    return line


def main():
    ap = argparse.ArgumentParser(description="FAS 运行日志查询")
    ap.add_argument("--file", default=DEFAULT)
    ap.add_argument("--trace")
    ap.add_argument("--cycle")
    ap.add_argument("--session")
    ap.add_argument("--subsys")
    ap.add_argument("--event")
    ap.add_argument("--level")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--mc", action="store_true")
    ap.add_argument("--grep", dest="pattern")
    ap.add_argument("--last", type=int, default=0, help="只显示最后 N 条")
    ap.add_argument("--json", action="store_true", help="输出原始 JSONL")
    args = ap.parse_args()

    recs = load_records(args.file)
    if not recs:
        print(f"没有记录（{args.file}）", file=sys.stderr)
        return 1

    def keep(r):
        if args.trace and r.get("trace_id") != args.trace:
            return False
        if args.cycle and r.get("cycle_id") != args.cycle:
            return False
        if args.session and r.get("session_id") != args.session:
            return False
        if args.subsys and r.get("subsystem") != args.subsys.lower():
            return False
        if args.event and r.get("event") != args.event:
            return False
        if args.level and r.get("level") != args.level.upper():
            return False
        if args.errors and r.get("level") not in ("ERROR", "CRITICAL"):
            return False
        if args.llm and r.get("subsystem") != "llm":
            return False
        if args.mc and r.get("subsystem") not in ("minecraft", "action"):
            return False
        if args.pattern:
            blob = json.dumps(r, ensure_ascii=False)
            if args.pattern not in blob:
                return False
        return True

    sel = [r for r in recs if keep(r)]
    if args.last:
        sel = sel[-args.last:]

    if args.llm and not (args.trace or args.cycle or args.session):
        # 汇总模式：按 purpose 聚合调用数/延迟/失败
        by = defaultdict(lambda: {"n": 0, "fail": 0, "ms": 0, "tok": 0})
        for r in sel:
            k = (r.get("data") or {}).get("purpose", "?")
            b = by[k]
            if r["event"] == "llm_call_finished":
                b["n"] += 1
                b["ms"] += (r.get("data") or {}).get("latency_ms") or 0
                b["tok"] += (r.get("data") or {}).get("total_tokens") or 0
            elif r["event"] in ("llm_call_failed",):
                b["fail"] += 1
        print(f"{'purpose':28s} {'次数':>5s} {'失败':>5s} {'均延迟ms':>8s} {'tokens':>8s}")
        for k, b in sorted(by.items(), key=lambda kv: -kv[1]["n"]):
            avg = b["ms"] // b["n"] if b["n"] else 0
            print(f"{k:28s} {b['n']:5d} {b['fail']:5d} {avg:8d} {b['tok']:8d}")
        if not by:
            print("（无 LLM 记录）")
        return 0

    if args.json:
        for r in sel:
            print(json.dumps(r, ensure_ascii=False, default=str))
    else:
        for r in sel:
            print(fmt_line(r))

    if not args.json:
        by_sub = Counter(r.get("subsystem") for r in sel)
        span = (sel[0]["ts"], sel[-1]["ts"]) if sel else None
        print(f"\n── {len(sel)} 条 "
              + (f"（{span[0]} … {span[1]}）" if span else "")
              + ("".join(f"  {k}:{v}" for k, v in by_sub.most_common(6))),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
