#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""chat_log_replay.py — 从 chat_log 重建丢失的记忆（事故恢复工具）

背景
----
2026-09-13 21:21，一次"一致性迁移"脚本误把 KnowledgeGraph.load() 当成实例方法
调用（它其实是 classmethod，返回**新实例**，那句等于没加载），随后 save() 用空图
覆写了 data/runtime_graph.json：894 节点 / 2136 边 → 162 节点。
已从 data/runtime_graph.json.bak.collab_20260906 恢复到 541 节点 / 753 边，
但 9/6 → 9/13 这一周学到的内容不在任何备份里。

data/chat_log.json 完整保存了每一轮的真实输入与回复，是这一周唯一的底稿。
本工具把它重新过一遍**在线那条记忆抽取路径**（app 的 /api/memory/replay）。

为什么是薄客户端而不是自包含脚本
--------------------------------
记忆写入有一套语义：模糊名称归并（"收获日2" vs "收获日2游玩"）、事件框架挂接
（时间锚点 + 父子事件）、缓冲区晋升、扩散引擎 name→node 索引同步。这些都封装在
app.py 的 _auto_consolidate_curiosity_knowledge 里。恢复脚本自己再实现一遍等于
开第二条写入路径，两条路早晚长歪。所以恢复必须走在线那条。

安全约束（不是建议，是事故教训）
--------------------------------
1. 只用 user_input 作记忆来源，**绝不用 system_response**。她的回复不是用户
   陈述，拿它抽取等于伪造用户经历——这类错误用户已经抓过两次。
   代码里有断言把这条钉死，并且显式请求在线端点（它也只收 source_text）。
2. 默认 dry-run：只抽不写，结果落到 data/recovery_candidates.json，
   每条带来源轮次时间与原文，可逐条审阅追溯。要写图必须显式 --apply。
3. 每轮都打印进度的同时写盘候选文件，中途可中断。
4. --apply 由 app 侧 _save(force=True) 原子落盘；再保险一层是 graph_model.save()
   新增的"意外缩小"守卫（节点数腰斩时自动留副本）。

用法
----
  python chat_log_replay.py                            # 干跑：统计 + 样例（不写图谱）
  python chat_log_replay.py --since 2026-09-06 --limit 20
  python chat_log_replay.py --exclude "天气|摄氏度|阿司匹林"   # 剔除测试/噪声消息
  python chat_log_replay.py --apply                    # 写入图谱（每轮进度可见）
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
CHAT_LOG = os.path.join(BASE, "data", "chat_log.json")
CANDIDATES = os.path.join(BASE, "data", "recovery_candidates.json")
API = "http://127.0.0.1:5000/api/memory/replay"

# 这些不是用户说的话，抽出来的东西一律是伪记忆
NON_USER_PREFIXES = ("(FAS 主动)", "(FAS主动)", "（FAS 主动）", "（FAS主动）")

BATCH = 8          # 每请求轮数（服务端上限 40）
TIMEOUT = 900      # 单请求超时（秒）——抽取是 LLM 调用，慢


def load_turns(since: str, until: str, min_len: int, excludes: list,
               dedup: bool = True) -> tuple:
    """取出可作记忆来源的轮次。

    dedup: 同一条输入只保留最早那次。理由不是省 LLM 调用，而是保真——
    实测 9/6→9/13 窗口 244 轮里只有 55 条唯一输入，前 9 条各重复 18-20 次
    （自动化测试反复回放）。不去重就会把同一句测试话反复写进她的记忆，
    真实对话反而被稀释。
    """
    with open(CHAT_LOG, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else (raw.get("items") or raw.get("log") or [])

    stats = {"total": len(items), "in_window": 0, "proactive": 0, "too_short": 0,
             "excluded": 0, "dup": 0, "kept": 0, "unique": 0}
    pats = [re.compile(p) for p in excludes]
    seen = set()
    turns = []
    for it in items:
        t = str(it.get("time") or "")
        if since and t < since:
            continue
        if until and t >= until:
            continue
        stats["in_window"] += 1
        text = (it.get("user_input") or "").strip()
        if not text or text.startswith(NON_USER_PREFIXES) or text.startswith("(主动)"):
            stats["proactive"] += 1
            continue
        if len(text) < min_len:
            stats["too_short"] += 1
            continue
        if any(p.search(text) for p in pats):
            stats["excluded"] += 1
            continue
        if dedup:
            if text in seen:
                stats["dup"] += 1
                continue
            seen.add(text)
        turns.append({"time": t, "source_text": text})
    stats["kept"] = len(turns)
    stats["unique"] = len(seen)
    return turns, stats


def post(turns: list, dry_run: bool) -> dict:
    body = json.dumps({"turns": turns, "dry_run": dry_run},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="从 chat_log 重建丢失的记忆")
    ap.add_argument("--since", default="2026-09-06", help="起始时间（含）")
    ap.add_argument("--until", default="", help="结束时间（不含）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少轮（0=不限）")
    ap.add_argument("--min-len", type=int, default=4, help="短于该长度跳过（寒暄噪声）")
    ap.add_argument("--exclude", default="", help="正则，命中则跳过（用 | 连接多条）")
    ap.add_argument("--apply", action="store_true", help="真正写入图谱（默认只干跑）")
    ap.add_argument("--no-dedup", action="store_true", help="不去重（默认同一条输入只保留最早那次）")
    ap.add_argument("--list-only", action="store_true",
                    help="只打印将要处理的轮次清单，不调用 LLM、不写任何文件")
    args = ap.parse_args()

    excludes = [p for p in args.exclude.split("|") if p.strip()] if args.exclude else []
    turns, stats = load_turns(args.since, args.until, args.min_len, excludes,
                              dedup=not args.no_dedup)
    if args.limit:
        turns = turns[:args.limit]

    print("=" * 74)
    print("chat_log 重放（事故恢复）—— 目标: 重建 9/6→9/13 丢失的记忆")
    print("=" * 74)
    print(f"窗口         : {args.since or '-'} → {args.until or '现在'}")
    print(f"总轮次       : {stats['total']}")
    print(f"  窗口内     : {stats['in_window']}")
    print(f"  FAS 主动   : {stats['proactive']}  （她的回复不是用户陈述，跳过）")
    print(f"  过短/寒暄  : {stats['too_short']}")
    print(f"  正则剔除   : {stats['excluded']}")
    print(f"  重复折叠   : {stats['dup']}")
    print(f"  本次处理   : {len(turns)}")

    if args.list_only:
        print("-" * 74)
        for i, t in enumerate(turns, 1):
            print(f"{i:3d}. {t['time'][5:16]}  {t['source_text'][:70]}")
        return 0

    print(f"模式         : {'APPLY（写入图谱）' if args.apply else 'DRY-RUN（只抽不写）'}")
    print("-" * 74)
    if not turns:
        print("没有可处理的轮次。")
        return 0

    dry = not args.apply
    all_results = []
    tot_n = tot_e = n_ok = n_err = 0
    t0 = time.time()

    for start in range(0, len(turns), BATCH):
        chunk = turns[start:start + BATCH]
        try:
            r = post(chunk, dry)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            print(f"  [批次 {start+1}-{start+len(chunk)}] HTTP {e.code}: {detail}")
            n_err += len(chunk)
            continue
        except Exception as e:
            print(f"  [批次 {start+1}-{start+len(chunk)}] 请求失败: {type(e).__name__}: {e}")
            print("       （app 在跑吗？恢复必须走在线路径——见文件头注释）")
            return 1

        n_ok += r.get("processed", 0)
        n_err += r.get("failed", 0)
        tot_n += r.get("added_nodes", 0)
        tot_e += r.get("added_edges", 0)
        for res in r.get("results", []):
            all_results.append(res)
            if res.get("error"):
                print(f"  × {res.get('time','')[5:16]} {str(res['error'])[:60]}")
                print(f"    {res.get('source_text','')[:64]}")
            else:
                print(f"  + {res.get('time','')[5:16]} "
                      f"+{res.get('added_nodes',0)}节点 +{res.get('added_edges',0)}边 "
                      f"({res.get('assertion_type','')})")
                print(f"    {res.get('source_text','')[:64]}")
                if res.get("node_ids"):
                    print(f"    → {', '.join(res['node_ids'][:5])}")

        # 每批落一次候选文件：中途中断也有可审阅的产物
        with open(CANDIDATES, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "window": {"since": args.since, "until": args.until},
                "applied": bool(args.apply),
                "stats": stats,
                "totals": {"processed": n_ok, "failed": n_err,
                           "added_nodes": tot_n, "added_edges": tot_e},
                "graph_nodes": r.get("graph_nodes"),
                "graph_edges": r.get("graph_edges"),
                "results": all_results,
            }, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    print("-" * 74)
    print(f"完成         : {n_ok} 轮成功 / {n_err} 轮失败，耗时 {dt:.0f}s "
          f"({dt / max(1, n_ok + n_err):.1f}s/轮)")
    print(f"图谱增量     : +{tot_n} 节点 / +{tot_e} 边"
          + ("" if args.apply else "（未写入，dry-run）"))
    print(f"图谱规模     : {r.get('graph_nodes')} 节点 / {r.get('graph_edges')} 边")
    print(f"候选文件     : {CANDIDATES}（含来源原话，可逐条审阅）")
    if not args.apply:
        print()
        print("这是干跑。审阅候选无误后，加 --apply 才会写入图谱。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
