#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_chat_log_replay.py — 记忆重放的安全约束

这些断言不是形式主义。2026-09-13 的事故链是：
  1. 迁移脚本误用 classmethod（load）→ 空图覆写运行时图谱（894→162 节点）；
  2. 恢复时若再从 chat_log 重放，而重放取错了来源（比如拿她的回复当"用户说的"），
     就会把不存在的用户经历写进她脑子里——这类错误用户已经抓过两次。

所以重放的边界必须由测试钉死，不能只写在注释里。
"""

import json
import os
import sys
import tempfile

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)

import chat_log_replay as R

_p = _f = 0


def check(name, cond, detail=""):
    global _p, _f
    if cond:
        _p += 1
        print(f"[PASS] {name}")
    else:
        _f += 1
        print(f"[FAIL] {name} {detail}")


def _write_log(items):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    return path


def main():
    print("=" * 70)
    print("记忆重放安全约束")
    print("=" * 70)

    # ── 1. 铁律：她的回复永远不进记忆来源 ──
    items = [
        {"time": "2026-09-07T10:00:00", "user_input": "我去驾校练车了",
         "system_response": "（FAS 自称：我今天也去了火星）"},
        {"time": "2026-09-07T11:00:00", "user_input": "(FAS 主动)",
         "system_response": "我想问问你在干嘛"},
        {"time": "2026-09-07T12:00:00", "user_input": "哦", "system_response": "嗯"},
        {"time": "2026-09-07T13:00:00", "user_input": "明天要考科目二",
         "system_response": "祝你顺利"},
    ]
    path = _write_log(items)
    orig = R.CHAT_LOG
    R.CHAT_LOG = path
    try:
        turns, stats = R.load_turns("", "", 4, [])
    finally:
        R.CHAT_LOG = orig
        os.unlink(path)

    check("只保留有用户原话的轮次", len(turns) == 2, f"得到 {len(turns)}")
    texts = [t["source_text"] for t in turns]
    check("用户原话被保留", "我去驾校练车了" in texts and "明天要考科目二" in texts, str(texts))
    check("FAS 主动发言被跳过", stats["proactive"] == 1, str(stats))
    check("过短输入被跳过", stats["too_short"] == 1, str(stats))
    check("system_response 一字未进载荷",
          all("火星" not in json.dumps(t, ensure_ascii=False) for t in turns))
    check("载荷字段只有 time/source_text",
          all(set(t.keys()) == {"time", "source_text"} for t in turns),
          str([sorted(t.keys()) for t in turns]))
    check("空窗口统计正确", stats["in_window"] == 4, str(stats))

    # ── 2. 时间窗口按 ISO 前缀比较 ──
    items2 = [
        {"time": "2026-09-05T23:00:00", "user_input": "窗口外的话"},
        {"time": "2026-09-06T00:01:00", "user_input": "窗口内的话"},
        {"time": "2026-09-10T12:00:00", "user_input": "结束前的话"},
        {"time": "2026-09-11T00:00:00", "user_input": "被 until 排除"},
    ]
    path2 = _write_log(items2)
    R.CHAT_LOG = path2
    try:
        turns2, _ = R.load_turns("2026-09-06", "2026-09-11", 4, [])
    finally:
        R.CHAT_LOG = orig
        os.unlink(path2)
    t2 = [t["source_text"] for t in turns2]
    check("since 生效", "窗口外的话" not in t2, str(t2))
    check("窗口内两条保留", t2 == ["窗口内的话", "结束前的话"], str(t2))

    # ── 3. 正则剔除（用于手工排除测试流量）──
    items3 = [{"time": "2026-09-09T10:00:00", "user_input": "帮我查一下明天的天气"},
              {"time": "2026-09-09T11:00:00", "user_input": "我把离合器踩熄火了"}]
    path3 = _write_log(items3)
    R.CHAT_LOG = path3
    try:
        turns3, stats3 = R.load_turns("", "", 4, ["天气"])
    finally:
        R.CHAT_LOG = orig
        os.unlink(path3)
    t3 = [t["source_text"] for t in turns3]
    check("正则剔除生效", t3 == ["我把离合器踩熄火了"], str(t3))
    check("剔除计数正确", stats3["excluded"] == 1, str(stats3))

    # ── 4. 走的是在线端点，不是自实现的写入路径 ──
    src = open(R.__file__, "r", encoding="utf-8").read()
    check("脚本走在线重放端点", "/api/memory/replay" in src)
    check("脚本不直接写图谱文件",
          "kg.save(" not in src and "upsert_node(" not in src and "add_edge(" not in src)
    check("脚本不自行初始化图谱/LLM",
          "merge_runtime_graph" not in src and "NLPProcessor(" not in src)

    # ── 5. app 侧端点存在且带铁律注释 ──
    app_src = open(os.path.join(os.path.dirname(R.__file__), "app.py"),
                   encoding="utf-8").read()
    check("app 有 /api/memory/replay 端点", "/api/memory/replay" in app_src)
    check("端点限制单批轮数", "单次最多 40 轮" in app_src)
    check("端点复用在线沉淀函数",
          "_auto_consolidate_curiosity_knowledge(kg, draft, buffer, text)" in app_src)
    check("端点用原轮次时间做锚点", "now=_t_when" in app_src)

    # ── 6. 抽取层支持时间覆盖（保真度）──
    nlp_src = open(os.path.join(os.path.dirname(R.__file__), "nlp_processor.py"),
                   encoding="utf-8").read()
    check("extract_assertion_graph 支持 now 覆盖", "now: \"datetime\" = None" in nlp_src)
    check("默认仍取当前时间", "(now or datetime.now())" in nlp_src)

    print("=" * 70)
    print(f"通过 {_p} / {_p + _f}")
    if _f:
        print(f"失败 {_f} 项")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
