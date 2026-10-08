# -*- coding: utf-8 -*-
"""一次性维护脚本：给陪伴仓的脚本式测试/研究脚本补上 FAS-Cognitive 同级仓路径。

只动 sys.path 引导行，不改任何测试逻辑/断言/参数。幂等：已有 _cog 标记的文件跳过。
"""
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
MARK = "# split_repos bootstrap: FAS-Cognitive sibling"
BLOCK = (
    MARK + "\n"
    "_cog = os.environ.get(\"FAS_COG_ROOT\") or os.path.join(\n"
    "    os.path.dirname(_r), \"FAS-Cognitive\")\n"
    "if os.path.isdir(_cog) and _cog not in sys.path:\n"
    "    sys.path.insert(0, _cog)\n"
)
PATTERN = re.compile(
    r"(sys\.path\.insert\(0, os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\))\n")

n = 0
for folder in ("tests", "scripts"):
    d = os.path.join(ROOT, folder)
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py") or f == "conftest.py":
            continue
        p = os.path.join(d, f)
        src = open(p, encoding="utf-8").read()
        if MARK in src or not PATTERN.search(src):
            continue
        repl = (
            "_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
            "sys.path.insert(0, _r)\n" + BLOCK)
        src2 = PATTERN.sub(repl, src, count=1)
        open(p, "w", encoding="utf-8").write(src2)
        n += 1
print(f"patched {n} files")
