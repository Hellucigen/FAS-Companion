import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 认知核心仓（同级 FAS-Cognitive，或 FAS_COG_ROOT 指定）
_COG = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_ROOT), "FAS-Cognitive")
if os.path.isdir(_COG) and _COG not in sys.path:
    sys.path.insert(0, _COG)
