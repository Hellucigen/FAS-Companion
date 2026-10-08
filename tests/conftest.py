# tests/conftest.py — pytest 收集假面
# ============================================================================
# 修复记录 L1-TH-01（research_audit/PATCH_LOG.md,2026-09-30,二期加固）：
#
# 结构事实（AST 全量扫描 2026-09-30,114 个测试文件）：
#   93 个文件是"脚本式回归"——模块顶层直接执行断言并 sys.exit() 报状态
#   （python tests/xxx.py 独立运行,退出码=结果,文件头注释写明）。pytest
#   收集必须 import,顶层 exit 即 SystemExit → INTERNALERROR,整个套件 0 测试。
#   一期修复用 17 项显式 collect_ignore,在 test_world_events.py 上被击穿
#   （第 18 个同型文件,INTERNALERROR 复现）→ 显式清单不可维护。
# 二期修复（本文件）：
#   1) 显式清单：real_* 实机脚本 + 无顶层 exit 但有连服务副作用的文件；
#   2) AST 自动分类：凡模块顶层（非函数内、非 __main__ 守卫内）裸 exit()/
#      sys.exit() 的文件自动跳过收集。新脚本式文件加入后自动被跳过,
#      无需改本文件；函数内或 main 守卫内的 exit 不受影响（pytest 不执行）。
# 效果：pytest tests/ 收集 13 个真 pytest 风格文件（87 个 test_* 函数）；
#       脚本式回归仍按各文件头注释独立运行。
# ============================================================================

import ast
import pathlib

# real_* 实机脚本（需真实服务/副作用,单独运行）；real_common 是公共桩。
_SCRIPT_EXPLICIT = [
    "real_0_connect.py",
    "real_a_movement.py",
    "real_b_social.py",
    "real_cdf_analysis.py",
    "real_common.py",
    "real_e_mining.py",
    "real_g_watch.py",
    "real_h_trials.py",
]


# 内置 pytest fixture 名集合:这些是可被 pytest 注入的合法函数参数;白名单
# 之外的 test_* 函数参数 = 脚本式装配参数(靠 __main__ 手动注入)→ 判脚本式。
_BUILTIN_FIXTURES = {
    "tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory", "monkeypatch",
    "capsys", "capsysbinary", "capfd", "capfdbinary", "caplog", "recwarn",
    "request", "cache", "pytestconfig", "record_property",
    "record_testsuite_property", "record_xml_attribute", "doctest_namespace",
    "subtests", "free_tcp_port", "free_tcp_port_factory", "free_udp_port",
    "free_udp_port_factory", "anyio_backend", "anyio_backend_name",
    "anyio_backend_options",
}


def _is_script_style(path: pathlib.Path) -> bool:
    """"脚本式回归"判定:
    1) 模块顶层(非函数内、非 __main__ 守卫内)含 exit()/sys.exit();或
    2) test_* 函数签名带非内置 fixture 参数(说明靠 __main__ 装配调用,
       单独收集必然 fixture-not-found,不是独立 pytest 测试)。
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return False

    func_depth = 0
    in_main_guard = False
    found = {"bad": False}

    def in_main_if(node):
        t = node.test
        if not isinstance(t, ast.Compare) or len(t.comparators) != 1:
            return False
        parts = (t.left, t.comparators[0])
        names = {p.id for p in parts if isinstance(p, ast.Name)}
        consts = {p.value for p in parts if isinstance(p, ast.Constant)}
        return "__main__" in consts and "__name__" in names

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            nonlocal func_depth
            # 规则 2:test_* 函数带非内置 fixture 形参 → 脚本式装配
            if n.name.startswith("test_"):
                for a in n.args.args:
                    if a.arg not in _BUILTIN_FIXTURES:
                        found["bad"] = True
            func_depth += 1
            self.generic_visit(n)
            func_depth -= 1

        def visit_AsyncFunctionDef(self, n):
            self.visit_FunctionDef(n)

        def visit_ClassDef(self, n):
            for s in n.body:
                self.generic_visit(s)

        def visit_If(self, n):
            nonlocal in_main_guard
            prev = in_main_guard
            if in_main_if(n):
                in_main_guard = True
            for s in n.body:
                self.generic_visit(s)
            in_main_guard = prev
            for s in n.orelse:
                self.generic_visit(s)

        def visit_Try(self, n):
            for s in n.body:
                self.generic_visit(s)
            for h in n.handlers:
                self.generic_visit(h)
            for s in n.orelse:
                self.generic_visit(s)
            for s in n.finalbody:
                self.generic_visit(s)

        def visit_Call(self, n):
            if func_depth == 0 and not in_main_guard:
                f = n.func
                if (isinstance(f, ast.Attribute) and f.attr == "exit") or (
                    isinstance(f, ast.Name) and f.id in ("exit", "quit")
                ):
                    found["bad"] = True
            self.generic_visit(n)

    _Visitor().visit(tree)
    return found["bad"]


_ignore = list(_SCRIPT_EXPLICIT)
for _p in sorted(pathlib.Path(__file__).parent.glob("*.py")):
    if _p.name == "conftest.py" or _p.name in _SCRIPT_EXPLICIT:
        continue
    if _is_script_style(_p):
        _ignore.append(_p.name)

collect_ignore = _ignore