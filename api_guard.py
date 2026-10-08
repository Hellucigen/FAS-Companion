# api_guard.py — API 访问守卫的纯判定逻辑（§18.2 #19，2026-09-22）
# ============================================================================
# 背景：服务全部路由无鉴权。默认监听 127.0.0.1 缓解了暴露面，但
# 浏览器 CSRF / DNS rebinding 仍可打到回环（恶意网页里的 JS 以
# 127.0.0.1 为"同源"发请求）。本模块给出两道独立防线的**纯函数**判定，
# app.py 只做 Flask 粘合——离线单测直接测这里，不 import app。
#
#   1. Origin 核验（常开）：带 Origin/Referer 的请求，其主机必须与
#      被请求主机一致或为回环名——恶意域名的跨源请求在这里被拒。
#   2. Token 核验（按需开）：token 生效时（config/FAS_API_TOKEN/
#      data/api_token.json 任一提供，或非回环绑定自动生成），**非回环
#      来源**必须携带匹配 token（X-FAS-Token 头 / Bearer / cookie /
#      ?token= 引导）。回环来源在 api_token_require_loopback=false
#      （默认）时不受影响——本地使用零摩擦的现状保持不变。
# ============================================================================

from urllib.parse import urlsplit

LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")
# 引导通道：这些路径不要求 token（静态外壳不含任何用户数据）
BOOTSTRAP_EXEMPT_PATHS = ("/", "/favicon.ico")


def is_loopback_addr(remote_addr: str) -> bool:
    addr = str(remote_addr or "").strip()
    if addr.startswith("127."):          # 127.0.0.0/8 整段
        return True
    if addr in ("::1", "::ffff:127.0.0.1"):
        return True
    if addr.startswith("::ffff:127."):   # v4-mapped
        return True
    return False


def host_of_url(url: str) -> str:
    """从 URL / "host:port" 串提取主机名（小写、去端口；失败返回空串）。"""
    s = str(url or "").strip()
    if not s:
        return ""
    try:
        if "://" not in s:
            s = "//" + s          # 裸 host[:port] 也走同一解析
        return (urlsplit(s).hostname or "").lower()
    except Exception:
        return ""


def _hosts_equal(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a in LOOPBACK_NAMES and b in LOOPBACK_NAMES:
        return True                      # 回环名的各种写法视为同源
    return a == b


def origin_allowed(origin: str, referer: str, request_host: str) -> bool:
    """无 Origin/Referer（curl/本地脚本/非浏览器）→ 放行；
    有则主机必须与请求主机一致（或同为回环名）。"""
    src = str(origin or "").strip() or str(referer or "").strip()
    if not src:
        return True
    return _hosts_equal(host_of_url(src), host_of_url(request_host))


def evaluate_request(*, remote_addr: str, path: str = "/",
                     origin: str = "", referer: str = "",
                     request_host: str = "",
                     token_expected: str = "",
                     token_supplied: str = "",
                     token_via_query: bool = False,
                     require_token_on_loopback: bool = False) -> dict:
    """统一裁决。返回 {allow, status, reason, mode}：
      mode ∈ allow / deny_origin / deny_token / bootstrap
    bootstrap = 非回环来源经 ?token= 引导通过——上层应下发 cookie，
    之后的 API 调用走 cookie/header 通道。token 不匹配一律 401，
    不回显期望值。静态外壳（BOOTSTRAP_EXEMPT_PATHS）不含用户数据，
    token 生效时也可匿名访问（浏览器需要先拿到它才能带 ?token=）。"""
    if not origin_allowed(origin, referer, request_host):
        return {"allow": False, "status": 403,
                "reason": "跨源请求被拒（Origin/Referer 与请求主机不一致）"
                          "——防 CSRF/DNS rebinding",
                "mode": "deny_origin"}
    token_on = bool(str(token_expected or "").strip())
    from_loop = is_loopback_addr(remote_addr)
    if token_on and (require_token_on_loopback or not from_loop):
        supplied = str(token_supplied or "").strip()
        matched = bool(supplied) and supplied == token_expected
        if matched:
            if not from_loop and token_via_query:
                return {"allow": True, "status": 200, "reason": "",
                        "mode": "bootstrap"}
            return {"allow": True, "status": 200, "reason": "",
                    "mode": "allow"}
        if str(path or "/") in BOOTSTRAP_EXEMPT_PATHS:
            return {"allow": True, "status": 200,
                    "reason": "静态外壳豁免（不含用户数据）",
                    "mode": "allow"}
        return {"allow": False, "status": 401,
                "reason": "缺少或错误的 API token（X-FAS-Token 头、"
                          "Authorization: Bearer、cookie 或 ?token= 任一）",
                "mode": "deny_token"}
    return {"allow": True, "status": 200, "reason": "", "mode": "allow"}
