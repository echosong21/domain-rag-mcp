"""domain-rag-mcp —— 把领域可配置的 RAG 能力暴露成标准 MCP 工具。

**已接通**

- registry 装配（8 类启动校验）
- **principal / scope 披露**：只把「已授予 scope」的工具加入可见列表
- 两个元工具：`meta.search_tools` / `meta.enable_tool`
- `enable_tool` 成功后动态 `enable()` 对应工具，并自动发 `tools/list_changed`
- 工具实现：discovery / knowledge / safety / ops 四组，共 7 个
- 降级与熔断（`resilience/`）
- 审计与指标（`observability/`）
- stdio 与 http 两种传输（`transport/`）

**关键设计：`enable` 是授予 scope，不是改会话状态**

MCP 2026-07-28 移除了协议级 session，且规定 `tools/list` 不得随连接变化
（但可因授权不同而不同）。所以「已披露工具」= 「已授予 scope 的工具」，
状态按 principal 存。

⚠️ **但实际协商的协议版本是 `2025-11-25`**（FastMCP 4.1.3 的 `initialize`
返回这个版本，传输层仍走握手与 `Mcp-Session-Id`）。「不按连接存状态」是
**应用层的设计选择**，不等于「我们在跑 2026-07-28」。两者别混为一谈 ——
前者我们能决定，后者取决于 SDK 支持到哪一版。

⚠️ 本 docstring 曾长期停留在「Step 2（披露层已接线）」，与实际代码脱节 ——
工具与降级/审计都已落地。2026-11-05 按实测更新。

⚠️ **另一处曾经的脱节（2026-11-07 修复）**：审计与指标的实现早就在
`observability/` 里，但**没有任何一处调用 `record_call`** —— 等于审计从未生效。
现已由 `_wrap_with_audit` 接线。同一次排查还发现 `create_server()` 本身
根本起不来（`health_check` 的 `probe: Probe` 让 Pydantic 生成不了 schema），
而 `--list-tools` 因为不走 `create_server` 一直掩盖着这个问题。
**教训：`--list-tools` 通过 ≠ server 能启动。** 判断 server 可用性必须真的起一次。
"""
from __future__ import annotations

import argparse
import functools
import inspect
import sys
import time
import uuid
from pathlib import Path

from fastmcp import FastMCP

from disclosure import (
    META_ENABLE,
    META_SEARCH,
    MemoryStore,
    PrincipalState,
    Profile,
    enable_tool,
    meta_tool_specs,
    search_tools,
)
from observability.audit import record_call
from registry.loader import build_registry
from registry.models import ToolSpec

ROOT = Path(__file__).resolve().parent
PROFILES_DIR = ROOT / "profiles"
DEFAULT_PROFILE = "health-agent"


# ── 审计包装 ─────────────────────────────────────────────────
def _bind_args(fn, args: tuple, kwargs: dict) -> dict | None:
    """把位置参数与关键字参数还原成「参数名 → 值」。

    审计要记的是「谁用什么参数调了什么」。记成一坨 `args=(...)` 就没法查 ——
    排查时要能直接看到 `{"domain": "health", "top_k": 3}`。
    """
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:  # noqa: BLE001
        return None


def _wrap_with_audit(fn, spec: ToolSpec, principal: str):
    """给工具调用加计时与审计。

    ⚠️ **必须用 `functools.wraps` 保住签名**：FastMCP 从函数签名生成参数 schema，
    如果包装器暴露 `*args, **kwargs`，**所有工具的 schema 会变成空对象** ——
    等于把工具废掉（Agent 看不到该传什么参数）。这是改这里最容易踩的坑。

    ⚠️ **工具内部把错误放在返回值的 `error` 字段里、不抛异常**（本仓一贯约定），
    所以不能只靠 `except` 判失败 —— 必须在返回值里也查一次。
    否则审计里所有失败都会记成 `status="ok"`，而「失败不可见」比没有审计更糟。
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex[:16]
        status, error, result = "ok", None, None

        def _write():
            record_call(
                trace_id=trace_id,
                principal=principal,
                tool=spec.name,
                args=_bind_args(fn, args, kwargs),
                duration_ms=(time.perf_counter() - t0) * 1000,
                status=status,
                error=error,
            )

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = f"{type(exc).__name__}: {exc}"[:200]
            _write()
            raise                       # 异常仍要上抛 —— 审计不是吞错的地方

        # 工具用返回值表达失败，这里必须识别，否则审计全是 ok
        if isinstance(result, dict):
            err = result.get("error")
            if err:
                status = "error"
                error = str(err)[:200]
            elif result.get("blocked"):
                # 安全拦截不是故障，但也不是成功，单独记一档
                status = "blocked"

        _write()
        return result

    return wrapper


# ── 工具注解 ─────────────────────────────────────────────────
def _annotations(spec: ToolSpec):
    """MCP 工具注解。规范要求客户端视其为 **untrusted**，所以它只是提示，
    真正的权限边界在 scope 校验（`can_use`）与 enable/disable。"""
    from mcp.types import ToolAnnotations

    return ToolAnnotations(
        title=spec.name,
        readOnlyHint=(spec.side_effect == "read"),
        destructiveHint=(spec.side_effect == "write"),
        idempotentHint=True,
    )


def _load_profile(name: str) -> Profile:
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"profile {name!r} 不存在：{path}。可用：{[p.stem for p in PROFILES_DIR.glob('*.yaml')]}"
        )
    return Profile.from_file(path)


# ── 装配 ─────────────────────────────────────────────────────
def create_server(
    name: str = "domain-rag-mcp",
    profile_name: str = DEFAULT_PROFILE,
) -> tuple[FastMCP, PrincipalState]:
    """装配 server：注册全部工具 → 只 enable 已授予的 → 挂两个元工具。"""
    registry, handlers = build_registry(root=ROOT)
    profile = _load_profile(profile_name)
    state = PrincipalState.bootstrap(profile)

    mcp = FastMCP(name)

    # ① 全部注册（registry 是事实源，注册不该取决于当前授权）
    for spec in registry.tools.values():
        mcp.tool(
            name=spec.name,
            description=spec.brief,
            tags=set(spec.tags) | {spec.scope, spec.batch},
            annotations=_annotations(spec),
        )(
            # 包装只加计时与审计，不改工具行为。
            # `functools.wraps` 保住签名 ⇒ FastMCP 仍能生成正确的参数 schema。
            _wrap_with_audit(handlers[spec.name], spec, profile_name)
        )

    # ② 未授予 scope 的工具立即 disable —— 这是「不可见」的实现
    ungranted = {
        spec.name for spec in registry.tools.values() if not state.can_use(spec.scope)
    }
    if ungranted:
        mcp.disable(names=ungranted)

    # ③ 元工具：始终可见（否则 Agent 拿不到「发现更多」的入口）
    def _search(query: str) -> dict:
        return search_tools(query, state, registry)

    def _enable(tool_name: str) -> dict:
        result = enable_tool(tool_name, state, registry)
        # 授予成功后把该 scope 下的**所有**工具打开
        # （scope 是权限单位，一个 scope 可能覆盖多个工具）
        if result.get("ok") and result.get("changed"):
            scope = result["scope"]
            names = {
                s.name for s in registry.tools.values() if s.scope == scope
            }
            mcp.enable(names=names)   # 自动发 tools/list_changed
        return result

    mcp.tool(
        name=META_SEARCH,
        description=next(t["description"] for t in meta_tool_specs() if t["name"] == META_SEARCH),
        tags={"meta", "discovery"},
    )(_search)

    mcp.tool(
        name=META_ENABLE,
        description=next(t["description"] for t in meta_tool_specs() if t["name"] == META_ENABLE),
        tags={"meta", "discovery"},
    )(_enable)

    return mcp, state


# ── CLI ──────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="domain-rag-mcp",
        description="把领域可配置的 RAG 检索能力暴露成 MCP 工具",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="用哪个 profile 的授权")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "http"],
        help="stdio 给本地客户端；http 给远程多客户端",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--list-tools", action="store_true",
        help="打印该 profile 下**可见**的工具后退出（不开客户端就能核对暴露面）",
    )
    args = parser.parse_args(argv)

    if args.list_tools:
        profile = _load_profile(args.profile)
        state = PrincipalState.bootstrap(profile)
        registry, _ = build_registry(root=ROOT)
        visible = registry.by_scope(state.scopes)
        print(f"profile = {profile.name}")
        print(f"已授予 scope = {sorted(state.scopes)}")
        print(f"可见工具 {len(visible)} / 总计 {len(registry.tools)}：\n")
        for spec in visible:
            print(f"  {spec.name:42s} [{spec.scope}]")
            print(f"  {'':42s} {spec.brief}")
        print(f"\n元工具（始终可见）: {META_SEARCH}, {META_ENABLE}")
        return 0

    mcp, _state = create_server(profile_name=args.profile)

    if args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())