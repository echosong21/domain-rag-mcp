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

⚠️ 本 docstring 曾长期停留在「Step 2（披露层已接线）」，与实际代码脱节 ——
工具与降级/审计都已落地。2026-11-05 按实测更新。
"""
from __future__ import annotations

import argparse
import sys
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
from registry.loader import build_registry
from registry.models import ToolSpec

ROOT = Path(__file__).resolve().parent
PROFILES_DIR = ROOT / "profiles"
DEFAULT_PROFILE = "health-agent"


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
        )(handlers[spec.name])

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