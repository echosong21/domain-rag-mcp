"""两个元工具：`search_tools` 与 `enable_tool`。

**它们为什么是元工具而不是普通工具**

普通工具的 schema 会随工具数量线性增长 context。元工具的 schema **恒定**
（两个工具、四个参数），却能让 Agent 触达任意多的工具 ——
这是渐进披露成立的算术基础。

**三条设计约束（都由规范推导而来）**

1. **先按 scope 过滤再搜索**
   不能"先搜全量再过滤"，否则「搜到了但调不动」会把 Agent 引向死路。
   规范允许 `tools/list` 因授权不同而不同，正是为了让"看不到"成为合法状态。

2. **只允许授予 `discovery_scopes` 内的 scope**
   否则 Agent 能通过 `enable_tool` 给自己提权到 `common.ops.write` ——
   权限系统就此失效。这条在 `PrincipalState.grant` 里强制。

3. **授予成功必须发通知**
   规范 2026-07-28：`tools/list` 变化只能经 `subscriptions/listen` 的
   `toolsListChanged` 送达，不能靠客户端轮询。
"""
from __future__ import annotations

from .principal import PrincipalState
from registry.models import Registry, ToolSpec

# 元工具自身的名字。⚠️ 它们**始终可见**，不参与 scope 过滤 ——
# 否则 Agent 拿不到"发现更多"的入口，披露就变成单向收窄。
META_SEARCH = "meta.search_tools"
META_ENABLE = "meta.enable_tool"
META_TOOL_NAMES = frozenset({META_SEARCH, META_ENABLE})


def meta_tool_specs() -> list[dict]:
    """元工具的 MCP 定义（供 server 注册）。"""
    return [
        {
            "name": META_SEARCH,
            "description": (
                "在本 principal 有权访问的范围内搜索工具。"
                "返回 [name, brief, scope] 摘要，不返回完整 schema —— "
                "确认需要后再调用 enable_tool 授予它。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自然语言或关键词，如「检索」「营养」",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": META_ENABLE,
            "description": (
                "授予某个工具所在 scope 的使用权，此后它会出现在 tools/list 中并可被调用。"
                "只能授予 search_tools 搜到的工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "工具名，如 common.knowledge.search",
                    },
                },
                "required": ["name"],
            },
        },
    ]


def search_tools(
    query: str,
    state: PrincipalState,
    registry: Registry,
    limit: int = 5,
) -> dict:
    """在 principal 有权发现的范围内搜索。

    返回**摘要**而非完整 schema：摘要约 30 token/个，
    完整 schema 约 450 token/个 —— 这是 15 倍的差距。
    """
    if not state.profile.allow_discovery:
        return {"tools": [], "note": "本 principal 未开启动态发现"}

    # 只搜 discovery_scopes 内的（不是所有已授予的）
    scopable = {s for s in state.profile.discovery_scopes}
    hits = registry.search(query, scopable, limit=limit)

    return {
        "tools": [
            {
                "name": s.name,
                "brief": s.brief,
                "scope": s.scope,
                "already_granted": state.can_use(s.scope),
                "batch": s.batch,
            }
            for s in hits
        ],
        "searched_scopes": sorted(scopable),
    }


def enable_tool(
    name: str,
    state: PrincipalState,
    registry: Registry,
) -> dict:
    """授予该工具所在 scope。

    返回 `changed` 表明是否发生了实际变化 —— 调用方据此决定要不要发
    `notifications/tools/list_changed`。重复授予是幂等的（返回 changed=false）。
    """
    spec: ToolSpec | None = registry.get(name)
    if spec is None:
        # ⚠️ 刻意不在错误信息里列出所有工具名：
        # 那会把未授权工具的清单泄露给无权访问的调用方
        return {"ok": False, "error": f"工具 {name!r} 不存在或不可见"}

    if spec.scope not in state.profile.discovery_scopes:
        return {
            "ok": False,
            "error": f"{name} 的 scope {spec.scope!r} 不在本 principal 的可发现范围内",
        }

    try:
        changed = state.grant(spec.scope)
    except PermissionError as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "name": name,
        "scope": spec.scope,
        "changed": changed,
        "message": (
            f"已授予 {spec.scope}，该 scope 下的工具现在可见并可调用"
            if changed
            else f"{spec.scope} 此前已授予，无需重复"
        ),
    }