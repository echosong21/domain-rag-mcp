"""principal（授权身份）与 scope 解析。

**核心语义：`enable` 是"授予 scope"，不是"改会话状态"**

MCP 规范 2026-07-28 有两条规定共同决定了这个设计：

1. **协议级 session 与 `Mcp-Session-Id` 头已被移除**
   ⇒ 状态不能挂在连接上，否则 http 多客户端下重连即丢
2. **`tools/list` 结果 MUST NOT 随连接变化，MAY 因授权（scope）不同而不同**
   ⇒ 想改变"agent 能看到哪些工具"，唯一合规的途径是**改授权**，不是改会话

所以本模块把「已披露的工具」建模为 **principal（profile 名）→ 已授予 scope 集合**。

这个建模带来三个直接好处：
- **合规**：眼见的变化都能解释成授权变化
- **可持久**：http 下重连后工具集一致（按 principal 存，不按连接）
- **可审计**：「谁在什么时候被授予了什么」有明确记录（`Grant.granted_by`）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ProfileError(Exception):
    """profile 配置非法。与 registry 一样，配错了必须启动即炸。"""


@dataclass(frozen=True)
class Grant:
    """一次 scope 授予。保留来源与时间，供审计与排查。"""

    scope: str
    granted_at: float
    granted_by: str          # "profile:initial" / "runtime:enable_tool" / "admin"

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "granted_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.granted_at)),
            "granted_by": self.granted_by,
        }


@dataclass(frozen=True)
class Profile:
    """一个 Agent 的授权声明（来自 `profiles/*.yaml`）。"""

    name: str
    initial_scopes: frozenset[str]
    discovery_scopes: frozenset[str]
    allowed_domains: frozenset[str] = frozenset()
    allow_discovery: bool = True
    allow_ops_write: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ProfileError("profile 必须有 name")
        if not self.initial_scopes:
            raise ProfileError(f"{self.name}: initial_scopes 不能为空")
        # 初始授予必须是可发现范围的子集 —— 否则「能用的工具搜不到」，
        # search_tools 会给出与实际不符的印象
        extra = self.initial_scopes - self.discovery_scopes
        if extra:
            raise ProfileError(
                f"{self.name}: initial_scopes 里有 scope 不在 discovery_scopes 内：{sorted(extra)}。"
                f"初始授予必须是可发现范围的子集，否则 search_tools 的结果会误导调用方"
            )

    @classmethod
    def from_file(cls, path: Path) -> "Profile":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ProfileError(f"{path.name} 顶层应为 mapping")

        def _set(key: str) -> frozenset[str]:
            return frozenset(raw.get(key) or [])

        return cls(
            name=raw.get("name") or path.stem,
            initial_scopes=_set("initial_scopes"),
            discovery_scopes=_set("discovery_scopes") or _set("initial_scopes"),
            allowed_domains=_set("allowed_domains"),
            allow_discovery=bool(raw.get("allow_discovery", True)),
            allow_ops_write=bool(raw.get("allow_ops_write", False)),
            description=raw.get("description") or "",
        )


@dataclass
class PrincipalState:
    """一个 principal 在运行期的已授予 scope 集合。"""

    profile: Profile
    grants: dict[str, Grant] = field(default_factory=dict)

    @classmethod
    def bootstrap(cls, profile: Profile) -> "PrincipalState":
        """按 profile 的初始声明授予第一批 scope。"""
        now = time.time()
        return cls(
            profile=profile,
            grants={
                scope: Grant(scope=scope, granted_at=now, granted_by="profile:initial")
                for scope in profile.initial_scopes
            },
        )

    # ── 查询 ─────────────────────────────────────────────────
    @property
    def scopes(self) -> frozenset[str]:
        return frozenset(self.grants)

    def can_use(self, scope: str) -> bool:
        """是否已授予该 scope。**这是所有权限校验的唯一入口。**"""
        return scope in self.grants

    def can_discover(self, scope: str) -> bool:
        """能否通过 search_tools 发现该 scope 下的工具。"""
        return self.profile.allow_discovery and scope in self.profile.discovery_scopes

    # ── 授予 ─────────────────────────────────────────────────
    def grant(self, scope: str, by: str = "runtime:enable_tool") -> bool:
        """授予一个 scope。返回是否**发生了实际变化**（用于决定要不要发通知）。

        ⚠️ 只允许授予 `discovery_scopes` 内的 scope。
        否则 Agent 可以通过 enable_tool 给自己提权到 `common.ops.write`
        —— 那等于把权限系统拆了。
        """
        if scope in self.grants:
            return False
        if not self.can_discover(scope):
            raise PermissionError(
                f"scope {scope!r} 不在 {self.profile.name} 的可发现范围内，拒绝授予"
            )
        self.grants[scope] = Grant(scope=scope, granted_at=time.time(), granted_by=by)
        return True

    def snapshot(self) -> dict:
        """给审计/调试用的快照。"""
        return {
            "principal": self.profile.name,
            "scopes": sorted(self.grants),
            "grants": {s: g.to_dict() for s, g in sorted(self.grants.items())},
        }