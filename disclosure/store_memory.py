"""principal → PrincipalState 的进程内存储（stdio 场景）。

**为什么 stdio 用内存、http 用 SQLite**

- `stdio`：单客户端、单进程、生命周期等同于连接 ⇒ 内存足够，且零依赖
- `streamable-http`：多客户端并发，且规范已移除协议级 session ⇒
  状态必须**按 principal 持久化**，否则客户端重连后已授予的 scope 全丢

两者对外接口一致（`get_or_create` / `reset` / `principals`），
所以上层 `meta_tools` 与 `server` 不需要知道自己跑在哪种传输上。
"""
from __future__ import annotations

from .principal import Profile, PrincipalState


class MemoryStore:
    """进程内存储。"""

    def __init__(self) -> None:
        self._states: dict[str, PrincipalState] = {}

    def get_or_create(self, profile: Profile) -> PrincipalState:
        """取 principal 的状态；首次访问时按 profile 的初始声明授予 scope。"""
        state = self._states.get(profile.name)
        if state is None:
            state = PrincipalState.bootstrap(profile)
            self._states[profile.name] = state
        return state

    def get(self, name: str) -> PrincipalState | None:
        return self._states.get(name)

    def principals(self) -> list[str]:
        return sorted(self._states)

    def reset(self) -> None:
        """清空（测试用：保证用例之间不串状态）。"""
        self._states.clear()