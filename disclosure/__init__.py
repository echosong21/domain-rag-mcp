"""披露层：principal / scope / 元工具。

对外只暴露 `MemoryStore`、`PrincipalState`、`Profile` 与两个元工具函数。
"""
from .meta_tools import (
    META_ENABLE,
    META_SEARCH,
    META_TOOL_NAMES,
    enable_tool,
    meta_tool_specs,
    search_tools,
)
from .principal import Grant, PrincipalState, Profile, ProfileError
from .store_memory import MemoryStore

__all__ = [
    "Grant",
    "META_ENABLE",
    "META_SEARCH",
    "META_TOOL_NAMES",
    "MemoryStore",
    "PrincipalState",
    "Profile",
    "ProfileError",
    "enable_tool",
    "meta_tool_specs",
    "search_tools",
]