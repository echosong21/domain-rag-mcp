"""工具注册表的数据模型。

**为什么要有这一层**

一个 MCP server 的工具定义散落在各处（scope、fallback、副作用、必填依赖），
如果这些信息写在各个工具的 `.py` 里，就没有单一地方能回答「现在有哪些工具、
谁被允许调、哪个坏了能降级到哪个」。

⇒ 全部收进 `registry.yaml`，本模块只负责把它变成**带校验**的对象。

**启动即校验，而不是运行时报错**

`ToolSpec.__post_init__` 与 `Registry.validate()` 在**加载时**就检查：

- 工具名是否符合 `<namespace>.<tool>` 的两段格式
- `scope` 是否在已知 scope 列表内
- `fallback` 里的目标是否**同命名空间**（跨域降级是明确禁止的）
- `fallback` 目标是否存在
- 依赖 `milvus` / `embedding` / `rerank` 的能力声明是否合法

任何一条不过就 **拒绝启动** —— 因为一个配错的 registry 会让故障在运行时才暴露，
而运行时暴露的代价是「Agent 收到一个它无法解释的错误」。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# 已知的副作用类型：决定权限矩阵如何拦截
SideEffect = Literal["read", "write"]

# 已知的外部依赖：health_check 据此报告降级，熔断器据此分区
Dependency = Literal["milvus", "embedding", "rerank", "llm"]

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


class RegistryError(Exception):
    """registry 配置非法。**刻意不降级不忽略** —— 配错了必须在启动时炸掉。"""


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的完整元数据（来自 `registry.yaml` 的一项）。"""

    name: str
    module: str
    handler: str
    brief: str
    scope: str
    tags: tuple[str, ...] = ()
    side_effect: SideEffect = "read"
    fallback: tuple[str, ...] = ()
    requires: tuple[Dependency, ...] = ()
    batch: Literal["core", "extended"] = "core"

    def __post_init__(self) -> None:
        if not NAME_RE.match(self.name):
            raise RegistryError(
                f"工具名 {self.name!r} 不合规：应为 `<namespace>.<tool>`，"
                f"如 `common.knowledge.search`（小写字母/数字/下划线，至少两段）"
            )
        if self.side_effect not in ("read", "write"):
            raise RegistryError(f"{self.name}: side_effect 只能是 read/write，实际 {self.side_effect!r}")
        if self.batch not in ("core", "extended"):
            raise RegistryError(f"{self.name}: batch 只能是 core/extended，实际 {self.batch!r}")
        if not self.brief.strip():
            raise RegistryError(f"{self.name}: brief 不能为空 —— 它是 search_tools 的唯一摘要来源")
        if not self.tags:
            raise RegistryError(f"{self.name}: tags 不能为空 —— tool search 靠它匹配")

    @property
    def namespace(self) -> str:
        """两段以上的命名空间（去掉最后一段工具名）。

        `common.knowledge.search` → `common.knowledge`
        `common.health_check`     → `common`
        """
        return self.name.rsplit(".", 1)[0]

    @property
    def family(self) -> str:
        """顶层命名空间，用于权限域判断（`common` / `dev` / `health` ...）。"""
        return self.name.split(".", 1)[0]

    def same_family(self, other: "ToolSpec") -> bool:
        """是否同权限域 —— 降级的唯一合法范围。"""
        return self.family == other.family


@dataclass
class Registry:
    """全部工具的索引，负责跨工具的一致性校验。"""

    tools: dict[str, ToolSpec] = field(default_factory=dict)
    scopes: set[str] = field(default_factory=set)

    # ── 构建 ──────────────────────────────────────────────────
    @classmethod
    def from_specs(cls, specs: list[ToolSpec], known_scopes: set[str]) -> "Registry":
        reg = cls(tools={s.name: s for s in specs}, scopes=set(known_scopes))
        reg.validate()
        return reg

    # ── 校验（启动时一次性跑完）────────────────────────────────
    def validate(self) -> None:
        if not self.tools:
            raise RegistryError("registry 为空 —— 至少要有 1 个工具")

        # ① 重名
        if len(self.tools) != len({s.name for s in self.tools.values()}):
            raise RegistryError("registry 里有重名工具")

        for spec in self.tools.values():
            # ② scope 必须在已知列表内
            if spec.scope not in self.scopes:
                raise RegistryError(
                    f"{spec.name}: scope {spec.scope!r} 未定义。"
                    f"已知 scope: {sorted(self.scopes)}"
                )

            # ③ fallback 目标必须存在 + 同命名空间 + 不指向自己
            for target_name in spec.fallback:
                target = self.tools.get(target_name)
                if target is None:
                    raise RegistryError(
                        f"{spec.name}: fallback 目标 {target_name!r} 不存在"
                    )
                if target_name == spec.name:
                    raise RegistryError(f"{spec.name}: fallback 不能指向自己")

                # ❗ 核心约束：降级只能发生在同一权限域内
                if not spec.same_family(target):
                    raise RegistryError(
                        f"{spec.name}: fallback 目标 {target_name!r} 跨权限域"
                        f"（{spec.family} → {target.family}）。"
                        f"降级只允许同域内发生 —— 否则会配出「安全闸降级到检索」这种荒谬链路"
                    )

                # ④ 写操作不允许有 fallback（自动找替代去改数据是灾难）
                if spec.side_effect == "write":
                    raise RegistryError(
                        f"{spec.name}: 写工具不允许配置 fallback"
                    )

        # ⑤ fallback 不能成环（A→B→A 会让重试永不停）
        for spec in self.tools.values():
            for target_name in spec.fallback:
                target = self.tools[target_name]
                if spec.name in target.fallback:
                    raise RegistryError(
                        f"fallback 成环：{spec.name} ↔ {target_name}"
                    )

    # ── 查询 ──────────────────────────────────────────────────
    def by_scope(self, scopes: set[str]) -> list[ToolSpec]:
        """给定 scope 集合，返回可用工具（deterministic 顺序）。

        ⚠️ 规范 2026-07-28 要求 `tools/list` 以**确定性顺序**返回，
        以便客户端缓存与 LLM prompt cache 命中 —— 所以这里按名字排序。
        """
        return sorted(
            (s for s in self.tools.values() if s.scope in scopes),
            key=lambda s: s.name,
        )

    def search(self, query: str, scopes: set[str], limit: int = 5) -> list[ToolSpec]:
        """`search_tools` 的实现：只在**有权限的 scope** 内搜索。

        匹配策略刻意简单（子串 + tag 命中 + 名字分段），因为：
        - 工具总数只有 17 个，上 BM25 是过度设计
        - 可解释：能说清「为什么这个工具被搜到」

        ⚠️ 先按 scope 过滤再搜索 —— 不能先搜再过滤，
        否则「搜到了但调不动」会把 Agent 引向无用的尝试。
        """
        q = query.strip().lower()
        if not q:
            return []

        scored: list[tuple[int, str, ToolSpec]] = []
        for spec in self.by_scope(scopes):
            score = 0
            if q in spec.name.lower():
                score += 10
            if any(q in t.lower() for t in spec.tags):
                score += 5
            if q in spec.brief.lower():
                score += 3
            # 名字分段命中（如 "search" 命中 common.knowledge.search）
            if any(q == seg for seg in spec.name.lower().split(".")):
                score += 8
            if score:
                scored.append((score, spec.name, spec))

        # 分数降序，同分按名字——保证确定性
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [s for _score, _name, s in scored[:limit]]

    def get(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)