"""从 `registry.yaml` 装配工具表。

**为什么拆成三步（读 yaml / 建 Registry / 解析 handler）**

因为它们的失败原因完全不同，分开后能给出精确的错误位置：

| 步骤 | 典型失败 | 错误信息 |
|---|---|---|
| `load_specs` | yaml 少字段、类型不对 | 「工具定义缺少字段 ['scope']」 |
| `Registry.from_specs` | 跨域 fallback、写工具带降级、scope 未定义 | 「fallback 目标跨权限域」 |
| `resolve_handlers` | 模块路径写错、handler 改名 | 「无法导入模块 tools.x.y」 |

合并成一步的话，所有错误会变成一句含糊的「registry 加载失败」。

**handler 在启动时解析而不是首次调用时**

为了让「模块路径写错」在启动就暴露。否则要等到 Agent 真的调用到那个工具
才会发现 —— 而在那之前它会一直出现在 `tools/list` 里，是被承诺过却不存在的能力。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from .models import Registry, RegistryError, ToolSpec

REGISTRY_FILE = Path(__file__).with_name("registry.yaml")

# 必填字段：缺任何一个都无法构成一个可调用的工具
REQUIRED_FIELDS = frozenset({"name", "module", "handler", "brief", "scope"})


def load_specs(path: Path = REGISTRY_FILE) -> tuple[list[ToolSpec], set[str]]:
    """读 yaml → `(specs, scopes)`。**不解析 handler**（便于纯配置单测）。"""
    if not path.is_file():
        raise RegistryError(f"registry 文件不存在：{path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RegistryError(f"{path.name} 顶层应为 mapping，实际 {type(raw).__name__}")

    scopes = set(raw.get("scopes") or [])
    if not scopes:
        raise RegistryError(f"{path.name} 未声明任何 scope")

    specs: list[ToolSpec] = []
    for idx, item in enumerate(raw.get("tools") or []):
        if not isinstance(item, dict):
            raise RegistryError(f"{path.name} 的 tools[{idx}] 应为 mapping")

        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            raise RegistryError(
                f"{path.name} 的 tools[{idx}] 缺少字段 {sorted(missing)}：{item.get('name', '<无名>')}"
            )

        specs.append(
            ToolSpec(
                name=item["name"],
                module=item["module"],
                handler=item["handler"],
                brief=item["brief"],
                scope=item["scope"],
                tags=tuple(item.get("tags") or ()),
                side_effect=item.get("side_effect", "read"),
                fallback=tuple(item.get("fallback") or ()),
                requires=tuple(item.get("requires") or ()),
                batch=item.get("batch", "core"),
            )
        )
    return specs, scopes


def resolve_handlers(
    specs: list[ToolSpec], root: Path
) -> dict[str, Callable[..., Any]]:
    """按 `module` + `handler` 动态导入，返回 `{工具名: 可调用对象}`。"""
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    handlers: dict[str, Callable[..., Any]] = {}
    for spec in specs:
        try:
            mod = importlib.import_module(spec.module)
        except ImportError as exc:
            raise RegistryError(
                f"{spec.name}: 无法导入模块 {spec.module!r}（{exc}）。"
                f"检查 registry.yaml 的 module 字段，以及 {root} 是否在 sys.path"
            ) from exc

        fn = getattr(mod, spec.handler, None)
        if fn is None:
            raise RegistryError(
                f"{spec.name}: 模块 {spec.module!r} 里没有 {spec.handler!r}"
            )
        if not callable(fn):
            raise RegistryError(f"{spec.name}: {spec.handler!r} 不是可调用对象")

        handlers[spec.name] = fn
    return handlers


def build_registry(
    path: Path = REGISTRY_FILE, root: Path | None = None
) -> tuple[Registry, dict[str, Callable[..., Any]]]:
    """完整装配：读 yaml → 建 Registry（校验）→ 解析 handler。

    返回 `(registry, handlers)`。任何一步失败都抛 `RegistryError`，
    调用方应当让它终止进程 —— 一个配错的 registry 不该被启动。
    """
    specs, scopes = load_specs(path)
    registry = Registry.from_specs(specs, scopes)          # ← 跨工具校验在这里
    handlers = resolve_handlers(specs, root or path.parents[1])
    return registry, handlers