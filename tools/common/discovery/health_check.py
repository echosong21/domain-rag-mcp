"""`common.discovery.health_check` —— 依赖与熔断状态探活。

**为什么这个工具必须存在**

Agent 调 `search` 之前无法知道 Milvus 是否可用。没有它，故障表现只能是
「调用超时」，Agent 只能盲试再被动降级；有了它，Agent 可以先查一次，
然后主动选择走基础检索。

**两条设计约束**

1. **从不抛异常** —— 一个探活工具自己挂掉是荒谬的。任何内部异常都折成
   状态字符串 `"unavailable"`，保证调用方总能拿到结构化结果。
2. **探测可注入** —— 真实探测要连网，测试必须离线。所以 `probe` 参数接受
   任意实现；默认用 `EnvProbe`（只读环境变量，不发起网络请求）。

⚠️ 默认探测**刻意不连网**：`health_check` 会被 Agent 频繁调用，
如果每次都真的去连 Milvus，它就变成了一个新的故障源和延迟源。
真正需要连通性验证时，由调用方注入深度探测器。
"""
from __future__ import annotations

import os
from typing import Protocol

# 三种状态，语义明确，不用布尔值（布尔值无法区分「没配」和「配了但连不上」）
STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_UNCONFIGURED = "unconfigured"
STATUS_DISABLED = "disabled"

_VALID = {STATUS_OK, STATUS_UNAVAILABLE, STATUS_UNCONFIGURED, STATUS_DISABLED}


class Probe(Protocol):
    """探测器协议。任何实现三个同名方法的对象都可以传进来。"""

    def milvus(self) -> str: ...
    def embedding(self) -> str: ...
    def rerank(self) -> str: ...


class EnvProbe:
    """只读环境变量的浅探测：**不发起任何网络请求**。

    判定规则：
    - 相关变量齐备 ⇒ `ok`（配置层面可用，不代表服务此刻活着）
    - 变量缺失   ⇒ `unconfigured`
    - 重排默认关闭 ⇒ `disabled`（这是正常状态，不是故障）
    """

    def __init__(self, env: dict[str, str] | None = None):
        self._env = env if env is not None else os.environ

    def _has(self, *keys: str) -> bool:
        return all(self._env.get(k) for k in keys)

    def milvus(self) -> str:
        return STATUS_OK if self._has("MILVUS_URL") else STATUS_UNCONFIGURED

    def embedding(self) -> str:
        ok = self._has("EMBED_MODEL", "EMBED_BASE_URL", "API_KEY")
        return STATUS_OK if ok else STATUS_UNCONFIGURED

    def rerank(self) -> str:
        enabled = (self._env.get("RERANK_ENABLED") or "").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if not enabled:
            return STATUS_DISABLED
        return STATUS_OK if self._has("DASHSCOPE_API_KEY") else STATUS_UNCONFIGURED


def _normalize(value: object) -> str:
    """把探测器返回的任何东西折成合法状态字符串。"""
    text = str(value).strip().lower()
    return text if text in _VALID else STATUS_UNAVAILABLE


# 默认探针。测试可替换这个模块级变量以注入假探针。
#
# ⚠️ 为什么注入点从「函数参数」改成「模块变量」：
# `Probe` 是 `Protocol`，Pydantic 无法为它生成 JSON Schema ——
# 而有参数注解的工具在 `create_server()` 注册时就会让 Pydantic 尝试生成 schema，
# 于是**整个 server 起不来**（实测抛 `PydanticSchemaGenerationError`）。
#
# 更根本的理由：MCP 的 Agent 本来就不该传探针对象。
# 把测试用的注入点暴露在工具签名上，等于把一个内部实现细节变成协议的一部分。
_DEFAULT_PROBE: Probe | None = None


def health_check(
    circuit_states: dict[str, str] | None = None,
) -> dict:
    """返回 `{milvus, embedding, rerank, circuit_states, degraded_tools}`。

    `circuit_states` 由熔断器在运行时注入（Step 3 接通）；当前默认空字典。
    """
    p = _DEFAULT_PROBE or EnvProbe()

    out: dict[str, object] = {}
    for name in ("milvus", "embedding", "rerank"):
        try:
            out[name] = _normalize(getattr(p, name)())
        except Exception:                       # noqa: BLE001 —— 探活绝不抛
            out[name] = STATUS_UNAVAILABLE

    states = dict(circuit_states or {})
    out["circuit_states"] = states
    # 非 closed 的工具即「当前处于降级」，让 Agent 能主动避开
    out["degraded_tools"] = sorted(t for t, st in states.items() if st != "closed")
    return out