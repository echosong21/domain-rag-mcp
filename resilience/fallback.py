"""降级链执行 —— 规范里**没有** fallback 概念，这是自研能力。

**为什么值得自研**

MCP 规范只要求 server「限流 / 鉴权 / 校验 / 净化」，对「依赖挂了怎么办」
一个字没写。而真实部署里，rerank 服务、embedding API、向量库任何一环都
会挂，Agent 只拿到一个超时错误是没用的。

所以要自己定义：**哪些失败可降级、降到哪、降到什么程度算成功**。

**四条硬约束（架构文档 §三）**

1. **降级目标必须已授予 scope** —— 否则主工具一挂，Agent 就白得了未授权工具的权限
2. **降级不递归** —— A→B 可以，A→B→C 不行（成环在 registry 加载期已禁止，
   此处再限制深度）
3. **写工具不降级** —— 自动找替代去改数据是灾难（registry 加载期已禁止配置）
4. **返回必带 `degraded` / `fallback_from` / `reason`** —— 调用方必须知道
   答案来自降级路径，否则会误判质量

**只捕 `ToolUnavailable`**

不是所有异常都该降级。参数错误、权限错误降级只会掩盖 bug。
只有「依赖不可用」这一类才降级 —— 所以调用方要显式抛 `ToolUnavailable`。
"""
from __future__ import annotations

import time
from typing import Any, Callable

from registry.models import Registry, ToolSpec

from .circuit import CircuitBreaker, CircuitState


class ToolUnavailable(Exception):
    """**可降级**的失败：依赖超时/连接失败/服务返回不可用。

    与其它异常的区别是语义：这些失败换一个工具可能就好了，
    而参数错误换工具还是错。
    """


class NoAvailableTool(Exception):
    """主工具与全部 fallback 都不可用。"""


CallFn = Callable[[str, dict], Any]


def call_with_fallback(
    name: str,
    args: dict,
    *,
    registry: Registry,
    breakers: dict[str, CircuitBreaker],
    scopes: frozenset[str],
    call_fn: CallFn,
) -> dict:
    """按 registry 的 fallback 链执行一次工具调用。

    `call_fn(工具名, 参数)` 是真正执行工具的函数（注入以便测试）。
    它应当抛 `ToolUnavailable` 表示「可降级失败」，其它异常原样上抛。
    """
    spec = registry.get(name)
    if spec is None:
        raise KeyError(f"工具 {name!r} 不在 registry 中")

    # ① 权限校验：未授予 scope 直接拒绝，**不降级**
    if spec.scope not in scopes:
        raise PermissionError(f"scope {spec.scope!r} 未授予，拒绝调用 {name}")

    primary_error: Exception | None = None

    # ② 主工具（受熔断保护）
    breaker = _breaker(breakers, name)
    if breaker.allow():
        try:
            result = call_fn(name, args)
            breaker.record_success()
            return _ok(result, degraded=False, circuit=breaker.state.value)
        except ToolUnavailable as exc:
            breaker.record_failure()
            primary_error = exc
        except Exception:
            # 非「不可用」类异常：不降级，直接上抛（否则会掩盖 bug）
            breaker.record_failure()
            raise
    else:
        # 熔断 OPEN：零等待跳过，直接进降级
        primary_error = ToolUnavailable(
            f"{name} 处于熔断状态（{breaker.snapshot()['opened_for_s']}s），已跳过"
        )

    # ③ 降级链（不递归：只用 spec.fallback，不再向下展开）
    for alt_name in spec.fallback:
        alt = registry.get(alt_name)
        if alt is None:
            continue
        # 约束①：目标必须已授予 scope，否则跳过（不能靠降级绕过权限）
        if alt.scope not in scopes:
            continue
        # 写工具不降级（registry 加载期已禁止，此处再加一道）
        if alt.side_effect == "write":
            continue

        alt_breaker = _breaker(breakers, alt_name)
        if not alt_breaker.allow():
            continue

        try:
            result = call_fn(alt_name, args)
            alt_breaker.record_success()
            return _ok(
                result,
                degraded=True,
                circuit=alt_breaker.state.value,
                fallback_from=name,
                reason=str(primary_error) if primary_error else "primary unavailable",
            )
        except Exception:                       # noqa: BLE001 —— 降级目标失败就试下一个
            alt_breaker.record_failure()
            continue

    # ④ 全链失败
    raise NoAvailableTool(
        f"{name} 及其 fallback {list(spec.fallback)} 全部不可用"
        + (f"：{primary_error}" if primary_error else "")
    ) from primary_error


# ── 内部 ─────────────────────────────────────────────────────
def _breaker(breakers: dict[str, CircuitBreaker], name: str) -> CircuitBreaker:
    """取（或惰性创建）某工具的熔断器。

    ⚠️ 一个工具一个熔断器：rerank 挂了不该连累 Milvus 检索。
    """
    breaker = breakers.get(name)
    if breaker is None:
        breaker = CircuitBreaker(name)
        breakers[name] = breaker
    return breaker


def _ok(result: Any, *, degraded: bool, circuit: str, **extra) -> dict:
    """统一包装返回，保证调用方总能判断「这是不是降级结果」。"""
    payload = result if isinstance(result, dict) else {"result": result}
    payload = dict(payload)
    payload.update(
        {
            "degraded": degraded,
            "circuit_state": circuit,
            "served_at": round(time.time(), 3),
        }
    )
    payload.update(extra)
    return payload


def circuit_snapshot(breakers: dict[str, CircuitBreaker]) -> dict[str, str]:
    """给 `health_check` 用：非 closed 的工具即当前处于降级。"""
    return {name: b.state.value for name, b in sorted(breakers.items())}