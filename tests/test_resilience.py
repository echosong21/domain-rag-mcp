"""容错层验收测试 —— 对应架构文档 §六 的 #7 #8 #9 #10。

**这四条为什么必须测**

降级和熔断是**故障路径**的代码。故障路径有个共同特点：正常时永远不执行，
所以写错了也不会被日常使用发现 —— 直到真出故障，才发现降级本身也是坏的。

`#10（降级不越权）`尤其危险：如果降级能绕过 scope 校验，那整套权限设计
在最需要它的时刻（故障时）恰好失效。
"""
from __future__ import annotations

import time

import pytest

from registry.models import Registry, ToolSpec
from resilience import (
    CircuitBreaker,
    CircuitState,
    NoAvailableTool,
    ToolUnavailable,
    call_with_fallback,
    circuit_snapshot,
)

SCOPES = {"common.knowledge", "common.safety", "common.ops.write"}


def _spec(name: str, scope: str, fallback=(), side_effect="read") -> ToolSpec:
    return ToolSpec(
        name=name, module="m", handler="h", brief="b",
        scope=scope, tags=("t",), fallback=tuple(fallback), side_effect=side_effect,
    )


@pytest.fixture
def registry() -> Registry:
    """deep_search --fallback--> search（你举的那个例子）。"""
    return Registry.from_specs(
        [
            _spec("common.knowledge.search", "common.knowledge"),
            _spec(
                "common.knowledge.deep_search", "common.knowledge",
                fallback=("common.knowledge.search",),
            ),
            _spec("common.safety.check_output", "common.safety"),
        ],
        SCOPES,
    )


# ── #7 降级生效 ─────────────────────────────────────────────
def test_fallback_serves_when_primary_fails(registry):
    calls = []

    def call_fn(name, args):
        calls.append(name)
        if name.endswith("deep_search"):
            raise ToolUnavailable("rerank 服务不可用")
        return {"hits": ["a", "b"]}

    out = call_with_fallback(
        "common.knowledge.deep_search", {"q": "x"},
        registry=registry, breakers={}, scopes=frozenset({"common.knowledge"}),
        call_fn=call_fn,
    )

    assert out["degraded"] is True
    assert out["fallback_from"] == "common.knowledge.deep_search"
    assert "rerank" in out["reason"]
    assert out["hits"] == ["a", "b"]
    # 主工具试过、备用也试过
    assert calls == ["common.knowledge.deep_search", "common.knowledge.search"]


def test_primary_success_is_not_marked_degraded(registry):
    out = call_with_fallback(
        "common.knowledge.deep_search", {},
        registry=registry, breakers={}, scopes=frozenset({"common.knowledge"}),
        call_fn=lambda n, a: {"hits": []},
    )
    assert out["degraded"] is False
    assert "fallback_from" not in out


# ── #8 熔断生效（零等待）──────────────────────────────────
def test_circuit_opens_after_threshold(registry):
    breakers: dict[str, CircuitBreaker] = {}

    def always_fail(name, args):
        raise ToolUnavailable("down")

    # 连续 3 次失败 ⇒ 熔断 OPEN
    for _ in range(3):
        with pytest.raises(NoAvailableTool):
            call_with_fallback(
                "common.knowledge.deep_search", {},
                registry=registry, breakers=breakers,
                scopes=frozenset({"common.knowledge"}), call_fn=always_fail,
            )
    assert breakers["common.knowledge.deep_search"].state is CircuitState.OPEN


def test_open_circuit_skips_primary_entirely(registry):
    """⚠️ 熔断的核心价值：OPEN 时**不产生主工具调用**（零等待）。"""
    breakers: dict[str, CircuitBreaker] = {}
    calls: list[str] = []

    def call_fn(name, args):
        calls.append(name)
        if name.endswith("deep_search"):
            raise ToolUnavailable("down")
        return {"hits": ["fallback-hit"]}

    # 先打到熔断
    for _ in range(3):
        call_with_fallback(
            "common.knowledge.deep_search", {}, registry=registry,
            breakers=breakers, scopes=frozenset({"common.knowledge"}), call_fn=call_fn,
        )
    calls.clear()

    # 第 4 次：熔断 OPEN ⇒ 主工具**一次都不该被调**
    out = call_with_fallback(
        "common.knowledge.deep_search", {}, registry=registry,
        breakers=breakers, scopes=frozenset({"common.knowledge"}), call_fn=call_fn,
    )
    assert out["degraded"] is True
    assert "deep_search" not in calls, f"熔断后不应再调主工具，实际 calls={calls}"
    assert calls == ["common.knowledge.search"]


# ── #9 熔断恢复 ────────────────────────────────────────────
def test_circuit_recovers_after_cooldown(registry):
    """冷却期结束 + 探针成功 ⇒ 回 CLOSED，主工具重新被调用。

    ⚠️ 用真实的短冷却期（0.05s）而不是 0：`recovery_timeout=0` 会让
    OPEN 在 `state` 属性一被读取时立刻转 HALF_OPEN，无法观察「曾进入 OPEN」。
    """
    b = CircuitBreaker("t", failure_threshold=2, recovery_timeout=0.05)

    b.record_failure(); b.record_failure()
    assert b.state is CircuitState.OPEN
    assert b.allow() is False, "冷却期内必须拒绝（这是熔断的价值）"

    time.sleep(0.06)                             # 冷却期结束
    assert b.allow() is True                     # 放一个探针
    assert b.allow() is False, "HALF_OPEN 只放一个探针"

    b.record_success()
    assert b.state is CircuitState.CLOSED
    assert b.allow() is True


def test_half_open_failure_reopens(registry):
    b = CircuitBreaker("t", failure_threshold=1, recovery_timeout=0.05)
    b.record_failure()
    assert b.state is CircuitState.OPEN

    time.sleep(0.06)
    assert b.allow() is True                     # 进 HALF_OPEN 放探针
    b.record_failure()                           # 探针失败 ⇒ 立刻再断
    assert b.state is CircuitState.OPEN
    assert b.allow() is False, "重断后应继续拒绝"


# ── #10 降级不越权（最关键）──────────────────────────────
def test_fallback_skipped_when_target_scope_not_granted(registry):
    """降级目标未授予 scope ⇒ 必须跳过，不能借故障期拿到权限。"""
    reg = Registry.from_specs(
        [
            _spec(
                "common.knowledge.deep_search", "common.knowledge",
                fallback=("common.safety.check_output",),   # 跨 scope 的 fallback
            ),
            _spec("common.safety.check_output", "common.safety"),
        ],
        SCOPES,
    )
    # 注意：Registry 允许「同 family」内的 fallback，所以这里能建起来

    with pytest.raises(NoAvailableTool):
        call_with_fallback(
            "common.knowledge.deep_search", {},
            registry=reg, breakers={},
            scopes=frozenset({"common.knowledge"}),     # 只有 knowledge，没有 safety
            call_fn=lambda n, a: (_ for _ in ()).throw(ToolUnavailable("down")),
        )


def test_permission_error_is_not_degraded(registry):
    """scope 未授予时**直接拒绝**，不该去试 fallback。"""
    with pytest.raises(PermissionError):
        call_with_fallback(
            "common.knowledge.deep_search", {},
            registry=registry, breakers={}, scopes=frozenset({"common.safety"}),
            call_fn=lambda n, a: {},
        )


def test_non_unavailable_exception_is_not_degraded(registry):
    """普通异常（如参数错误）不降级 —— 换工具也还是错，降级只会掩盖 bug。"""
    def boom(name, args):
        raise ValueError("参数错误")

    with pytest.raises(ValueError):
        call_with_fallback(
            "common.knowledge.deep_search", {},
            registry=registry, breakers={}, scopes=frozenset({"common.knowledge"}),
            call_fn=boom,
        )


# ── 观测 ───────────────────────────────────────────────────
def test_circuit_snapshot_reports_states(registry):
    breakers: dict[str, CircuitBreaker] = {}
    b = CircuitBreaker("common.knowledge.deep_search", failure_threshold=1)
    b.record_failure()
    breakers["common.knowledge.deep_search"] = b

    snap = circuit_snapshot(breakers)
    assert snap["common.knowledge.deep_search"] == "open"

    # health_check 会据此把该工具标进 degraded_tools
    assert "common.knowledge.deep_search" in snap


def test_breaker_snapshot_fields():
    b = CircuitBreaker("x", failure_threshold=2, recovery_timeout=30)
    b.record_failure()
    snap = b.snapshot()
    assert snap["state"] == "closed"
    assert snap["consecutive_failures"] == 1
    assert snap["threshold"] == 2
    assert snap["recovery_timeout_s"] == 30
