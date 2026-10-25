"""三态熔断器（CLOSED / OPEN / HALF_OPEN）。

**为什么必须有 OPEN 状态**

降级设计最常见的错误是「只捕异常」：主工具抛错就切备用。问题是**故障期间
每次调用都要先白等一次超时**。假设 rerank 服务挂了、超时 10s、一天 500 次调用
—— 光等待就烧掉 1.4 小时。

熔断器把「已知坏了」记成状态：连续失败到阈值后直接 `OPEN`，
此后**零等待**跳过主工具。冷却期后再放一个探针试探恢复。

**为什么是半开而不是直接关回**

直接关回会让故障服务在恢复瞬间被全部流量打回去（雪崩）。`HALF_OPEN` 只放
**一个**探针：成功才真恢复，失败立刻再断。

**状态机**

    CLOSED ──连续失败≥阈值──▶ OPEN
      ▲                        │
      │                    冷却期结束
      │                        ▼
      └────探针成功──── HALF_OPEN
                               │
                          探针失败
                               ▼
                             OPEN（重置冷却）

**线程安全**：`_lock` 保护全部状态迁移。MCP server 会在并发请求下调用它。
"""
from __future__ import annotations

import threading
import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"          # 正常放行
    OPEN = "open"              # 直接拒绝（零等待）
    HALF_OPEN = "half_open"    # 只放一个探针


class CircuitBreaker:
    """按工具实例化的熔断器。

    ⚠️ **一个工具一个熔断器**，不是全局一个 —— 否则 rerank 挂了会把
    Milvus 检索也一起断掉，那是过度熔断。
    """

    __slots__ = (
        "_name", "_state", "_failures", "_opened_at", "_half_open_inflight",
        "_failure_threshold", "_recovery_timeout", "_lock",
    )

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold 必须 ≥ 1")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout 不能为负")

        self._name = name
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_inflight = False
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._lock = threading.Lock()

    # ── 查询 ─────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        """当前状态（会根据时钟自动从 OPEN 迁到 HALF_OPEN）。"""
        with self._lock:
            self._maybe_half_open()
            return self._state

    # ── 放行判定 ─────────────────────────────────────────────
    def allow(self) -> bool:
        """是否放行本次调用。

        - `CLOSED`：放行
        - `OPEN` 且冷却未结束：**拒绝**（这是熔断的价值所在）
        - `OPEN` 且冷却已结束：转 `HALF_OPEN`，只放**第一个**探针
          （并发的第二个请求会被拒，避免恢复瞬间被流量打崩）
        """
        with self._lock:
            self._maybe_half_open()

            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN:
                return False
            # HALF_OPEN：只放一个探针
            if self._half_open_inflight:
                return False
            self._half_open_inflight = True
            return True

    # ── 结果上报 ─────────────────────────────────────────────
    def record_success(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                # 探针成功 ⇒ 真恢复
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._half_open_inflight = False
            elif self._state is CircuitState.CLOSED:
                self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                # 探针失败 ⇒ 立刻再断，重置冷却
                self._trip()
                return

            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._trip()

    # ── 内部 ─────────────────────────────────────────────────
    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._half_open_inflight = False

    def _maybe_half_open(self) -> None:
        if self._state is not CircuitState.OPEN:
            return
        if time.monotonic() - self._opened_at >= self._recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._half_open_inflight = False

    # ── 观测 ─────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            self._maybe_half_open()
            return {
                "tool": self._name,
                "state": self._state.value,
                "consecutive_failures": self._failures,
                "opened_for_s": (
                    round(time.monotonic() - self._opened_at, 1)
                    if self._state is not CircuitState.CLOSED else 0.0
                ),
                "threshold": self._failure_threshold,
                "recovery_timeout_s": self._recovery_timeout,
            }

    def reset(self) -> None:
        """测试与运维用：强制恢复。"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = 0.0
            self._half_open_inflight = False