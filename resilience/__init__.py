"""容错层：熔断器与降级链。

⚠️ 这一层是**规范外**的自研能力 —— MCP 规范没有 fallback / circuit breaker
的任何规定，只要求「限流 / 鉴权 / 校验 / 净化」。所以这里的设计没有标准可抄，
每条约束都在这两个模块的 docstring 里写明了理由。
"""
from .circuit import CircuitBreaker, CircuitState
from .fallback import (
    NoAvailableTool,
    ToolUnavailable,
    call_with_fallback,
    circuit_snapshot,
)

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "NoAvailableTool",
    "ToolUnavailable",
    "call_with_fallback",
    "circuit_snapshot",
]