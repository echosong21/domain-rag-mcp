"""可观测性层：审计 / 聚合 / 报告。

⚠️ MCP 2026-07-28 已弃用 `logging` capability（SEP-2577），
官方替代方向是 OpenTelemetry。本层用 JSONL + 聚合实现同等能力，
但不引入 OTel 依赖 —— 单机 MVP 下它带来的是运维复杂度而非价值。
切换条件：当需要跨服务链路追踪时（届时再接 OTel）。
"""
from .audit import read_records, record_call, summarize_args
from .metrics import SCHEMA_VERSION, aggregate, window_labels
from .report import build_stats, render_html

__all__ = [
    "SCHEMA_VERSION",
    "aggregate",
    "build_stats",
    "read_records",
    "record_call",
    "render_html",
    "summarize_args",
    "window_labels",
]