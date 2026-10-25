"""指标聚合 —— 把审计 JSONL 折成 `stats.json`。

**schema 是与监测面板的契约**

面板只认这一个文件结构。字段增删会直接让面板读空，所以：
- `SCHEMA_VERSION` 变化时面板必须同步
- `stats` 输出的键集合有单测断言（见 `test_observability.py`）

**为什么保留 `degrade_rate` 与 `circuit_open_count`**

它们不是「顺带统计」的量：
- `degrade_rate` 高 ⇒ 某条外部依赖不健康（也许该调整数据源）
- `circuit_open_count` > 0 ⇒ 熔断真的触发过（说明降级设计被实战检验过）

**为什么有 `denied`**

它是权限设计是否合理的信号。健康 Agent 被拒次数异常高，
说明 profile 的 `discovery_scopes` 配窄了 —— 给的工具不够用，
Agent 只能反复试探。
"""
from __future__ import annotations

import time
from collections import defaultdict

SCHEMA_VERSION = 1


def _percentile(values: list[float], fraction: float) -> float:
    """线性取分位。空列表返回 0.0（面板要能处理"没有数据"）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return round(ordered[idx], 1)


def _bucket(ts: str) -> str:
    """按小时分桶：`2026-10-25T21:07:33` → `2026-10-25T21:00`。"""
    return ts[:13] + ":00" if len(ts) >= 13 else ts


def aggregate(records: list[dict], since_label: str = "", until_label: str = "") -> dict:
    """把审计记录折成 stats.json 结构。"""
    tool_durations: dict[str, list[float]] = defaultdict(list)
    tool_calls: dict[str, int] = defaultdict(int)
    tool_errors: dict[str, int] = defaultdict(int)
    tool_degraded: dict[str, int] = defaultdict(int)
    tool_open: dict[str, int] = defaultdict(int)

    principal_calls: dict[str, int] = defaultdict(int)
    principal_denied: dict[str, int] = defaultdict(int)

    timeline: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "errors": 0})

    total_errors = total_degraded = total_denied = 0

    for rec in records:
        tool = rec.get("tool") or "?"
        principal = rec.get("principal") or "?"
        status = rec.get("status") or "ok"
        is_error = status != "ok"
        is_denied = status == "denied"
        is_degraded = bool(rec.get("degraded"))

        tool_calls[tool] += 1
        tool_durations[tool].append(float(rec.get("duration_ms") or 0.0))
        if is_error:
            tool_errors[tool] += 1
        if is_degraded:
            tool_degraded[tool] += 1
        if rec.get("circuit_state") == "open":
            tool_open[tool] += 1

        principal_calls[principal] += 1
        if is_denied:
            principal_denied[principal] += 1

        bucket = _bucket(rec.get("ts") or "")
        timeline[bucket]["calls"] += 1
        if is_error:
            timeline[bucket]["errors"] += 1

        total_errors += int(is_error)
        total_degraded += int(is_degraded)
        total_denied += int(is_denied)

    calls = len(records)

    by_tool = []
    for tool, n in sorted(tool_calls.items(), key=lambda kv: (-kv[1], kv[0])):
        durations = tool_durations[tool]
        by_tool.append({
            "tool": tool,
            "calls": n,
            "errors": tool_errors[tool],
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
            "p99_ms": _percentile(durations, 0.99),
            "degrade_rate": round(tool_degraded[tool] / n, 4) if n else 0.0,
            "circuit_open_count": tool_open[tool],
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "window": {"since": since_label, "until": until_label},
        "totals": {
            "calls": calls,
            "errors": total_errors,
            "error_rate": round(total_errors / calls, 4) if calls else 0.0,
            "degrade_rate": round(total_degraded / calls, 4) if calls else 0.0,
            "denied": total_denied,
        },
        "by_tool": by_tool,
        "by_principal": [
            {"principal": p, "calls": n, "denied": principal_denied[p]}
            for p, n in sorted(principal_calls.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "timeline": [
            {"bucket": b, **timeline[b]}
            for b in sorted(timeline)
        ],
    }


def window_labels(since_seconds: float | None) -> tuple[str, str]:
    """生成 window 字段的可读标签。"""
    now = time.time()
    until = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
    since = (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - since_seconds))
        if since_seconds else ""
    )
    return since, until