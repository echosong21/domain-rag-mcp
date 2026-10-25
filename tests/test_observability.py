"""可观测性验收测试 —— 对应架构文档 §六 的 #11 #12。

**#11 的核心不是「字段齐全」，而是「不泄露原文」**

审计日志如果照抄参数原文，它就变成了一份未经同意收集的用户数据副本。
所以 `test_args_summary_redacts_free_text` 是本组最重要的测试。
"""
from __future__ import annotations

import json

import pytest

from observability import (
    SCHEMA_VERSION,
    aggregate,
    build_stats,
    read_records,
    record_call,
    render_html,
    summarize_args,
)


# ── #11 审计字段完整 + 不泄露原文 ──────────────────────────
def test_args_summary_redacts_free_text():
    """自由文本只记长度 —— 参数可能含用户隐私（健康数据）。"""
    summary = summarize_args({"query": "我最近血压有点高，晚饭该吃什么比较好"})
    assert summary["query"] == {"len": 18}
    # 原文绝不能出现在摘要里
    assert "血压" not in json.dumps(summary, ensure_ascii=False)


def test_args_summary_keeps_enums():
    """短枚举值保留原文 —— 抓问题时不用猜 `mode` 是什么。"""
    summary = summarize_args({"mode": "single", "top_k": 5, "flag": True, "x": None})
    assert summary["mode"] == "single"
    assert summary["top_k"] == 5
    assert summary["flag"] is True
    assert summary["x"] is None


def test_args_summary_handles_containers():
    summary = summarize_args({"items": [1, 2, 3], "opts": {"a": 1, "b": 2}})
    assert summary["items"] == {"count": 3}
    assert summary["opts"] == {"keys": 2}


def test_record_call_writes_all_fields(tmp_path):
    log = tmp_path / "calls.jsonl"
    rec = record_call(
        trace_id="t-1", principal="health-agent", tool="common.knowledge.search",
        args={"query": "米饭多少热量"}, duration_ms=234.56,
        status="ok", degraded=True, fallback_from="common.knowledge.deep_search",
        circuit_state="closed", log_path=log,
    )
    for field in ("trace_id", "ts", "principal", "tool", "args_summary",
                  "status", "duration_ms", "degraded", "fallback_from",
                  "circuit_state", "error"):
        assert field in rec, f"审计记录缺少字段 {field}"
    assert rec["duration_ms"] == 234.6          # 四舍五入到 0.1ms
    assert rec["args_summary"]["query"] == {"len": 6}

    # 落盘可解析
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace_id"] == "t-1"


def test_read_records_skips_broken_lines(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text(
        '{"tool":"a","ts":"2026-10-25T10:00:00"}\n'
        "这不是 JSON\n"
        '{"tool":"b","ts":"2026-10-25T10:01:00"}\n',
        encoding="utf-8",
    )
    records = read_records(log)
    assert [r["tool"] for r in records] == ["a", "b"], "坏行应被跳过而不是让整份日志不可读"


# ── #12 stats schema 与面板契约一致 ────────────────────────
def _records() -> list[dict]:
    return [
        {"tool": "search", "principal": "health", "status": "ok",
         "duration_ms": 100.0, "degraded": False, "circuit_state": "closed",
         "ts": "2026-10-25T10:05:00"},
        {"tool": "search", "principal": "health", "status": "ok",
         "duration_ms": 300.0, "degraded": True, "circuit_state": "closed",
         "ts": "2026-10-25T10:35:00"},
        {"tool": "reindex", "principal": "dev", "status": "denied",
         "duration_ms": 0.0, "degraded": False, "circuit_state": "open",
         "ts": "2026-10-25T11:05:00"},
    ]


def test_stats_schema_keys():
    """面板只认这些键 —— 少一个它就渲染不出来。"""
    stats = aggregate(_records())

    assert stats["schema_version"] == SCHEMA_VERSION
    assert set(stats) == {"schema_version", "window", "totals", "by_tool",
                          "by_principal", "timeline"}
    assert set(stats["totals"]) == {"calls", "errors", "error_rate",
                                    "degrade_rate", "denied"}
    for row in stats["by_tool"]:
        assert set(row) == {"tool", "calls", "errors", "p50_ms", "p95_ms",
                            "p99_ms", "degrade_rate", "circuit_open_count"}
    for row in stats["by_principal"]:
        assert set(row) == {"principal", "calls", "denied"}
    for row in stats["timeline"]:
        assert set(row) == {"bucket", "calls", "errors"}


def test_stats_aggregation_values():
    stats = aggregate(_records())
    assert stats["totals"]["calls"] == 3
    assert stats["totals"]["errors"] == 1          # denied 也算 error
    assert stats["totals"]["denied"] == 1
    assert stats["totals"]["degrade_rate"] == pytest.approx(1 / 3, abs=1e-3)

    by_tool = {r["tool"]: r for r in stats["by_tool"]}
    assert by_tool["search"]["calls"] == 2
    assert by_tool["search"]["degrade_rate"] == 0.5
    assert by_tool["reindex"]["circuit_open_count"] == 1

    # 分桶按小时
    buckets = {r["bucket"] for r in stats["timeline"]}
    assert buckets == {"2026-10-25T10:00", "2026-10-25T11:00"}


def test_stats_handles_empty_input():
    """空数据不能崩 —— 面板会拿到 calls=0 的合法结构。"""
    stats = aggregate([])
    assert stats["totals"]["calls"] == 0
    assert stats["totals"]["error_rate"] == 0.0
    assert stats["by_tool"] == []
    assert stats["timeline"] == []


def test_build_stats_reads_from_disk(tmp_path):
    log = tmp_path / "c.jsonl"
    record_call(trace_id="t", principal="p", tool="tool.a", args={},
                duration_ms=50.0, log_path=log)
    stats = build_stats(str(log), since="all")
    assert stats["totals"]["calls"] == 1
    assert stats["by_tool"][0]["tool"] == "tool.a"


def test_render_html_is_self_contained():
    """单文件报告：断网也能打开，所以数据必须内嵌。"""
    html = render_html(aggregate(_records()))
    assert "<!doctype html>" in html
    assert 'id="stats-data"' in html
    assert "search" in html
    assert "http://" not in html.split("</style>")[1], "不能引用外部资源"
