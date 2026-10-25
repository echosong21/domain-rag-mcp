"""`stats` 命令 —— 把审计日志折成可消费的报告。

**两种输出，两种用途**

- `--format json`：给**监测面板**消费（schema 见 `metrics.SCHEMA_VERSION`）
- `--format html`：单文件自包含报告，双击就能看（不依赖面板）

两者数据同源，区别只是呈现。这样面板还没做好时也能先看数。

**为什么不在这里做实时刷新**

数据是快照。刷新 = 重新生成。实时轮询会引入长连接、增量计算、
前端状态同步 —— 单机 MVP 不需要这些复杂度，而它们的维护成本是持续的。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import DEFAULT_LOG, read_records
from .metrics import aggregate, window_labels

_PERIODS = {"1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400}


def _parse_period(text: str) -> float | None:
    """`24h` / `7d` / `all` → 秒。非法值按 None（= 全部）处理。"""
    text = (text or "").strip().lower()
    if text in ("", "all", "none"):
        return None
    if text in _PERIODS:
        return float(_PERIODS[text])
    return None


def build_stats(log_path: str | None = None, since: str = "24h") -> dict:
    seconds = _parse_period(since)
    since_ts = None
    if seconds is not None:
        import time
        since_ts = time.time() - seconds

    records = read_records(log_path or DEFAULT_LOG, since=since_ts)
    since_label, until_label = window_labels(seconds)
    return aggregate(records, since_label, until_label)


def render_html(stats: dict) -> str:
    """极简自包含 HTML —— 够看，不追求样式。

    真正的可视化面板消费 `--format json`，不解析这个 HTML。
    """
    rows = "".join(
        f"<tr><td>{t['tool']}</td><td>{t['calls']}</td><td>{t['errors']}</td>"
        f"<td>{t['p50_ms']}</td><td>{t['p95_ms']}</td><td>{t['p99_ms']}</td>"
        f"<td>{t['degrade_rate']:.2%}</td><td>{t['circuit_open_count']}</td></tr>"
        for t in stats["by_tool"]
    )
    tot = stats["totals"]
    payload = json.dumps(stats, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>MCP 调用统计</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#222}}
table{{border-collapse:collapse;margin-top:1rem}}
th,td{{border:1px solid #ddd;padding:.4rem .8rem;text-align:right}}
th{{background:#f5f5f5}} td:first-child,th:first-child{{text-align:left}}
.bad{{color:#b00}} .warn{{color:#b60}}
</style></head><body>
<h1>MCP 调用统计</h1>
<p>{stats['window']['since']} ~ {stats['window']['until']}</p>
<ul>
<li>调用量 <b>{tot['calls']}</b></li>
<li>错误率 <b class="{'bad' if tot['error_rate'] > 0.05 else ''}">{tot['error_rate']:.2%}</b></li>
<li>降级率 <b class="{'warn' if tot['degrade_rate'] > 0.10 else ''}">{tot['degrade_rate']:.2%}</b></li>
<li>权限拒绝 <b>{tot['denied']}</b></li>
</ul>
<table><thead><tr><th>工具</th><th>调用</th><th>错误</th>
<th>P50 (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th><th>降级率</th><th>熔断次数</th>
</tr></thead><tbody>{rows}</tbody></table>
<script type="application/json" id="stats-data">{payload}</script>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stats", description="MCP 调用统计")
    parser.add_argument("--since", default="24h", help="1h / 6h / 24h / 7d / all")
    parser.add_argument("--format", default="json", choices=["json", "html"])
    parser.add_argument("--log", default=None, help="审计日志路径（默认 logs/mcp_calls.jsonl）")
    parser.add_argument("--out", default=None, help="写文件而不打印")
    args = parser.parse_args(argv)

    stats = build_stats(args.log, args.since)
    text = (
        json.dumps(stats, ensure_ascii=False, indent=2)
        if args.format == "json"
        else render_html(stats)
    )

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())