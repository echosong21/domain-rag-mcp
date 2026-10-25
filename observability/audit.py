"""调用审计 —— 每次 `tools/call` 落一条 JSONL。

**为什么参数只记摘要、不记原文**

MCP 工具的参数里可能有用户隐私：健康场景下 `query="我最近血压高怎么办"`、
`text="我的检查报告说..."`。审计日志如果照抄原文，它就变成了一份**未经同意
收集的用户数据副本** —— 而这正是数据最小化原则要避免的。

所以 `summarize_args` 的策略是：

| 参数类型 | 记录什么 |
|---|---|
| 短字符串（≤24 字符、无空格） | **原值**（当作枚举，如 `mode="single"`） |
| 长字符串 | **只有长度**（内容不落盘） |
| 数字/布尔/None | 原值（本身不含隐私） |
| 列表/字典 | 只有元素个数 |

**为什么不做采样**

采样会让「出问题时恰好没记」变成常态。审计日志必须是完整序列 ——
它的价值在于事后能重建现场。

**为什么用 JSONL 而不是数据库**

单机 MVP 下 JSONL 有三个好处：追加写无锁、损坏只影响一行、
`grep`/`jq` 直接可用。等真的需要并发查询再换（这是可判定的切换条件）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# 短到这个长度且无空格 ⇒ 当作枚举值记录原文
ENUM_MAX_LEN = 24

DEFAULT_LOG = Path("logs") / "mcp_calls.jsonl"


def summarize_args(args: dict | None) -> dict:
    """把参数压成**不含原文**的摘要。"""
    if not args:
        return {}
    out: dict[str, object] = {}
    for key, value in args.items():
        if isinstance(value, str):
            # ⚠️ 判定枚举必须同时满足「纯 ASCII + 短 + 无空格」。
            # 曾经只用「短 + 无空格」，结果中文句子天然没有空格，
            # 于是「我最近血压有点高」被当成枚举**原样写进了审计日志** ——
            # 这是真实的隐私泄露，由 test_args_summary_redacts_free_text 抓到。
            if value.isascii() and len(value) <= ENUM_MAX_LEN and " " not in value:
                out[key] = value                       # 枚举：如 mode="single"
            else:
                out[key] = {"len": len(value)}         # 自由文本（含中文）：只记长度
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple, set)):
            out[key] = {"count": len(value)}
        elif isinstance(value, dict):
            out[key] = {"keys": len(value)}
        else:
            out[key] = {"type": type(value).__name__}
    return out


def record_call(
    *,
    trace_id: str,
    principal: str,
    tool: str,
    args: dict | None,
    duration_ms: float,
    status: str = "ok",
    degraded: bool = False,
    fallback_from: str | None = None,
    circuit_state: str = "closed",
    error: str | None = None,
    log_path: Path | str | None = None,
) -> dict:
    """追加一条审计记录，返回写下的那个 dict（便于测试断言）。"""
    record = {
        "trace_id": trace_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "principal": principal,
        "tool": tool,
        "args_summary": summarize_args(args),
        "status": status,
        "duration_ms": round(duration_ms, 1),
        "degraded": bool(degraded),
        "fallback_from": fallback_from,
        "circuit_state": circuit_state,
        "error": error,
    }

    path = Path(log_path) if log_path else DEFAULT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_records(log_path: Path | str | None = None, since: float | None = None) -> list[dict]:
    """读回审计记录。坏行跳过（一行损坏不该让整份日志不可读）。"""
    path = Path(log_path) if log_path else DEFAULT_LOG
    if not path.is_file():
        return []

    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is not None:
                try:
                    ts = time.mktime(time.strptime(rec["ts"], "%Y-%m-%dT%H:%M:%S"))
                except (KeyError, ValueError):
                    continue
                if ts < since:
                    continue
            out.append(rec)
    return out