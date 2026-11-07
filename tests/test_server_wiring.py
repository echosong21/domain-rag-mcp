"""`server.py` 的接线测试：server 真的能建起来、审计真的会被写。

**这两条测试存在的理由，是它们各自守住了一个真实发生过的「静默失效」**

### ① `create_server()` 曾经根本起不来

`health_check(probe: Probe | None)` 里的 `Probe` 是 `Protocol`，Pydantic 无法为它
生成 JSON Schema，于是注册工具时抛 `PydanticSchemaGenerationError` —— **整个 server
起不来**。

**为什么长期没被发现**：`--list-tools` 只读 registry、不建 FastMCP，
完全绕过了这条路径。所以「12 个工具可见」这个结论一直是空转的。

**⇒ 铁律：`--list-tools` 通过 ≠ server 能启动。判断可用性必须真的建一次。**

### ② `record_call` 曾经零调用

`observability/` 里 audit / metrics / report 三件套都写完了，
但**没有任何一处真正调用 `record_call`** —— 审计日志永远是空的。

「有实现」和「有接线」是两件事，而且后者更容易漏：前者能被 import 检查发现，
后者只能靠真的调一次再回读产物。

**⇒ 铁律：可观测性要测「产物里有没有这条记录」，不是测「模块能不能 import」。**
"""
from __future__ import annotations

import json

import pytest

import observability.audit as audit_mod
import server as server_mod


# ── ① server 能建起来 ──────────────────────────────────────
def test_create_server_actually_builds():
    """`create_server()` 必须真的能建起来，而不只是 registry 能加载。

    `health_check` 的 `probe: Probe`（Protocol）曾让这里抛
    `PydanticSchemaGenerationError`，而 `--list-tools` 绕过了这条路径，
    所以问题被掩盖了很久。
    """
    mcp, state = server_mod.create_server(profile_name="dev-agent")

    assert mcp is not None
    assert state.scopes, "principal 应当持有 scope"


def test_create_server_works_for_health_agent_too():
    """两个 profile 都要能建 —— 只测一个会漏掉 profile 相关的错误。"""
    mcp, state = server_mod.create_server(profile_name="health-agent")
    assert mcp is not None
    # health-agent 没有 ops.write，这是它的权限边界
    assert "common.ops.write" not in state.scopes


# ── ② 审计真的会被写 ───────────────────────────────────────
def test_wrapped_tool_call_writes_audit_record(tmp_path, monkeypatch):
    """包装后的工具调用必须留下审计记录。

    测「产物里有没有这条记录」，而不是「record_call 能不能 import」——
    后者在实现早就存在、却零调用的情况下依然会通过。
    """
    log = tmp_path / "mcp_calls.jsonl"
    monkeypatch.setattr(audit_mod, "DEFAULT_LOG", log)

    spec = server_mod.ToolSpec.__new__(server_mod.ToolSpec) if False else None  # noqa: F841

    def sample_tool(query: str, top_k: int = 3) -> dict:
        return {"hits": [], "n_hits": 0}

    # 用真实的 ToolSpec 构造，避免依赖私有字段顺序
    from registry.models import ToolSpec

    spec = ToolSpec(
        name="test.sample", module="x", handler="sample_tool", brief="示例",
        scope="common.knowledge", tags=("t",), side_effect="read",
        requires=(), batch="core",
    )

    wrapped = server_mod._wrap_with_audit(sample_tool, spec, "dev-agent")
    wrapped("盐", top_k=2)

    assert log.exists(), "审计日志应当被创建"
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1

    rec = records[0]
    assert rec["tool"] == "test.sample"
    assert rec["principal"] == "dev-agent"
    assert rec["status"] == "ok"
    assert rec["duration_ms"] >= 0
    # 参数要还原成「名字 → 值」，而不是一坨 tuple
    assert rec["args_summary"]["top_k"] == 2
    assert rec["trace_id"]


def test_wrapped_tool_records_error_status_from_return_value(tmp_path, monkeypatch):
    """工具用返回值的 `error` 字段表达失败时，审计必须记成 error。

    如果只靠 `except` 判失败，本仓所有工具都会记成 `status="ok"` ——
    因为它们的约定就是不抛异常。「失败不可见」比没有审计更糟。
    """
    log = tmp_path / "mcp_calls.jsonl"
    monkeypatch.setattr(audit_mod, "DEFAULT_LOG", log)

    from registry.models import ToolSpec

    spec = ToolSpec(
        name="test.failing", module="x", handler="failing", brief="示例",
        scope="common.knowledge", tags=("t",), side_effect="read",
        requires=(), batch="core",
    )

    def failing_tool(query: str) -> dict:
        return {"error": "Milvus 连接失败"}

    server_mod._wrap_with_audit(failing_tool, spec, "dev-agent")("x")

    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["status"] == "error"
    assert "Milvus" in rec["error"]


def test_wrapped_tool_records_blocked_status(tmp_path, monkeypatch):
    """安全拦截要单独记一档 —— 它不是故障，也不是成功。"""
    log = tmp_path / "mcp_calls.jsonl"
    monkeypatch.setattr(audit_mod, "DEFAULT_LOG", log)

    from registry.models import ToolSpec

    spec = ToolSpec(
        name="test.blocked", module="x", handler="blocked", brief="示例",
        scope="common.knowledge", tags=("t",), side_effect="read",
        requires=(), batch="core",
    )

    def blocked_tool(query: str) -> dict:
        return {"blocked": True, "answer": "已拦截"}

    server_mod._wrap_with_audit(blocked_tool, spec, "dev-agent")("x")

    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["status"] == "blocked"


def test_wrapped_tool_reraises_and_still_audits(tmp_path, monkeypatch):
    """抛异常时要先记审计再上抛 —— 审计不能吞掉异常。"""
    log = tmp_path / "mcp_calls.jsonl"
    monkeypatch.setattr(audit_mod, "DEFAULT_LOG", log)

    from registry.models import ToolSpec

    spec = ToolSpec(
        name="test.boom", module="x", handler="boom", brief="示例",
        scope="common.knowledge", tags=("t",), side_effect="read",
        requires=(), batch="core",
    )

    def boom(query: str) -> dict:
        raise RuntimeError("炸了")

    with pytest.raises(RuntimeError, match="炸了"):
        server_mod._wrap_with_audit(boom, spec, "dev-agent")("x")

    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["status"] == "error"
    assert "RuntimeError" in rec["error"]


# ── ③ 包装不能丢参数 schema ────────────────────────────────
def test_wrapper_preserves_signature():
    """包装器必须用 `functools.wraps` 保住签名。

    FastMCP 从函数签名生成参数 schema，`*args, **kwargs` 会让**所有工具的
    schema 变空** —— 等于把工具废掉（Agent 看不到该传什么参数）。
    """
    import inspect

    from registry.models import ToolSpec

    spec = ToolSpec(
        name="test.sig", module="x", handler="sig", brief="示例",
        scope="common.knowledge", tags=("t",), side_effect="read",
        requires=(), batch="core",
    )

    def sample(query: str, top_k: int = 5) -> dict:
        return {}

    wrapped = server_mod._wrap_with_audit(sample, spec, "dev-agent")
    params = list(inspect.signature(wrapped).parameters)

    assert params == ["query", "top_k"], f"签名被破坏：{params}"