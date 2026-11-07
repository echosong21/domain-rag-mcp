"""注册表与降级约束的单测。

**这组测试守的是什么**

不是「代码能跑」，而是**几条设计约束不允许被后续改动破坏**：

1. 降级只能同权限域内发生（禁止「安全闸降级到检索」）
2. 写工具不允许有 fallback（自动找替代去改数据是灾难）
3. 降级不成环
4. `search_tools` 只在**有权限的 scope** 内搜索
5. `tools/list` 顺序必须确定（规范 hard 要求，否则客户端缓存失效）

**为什么这些必须是测试而不是注释**

它们是「靠人记住才能维持」的约定。一旦某次加工具时忘了，故障不会立刻出现 ——
要等到线上真的降级那一刻。测试能把这个时间差抹掉。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from registry.loader import build_registry, load_specs, resolve_handlers
from registry.models import Registry, RegistryError, ToolSpec

# ── 构造合法 spec 的小工具 ────────────────────────────────────
SCOPES = {"common.discovery", "common.knowledge", "common.ops.write", "dev.eval"}


def spec(name: str, **kw) -> ToolSpec:
    """造一个最小可用的 ToolSpec，只覆盖本测试关心的字段。"""
    base = dict(
        name=name,
        module=kw.pop("module", "tools.common.discovery.list_domains"),
        handler=kw.pop("handler", "list_domains"),
        brief=kw.pop("brief", "测试用工具"),
        scope=kw.pop("scope", "common.discovery"),
        tags=kw.pop("tags", ("测试", "test")),
    )
    base.update(kw)
    return ToolSpec(**base)


# ── 1. 真实 registry.yaml 能装配 ──────────────────────────────
def test_real_registry_loads():
    """仓库里的 registry.yaml 必须能加载，且有 2 个核心工具。"""
    registry, handlers = build_registry()

    # 精确数量是**有意的摩擦**：加/删工具时必须来改这一行，
    # 等于强制一次「我确实要增删」的确认。改成「>= N」会失去这个作用。
    # 变更记录：7 → 8（2026-11-07 新增 common.knowledge.search）
    #           8 → 9（2026-11-07 新增 common.knowledge.search_multi）
    #           9 → 10（2026-11-07 新增 common.knowledge.answer）
    assert len(registry.tools) == 10
    assert "common.discovery.list_domains" in registry.tools
    assert "common.discovery.health_check" in registry.tools
    # handler 必须真能解析到可调用对象
    assert set(handlers) == set(registry.tools)
    assert all(callable(fn) for fn in handlers.values())


def test_business_scopes_have_tools():
    """业务类 scope 不能声明了却空着。

    `common.knowledge` 曾长期挂在 scopes 里却一个工具都没有 —— 一个 RAG 服务
    没有「检索」工具是明显缺口，而且这种「声明了但空着」的状态不会报错，
    只会让人误以为能力已存在。

    ⚠️ 只查 `common.*`：`dev.eval` / `dev.feedback` 是**有意留白**的
    （开发期能力，工具待建）。留白必须能被解释，所以这里不写「允许任意空」，
    而是明确限定范围 —— 否则「声明了但空着」会悄悄累积。
    """
    specs, scopes = load_specs()
    used = {s.scope for s in specs}
    empty_business = {s for s in scopes if s.startswith("common.")} - used
    assert not empty_business, (
        f"业务 scope 声明了但没有工具：{sorted(empty_business)}"
    )


def test_scopes_declared():
    """registry.yaml 声明的 scope 必须覆盖实际用到的。"""
    specs, scopes = load_specs()
    assert scopes, "必须声明 scope"
    for s in specs:
        assert s.scope in scopes, f"{s.name} 用了未声明的 scope {s.scope}"


# ── 2. 降级只能同权限域内（核心约束）──────────────────────────
def test_fallback_must_be_same_family():
    """跨域降级必须在**启动时**炸掉，而不是运行时才发现。"""
    specs = [
        spec("common.knowledge.search", scope="common.knowledge"),
        spec("dev.eval.run_query_eval", scope="dev.eval"),
        spec(
            "common.knowledge.deep_search",
            scope="common.knowledge",
            fallback=("dev.eval.run_query_eval",),   # ← 从 common 降到 dev，非法
        ),
    ]
    with pytest.raises(RegistryError, match="跨权限域"):
        Registry.from_specs(specs, SCOPES)


def test_fallback_within_family_is_allowed():
    """同域内降级应当通过 —— 这是 deep_search → search 的合法形态。"""
    specs = [
        spec("common.knowledge.search", scope="common.knowledge"),
        spec(
            "common.knowledge.deep_search",
            scope="common.knowledge",
            fallback=("common.knowledge.search",),
        ),
    ]
    reg = Registry.from_specs(specs, SCOPES)          # 不抛即通过
    assert reg.get("common.knowledge.deep_search").fallback == ("common.knowledge.search",)


# ── 3. 写工具不允许降级 ──────────────────────────────────────
def test_write_tool_cannot_have_fallback():
    specs = [
        spec("common.knowledge.search", scope="common.knowledge"),
        spec(
            "common.ops.reindex",
            scope="common.ops.write",
            side_effect="write",
            fallback=("common.knowledge.search",),
        ),
    ]
    with pytest.raises(RegistryError, match="写工具不允许"):
        Registry.from_specs(specs, SCOPES)


# ── 4. 其它启动期校验 ────────────────────────────────────────
def test_fallback_target_must_exist():
    specs = [spec("common.knowledge.search", scope="common.knowledge",
                  fallback=("common.knowledge.nope",))]
    with pytest.raises(RegistryError, match="不存在"):
        Registry.from_specs(specs, SCOPES)


def test_fallback_cannot_point_to_self():
    specs = [spec("common.knowledge.search", scope="common.knowledge",
                  fallback=("common.knowledge.search",))]
    with pytest.raises(RegistryError, match="不能指向自己"):
        Registry.from_specs(specs, SCOPES)


def test_fallback_cycle_rejected():
    """A→B→A 会让重试永不停止，必须拒绝。"""
    specs = [
        spec("common.knowledge.a", scope="common.knowledge", fallback=("common.knowledge.b",)),
        spec("common.knowledge.b", scope="common.knowledge", fallback=("common.knowledge.a",)),
    ]
    with pytest.raises(RegistryError, match="成环"):
        Registry.from_specs(specs, SCOPES)


def test_unknown_scope_rejected():
    specs = [spec("common.knowledge.search", scope="common.nonexistent")]
    with pytest.raises(RegistryError, match="未定义"):
        Registry.from_specs(specs, SCOPES)


def test_bad_tool_name_rejected():
    """工具名必须是 `<namespace>.<tool>` 至少两段。"""
    with pytest.raises(RegistryError, match="不合规"):
        spec("search")


def test_empty_tags_rejected():
    """tags 为空会让 tool search 永远搜不到它，属于配置错误。"""
    with pytest.raises(RegistryError, match="tags 不能为空"):
        ToolSpec(
            name="common.knowledge.search",
            module="m", handler="h", brief="b",
            scope="common.knowledge", tags=(),
        )


# ── 5. 披露相关：search 只在有权限的 scope 内搜 ───────────────
def _demo_registry() -> Registry:
    specs = [
        spec("common.knowledge.search", scope="common.knowledge",
             tags=("检索", "search")),
        spec("common.knowledge.deep_search", scope="common.knowledge",
             tags=("深度", "deep", "检索")),
        spec("dev.eval.get_report", scope="dev.eval", tags=("评测", "eval", "report")),
        spec("common.discovery.health_check", scope="common.discovery",
             tags=("健康", "health")),
    ]
    return Registry.from_specs(specs, SCOPES)


def test_search_respects_scopes():
    """health-agent 搜「评测」应该**搜不到** dev.eval 的工具。

    关键：不是「搜到了但调不动」，而是根本不出现在结果里 ——
    后者会把 Agent 引向无用的尝试。
    """
    reg = _demo_registry()
    agent_scopes = {"common.knowledge", "common.discovery"}

    assert reg.search("评测", agent_scopes) == []
    # 同一个查询，给足权限就能搜到
    assert [s.name for s in reg.search("评测", agent_scopes | {"dev.eval"})] == [
        "dev.eval.get_report"
    ]


def test_search_hits_name_and_tags():
    reg = _demo_registry()
    names = {s.name for s in reg.search("检索", {"common.knowledge"})}
    assert names == {"common.knowledge.search", "common.knowledge.deep_search"}

    # 精确的工具名分段命中应排在最前
    hits = reg.search("search", {"common.knowledge"})
    assert hits[0].name == "common.knowledge.search"


def test_by_scope_is_deterministic():
    """规范 2026-07-28 要求 tools/list 以确定性顺序返回。"""
    reg = _demo_registry()
    scopes = {"common.knowledge", "dev.eval", "common.discovery"}
    first = [s.name for s in reg.by_scope(scopes)]
    second = [s.name for s in reg.by_scope(scopes)]
    assert first == second == sorted(first)


def test_by_scope_filters():
    reg = _demo_registry()
    names = {s.name for s in reg.by_scope({"common.discovery"})}
    assert names == {"common.discovery.health_check"}


# ── 6. handler 解析失败要在启动时报 ──────────────────────────
def test_missing_module_rejected():
    """模块路径写错必须启动即失败，而不是等 Agent 调到它。"""
    bad = ToolSpec(
        name="common.knowledge.search",
        module="tools.common.discovery.no_such_module",
        handler="search",
        brief="b", scope="common.discovery", tags=("t",),
    )
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(RegistryError, match="无法导入模块"):
        resolve_handlers([bad], root)


def test_missing_attr_rejected():
    """模块在、但 handler 名写错，也要在启动时报。"""
    bad = ToolSpec(
        name="common.discovery.list_domains",
        module="tools.common.discovery.list_domains",
        handler="no_such_function",
        brief="b", scope="common.discovery", tags=("t",),
    )
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(RegistryError, match="没有"):
        resolve_handlers([bad], root)
