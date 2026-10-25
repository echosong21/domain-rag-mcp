"""披露层的验收测试 —— 对应架构文档 §六 的 #3 #4 #6 三条命门。

**这组测试守的三件事**

1. **未授予 scope 不可调用**（#3）
2. **跨 profile 隔离**（#4）
3. **search 只搜权限内**（#6）

**为什么单独测 `enable_tool` 的提权路径**

`enable_tool` 是 Agent 唯一能主动改自己权限的入口。如果它能授予
`discovery_scopes` 之外的 scope，那整套权限设计就形同虚设 ——
Agent 可以自己把 `common.ops.write` 授给自己，然后调 `reindex` 改数据。
这条比「功能对不对」重要得多。
"""
from __future__ import annotations

import pytest

from disclosure import (
    MemoryStore,
    PrincipalState,
    Profile,
    ProfileError,
    enable_tool,
    search_tools,
)
from registry.models import Registry, ToolSpec

# ── 测试夹具：两个 profile + 一份工具表 ──────────────────────
SCOPES = {
    "common.discovery", "common.knowledge", "common.safety",
    "common.ops.read", "common.ops.write", "dev.eval",
}


def _spec(name: str, scope: str, tags=("测试",)) -> ToolSpec:
    return ToolSpec(
        name=name, module="m", handler="h",
        brief=f"{name} 的摘要", scope=scope, tags=tags,
    )


@pytest.fixture
def registry() -> Registry:
    specs = [
        _spec("common.knowledge.search", "common.knowledge", ("检索", "search")),
        _spec("common.knowledge.deep_search", "common.knowledge", ("深度", "deep")),
        _spec("common.safety.check_input", "common.safety", ("安全", "safety")),
        _spec("common.ops.get_index_status", "common.ops.read", ("状态", "status")),
        _spec("common.ops.reindex", "common.ops.write", ("重建", "reindex")),
        _spec("dev.eval.get_report", "dev.eval", ("评测", "eval")),
    ]
    return Registry.from_specs(specs, SCOPES)


@pytest.fixture
def health_profile() -> Profile:
    return Profile(
        name="health-agent",
        initial_scopes=frozenset({"common.discovery", "common.knowledge"}),
        discovery_scopes=frozenset(
            {"common.discovery", "common.knowledge", "common.safety", "common.ops.read"}
        ),
        allowed_domains=frozenset({"health"}),
        allow_discovery=True,
        allow_ops_write=False,
    )


@pytest.fixture
def dev_profile() -> Profile:
    return Profile(
        name="dev-agent",
        initial_scopes=frozenset({"common.discovery"}),
        discovery_scopes=frozenset(SCOPES),      # 全部
        allow_discovery=True,
        allow_ops_write=True,
    )


# ── profile 自身校验 ────────────────────────────────────────
def test_profile_rejects_initial_outside_discovery():
    """initial 必须是 discovery 的子集，否则「能用的搜不到」。"""
    with pytest.raises(ProfileError, match="不在 discovery_scopes 内"):
        Profile(
            name="bad",
            initial_scopes=frozenset({"common.knowledge"}),
            discovery_scopes=frozenset({"common.discovery"}),
        )


def test_bootstrap_grants_declared_scopes(health_profile):
    state = PrincipalState.bootstrap(health_profile)
    assert state.scopes == frozenset({"common.discovery", "common.knowledge"})
    assert state.can_use("common.knowledge")
    assert not state.can_use("common.ops.write")


# ── 命门 #3：未授予不可调用 ─────────────────────────────────
def test_ungranted_scope_is_not_usable(health_profile):
    state = PrincipalState.bootstrap(health_profile)
    assert not state.can_use("common.ops.write")
    assert not state.can_use("dev.eval")


# ── 命门 #4：跨 profile 隔离 ────────────────────────────────
def test_two_profiles_have_different_scope_sets(health_profile, dev_profile):
    """两个 profile 的**可发现范围**必须不同 —— 这是隔离的根据。

    ⚠️ 断言的是 `discovery_scopes` 而不是已授予的 `scopes`：
    `dev.eval` 对 dev-agent 是「可发现但需 enable」，对 health-agent 则是
    「根本不可发现」。隔离生效在**可发现范围**这一层，不在授予状态。
    """
    h = PrincipalState.bootstrap(health_profile)
    d = PrincipalState.bootstrap(dev_profile)

    assert "dev.eval" not in h.profile.discovery_scopes
    assert "common.ops.write" not in h.profile.discovery_scopes
    assert "dev.eval" in d.profile.discovery_scopes
    assert "common.ops.write" in d.profile.discovery_scopes

    # 两者的已授予集合也不同（initial 声明不同）
    assert h.scopes != d.scopes


# ── 命门 #6：search 只搜权限内 ──────────────────────────────
def test_search_excludes_unauthorized_scopes(registry, health_profile):
    """health-agent 搜「评测」应该**搜不到** dev.eval 的工具。"""
    state = PrincipalState.bootstrap(health_profile)
    out = search_tools("评测", state, registry)
    assert out["tools"] == [], "未授权 scope 下的工具不应出现在搜索结果里"


def test_search_finds_within_authorized(registry, health_profile):
    state = PrincipalState.bootstrap(health_profile)
    out = search_tools("检索", state, registry)
    names = {t["name"] for t in out["tools"]}
    assert "common.knowledge.search" in names
    assert all(t["scope"] in state.profile.discovery_scopes for t in out["tools"])


def test_search_flags_already_granted(registry, health_profile):
    """已在 initial 里的工具，搜索结果要标 already_granted=True（省一次 enable）。"""
    state = PrincipalState.bootstrap(health_profile)
    out = search_tools("检索", state, registry)
    hit = next(t for t in out["tools"] if t["name"] == "common.knowledge.search")
    assert hit["already_granted"] is True


# ── enable_tool：正常路径 + 提权拒绝 ────────────────────────
def test_enable_tool_grants_within_scope(registry, health_profile):
    state = PrincipalState.bootstrap(health_profile)
    assert not state.can_use("common.safety")

    out = enable_tool("common.safety.check_input", state, registry)
    assert out["ok"] is True
    assert out["changed"] is True
    assert state.can_use("common.safety")


def test_enable_tool_is_idempotent(registry, health_profile):
    state = PrincipalState.bootstrap(health_profile)
    enable_tool("common.safety.check_input", state, registry)
    again = enable_tool("common.safety.check_input", state, registry)
    assert again["ok"] is True
    assert again["changed"] is False, "重复授予不应产生变化（否则会重复发通知）"


def test_enable_tool_cannot_escalate(registry, health_profile):
    """❗核心安全测试：不能通过 enable_tool 给自己授予 scope 外的权限。"""
    state = PrincipalState.bootstrap(health_profile)

    # common.ops.write 不在 health-agent 的 discovery_scopes 内
    out = enable_tool("common.ops.reindex", state, registry)
    assert out["ok"] is False
    assert "可发现范围" in out["error"]
    assert not state.can_use("common.ops.write"), "绝不能因一次 enable 就拿到写权限"


def test_enable_tool_rejects_dev_scope(registry, health_profile):
    """dev.* 也不该被健康 Agent 拿到。"""
    state = PrincipalState.bootstrap(health_profile)
    out = enable_tool("dev.eval.get_report", state, registry)
    assert out["ok"] is False
    assert not state.can_use("dev.eval")


def test_enable_tool_unknown_name_does_not_leak(registry, health_profile):
    """错误信息不能泄露「存在但无权访问」的工具清单。"""
    state = PrincipalState.bootstrap(health_profile)
    out = enable_tool("common.ops.reindex", state, registry)
    # 不能把 SCOPES 或其它工具名列出来
    assert "common.knowledge" not in out["error"]


# ── store 的行为 ────────────────────────────────────────────
def test_store_reuses_state_per_principal(health_profile, dev_profile):
    store = MemoryStore()
    a = store.get_or_create(health_profile)
    b = store.get_or_create(health_profile)
    assert a is b, "同一 principal 必须复用同一状态（否则授予会丢）"

    c = store.get_or_create(dev_profile)
    assert c is not a

    assert store.principals() == ["dev-agent", "health-agent"]


def test_store_reset(health_profile):
    store = MemoryStore()
    store.get_or_create(health_profile)
    store.reset()
    assert store.principals() == []
    assert store.get("health-agent") is None


def test_state_snapshot_is_auditable(health_profile):
    """快照要能回答「谁在什么时候被授予了什么」。"""
    state = PrincipalState.bootstrap(health_profile)
    snap = state.snapshot()
    assert snap["principal"] == "health-agent"
    assert set(snap["scopes"]) == {"common.discovery", "common.knowledge"}
    for scope, grant in snap["grants"].items():
        assert grant["granted_by"] == "profile:initial"
        assert grant["granted_at"], "必须有授予时间"