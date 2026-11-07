"""`common.knowledge.answer` 的验收测试。

**这组测试守的四件事**

1. **安全拦截必须真的拦** —— 命中禁用断言时 `blocked=True`，且**原始回答不出现在
   返回里**。如果原始文本被一并返回，调用方（或日志、或模型）就能看到它，拦截形同虚设。
2. **没材料不调模型** —— 检索为空时直接返回兜底话术。这是一次必然无用且昂贵的调用，
   省掉它是这个工具相对"调用方自己拼"的实际价值之一。
3. **`blocked` 与 `error` 分开** —— 前者是正常业务结果（换说法可解），后者是系统故障
   （重试才有意义）。混成一个字段，调用方就无从决策。
4. **免责声明按 position 注入** —— `prepend` 在前、`append` 在后，且顺序稳定。

**为什么这些不依赖真实 Milvus / LLM**

安全闸与编排是纯逻辑，它们的正确性不该由「能不能连上向量库」决定。
真实链路的验证在端到端实测里做，不在单测里。
"""
from __future__ import annotations

import pytest

from tools.common import knowledge


# ── 夹具 ────────────────────────────────────────────────────
@pytest.fixture
def domains_dir(tmp_path, monkeypatch):
    """一个带 prompt / disclaimers / safety 的领域配置。"""
    (tmp_path / "health.yaml").write_text(
        """domain:
  key: health
  language: zh
forms:
  - key: guides
collection: test_collection
prompt:
  system: |
    只依据材料回答；没有就说"{no_answer}"。不要做诊断。
  no_answer: 材料中未提及
disclaimers:
  - text: 仅供参考，不构成医疗建议。
    position: prepend
  - text: 如不适请就医。
    position: append
safety:
  forbidden_claims:
    - 可以治愈
    - 无需就医
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(knowledge.ENV_DOMAINS_DIR, str(tmp_path))
    return tmp_path


def _hit(text: str, src: str = "guides/a.md") -> dict:
    return {"score": 0.8, "source": src, "chunk_index": 0, "text": text}


@pytest.fixture
def stub_retrieval(monkeypatch):
    """把 search 打桩，可按用例改变返回的 hits。"""
    state: dict = {"hits": [_hit("每天盐不超过 5 克。")], "error": None}

    def fake_search(query, domain=None, top_k=5, min_score=0.0, milvus_url=None):
        if state["error"]:
            return {"error": state["error"]}
        return {
            "hits": state["hits"],
            "n_hits": len(state["hits"]),
            "domain": "health",
            "collection": "test_collection",
            "dropped": 0,
            "stage_ms": {"embed": 1.0, "search": 2.0},
        }

    monkeypatch.setattr(knowledge, "search", fake_search)
    return state


class _FakeResp:
    def __init__(self, content: str, in_tok: int = 10, out_tok: int = 5):
        self.content = content
        self.usage_metadata = {"input_tokens": in_tok, "output_tokens": out_tok}


@pytest.fixture
def stub_llm(monkeypatch):
    """把 LLM 打桩，并记录收到的 messages。"""
    state: dict = {"content": "每天盐摄入应低于 5 克。", "error": None, "messages": None}

    class _FakeLLM:
        def invoke(self, messages):
            state["messages"] = messages
            if state["error"]:
                raise RuntimeError(state["error"])
            return _FakeResp(state["content"])

    monkeypatch.setattr(knowledge, "_llm_client", lambda: (_FakeLLM(), None))
    return state


# ── 参数校验 ────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"question": ""}, "question 不能为空"),
        ({"question": "x" * 2001}, "question 过长"),
        ({"question": "ok", "top_k": 0}, "top_k 超出范围"),
        ({"question": "ok", "top_k": 21}, "top_k 超出范围"),
        ({"question": "ok", "top_k": "3"}, "top_k 必须是整数"),
    ],
    ids=["empty", "too-long", "topk-zero", "topk-over", "topk-str"],
)
def test_answer_param_validation(domains_dir, kwargs, fragment):
    out = knowledge.answer(domain="health", **kwargs)
    assert "error" in out
    assert fragment in out["error"]


def test_answer_rejects_unknown_domain(domains_dir):
    out = knowledge.answer(question="ok", domain="nope")
    assert "不存在" in out["error"]


def test_answer_propagates_retrieval_error(domains_dir, stub_retrieval):
    """检索失败要原样上抛成 error，不能当成「没材料」处理。

    这两个混淆的后果很严重：系统故障会被当成「知识库确实没有这条」，
    调用方于是去改问题而不是去排障。
    """
    stub_retrieval["error"] = "Milvus 连接失败"
    out = knowledge.answer(question="ok", domain="health")
    assert "error" in out
    assert "Milvus 连接失败" in out["error"]


# ── 正常路径 ────────────────────────────────────────────────
def test_answer_success_shape(domains_dir, stub_retrieval, stub_llm):
    out = knowledge.answer(question="盐吃多少合适", domain="health")

    assert "error" not in out
    assert "blocked" not in out
    assert "5 克" in out["answer"]
    assert len(out["contexts"]) == 1
    assert out["safety"]["output"]["allowed"] is True
    assert out["usage"]["input_tokens"] == 10
    assert out["domain"] == "health"
    assert "retrieve" in out["stage_ms"]


def test_disclaimers_injected_in_configured_position(domains_dir, stub_retrieval, stub_llm):
    """prepend 的在前、append 的在后。顺序错了等于改变声明的优先级。"""
    out = knowledge.answer(question="盐", domain="health")
    answer = out["answer"]

    assert answer.startswith("仅供参考，不构成医疗建议。")
    assert answer.endswith("如不适请就医。")
    assert "5 克" in answer


def test_prompt_includes_material_with_source(domains_dir, stub_retrieval, stub_llm):
    """材料必须带来源进 prompt —— 否则模型无法引用，也无法追责。"""
    knowledge.answer(question="盐", domain="health")
    user_msg = [m for m in stub_llm["messages"] if m["role"] == "user"][0]["content"]

    assert "guides/a.md#0" in user_msg
    assert "每天盐不超过 5 克" in user_msg
    assert "问题：盐" in user_msg


def test_no_answer_placeholder_substituted(domains_dir, stub_retrieval, stub_llm):
    """`{no_answer}` 要按领域配置替换，不能把占位符原样丢给模型。"""
    knowledge.answer(question="盐", domain="health")
    system_msg = [m for m in stub_llm["messages"] if m["role"] == "system"][0]["content"]

    assert "材料中未提及" in system_msg
    assert "{no_answer}" not in system_msg


# ── 没材料 → 不调模型 ───────────────────────────────────────
def test_empty_contexts_skips_generation(domains_dir, stub_retrieval, stub_llm):
    """检索为空时不该调用模型。

    这既省成本，也避免「没有依据却让模型自由发挥」这个最危险的组合。
    """
    stub_retrieval["hits"] = []
    out = knowledge.answer(question="完全不相关的问题", domain="health")

    assert stub_llm["messages"] is None, "不应该调用 LLM"
    assert out["skipped_generation"] == "no_context"
    assert "材料中未提及" in out["answer"]
    # 兜底话术同样要带免责声明
    assert "仅供参考" in out["answer"]
    assert out["usage"]["input_tokens"] == 0


# ── 安全闸 ──────────────────────────────────────────────────
def test_forbidden_claim_blocks_and_hides_raw_answer(domains_dir, stub_retrieval, stub_llm):
    """命中禁用断言时必须拦截，**且原始回答不能出现在返回里**。

    如果原始文本被返回，调用方（或日志、或下一个模型）就能看到它 —— 拦截形同虚设。
    """
    raw = "这个方法可以治愈你的问题。"
    stub_llm["content"] = raw

    out = knowledge.answer(question="怎么治", domain="health")

    assert out.get("blocked") is True
    assert "error" not in out, "拦截不是系统错误"
    assert out["safety"]["output"]["allowed"] is False
    assert "可以治愈" in out["safety"]["output"]["matched"]
    # 关键断言：越界内容不得泄露
    assert raw not in out["answer"]
    assert "治愈" not in out["answer"]
    # 但拦截话术仍要带免责声明
    assert "仅供参考" in out["answer"]


def test_allowed_answer_passes_through(domains_dir, stub_retrieval, stub_llm):
    """不含禁用断言的正常回答不该被误拦。"""
    stub_llm["content"] = "建议每天盐摄入低于 5 克。"
    out = knowledge.answer(question="盐", domain="health")

    assert "blocked" not in out
    assert out["safety"]["output"]["allowed"] is True
    assert out["safety"]["output"]["matched"] == []


def test_check_output_matches_substring():
    """判定是子串匹配 —— 这是刻意的确定性设计，不是模糊判断。"""
    cfg = {"safety": {"forbidden_claims": ["可以治愈", "无需就医"]}}

    assert knowledge._check_output("这个药可以治愈", cfg)["allowed"] is False
    assert knowledge._check_output("建议每天 5 克", cfg)["allowed"] is True
    # 记录一个已知的误伤：否定句也会命中。宁可误伤也不放过。
    assert knowledge._check_output("不能说可以治愈", cfg)["allowed"] is False


# ── 故障分支 ────────────────────────────────────────────────
def test_llm_failure_reports_error(domains_dir, stub_retrieval, stub_llm):
    stub_llm["error"] = "connection reset"
    out = knowledge.answer(question="盐", domain="health")

    assert "error" in out
    assert "LLM 调用失败" in out["error"]
    assert "connection reset" in out["error"]


def test_missing_llm_config_names_the_vars(domains_dir, stub_retrieval, monkeypatch):
    """LLM 配置缺失要一次说全三个变量，而不是只报第一个让人来回试。"""
    for var in ("API_KEY", "DASHSCOPE_API_KEY", "BASE_URL", "MODEL"):
        monkeypatch.delenv(var, raising=False)

    out = knowledge.answer(question="盐", domain="health")
    assert "LLM 配置缺失" in out["error"]
    for var in ("API_KEY", "BASE_URL", "MODEL"):
        assert var in out["error"]


# ── prompt 组装（纯函数）────────────────────────────────────
def test_build_messages_omits_empty_system():
    """没有 system prompt 时不该塞一条空消息进去。"""
    msgs = knowledge._build_messages("q", [_hit("t")], {"prompt": {}})
    assert [m["role"] for m in msgs] == ["user"]


def test_apply_disclaimers_keeps_order():
    cfg = {
        "disclaimers": [
            {"text": "A", "position": "prepend"},
            {"text": "B", "position": "prepend"},
            {"text": "C", "position": "append"},
        ]
    }
    assert knowledge._apply_disclaimers("X", cfg) == "A\n\nB\n\nX\n\nC"


def test_apply_disclaimers_ignores_malformed_entries():
    cfg = {"disclaimers": [{"text": "A", "position": "prepend"}, {"noposition": 1}, "str"]}
    assert knowledge._apply_disclaimers("X", cfg) == "A\n\nX"