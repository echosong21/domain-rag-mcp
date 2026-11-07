"""`common.knowledge.answer_planned` 与 `common.knowledge.explain` 的验收测试。

**这组测试守的四件事**

1. **拆分规则的行为边界**（6 个正反例）—— 阈值取错会把「高血压/糖尿病」这类
   正常多焦点问题漏掉，也会把「牙齿和牙周」这种词内误切。两个方向都测。
2. **`empty_subtasks` 必须报出来** —— 多焦点问题的部分失败是最危险的情况：
   回答读起来通顺，调用方看不出只答了一半。
3. **全部子任务都没召回时不调模型** —— 与 `answer` 同一条原则。
4. **`explanation` 是规则生成的** —— 不依赖模型，同样的输入永远同样的输出。

**为什么不测真实拆分+检索的联合效果**

那需要 Milvus + 真实语料 + 真实 LLM，属于端到端实测。
单测守的是「逻辑分支对不对」，端到端守的是「效果好不好」，两者不能互相替代。
"""
from __future__ import annotations

import pytest

from tools.common import knowledge


# ── 夹具 ────────────────────────────────────────────────────
@pytest.fixture
def domains_dir(tmp_path, monkeypatch):
    (tmp_path / "health.yaml").write_text(
        """domain:
  key: health
  language: zh
forms:
  - key: guides
  - key: faq
collection: test_collection
prompt:
  system: |
    只依据材料回答；没有就说"{no_answer}"。
  no_answer: 材料中未提及
disclaimers:
  - text: 仅供参考。
    position: prepend
safety:
  forbidden_claims:
    - 可以治愈
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(knowledge.ENV_DOMAINS_DIR, str(tmp_path))
    return tmp_path


def _hit(text: str, src: str = "guides/a.md", idx: int = 0) -> dict:
    return {"score": 0.8, "source": src, "chunk_index": idx, "text": text}


@pytest.fixture
def stub_search(monkeypatch):
    """按子问题返回不同数量的材料，模拟「一半有材料、一半没有」。"""
    state: dict = {"by_query": {}, "default": [_hit("材料A")], "error": None}

    def fake(query, domain=None, top_k=5, min_score=0.0, milvus_url=None):
        if state["error"]:
            return {"error": state["error"]}
        hits = state["by_query"].get(query, state["default"])
        return {
            "hits": hits, "n_hits": len(hits), "domain": "health",
            "collection": "test_collection", "dropped": 0,
            "stage_ms": {"embed": 1.0, "search": 2.0},
        }

    monkeypatch.setattr(knowledge, "search", fake)
    return state


@pytest.fixture
def stub_generate(monkeypatch):
    """把生成打桩，记录收到的材料。"""
    state: dict = {"content": "答：材料A 说…", "calls": 0, "hits": None}

    def fake_generate(question, hits, cfg):
        state["calls"] += 1
        state["hits"] = hits
        return {
            "answer": state["content"],
            "safety": {"input": {"allowed": True},
                       "output": {"allowed": True, "matched": []}},
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "generate_ms": 7.0,
        }

    monkeypatch.setattr(knowledge, "_generate", fake_generate)
    return state


# ════════════════════════════════════════════════════════════
# 拆分规则
# ════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    ("question", "expect_n"),
    [
        # 应当拆（真多焦点）
        ("合同违约和侵权责任的赔偿标准分别是什么", 2),
        ("高血压以及糖尿病的饮食注意事项", 2),
        ("菠菜的热量、糙米的热量分别是多少", 2),
        # 不应拆
        ("成年人每天盐摄入量上限是多少", 1),   # 无连接词
        ("牙齿和牙周", 1),                     # 「和」在词内部，两侧各 2 字
        ("几点了", 1),                         # 太短
        ("", 0),                               # 空串
    ],
    ids=[
        "dual-and", "dual-yiji", "dual-comma",
        "single", "inside-word", "too-short", "empty",
    ],
)
def test_split_question_boundaries(question, expect_n):
    assert len(knowledge._split_question(question)) == expect_n


def test_split_never_returns_more_than_max():
    """超长连接词链要被截断到 max_subtasks，而不是返回一长串碎片。"""
    q = "甲和乙和丙和丁和戊"
    parts = knowledge._split_question(q, max_subtasks=2)
    assert len(parts) <= 2


def test_split_falls_back_to_original_when_cannot_split():
    """拆不出来时返回原问题本身，不能返回空列表 ——
    否则调用方会以为「没有子任务」，进而什么都不检索。"""
    q = "成年人每天盐摄入量上限是多少"
    assert knowledge._split_question(q) == [q]


# ════════════════════════════════════════════════════════════
# answer_planned
# ════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"question": ""}, "question 不能为空"),
        ({"question": "x" * 2001}, "question 过长"),
        ({"question": "ok", "per_task": 0}, "per_task 超出范围"),
        ({"question": "ok", "per_task": 21}, "per_task 超出范围"),
        ({"question": "ok", "max_subtasks": 0}, "max_subtasks 超出范围"),
        ({"question": "ok", "max_subtasks": 4}, "max_subtasks 超出范围"),
    ],
    ids=["empty", "too-long", "per-task-zero", "per-task-over",
         "max-sub-zero", "max-sub-over"],
)
def test_answer_planned_param_validation(domains_dir, kwargs, fragment):
    out = knowledge.answer_planned(domain="health", **kwargs)
    assert "error" in out
    assert fragment in out["error"]


def test_answer_planned_splits_and_reports_per_task(domains_dir, stub_search, stub_generate):
    """多焦点问题要拆开、各自检索，并把每路的条数报出来。"""
    stub_search["by_query"] = {
        "合同违约": [_hit("违约条款"), _hit("违约赔偿", "guides/b.md", 1)],
        "侵权责任的赔偿标准": [_hit("侵权赔偿", "faq/c.md")],
    }
    out = knowledge.answer_planned(
        "合同违约和侵权责任的赔偿标准", domain="health", per_task=3
    )

    assert "error" not in out
    assert out["subtasks"] == ["合同违约", "侵权责任的赔偿标准"]
    assert out["per_task_n"] == {"合同违约": 2, "侵权责任的赔偿标准": 1}
    assert out["empty_subtasks"] == []
    assert out["collection"] == "test_collection"


def test_answer_planned_reports_empty_subtasks(domains_dir, stub_search, stub_generate):
    """一半子任务没召回到时，必须显式报出来。

    这是本工具比直接调 `answer` 多出来的核心价值：
    没有它，「答了一半」会被完全掩盖 —— 回答读起来是通顺的。
    """
    stub_search["by_query"] = {
        "合同违约": [],                       # 这一半没有
        "侵权责任的赔偿标准": [_hit("侵权赔偿")],
    }
    out = knowledge.answer_planned(
        "合同违约和侵权责任的赔偿标准", domain="health"
    )

    assert out["empty_subtasks"] == ["合同违约"]
    assert out["per_task_n"]["合同违约"] == 0
    # 仍应生成 —— 有一半材料可用
    assert stub_generate["calls"] == 1


def test_answer_planned_skips_generation_when_nothing_retrieved(domains_dir, stub_search, stub_generate):
    """所有子任务都没材料时不调模型（与 answer 同一条原则）。"""
    stub_search["default"] = []
    out = knowledge.answer_planned(
        "合同违约和侵权责任的赔偿标准", domain="health"
    )

    assert stub_generate["calls"] == 0
    assert out["skipped_generation"] == "no_context"
    assert "材料中未提及" in out["answer"]
    # 兜底话术也要带免责声明
    assert "仅供参考" in out["answer"]


def test_answer_planned_fuses_across_subtasks(domains_dir, stub_search, stub_generate):
    """两条子任务都命中同一块时，融合后去重且该块排第一（多路共识）。"""
    shared = _hit("共享材料", "guides/shared.md")
    stub_search["by_query"] = {
        "合同违约": [shared],
        "侵权责任的赔偿标准": [shared],
    }
    out = knowledge.answer_planned(
        "合同违约和侵权责任的赔偿标准", domain="health"
    )

    assert len(out["contexts"]) == 1, "同一块被两路命中应去重"
    assert out["contexts"][0]["source"] == "guides/shared.md"
    # 融合后应记录它来自两个子任务
    assert len(out["contexts"][0]["from_subtask"]) == 2


def test_answer_planned_propagates_error(domains_dir, stub_search, stub_generate):
    stub_search["error"] = "Milvus 连接失败"
    out = knowledge.answer_planned(
        "合同违约和侵权责任的赔偿标准", domain="health"
    )
    assert "error" in out
    assert "Milvus 连接失败" in out["error"]


def test_answer_planned_single_focus_still_works(domains_dir, stub_search, stub_generate):
    """单焦点问题不拆，直接走一遍（长度 1 是合法结果，不是错误）。"""
    out = knowledge.answer_planned("成年人每天盐摄入量是多少", domain="health")
    assert out["subtasks"] == ["成年人每天盐摄入量是多少"]
    assert "error" not in out


# ════════════════════════════════════════════════════════════
# explain
# ════════════════════════════════════════════════════════════
@pytest.fixture
def stub_multi_retrieve(monkeypatch):
    """打桩多路召回：guides 有 2 条、faq 有 1 条、其余空。"""
    monkeypatch.setattr(knowledge, "_embed_query", lambda q: ([0.1], 1.0, None))
    monkeypatch.setenv(knowledge.ENV_MILVUS_URL, "http://stub")
    monkeypatch.setattr(knowledge, "_milvus_client", lambda: (object(), None))

    def fake(client, qvec, collection, per_form, forms, max_workers=4):
        docs = {
            "guides": [_hit("g0", "guides/x.md", 0), _hit("g1", "guides/y.md", 1)],
            "faq": [_hit("f0", "faq/z.md", 0)],
        }
        for f in forms:
            docs.setdefault(f, [])
        return docs, {f: 10.0 for f in forms}, 12.0

    monkeypatch.setattr(knowledge, "_retrieve_multi", fake)
    return fake


def test_explain_reports_route_details(domains_dir, stub_multi_retrieve):
    out = knowledge.explain("盐摄入", domain="health")

    assert "error" not in out
    assert len(out["per_task"]) == 1
    pt = out["per_task"][0]
    assert pt["per_form_n"]["guides"] == 2
    assert pt["per_form_n"]["faq"] == 1
    assert pt["top1"]["guides"] == 0.8
    assert out["fused"], "融合结果不应为空"


def test_explain_does_not_generate(domains_dir, stub_multi_retrieve, monkeypatch):
    """explain 不生成回答 —— 它不该碰 LLM。"""
    called = {"llm": False}
    monkeypatch.setattr(knowledge, "_llm_client",
                        lambda: (called.update(llm=True), (None, "should not be called"))[1])

    out = knowledge.explain("盐摄入", domain="health")
    assert "error" not in out
    assert called["llm"] is False, "explain 不应初始化 LLM 客户端"


def test_explain_allowed_without_llm_config(domains_dir, stub_multi_retrieve, monkeypatch):
    """没配 LLM 也该能用 —— 它是排查工具，不能依赖生成侧的配置。"""
    for var in ("API_KEY", "BASE_URL", "MODEL"):
        monkeypatch.delenv(var, raising=False)
    out = knowledge.explain("盐摄入", domain="health")
    assert "error" not in out


def test_explain_explanation_is_deterministic(domains_dir, stub_multi_retrieve):
    """同一输入两次调用，解释文本必须完全一致（规则生成，不经模型）。"""
    a = knowledge.explain("盐摄入", domain="health")["explanation"]
    b = knowledge.explain("盐摄入", domain="health")["explanation"]
    assert a == b
    assert a.endswith("。")


def test_explain_empty_result_explains_why():
    """全空时要给出可操作的归因，而不是空字符串。"""
    text = knowledge._explain_once(
        ["q"], [{"subtask": "q", "per_form_n": {"guides": 0}, "per_form_ms": {},
                 "wall_ms": 1.0, "top1": {"guides": None}, "embed_ms": 1.0}], []
    )
    assert "未召回" in text
    assert "过滤器" in text or "数据" in text


def test_explain_reports_ties_explicitly():
    """多条并列同分时必须说出来，不能只描述「首条」。

    实测踩过：4 路各出 1 条、RRF 分数完全相同，初版解释读起来像
    「只有 guides 一路有结果」—— 会把人引向「去查其它形态为什么没召回」，
    而实际是各形态都召回到了、只是互不重叠。
    """
    fused = [
        {"source": "guides/a.md", "chunk_index": 0, "score": 0.016393,
         "ranks": {"guides": 1}},
        {"source": "faq/b.md", "chunk_index": 0, "score": 0.016393,
         "ranks": {"faq": 1}},
        {"source": "tables/c.md", "chunk_index": 0, "score": 0.016393,
         "ranks": {"tables": 1}},
    ]
    per_task = [{"subtask": "q", "per_form_n": {"guides": 2, "faq": 2, "tables": 2},
                 "per_form_ms": {}, "wall_ms": 1.0, "top1": {}, "embed_ms": 1.0}]

    text = knowledge._explain_once(["q"], per_task, fused)

    assert "并列" in text
    assert "3 条" in text
    # 关键：不得暗示「其它形态没有东西」
    assert "只有" not in text or "互不重叠" in text


def test_explain_reports_consensus_when_one_doc_hits_multiple_routes():
    """单条在多路都出现（真共识）时，解释要说「多路共识」。"""
    fused = [
        {"source": "guides/x.md", "chunk_index": 0, "score": 0.032,
         "ranks": {"guides": 1, "faq": 1}},
        {"source": "faq/y.md", "chunk_index": 0, "score": 0.016, "ranks": {"faq": 2}},
    ]
    per_task = [{"subtask": "q", "per_form_n": {"guides": 1, "faq": 2},
                 "per_form_ms": {}, "wall_ms": 1.0, "top1": {}, "embed_ms": 1.0}]

    text = knowledge._explain_once(["q"], per_task, fused)
    assert "多路共识" in text
    assert "并列" not in text


def test_explain_disclaimer_not_applied(domains_dir, stub_multi_retrieve):
    """explain 返回的是检索结构，不是回答 —— 不该掺免责声明。"""
    out = knowledge.explain("盐摄入", domain="health")
    dump = str(out)
    assert "仅供参考" not in dump


def test_explain_rejects_undeclared_form(domains_dir, stub_multi_retrieve):
    out = knowledge.explain("盐摄入", domain="health", forms=["nope"])
    assert "未声明的形态" in out["error"]