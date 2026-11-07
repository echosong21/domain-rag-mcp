"""`common.knowledge.search` 的验收测试。

**这组测试守的三件事**

1. **配置不全时给可诊断的错误** —— 不是抛异常，也不是返回空结果。
   检索类工具最危险的失败模式是「查错了库但结果看起来正常」，
   所以每种缺失都要明确指出缺哪个变量。
2. **参数在动手前校验** —— 不接受跑了一半才发现 top_k 越界。
3. **阈值过滤要能区分「没召回」和「被过滤掉」** —— `dropped` 字段。

**为什么不依赖真实 Milvus / embedding**

这些用例跑在 CI 上，那里没有向量库也没有 API key。同时，「参数校验」
和「配置缺失的分支」本来就不该需要真后端 —— 把它们和真实检索解耦，
才能保证这部分逻辑任何时候都能验证。

真实链路的验证在端到端实测里做（见 README 的实测记录），不在单测里。
"""
from __future__ import annotations

import pytest

from tools.common import knowledge


# ── 夹具：一个最小的领域配置目录 ────────────────────────────
@pytest.fixture
def domains_dir(tmp_path, monkeypatch):
    """造一个只含 health 域的配置目录，并指向它。

    ⚠️ 用三引号多行而不是逐行字符串拼接：以冒号结尾的配置行后紧跟转义换行时，
    泄露扫描的绝对路径启发式规则会把它误判成盘符路径（实误）。
    """
    (tmp_path / "health.yaml").write_text(
        """domain:
  key: health
  display_name: TestHealth
  language: zh
forms:
  - key: guides
  - key: faq
collection: test_collection
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(knowledge.ENV_DOMAINS_DIR, str(tmp_path))
    return tmp_path


# ── 参数校验 ────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"query": ""}, "query 不能为空"),
        ({"query": "   "}, "query 不能为空"),
        ({"query": None}, "query 不能为空"),
        ({"query": "x" * 2001}, "query 过长"),
        ({"query": "ok", "top_k": 0}, "top_k 超出范围"),
        ({"query": "ok", "top_k": 21}, "top_k 超出范围"),
        ({"query": "ok", "top_k": "5"}, "top_k 必须是整数"),
        ({"query": "ok", "min_score": -0.1}, "min_score 超出范围"),
        ({"query": "ok", "min_score": 1.1}, "min_score 超出范围"),
    ],
    # ⚠️ 必须给 ASCII id。默认 id 会把中文转义成 `\u8d85` 这种形式，
    # 而反斜杠在 Windows 路径里是分隔符 —— conftest 的 tmp_path 会因此
    # 建目录失败（实测：FileNotFoundError WinError 3）。
    ids=[
        "empty", "spaces", "none", "too-long",
        "topk-zero", "topk-over", "topk-str",
        "min-score-neg", "min-score-over",
    ],
)
def test_param_validation_rejects_before_any_io(domains_dir, kwargs, fragment):
    """非法参数必须在碰后端之前就被拒。"""
    out = knowledge.search(domain="health", **kwargs)
    assert "error" in out
    assert fragment in out["error"]


def test_bool_top_k_is_rejected(domains_dir):
    """`True` 在 Python 里是 int 的子类 —— 不能让它混过类型检查。"""
    out = knowledge.search(query="ok", domain="health", top_k=True)
    assert "error" in out
    assert "top_k 必须是整数" in out["error"]


# ── 领域配置缺失 ────────────────────────────────────────────
def test_missing_domains_dir_mentions_env_var(monkeypatch):
    """目录没配时，错误要指名是哪个环境变量。"""
    monkeypatch.delenv(knowledge.ENV_DOMAINS_DIR, raising=False)
    out = knowledge.search(query="ok", domain="health")
    assert knowledge.ENV_DOMAINS_DIR in out["error"]


def test_no_domain_does_not_guess_default(domains_dir):
    """不传 domain 时不猜默认值，而是把可选项列出来。"""
    out = knowledge.search(query="ok")
    assert "缺少参数 domain" in out["error"]
    assert "health" in out["error"]


def test_unknown_domain_lists_available(domains_dir):
    out = knowledge.search(query="ok", domain="nonexistent")
    assert "不存在" in out["error"]
    assert "health" in out["error"]


def test_domain_without_collection_is_rejected(tmp_path, monkeypatch):
    """配置缺 collection 时不能继续 —— 否则会查到一个猜出来的集合名。"""
    (tmp_path / "broken.yaml").write_text(
        """domain:
  key: broken
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(knowledge.ENV_DOMAINS_DIR, str(tmp_path))
    out = knowledge.search(query="ok", domain="broken")
    assert "未声明 collection" in out["error"]


def test_malformed_yaml_reports_instead_of_raising(tmp_path, monkeypatch):
    """坏 yaml 不能把工具打挂 —— 返回错误让调用方决定怎么办。"""
    (tmp_path / "bad.yaml").write_text("domain: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv(knowledge.ENV_DOMAINS_DIR, str(tmp_path))
    out = knowledge.search(query="ok", domain="bad")
    assert "error" in out


# ── 后端不可用 ──────────────────────────────────────────────
def test_missing_embedding_config_names_the_var(domains_dir, monkeypatch):
    for var in ("API_KEY", "DASHSCOPE_API_KEY", "EMBED_BASE_URL", "EMBED_MODEL"):
        monkeypatch.delenv(var, raising=False)
    out = knowledge.search(query="ok", domain="health")
    assert "embedding 配置缺失" in out["error"]
    # 三个都要点名，只报一个会让人来回试
    for var in ("API_KEY", "EMBED_BASE_URL", "EMBED_MODEL"):
        assert var in out["error"]


def test_missing_milvus_url_names_the_var(domains_dir, monkeypatch):
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("EMBED_BASE_URL", "http://x")
    monkeypatch.setenv("EMBED_MODEL", "m")
    monkeypatch.delenv(knowledge.ENV_MILVUS_URL, raising=False)
    monkeypatch.setattr(
        knowledge, "_embed_query", lambda q: ([0.1, 0.2], 1.0, None)
    )
    out = knowledge.search(query="ok", domain="health")
    assert knowledge.ENV_MILVUS_URL in out["error"]


# ── 正常路径（打桩后端）─────────────────────────────────────
@pytest.fixture
def stub_backend(monkeypatch):
    """把 embed / milvus / 检索全打桩，只验证编排逻辑。"""
    monkeypatch.setattr(
        knowledge, "_embed_query", lambda q: ([0.1, 0.2], 3.5, None)
    )
    monkeypatch.setenv(knowledge.ENV_MILVUS_URL, "http://stub")
    monkeypatch.setattr(knowledge, "_milvus_client", lambda: (object(), None))

    captured: dict = {}

    def fake_retrieve(client, qvec, collection, top_k, form=None):
        captured.update(
            qvec=qvec, collection=collection, top_k=top_k, form=form
        )
        return (
            [
                {"score": 0.9, "source": "guides/a.md", "chunk_index": 0, "text": "high"},
                {"score": 0.5, "source": "faq/b.md", "chunk_index": 1, "text": "mid"},
                {"score": 0.1, "source": "faq/c.md", "chunk_index": 2, "text": "low"},
            ],
            12.3,
        )

    monkeypatch.setattr(knowledge, "_retrieve_one", fake_retrieve)
    return captured


def test_success_returns_hits_and_domain_metadata(domains_dir, stub_backend):
    out = knowledge.search(query="  盐上限  ", domain="health", top_k=3)

    assert "error" not in out
    assert out["n_hits"] == 3
    assert out["domain"] == "health"
    assert out["collection"] == "test_collection"
    assert out["dropped"] == 0
    assert out["stage_ms"]["embed"] == 3.5
    assert out["stage_ms"]["search"] == 12.3

    # query 应当被 strip 后再 embed，且不按形态过滤（单路检索）
    assert stub_backend["qvec"] == [0.1, 0.2]
    assert stub_backend["top_k"] == 3
    assert stub_backend["form"] is None


def test_min_score_filters_and_reports_dropped(domains_dir, stub_backend):
    """阈值过滤必须报告丢了几条 —— 否则调用方分不清「没召回」和「被过滤」。"""
    out = knowledge.search(query="ok", domain="health", top_k=3, min_score=0.4)

    assert out["n_hits"] == 2
    assert out["dropped"] == 1
    assert all(h["score"] >= 0.4 for h in out["hits"])


def test_min_score_zero_keeps_everything(domains_dir, stub_backend):
    out = knowledge.search(query="ok", domain="health", top_k=3, min_score=0.0)
    assert out["n_hits"] == 3
    assert out["dropped"] == 0


def test_search_failure_is_reported_not_raised(domains_dir, stub_backend, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("collection not loaded")

    monkeypatch.setattr(knowledge, "_retrieve_one", boom)
    out = knowledge.search(query="ok", domain="health")
    assert "Milvus 检索失败" in out["error"]
    assert "collection not loaded" in out["error"]


# ── 纯函数直接验证 ──────────────────────────────────────────
def test_to_raw_maps_milvus_hit_shape():
    """Milvus 的命中结构是嵌套的，映射写错会让上层拿到 None。"""
    hits = [
        {
            "distance": 0.87654,
            "entity": {"text": "正文", "source": "guides/x.md", "chunk_index": 7},
        }
    ]
    out = knowledge._to_raw(hits)
    assert out == [
        {"score": 0.8765, "source": "guides/x.md", "chunk_index": 7, "text": "正文"}
    ]


def test_form_keys_preserves_config_order():
    """形态顺序要稳定 —— 它影响 per_form 结果的合并顺序与可复现性。"""
    cfg = {
        "forms": [
            {"key": "guides"},
            {"key": "faq"},
            {"key": "tables"},
        ]
    }
    assert knowledge._form_keys(cfg) == ["guides", "faq", "tables"]


def test_form_keys_skips_malformed_entries():
    cfg = {"forms": [{"key": "guides"}, {"nokey": 1}, "string", {"key": "faq"}]}
    assert knowledge._form_keys(cfg) == ["guides", "faq"]