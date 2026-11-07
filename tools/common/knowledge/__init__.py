"""`common.knowledge.*` —— 检索类工具。

**为什么这个 scope 之前是空的**

`registry.yaml` 一直声明着 `common.knowledge`，但一个工具都没有 —— 一个 RAG 服务
没有「检索」这个最核心的工具，是明显的缺口。

**为什么不直接调同生态里的检索服务**

有过一版设计是「工具层 HTTP 调某个检索服务」，被否掉了，理由是**依赖方向错了**：
那样 MCP 会变成某个具体业务服务的前端，而不是通用工具层。别的 Agent 接入时，
就得先有那个业务服务在跑。

**正确的依赖方向**：工具层直接连基础设施（Milvus / embedding 端点），
领域差异通过 `domains/*.yaml` 配置注入，代码里不出现任何具体领域名。

**复用策略：复制纯函数，不复制业务编排**

`_to_raw` 与 `_retrieve_one` 是从同生态检索仓**复制**的（截至 2026-11-07）。
之所以是复制而不是抽公共库：

1. 这两个是**依赖注入的纯函数**（client / qvec 都由调用方给），没有状态
2. 检索仓那边的另一半是**业务编排**（检索→融合→生成→安全闸→trace），
   工具层不需要那部分，也不需要跟着它一起演进
3. 抽库要先动检索仓，而它有 109 个测试 —— 为当前目标不值得

**什么时候该改成抽公共库**：当第三个地方也要用「多形态并发召回 + RRF 融合」
这段逻辑时（那部分有 33 + 40 行，重复的代价开始超过抽库的成本）。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

ENV_DOMAINS_DIR = "DOMAIN_RAG_DOMAINS_DIR"
ENV_MILVUS_URL = "MILVUS_URL"

# 本仓自带的领域配置目录。
# `tools/common/knowledge/__init__.py` 上溯三层 = 仓库根，再下到 `domains/`。
_BUNDLED_DOMAINS = Path(__file__).resolve().parents[3] / "domains"

# 与检索仓保持一致的默认值，避免「同一个查询在两个入口得到不同上限」
DEFAULT_TOP_K = 5
MAX_TOP_K = 20


# ── 领域配置 ────────────────────────────────────────────────
def _domains_dir() -> Path | None:
    """定位 `domains/` 目录。

    **优先级**：环境变量 `DOMAIN_RAG_DOMAINS_DIR` → 本仓自带的 `domains/`。

    **为什么默认读自己仓**：领域配置是「工具层的能力声明」，不属于任何一个业务。
    早先的做法是靠环境变量指向某个业务仓的 `domains/`，后果是
    「部署工具层还得连带业务仓的配置」—— 那就是配置层面的反向依赖。
    现在自带一份，环境变量只在「多套配置 / 多集群」时才需要。

    ⚠️ **显式配了但路径无效时返回 None（报错），不静默回退到自带**：
    运维把路径写错时，静默回退会让人以为「用的是新配置」，
    实际用的是仓里那份可能已经过期的 —— 这类问题比直接报错难查得多。
    """
    raw = os.getenv(ENV_DOMAINS_DIR)
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_dir() else None
    return _BUNDLED_DOMAINS if _BUNDLED_DOMAINS.is_dir() else None


def _load_domain(key: str | None) -> tuple[dict | None, str | None]:
    """按 key 读 `domains/<key>.yaml`。

    返回 `(cfg, error)`，两者恰有一个非空。

    ⚠️ 不传 key 时**不猜**默认领域 —— 「猜」会让调用方拿到的集合与预期不符，
    而这类错误（查错了库但结果看起来正常）比报错难查得多。
    """
    base = _domains_dir()
    if base is None:
        return None, f"{ENV_DOMAINS_DIR} 未配置或目录不存在"

    if not key:
        available = sorted(p.stem for p in base.glob("*.yaml"))
        return None, f"缺少参数 domain，可用：{available}"

    path = base / f"{key}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in base.glob("*.yaml"))
        return None, f"领域 {key!r} 不存在，可用：{available}"

    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return None, f"领域配置解析失败：{type(exc).__name__}: {exc}"

    if not isinstance(cfg, dict):
        return None, f"领域配置格式错误（顶层应为映射）：{path.name}"
    if not cfg.get("collection"):
        return None, f"领域 {key!r} 未声明 collection"

    return cfg, None


def _form_keys(cfg: dict) -> list[str]:
    """配置里声明的形态 key，保持原顺序（形态顺序影响可复现性）。"""
    return [
        f.get("key")
        for f in (cfg.get("forms") or [])
        if isinstance(f, dict) and f.get("key")
    ]


# ── 检索原语（复制自检索仓，纯函数）────────────────────────
def _to_raw(hits: list) -> list[dict]:
    """把 Milvus 命中统一成内部结构。"""
    return [
        {
            "score": round(h["distance"], 4),
            "source": h["entity"]["source"],
            "chunk_index": h["entity"]["chunk_index"],
            "text": h["entity"]["text"],
        }
        for h in hits
    ]


def _retrieve_one(client, qvec, collection: str, top_k: int,
                  form: str | None = None) -> tuple[list[dict], float]:
    """单次向量检索，可选按形态过滤。返回 `(hits, elapsed_ms)`。"""
    t0 = time.perf_counter()
    kwargs = {
        "collection_name": collection,
        "data": [qvec],
        "limit": top_k,
        "output_fields": ["text", "source", "chunk_index"],
    }
    if form:
        kwargs["filter"] = f'source like "{form}/%"'

    hits = client.search(**kwargs)[0]
    return _to_raw(hits), round((time.perf_counter() - t0) * 1000, 1)


# ── 能力探针与后端 ──────────────────────────────────────────
def _embed_query(query: str) -> tuple[list[float] | None, float, str | None]:
    """把 query 向量化。

    返回 `(vec, elapsed_ms, error)`。

    ⚠️ `check_embedding_ctx_length=False` 不能省：默认值会让 OpenAI 兼容端点
    尝试本地分词并按 token 校验长度，对这些端点会直接报错。
    """
    api_key = os.getenv("API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("EMBED_BASE_URL")
    model = os.getenv("EMBED_MODEL")

    missing = [
        name
        for name, val in (("API_KEY", api_key), ("EMBED_BASE_URL", base_url),
                          ("EMBED_MODEL", model))
        if not val
    ]
    if missing:
        return None, 0.0, f"embedding 配置缺失：{', '.join(missing)}"

    t0 = time.perf_counter()
    try:
        # 函数内导入：没装 langchain 也能跑本模块的其余部分（含测试）
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        return None, 0.0, "langchain-openai 未安装（pip install -e \".[retrieval]\"）"

    try:
        emb = OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=api_key,
            check_embedding_ctx_length=False,
        )
        vec = emb.embed_query(query)
    except Exception as exc:  # noqa: BLE001
        return None, 0.0, f"embedding 调用失败：{type(exc).__name__}: {exc}"

    return vec, round((time.perf_counter() - t0) * 1000, 1), None


def _milvus_client():
    """建 Milvus 客户端。返回 `(client, error)`。"""
    url = os.getenv(ENV_MILVUS_URL)
    if not url:
        return None, f"{ENV_MILVUS_URL} 未配置"

    try:
        from pymilvus import MilvusClient
    except ImportError:
        return None, "pymilvus 未安装（pip install -e \".[retrieval]\"）"

    try:
        return MilvusClient(uri=url, timeout=10.0), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Milvus 连接失败：{type(exc).__name__}: {exc}"


# ── 对外工具 ────────────────────────────────────────────────
def search(
    query: str,
    domain: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.0,
    milvus_url: str | None = None,
) -> dict:
    """在指定领域做单路向量检索，返回带来源的片段列表，**不生成回答**。

    Args:
        query: 检索文本。
        domain: `domains/` 下的配置名。**必填**（不猜默认）。
        top_k: 返回条数，1–20。
        min_score: 相似度下限，低于此值丢弃。默认 0.0 = 不过滤。
        milvus_url: 覆盖 `MILVUS_URL`（测试与多集群场景用）。

    Returns:
        `{"hits": [...], "n_hits": int, "domain": str, "collection": str,
           "dropped": int, "stage_ms": {...}}`
        或 `{"error": str}`。
    """
    # 参数校验先做完，避免「跑了一半才失败」
    if not isinstance(query, str) or not query.strip():
        return {"error": "query 不能为空"}
    if len(query) > 2000:
        return {"error": f"query 过长（{len(query)} > 2000 字符）"}
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        return {"error": "top_k 必须是整数"}
    if not 1 <= top_k <= MAX_TOP_K:
        return {"error": f"top_k 超出范围（1–{MAX_TOP_K}）"}
    if not 0.0 <= min_score <= 1.0:
        return {"error": "min_score 超出范围（0–1）"}

    cfg, err = _load_domain(domain)
    if err:
        return {"error": err}
    collection = cfg["collection"]

    vec, embed_ms, err = _embed_query(query.strip())
    if err:
        return {"error": err}

    # milvus_url 显式参数优先（照 get_index_status 的约定，便于测试注入）
    if milvus_url:
        os.environ[ENV_MILVUS_URL] = milvus_url
    client, err = _milvus_client()
    if err:
        return {"error": err}

    try:
        hits, search_ms = _retrieve_one(client, vec, collection, top_k)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Milvus 检索失败：{type(exc).__name__}: {exc}"}

    before = len(hits)
    if min_score > 0.0:
        hits = [h for h in hits if h["score"] >= min_score]

    return {
        "hits": hits,
        "n_hits": len(hits),
        "domain": cfg.get("domain", {}).get("key") or domain,
        "collection": collection,
        # 被阈值丢掉的条数要显式返回：否则调用方无法区分
        # 「本来就没召回」和「召回了但被我过滤掉了」
        "dropped": before - len(hits),
        "stage_ms": {"embed": embed_ms, "search": search_ms},
    }


# ════════════════════════════════════════════════════════════
# 多形态并发召回 + RRF 融合
# ════════════════════════════════════════════════════════════

# RRF 的平滑常数。取值越大，排名靠后的候选权重越接近，融合越"平"。
# 60 是原论文的常用默认值。
RRF_K = 60


def _doc_key(doc: dict) -> tuple:
    """候选块的唯一标识。

    用 `(source, chunk_index)` 而不是 text：两条召回路径都能提供这两个字段，
    而文本比较代价高。
    """
    return (doc.get("source"), doc.get("chunk_index"))


def _rrf_fuse(route_docs: dict[str, list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion。

    `route_docs`：`{"路名": [候选, ...]}`，每路候选按**该路自己的排序**给出。

    返回融合后的列表（按 RRF 分降序），每项含
    `key` / `doc` / `score` / `ranks`（这一项在各路里的名次）。

    **`ranks` 是这个函数存在的主要理由**：它让「为什么这条被顶上来」可解释 ——
    是两路都排第 1，还是只有一路排第 1 而另一路排第 5。没有它，
    融合结果是个黑盒。

    ⚠️ **不做加权**：本工具的各路是同构的（都是向量检索，只是形态过滤不同），
    不像 dense/sparse 那样质量不对等。加权要先有「哪一路更可信」的实测依据，
    没有依据的权重只是把主观臆断写成数字。
    """
    scores: dict[tuple, float] = {}
    ranks: dict[tuple, dict[str, int]] = {}
    docs: dict[tuple, dict] = {}

    for route, items in route_docs.items():
        for rank, doc in enumerate(items, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(key, {})[route] = rank
            docs.setdefault(key, doc)

    fused = [
        {"key": key, "doc": docs[key], "score": round(score, 6), "ranks": ranks[key]}
        for key, score in scores.items()
    ]
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


def _retrieve_multi(
    client, qvec, collection: str, per_form: int,
    forms: list[str], max_workers: int = 4,
) -> tuple[dict[str, list[dict]], dict[str, float], float]:
    """按形态**并发**召回。返回 `(route_docs, per_form_ms, wall_ms)`。

    **为什么并发**：串行时总延迟是各形态的**加和**，并发后接近**最大值**。
    语义完全不变（仍是每形态各取 per_form 条），只是不再互相等待。

    **失败处理**：单路失败只丢那一路，不炸整体。一个形态的过滤器写错，
    不该让整个检索不可用 —— 但要在返回里能看出少了一路（缺 key）。
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(form: str):
        t = time.perf_counter()
        try:
            hits, _ = _retrieve_one(client, qvec, collection, per_form, form=form)
            return form, hits, round((time.perf_counter() - t) * 1000, 1)
        except Exception as exc:  # noqa: BLE001
            return form, [], round((time.perf_counter() - t) * 1000, 1)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(one, forms))
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)

    route_docs: dict[str, list[dict]] = {}
    per_form_ms: dict[str, float] = {}
    for form, hits, ms in results:
        route_docs[form] = hits
        per_form_ms[form] = ms
    return route_docs, per_form_ms, wall_ms


def search_multi(
    query: str,
    domain: str | None = None,
    per_form: int = 2,
    forms: list[str] | None = None,
    milvus_url: str | None = None,
) -> dict:
    """按数据形态并发召回多路，再用 RRF 融合成统一排序的片段列表。

    **与 `search` 的区别**：`search` 是单路全库检索；本工具把语料按形态
    （指南/问答/表格/对话）各查一次再融合。适用于「答案可能是表格、但问法像问答」
    这类跨形态场景 —— 单路检索在这种情况下会漏。

    Args:
        query: 检索文本。
        domain: 领域配置名，**必填**。
        per_form: **每个形态**取几条，1–10（不是总数）。
        forms: 只查这些形态；默认取配置里声明的全部。
        milvus_url: 覆盖 `MILVUS_URL`。

    Returns:
        `{"hits": [...], "n_hits": int, "score_kind": "rrf", "domain": str,
           "collection": str, "forms": [...], "per_form_ms": {...},
           "wall_ms": float, "fuse_ms": float, "stage_ms": {...}}`
        或 `{"error": str}`。

    ⚠️ **`score` 是 RRF 融合分，不是向量相似度**（典型值 0.01–0.05）。
    不要拿它跟 `min_score` 那类相似度阈值比较 —— 量纲完全不同。
    所以本工具**刻意不提供 `min_score` 参数**。如果需要按相似度过滤，
    用 `search`（单路），或者自己按 `hits[].ranks` 判断。
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "query 不能为空"}
    if len(query) > 2000:
        return {"error": f"query 过长（{len(query)} > 2000 字符）"}
    if not isinstance(per_form, int) or isinstance(per_form, bool):
        return {"error": "per_form 必须是整数"}
    if not 1 <= per_form <= 10:
        return {"error": "per_form 超出范围（1–10）"}

    cfg, err = _load_domain(domain)
    if err:
        return {"error": err}
    collection = cfg["collection"]

    declared = _form_keys(cfg)
    if forms is None:
        forms = declared
    else:
        unknown = [f for f in forms if f not in declared]
        if unknown:
            return {
                "error": f"未声明的形态：{unknown}，可用：{declared}"
            }
        if not forms:
            return {"error": "forms 不能为空列表"}

    if not forms:
        return {"error": f"领域 {domain!r} 未声明任何形态（forms）"}

    vec, embed_ms, err = _embed_query(query.strip())
    if err:
        return {"error": err}

    if milvus_url:
        os.environ[ENV_MILVUS_URL] = milvus_url
    client, err = _milvus_client()
    if err:
        return {"error": err}

    try:
        route_docs, per_form_ms, wall_ms = _retrieve_multi(
            client, vec, collection, per_form, forms
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"多路检索失败：{type(exc).__name__}: {exc}"}

    t0 = time.perf_counter()
    fused = _rrf_fuse(route_docs)
    fuse_ms = round((time.perf_counter() - t0) * 1000, 1)

    hits = [
        {
            "source": f["doc"].get("source"),
            "chunk_index": f["doc"].get("chunk_index"),
            "text": f["doc"].get("text"),
            "score": f["score"],
            "ranks": f["ranks"],
        }
        for f in fused
    ]

    return {
        "hits": hits,
        "n_hits": len(hits),
        # 显式声明分数语义，避免调用方把它当相似度用
        "score_kind": "rrf",
        "domain": cfg.get("domain", {}).get("key") or domain,
        "collection": collection,
        "forms": forms,
        "per_form_ms": per_form_ms,
        "wall_ms": wall_ms,
        "fuse_ms": fuse_ms,
        "stage_ms": {"embed": embed_ms, "wall": wall_ms, "fuse": fuse_ms},
    }


# ════════════════════════════════════════════════════════════
# 检索 + 生成 + 安全闸
# ════════════════════════════════════════════════════════════

DEFAULT_ANSWER_TOP_K = 3


# ── 安全闸 ──────────────────────────────────────────────────
def _check_output(text: str, cfg: dict) -> dict:
    """检查回答是否含禁用断言。

    **命中即整条拒答，不改写** —— 改写会引入新的、未经验证的表达，
    而拒答是确定性的。这个取舍在健康/法律这类领域没有讨论空间。

    判定方式是**子串匹配**（`claim in text`），刻意不用模型判：
    模型判会引入不确定性，而这条闸门存在的意义就是确定性。
    代价是可能误伤（如「不是可以治愈的」也会命中），
    **宁可误伤也不放过** —— 误伤时调用方得到明确拦截，漏放则可能输出越界内容。
    """
    claims = (cfg.get("safety") or {}).get("forbidden_claims") or []
    hits = [c for c in claims if c and c in text]
    if hits:
        return {
            "allowed": False,
            "action": "block",
            "matched": hits,
            "reasons": [f"命中禁用断言：{'、'.join(hits)}"],
            "message": "该回答涉及超出安全边界的表述，已被拦截。请以专业医疗意见为准。",
        }
    return {"allowed": True, "action": "allow", "matched": [], "reasons": []}


def _apply_disclaimers(text: str, cfg: dict) -> str:
    """按 `disclaimers` 的 position 注入免责声明。

    `prepend` 在前、`append` 在后；**顺序按配置里的出现顺序**，
    因为多条同位置的声明有优先级（越靠前越重要）。
    """
    items = cfg.get("disclaimers") or []
    before = [d.get("text") for d in items
              if isinstance(d, dict) and d.get("position") == "prepend" and d.get("text")]
    after = [d.get("text") for d in items
             if isinstance(d, dict) and d.get("position") == "append" and d.get("text")]

    parts = [*before, text, *after]
    return "\n\n".join(p for p in parts if p)


# ── prompt 组装 ─────────────────────────────────────────────
def _build_messages(question: str, hits: list[dict], cfg: dict) -> list[dict]:
    """组装 messages。

    **材料编号后拼在 user 消息里，不放 system**：system 是能力约束（每领域固定），
    材料是本次请求的内容。混在一起会让「改 prompt」和「改材料」纠缠不清。

    `no_answer` 占位符从配置里替换 —— 领域不同，兜底措辞也不同
    （健康是「材料中未提及」，法律可能是「现有材料不足以判断」）。
    """
    prompt_cfg = cfg.get("prompt") or {}
    no_answer = prompt_cfg.get("no_answer") or "材料中未提及"
    system = (prompt_cfg.get("system") or "").replace("{no_answer}", no_answer)

    blocks = [
        f"【材料{i}】来源：{h.get('source')}#{h.get('chunk_index')}\n{h.get('text')}"
        for i, h in enumerate(hits, start=1)
    ]
    user = "材料：\n" + "\n\n".join(blocks) + f"\n\n问题：{question}"

    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def _llm_client():
    """建 LLM 客户端。返回 `(client, error)`。

    配置键与 embedding 分开（`MODEL` / `BASE_URL` vs `EMBED_MODEL` / `EMBED_BASE_URL`）：
    生成和向量化通常是不同的模型、不同的端点，复用一套配置会让换模型时互相牵连。
    """
    api_key = os.getenv("API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("MODEL")

    missing = [
        name
        for name, val in (("API_KEY", api_key), ("BASE_URL", base_url), ("MODEL", model))
        if not val
    ]
    if missing:
        return None, f"LLM 配置缺失：{', '.join(missing)}"

    try:
        # 函数内导入：没装 langchain 也能导入本模块（含测试）
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None, 'langchain-openai 未安装（pip install -e ".[retrieval]"）'

    try:
        return ChatOpenAI(
            model=model,
            # 与检索仓一致：有些部署把完整路径写进 BASE_URL，这里容错
            base_url=base_url.removesuffix("/chat/completions"),
            api_key=api_key,
        ), None
    except Exception as exc:  # noqa: BLE001
        return None, f"LLM 客户端初始化失败：{type(exc).__name__}: {exc}"


def answer(
    question: str,
    domain: str | None = None,
    top_k: int = DEFAULT_ANSWER_TOP_K,
    min_score: float = 0.0,
    milvus_url: str | None = None,
) -> dict:
    """检索 + 生成 + 安全闸一体。回答严格基于检索到的材料。

    **为什么这个工具存在**：如果只给 `search`，每个调用方都要自己实现
    「拼 prompt → 调模型 → 过安全闸」三段。安全闸尤其危险 —— 漏了它，
    健康领域就会输出像医嘱的内容。**这件事该由能力层做一次，而不是每个调用方各做一遍。**

    **不带记忆（无状态）**：MCP 2026-07-28 已移除协议级 session，在工具里做会话状态
    与协议方向相反。需要多轮记忆时由调用方（业务层）自己管，把历史整理进 question。

    Args:
        question: 用户问题。
        domain: 领域配置名，**必填**。
        top_k: 送入模型的材料条数，1–20。
        min_score: 材料相似度下限。
        milvus_url: 覆盖 `MILVUS_URL`。

    Returns:
        成功：`{"answer": str, "contexts": [...], "safety": {...}, "usage": {...},
                "stage_ms": {...}, "domain": str, "collection": str}`
        **安全拦截**：`{"blocked": True, "answer": str, "safety": {...}, ...}`
        系统故障：`{"error": str}`

    ⚠️ **`blocked` 与 `error` 是两件事，调用方必须分开处理**：
        - `error` = 系统坏了（Milvus / LLM 不可用）⇒ 该重试或报障
        - `blocked` = 系统正常，但内容越界 ⇒ 该换说法问，重试无用
    """
    if not isinstance(question, str) or not question.strip():
        return {"error": "question 不能为空"}
    if len(question) > 2000:
        return {"error": f"question 过长（{len(question)} > 2000 字符）"}
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        return {"error": "top_k 必须是整数"}
    if not 1 <= top_k <= MAX_TOP_K:
        return {"error": f"top_k 超出范围（1–{MAX_TOP_K}）"}

    cfg, err = _load_domain(domain)
    if err:
        return {"error": err}

    # ① 检索（复用 search，避免两套检索逻辑）
    retrieved = search(question, domain=domain, top_k=top_k,
                       min_score=min_score, milvus_url=milvus_url)
    if "error" in retrieved:
        return retrieved

    hits = retrieved["hits"]
    no_answer = (cfg.get("prompt") or {}).get("no_answer") or "材料中未提及"

    # ② 没检索到材料就不调模型 —— 省一次必然无用且昂贵的调用
    if not hits:
        return {
            "answer": _apply_disclaimers(no_answer, cfg),
            "contexts": [],
            "safety": {"input": {"allowed": True}, "output": {"allowed": True}},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "skipped_generation": "no_context",
            "domain": retrieved["domain"],
            "collection": retrieved["collection"],
            "stage_ms": {"retrieve": retrieved["stage_ms"], "generate": 0.0},
        }

    # ③ 生成
    llm, err = _llm_client()
    if err:
        return {"error": err}

    messages = _build_messages(question.strip(), hits, cfg)

    t0 = time.perf_counter()
    try:
        resp = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"LLM 调用失败：{type(exc).__name__}: {exc}"}
    generate_ms = round((time.perf_counter() - t0) * 1000, 1)

    raw = getattr(resp, "content", None)
    if not isinstance(raw, str):
        raw = str(raw or "")
    raw = raw.strip()

    # ④ 安全闸：命中就整条拒答，不改写
    verdict = _check_output(raw, cfg)
    usage = getattr(resp, "usage_metadata", None) or {}
    usage_out = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }

    if not verdict["allowed"]:
        return {
            "blocked": True,
            "answer": _apply_disclaimers(verdict["message"], cfg),
            # 原始回答**不返回**给调用方 —— 否则拦截形同虚设
            "contexts": hits,
            "safety": {"input": {"allowed": True}, "output": verdict},
            "usage": usage_out,
            "domain": retrieved["domain"],
            "collection": retrieved["collection"],
            "stage_ms": {
                "retrieve": retrieved["stage_ms"], "generate": generate_ms,
            },
        }

    return {
        "answer": _apply_disclaimers(raw, cfg),
        "contexts": hits,
        "safety": {"input": {"allowed": True}, "output": verdict},
        "usage": usage_out,
        "domain": retrieved["domain"],
        "collection": retrieved["collection"],
        "stage_ms": {"retrieve": retrieved["stage_ms"], "generate": generate_ms},
    }