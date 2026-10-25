"""`common.ops.*` —— 索引运维工具。

**为什么 reindex 是唯一允许存在的写工具**

完全不给写操作会脱离真实需求：语料更新后必须重建索引，否则知识库会越来越旧。
但写操作又是最危险的，所以三重限制：

1. 只需 `common.ops.write` scope（health-agent 的 discovery_scopes 里没有它）
2. 默认 `dry_run=True` —— 必须显式传 false 才真的动手
3. 不参与降级（registry 加载期已禁止给写工具配 fallback）

**为什么不在这里实现建库逻辑**

建库属于仓储层职责（读语料 → 切分 → embedding → 写入），
MCP 只是暴露面。所以这里调子进程跑既有脚本，路径由环境变量注入 ——
避免把仓储路径硬编码进协议层。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

API_TIMEOUT_S = 5.0


def _milvus_url(explicit: str | None = None) -> str | None:
    return explicit or os.getenv("MILVUS_URL")


def get_index_status(
    collection: str,
    milvus_url: str | None = None,
) -> dict:
    """查某个集合的行数、索引类型、状态。

    不需要领域配置 —— 直接按集合名查，这样它对任何域都适用。
    """
    url = _milvus_url(milvus_url)
    if not url:
        return {"error": "MILVUS_URL 未配置", "collection": collection}

    try:
        from pymilvus import MilvusClient
    except ImportError:
        return {"error": "pymilvus 未安装", "collection": collection}

    try:
        client = MilvusClient(uri=url, timeout=API_TIMEOUT_S)
        if collection not in client.list_collections():
            return {"collection": collection, "exists": False}

        stats = client.get_collection_stats(collection_name=collection)
        out = {
            "collection": collection,
            "exists": True,
            "rows": stats.get("row_count"),
        }

        try:
            indexes = client.list_indexes(collection_name=collection)
            detail = []
            for name in indexes:
                info = client.describe_index(collection_name=collection, index_name=name)
                detail.append({
                    "index": name,
                    "type": info.get("index_type"),
                    "metric": info.get("metric_type"),
                    "state": info.get("state"),
                })
            out["indexes"] = detail
        except Exception:            # noqa: BLE001 —— 索引信息是附加，查不到不影响主结果
            out["indexes"] = []

        return out
    except Exception as exc:         # noqa: BLE001
        # ⚠️ 探活类工具不抛异常，把失败折成结构化字段（与 health_check 一致）
        return {"collection": collection, "error": f"{type(exc).__name__}: {exc}"}


def reindex(
    domain: str,
    dry_run: bool = True,
    repo_path: str | None = None,
) -> dict:
    """重建某领域的索引。

    `repo_path` 应指向含 `scripts/build_index.py` 的仓储根目录；
    未提供时读环境变量 `DOMAIN_RAG_REPO`。

    ⚠️ **默认 dry_run=True**：只回报将要执行什么，不动数据。
    """
    repo = Path(repo_path or os.getenv("DOMAIN_RAG_REPO") or "")
    script = repo / "scripts" / "build_index.py"

    plan = {
        "domain": domain,
        "repo": str(repo) if repo else None,
        "script": str(script) if repo else None,
        "dry_run": dry_run,
    }

    if not repo or not repo.is_dir():
        return {**plan, "ok": False,
                "error": "仓储路径未配置（设置 DOMAIN_RAG_REPO 或传 repo_path）"}
    if not script.is_file():
        return {**plan, "ok": False,
                "error": f"未找到建库脚本：{script}"}

    if dry_run:
        return {**plan, "ok": True, "message": "预演：未执行。传 dry_run=false 才会真的重建"}

    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--corpus", "./corpus", "--domain", domain],
            cwd=str(repo), capture_output=True, text=True, timeout=600,
            env={**os.environ, "DOMAIN": domain},
        )
    except subprocess.TimeoutExpired:
        return {**plan, "ok": False, "error": "建库超时（>600s）"}
    except Exception as exc:         # noqa: BLE001
        return {**plan, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        **plan,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }