"""`common.discovery.list_domains` —— 列出可用知识领域。

**数据来源**

读 `domains/*.yaml`（与产品仓同一份配置文件），不硬编码任何领域名。

**为什么目录要外部注入而不是写死**

同一个 MCP server 应当能服务多个领域目录（比如健康语料和生产环境语料分开放）。
所以目录从 `DOMAIN_RAG_DOMAINS_DIR` 环境变量取；没配就返回空列表，
由调用方注入参数。

⚠️ 返回空列表而不是抛异常：一个「发现」工具在没配置时不应该是错误，
调用方看到空列表就能判断「这个 server 还没接语料」。
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ENV_DOMAINS_DIR = "DOMAIN_RAG_DOMAINS_DIR"


def _resolve_base(domains_dir: str | None) -> Path | None:
    """优先用显式参数，其次环境变量。都没有或不存在就返回 None。"""
    raw = domains_dir or os.getenv(ENV_DOMAINS_DIR)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _parse_one(path: Path) -> dict | None:
    """解析单个领域配置。坏文件返回 None（跳过），不让它拖垮整个列表。"""
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:                        # noqa: BLE001
        return None
    if not isinstance(cfg, dict):
        return None

    domain = cfg.get("domain") or {}
    forms = [
        f.get("key")
        for f in (cfg.get("forms") or [])
        if isinstance(f, dict) and f.get("key")
    ]
    return {
        "key": domain.get("key") or path.stem,
        "name": domain.get("name"),
        "language": domain.get("language"),
        "forms": forms,
        "collection": cfg.get("collection"),
        # 块数需要查索引，Step 4 接上后再填；现在明确给 null 而不是编一个数
        "chunk_count": None,
    }


def list_domains(domains_dir: str | None = None) -> list[dict]:
    """列出领域。按 key 排序，保证**确定性顺序**（规范要求 tools/list 与结果稳定）。"""
    base = _resolve_base(domains_dir)
    if base is None:
        return []

    items = []
    for path in sorted(base.glob("*.yaml")):
        parsed = _parse_one(path)
        if parsed is not None:
            items.append(parsed)

    return sorted(items, key=lambda d: d["key"])