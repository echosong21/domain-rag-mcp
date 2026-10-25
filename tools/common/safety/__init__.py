"""`common.safety.*` —— 领域安全闸。

**三个工具对应三道检查**

| 工具 | 检查什么 | 对应风险 |
|---|---|---|
| `check_input` | 用户输入是否要先转介（急症） | 把急症当普通咨询答 |
| `check_output` | 生成的回答里有没有禁止断言 | 把建议说成医嘱 |
| `get_rules` | 该领域的规则本身 | 调用方无法预知边界，只能撞错 |

**为什么用「完整短语」而不是关键词**

`forbidden_claims` 里若是关键词「治愈」，那「**不能**治愈」这种正确表述会被误杀。
所以配置里存完整短语（「可以治愈」），匹配时也按短语 —— 让否定句天然安全。

**为什么 check_input 与 check_output 不合并**

两个方向的语义不同：输入闸只管「要不要转介急症」，输出闸管「有没有越界断言」。
合并成 `check(text, direction)` 后，调用方极易在错误的时机用错方向 ——
而这个错误在故障时才暴露。

⚠️ 本模块**不实现**整条问答链路的安全策略，只提供可被调用的检查原语；
真正的「拒答 / 改写 / 注入免责声明」由调用方根据返回的 action 决定。
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# 急症信号：命中即转介，不进入正常问答。
# 刻意写死在代码里而不是配置里 —— 这是「不该因为换领域就消失」的底线。
EMERGENCY_SIGNS: tuple[str, ...] = (
    "胸痛", "胸闷伴有出汗", "呼吸困难", "喘不上气", "意识模糊", "昏迷",
    "晕厥", "大出血", "抽搐", "中毒",
)


def _domains_dir(explicit: str | None = None) -> Path | None:
    raw = explicit or os.getenv("DOMAIN_RAG_DOMAINS_DIR")
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


def _load_domain(domain: str, domains_dir: str | None = None) -> dict | None:
    base = _domains_dir(domains_dir)
    if base is None:
        return None
    path = base / f"{domain}.yaml"
    if not path.is_file():
        return None
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:                # noqa: BLE001
        return None
    return cfg if isinstance(cfg, dict) else None


def _safety(cfg: dict) -> dict:
    s = cfg.get("safety") or {}
    return {
        "forbidden_claims": list(s.get("forbidden_claims") or []),
        "disclaimers": list(cfg.get("disclaimers") or []),
        "max_plan_delta": s.get("max_plan_delta_per_day"),
    }


def get_rules(domain: str, domains_dir: str | None = None) -> dict:
    """把该领域的边界规则**透明化**，让调用方提前知道能问什么、不能答什么。"""
    cfg = _load_domain(domain, domains_dir)
    if cfg is None:
        return {
            "domain": domain, "found": False,
            "hint": "设置 DOMAIN_RAG_DOMAINS_DIR 指向含 domains/*.yaml 的目录",
        }
    rules = _safety(cfg)
    return {
        "domain": domain,
        "found": True,
        **rules,
        "emergency_signs": list(EMERGENCY_SIGNS),
    }


def check_input(
    text: str,
    domain: str,
    domains_dir: str | None = None,
) -> dict:
    """输入闸。返回 `action`：`pass` / `refuse` / `refer`。

    `refer` = 命中急症，应建议立即就医且**不进入正常回答生成**。
    """
    if not text or not text.strip():
        return {"action": "pass", "reason": "空输入"}

    # ① 急症优先：哪怕同时含有禁止断言，也该先转介
    hit = [s for s in EMERGENCY_SIGNS if s in text]
    if hit:
        return {
            "action": "refer",
            "rule": hit[0],
            "reason": f"输入含急症信号「{hit[0]}」，应建议立即就医",
        }

    # ② 禁止断言（用户主动问「是不是可以治愈」这类）
    cfg = _load_domain(domain, domains_dir)
    if cfg is None:
        return {"action": "pass", "reason": "领域配置不可用，跳过禁止断言检查"}

    for claim in _safety(cfg)["forbidden_claims"]:
        if claim and claim in text:
            return {
                "action": "refuse",
                "rule": claim,
                "reason": f"输入触及禁止断言「{claim}」",
            }

    return {"action": "pass"}


def check_output(
    text: str,
    domain: str,
    domains_dir: str | None = None,
) -> dict:
    """输出闸。返回 `action`：`pass` / `refuse`。

    ⚠️ **命中即整条拒答，不做改写**。
    改写等于把已被判定为不安全的内容重新交给模型生成，结果不可预测；
    拒答是确定性的、可审计的。
    """
    if not text or not text.strip():
        return {"action": "pass", "reason": "空输出"}

    cfg = _load_domain(domain, domains_dir)
    if cfg is None:
        return {"action": "pass", "reason": "领域配置不可用，跳过禁止断言检查"}

    rules = _safety(cfg)
    violations = [c for c in rules["forbidden_claims"] if c and c in text]
    if violations:
        return {
            "action": "refuse",
            "rule": violations[0],
            "reason": f"输出含禁止断言 {violations}，应整条拒答（不改写）",
        }

    return {
        "action": "pass",
        "disclaimers_required": [d.get("text") for d in rules["disclaimers"] if isinstance(d, dict)],
    }