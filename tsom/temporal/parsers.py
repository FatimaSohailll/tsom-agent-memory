"""Defensive parser for fact-extraction output ("KEY: value" blocks).

Follows the style of third_party/amem/llm_text_parsers.py: never raise on bad
model output, skip what cannot be read, fall back to an empty result.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

_KEYS = ("ENTITY", "ATTRIBUTE", "VALUE", "VALID_FROM", "VALID_UNTIL", "SOURCE")
_LINE = re.compile(
    r"^[\s*\-•>#]*(?:\d+[.)]\s+)?[*_`]*(FACT\s*\d*|" + "|".join(_KEYS) + r")[*_`]*\s*:[*_`]*\s*(.*)$",
    re.IGNORECASE,
)
_NONE = {"", "none", "null", "n/a", "na", "unknown", "nil", "-"}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip("*_`").strip())


def _blocks(response: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    cur: Dict[str, str] = {}
    for line in (response or "").splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        key, val = m.group(1).upper(), _clean(m.group(2))
        if key.startswith("FACT"):  # header: start a new block
            if cur:
                blocks.append(cur)
            cur = {}
            continue
        if key in cur:  # repeated key: the header was missing
            blocks.append(cur)
            cur = {}
        cur[key] = val
    if cur:
        blocks.append(cur)
    return blocks


def _opt(value: Optional[str]) -> Optional[str]:
    if value is None or value.lower() in _NONE:
        return None
    return value


def parse_facts(response: str, with_source: bool = False) -> List[dict]:
    """Parse extractor output into dicts with entity, attribute, value,
    valid_from, valid_until (and "source": int | None when with_source).

    Blocks missing ENTITY, ATTRIBUTE or VALUE are skipped; NONE becomes None
    for the validity fields. Returns [] for NO_FACTS or unreadable output.
    """
    facts: List[dict] = []
    try:
        for b in _blocks(response):
            entity, attribute, value = b.get("ENTITY", ""), b.get("ATTRIBUTE", ""), b.get("VALUE", "")
            if not (entity and attribute and value):
                continue
            fact = {
                "entity": entity,
                "attribute": attribute,
                "value": value,
                "valid_from": _opt(b.get("VALID_FROM")),
                "valid_until": _opt(b.get("VALID_UNTIL")),
            }
            if with_source:
                m = re.search(r"\d+", b.get("SOURCE", ""))
                fact["source"] = int(m.group()) if m else None
            facts.append(fact)
    except Exception:
        return []
    return facts
