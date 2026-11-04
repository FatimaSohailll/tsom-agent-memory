"""Temporal retrieval: A-Mem's notes plus the facts (and their history) behind them."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..contracts import Fact
from .dates import date_key
from .normalize import norm

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
# "may" is also a verb, so it counts only next to a number ("May 2020", "5th May").
_MONTH = re.compile(r"\b(january|february|march|april|june|july|august|september|october|"
                    r"november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b",
                    re.IGNORECASE)
_MAY = re.compile(r"\bmay\s+\d|\d(?:st|nd|rd|th)?\s+may\b", re.IGNORECASE)
_PAST = re.compile(r"\b(in the past|used to|previously|before|back in|at that time|originally)\b",
                   re.IGNORECASE)


def classify_query(query: str) -> Tuple[str, Optional[str]]:
    """("historical" | "current", target_date). target_date is "YYYY" or
    "YYYY-MM" for a historical query that names a year, else None."""
    year = _YEAR.search(query)
    month = _MONTH.search(query)
    month_no = None
    if month:
        month_no = [m[:3] for m in _MONTHS].index(month.group(1).lower()[:3]) + 1
    elif _MAY.search(query):
        month_no = 5
    if not (year or month_no or _PAST.search(query)):
        return "current", None
    if not year:
        return "historical", None
    target = year.group(0)
    return "historical", f"{target}-{month_no:02d}" if month_no else target


def _cmp(a: tuple, b: tuple) -> int:
    k = min(len(a), len(b))
    return (a[:k] > b[:k]) - (a[:k] < b[:k])


def _outside(target: tuple, fact: Fact) -> bool:
    """True only if `target` is provably outside [valid_from, valid_until).
    Unstated or unparseable bounds are open."""
    start, end = date_key(fact.valid_from), date_key(fact.valid_until)
    if start is not None and _cmp(target, start) < 0:
        return True
    if end is not None:
        c = _cmp(target, end)
        if c > 0 or (c == 0 and len(target) == len(end)):  # end is exclusive
            return True
    return False


class TemporalRetriever:
    def __init__(self, memory, max_facts: int = 30):
        self.memory = memory  # a MemorySystem
        self.max_facts = max_facts

    def search(self, query: str, k: int) -> dict:
        amem = self.memory.amem
        store = self.memory.store
        sid = self.memory.sample_id or ""
        query_type, target = classify_query(query)

        notes_str = amem.find_related_memories_raw(query, k=k)
        ids = list(amem.memories.keys())
        indices = amem.retriever.search(query, k) if amem.memories else []
        note_ids = [ids[int(i)] for i in indices if 0 <= int(i) < len(ids)]

        # Facts of the retrieved notes, expanded to their full history; then
        # every fact of any entity the query names (helps multi-hop questions).
        groups: Dict[Tuple[str, str], List[Fact]] = {}
        for f in store.facts_for_notes(sid, note_ids):
            key = (norm(f.entity), norm(f.attribute))
            if key not in groups:
                groups[key] = store.fact_history(sid, f.entity, f.attribute)
        q = query.lower()
        for entity in store.entity_keys(sid):
            if len(entity) >= 2 and re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", q):
                for f in store.entity_facts(sid, entity):
                    groups.setdefault((norm(f.entity), norm(f.attribute)), [])
                    if f.fact_id not in {g.fact_id for g in groups[(norm(f.entity), norm(f.attribute))]}:
                        groups[(norm(f.entity), norm(f.attribute))].append(f)

        lines: List[str] = []
        for history in groups.values():
            history = sorted(history, key=lambda f: f.observed_at)
            lines += [self._format(store, sid, f) for f in self._filter(history, query_type, target)]
        return {"notes": notes_str, "facts": "\n".join(lines[: self.max_facts]),
                "query_type": query_type}

    @staticmethod
    def _filter(history: List[Fact], query_type: str, target: Optional[str]) -> List[Fact]:
        live = [f for f in history if f.status != "retracted"]
        if query_type == "current":
            return [f for f in live if f.status == "active"]
        key = date_key(target) if target else None
        dated = any(date_key(f.valid_from) or date_key(f.valid_until) for f in live)
        if key is None or not dated:
            return live  # no way to place it in time: show the whole history
        return [f for f in live if not _outside(key, f)]

    @staticmethod
    def _format(store, sid: str, f: Fact) -> str:
        if f.status == "active":
            return f"[CURRENT] {f.entity} | {f.attribute} | {f.value} (step {f.observed_at})"
        later = store.get_fact(f.superseded_by) if f.superseded_by else None
        replaced = f", replaced at step {later.observed_at}" if later else ""
        return f"[PAST] {f.entity} | {f.attribute} | {f.value} (step {f.observed_at}{replaced})"
