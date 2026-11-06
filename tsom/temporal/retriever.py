"""Temporal retrieval: A-Mem's notes plus the facts (and their history) behind them."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..contracts import Fact
from .dates import date_key, effective_start
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


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    return vector / length if length else vector


def _cmp(a: tuple, b: tuple) -> int:
    k = min(len(a), len(b))
    return (a[:k] > b[:k]) - (a[:k] < b[:k])


def _bounds(fact: Fact, by_id: Dict[str, Fact]) -> Tuple[Optional[tuple], Optional[tuple]]:
    """When `fact` held: its stated dates, else the moment a replacement took over.
    A session date only bounds a replacement, and is only compared, never stored.
    A first value has no invented start: learning it in 2023 does not mean it began then."""
    start, end = date_key(fact.valid_from), date_key(fact.valid_until)
    if start is None and fact.supersedes:
        start = effective_start(fact)
    if end is None and fact.superseded_by in by_id:
        end = effective_start(by_id[fact.superseded_by])
    return start, end


def _outside(target: tuple, start: Optional[tuple], end: Optional[tuple]) -> bool:
    """True only if `target` is provably outside [start, end).
    Unknown bounds are open."""
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
        self._vectors: Dict[str, np.ndarray] = {}  # fact_id -> unit embedding

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
        named: Set[Tuple[str, str]] = set()
        for entity in store.entity_keys(sid):
            if len(entity) >= 2 and re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", q):
                for f in store.entity_facts(sid, entity):
                    key = (norm(f.entity), norm(f.attribute))
                    named.add(key)
                    groups.setdefault(key, [])
                    if f.fact_id not in {g.fact_id for g in groups[key]}:
                        groups[key].append(f)

        shown: Dict[Tuple[str, str], List[Fact]] = {}
        for key, history in groups.items():
            kept = self._filter(sorted(history, key=lambda f: f.observed_at), query_type, target)
            if kept:
                shown[key] = kept
        lines = {key: [self._format(store, sid, f) for f in facts] for key, facts in shown.items()}
        if sum(map(len, lines.values())) > self.max_facts:
            kept_lines = self._select(lines, self._scores(query, shown), named, self.max_facts)
        else:
            kept_lines = [line for group in lines.values() for line in group]
        return {"notes": notes_str, "facts": "\n".join(kept_lines), "query_type": query_type}

    def history(self, entity: str) -> str:
        """Every fact held about `entity`, oldest first, without retracted ones."""
        store, sid = self.memory.store, self.memory.sample_id or ""
        facts = [f for f in store.entity_facts(sid, entity) if f.status != "retracted"]
        return "\n".join(self._format(store, sid, f) for f in facts[-self.max_facts:])

    def _scores(self, query: str, shown: Dict[Tuple[str, str], List[Fact]]) -> Dict[Tuple[str, str], float]:
        """Each group's best cosine similarity to the query, with the embedder A-Mem uses."""
        encode = self.memory.amem.retriever.model.encode
        missing = [f for facts in shown.values() for f in facts if f.fact_id not in self._vectors]
        if missing:
            texts = [f"{f.entity} {f.attribute} {f.value}".replace("_", " ") for f in missing]
            for f, vector in zip(missing, encode(texts)):
                self._vectors[f.fact_id] = _unit(vector)
        target = _unit(encode([query])[0])
        return {key: max(float(self._vectors[f.fact_id] @ target) for f in facts)
                for key, facts in shown.items()}

    @staticmethod
    def _select(lines: Dict[Tuple[str, str], List[str]], scores: Dict[Tuple[str, str], float],
                named: Set[Tuple[str, str]], limit: int) -> List[str]:
        """The best-scoring groups' lines, at most `limit`; a named entity wins a tie.
        A group that no longer fits keeps its newest lines."""
        kept: List[str] = []
        for key in sorted(lines, key=lambda key: (-scores[key], key not in named)):
            room = limit - len(kept)
            if room <= 0:
                break
            kept += lines[key][-room:]
        return kept

    @staticmethod
    def _filter(history: List[Fact], query_type: str, target: Optional[str]) -> List[Fact]:
        live = [f for f in history if f.status != "retracted"]
        if query_type == "current":
            return [f for f in live if f.status == "active"]
        key = date_key(target) if target else None
        by_id = {f.fact_id: f for f in history}
        bounds = {f.fact_id: _bounds(f, by_id) for f in live}
        if key is None or not any(start or end for start, end in bounds.values()):
            return live  # no way to place it in time: show the whole history
        return [f for f in live if not _outside(key, *bounds[f.fact_id])]

    @staticmethod
    def _format(store, sid: str, f: Fact) -> str:
        if f.status == "active":
            return f"[CURRENT] {f.entity} | {f.attribute} | {f.value} (step {f.observed_at})"
        later = store.get_fact(f.superseded_by) if f.superseded_by else None
        if later is None:
            note = ""
        elif f.fact_id in later.supersedes:
            note = f", replaced at step {later.observed_at}"
        else:  # learned late: it ended before `later`, which never replaced it
            note = f", earlier than the value from step {later.observed_at}"
        return f"[PAST] {f.entity} | {f.attribute} | {f.value} (step {f.observed_at}{note})"
