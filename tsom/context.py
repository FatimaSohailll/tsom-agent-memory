"""Retrieval context with validity annotations (Gap 3).

retrieve_blocks copies A-Mem's find_related_memories_raw loop exactly, but
keeps which note each block came from, so joining the blocks gives A-Mem's
string byte for byte. fact_context then builds a "Current facts" block and
per-note OUTDATED markers from the fact store; render puts them together.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .contracts import Fact
from .schema import is_single_valued
from .store import Store

MAX_FACT_LINES = 15


def _block(note) -> str:
    return ("talk start time:" + note.timestamp +
            "memory content: " + note.content +
            "memory context: " + note.context +
            "memory keywords: " + str(note.keywords) +
            "memory tags: " + str(note.tags) + "\n")


def retrieve_blocks(amem, query: str, k: int) -> List[Tuple[str, str]]:
    """[(note_id, block_text)] in A-Mem's order, quirks included.

    Like find_related_memories_raw: the neighbour counter j restarts for every
    hit, neighbours are not de-duplicated, and the break comes after the append,
    so up to k + 1 neighbours are listed per hit.
    """
    if not amem.memories:
        return []
    indices = amem.retriever.search(query, k)
    all_memories = list(amem.memories.values())
    blocks = []
    for i in indices:
        j = 0
        blocks.append((all_memories[i].id, _block(all_memories[i])))
        for neighbor in all_memories[i].links:
            blocks.append((all_memories[neighbor].id, _block(all_memories[neighbor])))
            if j >= k:
                break
            j += 1
    return blocks


def _mentioned(subjects: List[str], text: str) -> List[str]:
    return [s for s in subjects if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text, re.IGNORECASE)]


def fact_context(store: Store, note_ids: List[str], text: str) -> dict:
    """Current-fact lines and per-note outdated markers for one search.

    note_ids are the retrieved notes (hits and neighbours); text is the question
    (or the query) used to find people mentioned by name.
    """
    attached = [f for f in store.facts_by_source(list(dict.fromkeys(note_ids))) if f.status != "rejected"]
    speakers = list(dict.fromkeys(
        _mentioned(store.subjects(), text) + [f.subject for f in attached]))
    facts = [f for f in store.facts_of(speakers) if f.status != "rejected"]
    by_id = {f.id: f for f in facts}
    order = {f.id: n for n, f in enumerate(facts)}  # insertion order
    replaced_by = {f.supersedes: f for f in facts if f.supersedes}
    retrieved = set(note_ids)
    speaker_rank = {s.lower(): n for n, s in enumerate(speakers)}

    def previous(fact: Fact) -> List[Fact]:
        chain, seen = [], set()
        while fact.supersedes and fact.supersedes in by_id and fact.supersedes not in seen:
            seen.add(fact.supersedes)
            fact = by_id[fact.supersedes]
            chain.append(fact)
        return chain

    def conflict_partner(fact: Fact) -> Optional[Fact]:
        other = by_id.get(fact.conflict_with) if fact.conflict_with else None
        if other is None:  # the older side of a conflict
            other = next((f for f in facts if f.conflict_with == fact.id and f.status == "active"), None)
        return other if other is not None and other.status == "active" else None

    # ---- current facts, one group per (speaker, predicate) ----
    groups: Dict[Tuple[str, str], List[Fact]] = {}
    for f in facts:
        if f.status == "active":
            groups.setdefault((f.subject.lower(), f.predicate), []).append(f)

    current = []
    for (_, predicate), active in groups.items():
        subject = active[0].subject
        related = active + [p for f in active for p in previous(f)]
        values = []
        for f in active:
            entry = {"fact": f.model_dump(), "previous": [], "conflicts_with": None}
            if is_single_valued(predicate):
                details = [f"since {f.valid_from}"]
                prev = previous(f)
                if prev:
                    details.append("previously " + ", ".join(f"{p.object} until {p.valid_to}" for p in prev))
                    entry["previous"] = [p.model_dump() for p in prev]
                other = conflict_partner(f)
                if other is not None:
                    details.append(f"unconfirmed: conflicts with {other.object}")
                    entry["conflicts_with"] = other.object
                values.append((f"{f.object} ({'; '.join(details)})", entry))
            else:
                values.append((f.object, entry))
        current.append({
            "subject": subject, "predicate": predicate, "kind": "active",
            "line": f"- {subject} {predicate} " + ", ".join(v for v, _ in values),
            "values": [e for _, e in values],
            "_attached": any(f.source_note_id in retrieved for f in related),
            "_newest": max(order[f.id] for f in related),
            "_rank": (speaker_rank[subject.lower()], min(order[f.id] for f in related)),
        })

    if len(current) > MAX_FACT_LINES:
        # Keep groups tied to retrieved notes, then the newest; corrections are newest.
        current.sort(key=lambda g: (not g["_attached"], -g["_newest"]))
        current = current[:MAX_FACT_LINES]
    current.sort(key=lambda g: g["_rank"])

    # Retired with no replacement: "no longer true", after the active lines.
    active_values = {(f.subject.lower(), f.predicate, f.object.strip().lower())
                     for f in facts if f.status == "active"}
    negated = [f for f in facts if f.status == "retired" and f.id not in replaced_by
               and (f.subject.lower(), f.predicate, f.object.strip().lower()) not in active_values]
    for f in negated[::-1][:max(0, MAX_FACT_LINES - len(current))][::-1]:
        current.append({"subject": f.subject, "predicate": f.predicate, "kind": "no_longer",
                        "line": f"- {f.subject} {f.predicate} {f.object} (no longer true since {f.valid_to})",
                        "values": [{"fact": f.model_dump(), "previous": [], "conflicts_with": None}]})
    for g in current:
        for key in ("_attached", "_newest", "_rank"):
            g.pop(key, None)

    # ---- outdated markers per retrieved note ----
    outdated: Dict[str, List[dict]] = {}
    for f in attached:
        if f.status != "retired":
            continue
        new = replaced_by.get(f.id)
        if new is not None:
            line = f"[OUTDATED as of {f.valid_to}: {f.predicate} {f.object} was replaced by {new.object}]"
        else:
            line = f"[OUTDATED as of {f.valid_to}: {f.predicate} {f.object} is no longer true]"
        outdated.setdefault(f.source_note_id, []).append({
            "note_id": f.source_note_id, "fact": f.model_dump(),
            "replaced_by": new.object if new is not None else None, "line": line,
        })

    return {"current": current, "outdated": outdated}


def render(blocks: List[Tuple[str, str]], ctx: dict) -> str:
    """The annotated context string: current facts first, then notes with markers."""
    notes = ""
    for note_id, text in blocks:
        notes += text
        for marker in ctx["outdated"].get(note_id, []):
            notes += marker["line"] + "\n"
    if not ctx["current"]:
        return notes
    return "Current facts:\n" + "".join(g["line"] + "\n" for g in ctx["current"]) + "\n" + notes
