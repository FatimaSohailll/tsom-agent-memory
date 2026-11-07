"""Active contradiction rate at session checkpoints, rebuilt from a run's fact table.

A contradiction is a key (entity, attribute) holding two or more different values at
once. Attributes that can hold several values are skipped. Each checkpoint also
counts every extracted fact as active. That is what A-Mem keeps, since it never
retires anything, and it uses the same extraction, so the two rates can be compared.

No model is needed, so a finished run can be measured from its SQLite file.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

from tsom.contracts import Fact
from tsom.store import Store
from tsom.temporal.normalize import norm


def _active_at(fact: Fact, step: int, by_id: Dict[str, Fact]) -> bool:
    """Whether `fact` was current at `step`: known by then and not yet replaced."""
    if fact.observed_at > step:
        return False
    replacement = by_id.get(fact.superseded_by)
    return replacement is None or replacement.observed_at > step


def _conflicts(facts: Iterable[Fact], skipped: set) -> int:
    """Keys holding more than one distinct value."""
    values = defaultdict(set)
    for f in facts:
        attribute = norm(f.attribute)
        if attribute not in skipped:
            values[norm(f.entity), attribute].add(norm(f.value))
    return sum(len(held) > 1 for held in values.values())


def _per_100(count: int, facts: int) -> float:
    return 100 * count / facts if facts else 0.0


def curve(store: Store, sample_id: str, multivalued: Iterable[str]) -> List[dict]:
    """One point per session with facts, in order."""
    facts = [f for key in store.entity_keys(sample_id) for f in store.entity_facts(sample_id, key)]
    by_id = {f.fact_id: f for f in facts}
    skipped = {norm(a) for a in multivalued}
    last_step: Dict[Optional[str], int] = {}
    for f in sorted(facts, key=lambda f: f.observed_at):
        last_step[f.observed_time] = f.observed_at
    points = []
    for session, step in last_step.items():
        known = [f for f in facts if f.observed_at <= step]
        active = [f for f in known if _active_at(f, step, by_id)]
        contradictions, without = _conflicts(active, skipped), _conflicts(known, skipped)
        points.append({"session": session, "step": step,
                       "facts": len(active), "contradictions": contradictions,
                       "rate": _per_100(contradictions, len(active)),
                       "all_facts": len(known), "without_consolidation": without,
                       "rate_without_consolidation": _per_100(without, len(known))})
    return points


def read_curve(db_path: Path, sample_id: str, multivalued: Iterable[str]) -> List[dict]:
    if not Path(db_path).exists():
        return []
    store = Store(str(db_path))
    try:
        return curve(store, sample_id, multivalued)
    finally:
        store.close()


def summarise_curves(curves: Dict[str, List[dict]]) -> Optional[dict]:
    """Totals across samples; the final rates are means of each sample's last checkpoint."""
    points = [p for c in curves.values() for p in c]
    if not points:
        return None
    finals = [c[-1] for c in curves.values() if c]
    return {"checkpoints": len(points),
            "checkpoints_lower": sum(p["rate"] < p["rate_without_consolidation"] for p in points),
            "final_rate": mean(p["rate"] for p in finals),
            "final_rate_without_consolidation": mean(p["rate_without_consolidation"] for p in finals),
            "per_sample": curves}


def contradiction_line(summary: Optional[dict]) -> str:
    if summary is None:
        return ""
    return (f"Active contradictions per 100 facts at the last checkpoint: {summary['final_rate']:.2f}, "
            f"against {summary['final_rate_without_consolidation']:.2f} with no consolidation "
            f"(lower at {summary['checkpoints_lower']} of {summary['checkpoints']} checkpoints).")
