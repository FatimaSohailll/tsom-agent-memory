"""Change scenarios: the data model, the loader and the answer scorer.

A scenario is a short multi-session dialogue in which one fact about a person
changes (or is added to, restated, or contradicted). run_scenarios reads them
through load_scenarios.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIOS = "data/scenarios/change_scenarios.jsonl"
DRAFTS = "data/scenarios/drafts.jsonl"

Kind = Literal["change", "cancel", "additive", "restate", "conflict"]
HAS_OLD = ("change", "cancel")  # kinds where an earlier value stops being true
Label = Literal["additive", "corrective", "redundant", "contradictory"]


class Turn(BaseModel):
    speaker: str
    text: str


class Session(BaseModel):
    date: str  # LoCoMo style, e.g. "6:30 pm on 14 June, 2023"
    turns: List[Turn] = Field(min_length=1)


class ScenarioFact(BaseModel):
    speaker: str
    predicate: str
    value: str
    session: int
    expected_label: Label
    replaces: Optional[str] = None


class Scenario(BaseModel):
    id: str
    kind: Kind
    reviewed: bool = False
    sessions: List[Session] = Field(min_length=2, max_length=4)
    facts: List[ScenarioFact] = Field(min_length=1)
    question: str
    question_alt: str
    current: List[str] = Field(min_length=1)
    old: List[str] = []

    @model_validator(mode="after")
    def _consistent(self):
        for f in self.facts:
            if not 0 <= f.session < len(self.sessions):
                raise ValueError(f"{self.id}: fact session {f.session} out of range")
        if any(a != a.lower() for a in self.current + self.old):
            raise ValueError(f"{self.id}: current/old answers must be lowercase")
        if self.kind in HAS_OLD and not self.old:
            raise ValueError(f"{self.id}: a {self.kind} scenario needs old answers")
        if self.kind not in HAS_OLD and self.old:
            raise ValueError(f"{self.id}: only change and cancel scenarios have old answers")
        return self


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path


def read_jsonl(path: str | Path) -> List[Scenario]:
    path = resolve(path)
    if not path.exists():
        return []
    return [Scenario.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: str | Path, scenarios: List[Scenario]):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(s.model_dump_json() + "\n" for s in scenarios), encoding="utf-8")


def load_scenarios(path: str | Path = DEFAULT_SCENARIOS, include_unreviewed: bool = False) -> List[Scenario]:
    """Reviewed scenarios only, unless include_unreviewed (for development)."""
    return [s for s in read_jsonl(path) if s.reviewed or include_unreviewed]


def answered(scenarios: List[Scenario], limit: Optional[int] = None) -> List[Scenario]:
    """The scenarios that get questions (conflict ones are skipped). With a limit, the kinds share it
    equally, taking each kind's scenarios in file order, so a small run still covers every kind;
    a kind with too few scenarios leaves its share to the others. The order of the file is kept."""
    kept = [s for s in scenarios if s.kind != "conflict"]
    if limit is None or limit >= len(kept):
        return kept
    by_kind: dict = {}
    for s in kept:
        by_kind.setdefault(s.kind, []).append(s)
    chosen, rounds = set(), 0
    while len(chosen) < limit:
        for group in by_kind.values():
            if rounds < len(group) and len(chosen) < limit:
                chosen.add(group[rounds].id)
        rounds += 1
    return [s for s in kept if s.id in chosen]


# ---- answer scoring ----

_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def normalise(text: str) -> str:
    return " ".join(str(text).lower().translate(_PUNCT).split())


def _has(answer: str, alias: str) -> bool:
    alias = normalise(alias)
    return bool(alias) and alias in answer


def first_session_values(scenario: Scenario) -> List[str]:
    """Values stated in session 1 that are still expected in the answer (for retention)."""
    current = {normalise(c) for c in scenario.current}
    return [f.value for f in scenario.facts if f.session == 0 and normalise(f.value) in current]


def score_answer(answer: str, scenario: Scenario) -> dict:
    """Substring matching on normalised text (lowercase, punctuation removed).

    correct: contains a current alias (additive: every current value).
    stale: an old alias and no current alias. hedged: both.
    verdict is exclusive (correct, hedged, stale or wrong) and is what
    consistency compares. retention is only set for additive and restate.
    """
    a = normalise(answer)
    if scenario.kind == "additive":
        any_current = all(_has(a, c) for c in scenario.current)
    else:
        any_current = any(_has(a, c) for c in scenario.current)
    any_old = any(_has(a, o) for o in scenario.old)
    verdict = ("hedged" if any_current and any_old else "correct" if any_current
               else "stale" if any_old else "wrong")
    retention = None
    if scenario.kind in ("additive", "restate"):
        retention = all(_has(a, v) for v in first_session_values(scenario))
    return {"correct": any_current, "stale": any_old and not any_current,
            "hedged": any_current and any_old, "verdict": verdict, "retention": retention}


# ---- reporting helpers ----

def kn(k: int, n: int) -> dict:
    return {"k": k, "n": n, "pct": round(100.0 * k / n, 1) if n else None}


def fmt(c: Optional[dict]) -> str:
    if not c or not c["n"]:
        return "–"
    return f"{c['k']}/{c['n']} ({c['pct']:.1f}%)"


def save_json(path: Path, data):
    """Write via a temp file so an interrupted run never leaves half a result."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
