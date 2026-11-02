"""The fact layer (Gap 1): write-time fact extraction and conflict labelling.

One LLM call per turn extracts facts about the speaker and labels each one
additive, redundant, corrective or contradictory. Plain Python rules then
decide what changes in the store. Nothing is ever deleted.

With use_classifier=False the same extraction call runs but its labels are
ignored ("newest value wins"), so comparing the two modes isolates labelling.
The layer needs only a Store and an LLM, not A-Mem, so it can be evaluated alone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from .contracts import Fact, FactOp
from .llm import LLM
from .schema import PREDICATES, is_single_valued
from .store import Store

log = logging.getLogger(__name__)

LABELS = ("additive", "redundant", "corrective", "contradictory")
NO_VALUE = ("none", "n/a", "na", "null", "-", "")

PROMPT_HEAD = "You extract facts about one speaker from a conversation turn and label each fact."

EXAMPLES = """Example 1
Speaker: Sara
Known facts: none
Turn: I just adopted a puppy!
FACT: has_pet | dog | additive | none

Example 2
Speaker: Sara
Known facts: lives_in: Paris
Turn: Paris has been home for years now.
FACT: lives_in | Paris | redundant | none

Example 3
Speaker: Sara
Known facts: works_at: Google
Turn: I left Google last month and joined Meta.
FACT: works_at | Meta | corrective | Google

Example 4
Speaker: Sara
Known facts: lives_in: Boston
Turn: The winters here in Denver are rough.
FACT: lives_in | Denver | contradictory | none

Example 5
Speaker: Sara
Known facts: likes: coffee; likes: jazz
Turn: I don't drink coffee anymore.
FACT: likes | none | corrective | coffee

Example 6
Speaker: Sara
Known facts: likes: jazz
Turn: Haha, see you tomorrow then!
NONE"""


@dataclass
class ParsedFact:
    predicate: str
    value: Optional[str]  # None for a pure negation ("no longer ...")
    label: str
    old: Optional[str]


def norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _clean(field: str) -> str:
    return field.strip().strip("\"'`").strip().rstrip(".,;:!").strip()


def parse_output(text: str) -> tuple[List[ParsedFact], bool]:
    """Parse the model's lines leniently. Returns (facts, said_none)."""
    facts, said_none = [], False
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if _clean(line).lower() == "none":
            said_none = True
            continue
        m = re.match(r"(?i)^fact\s*:\s*(.*)$", line)
        if not m:
            continue
        parts = [_clean(p) for p in m.group(1).split("|")]
        if len(parts) == 3:
            parts.append("none")
        if len(parts) != 4:
            continue
        predicate = re.sub(r"\s+", "_", parts[0].lower())
        label = parts[2].lower()
        if predicate not in PREDICATES or label not in LABELS:
            log.debug("dropped fact line %r", line)
            continue
        value = None if parts[1].lower() in NO_VALUE else parts[1]
        old = None if parts[3].lower() in NO_VALUE else parts[3]
        facts.append(ParsedFact(predicate, value, label, old))
    return facts, said_none


class FactLayer:
    def __init__(self, store: Store, llm: LLM, use_classifier: bool):
        self.store = store
        self.llm = llm
        self.use_classifier = use_classifier
        self.parse_failures = 0  # running total; callers diff it per write
        self.last_raw: List[str] = []  # raw model outputs for the last turn (for inspection)

    # ---- extraction ----

    def build_prompt(self, speaker: str, text: str, session_date: str) -> str:
        active = self.store.active_facts(speaker)
        known = "\n".join(f"- {f.predicate}: {f.object}" for f in active) or "- none"
        predicates = ", ".join(f"{p} ({kind})" for p, kind in PREDICATES.items())
        return f"""{PROMPT_HEAD}

Allowed predicates (single = one value at a time, multi = many values at once):
{predicates}

Labels:
additive = new information that adds to what is known (a new predicate, or an extra value of a multi-valued predicate)
redundant = restates a value that is already active
corrective = the speaker says an earlier value is no longer true or was replaced ("moved", "switched", "no longer", "used to")
contradictory = conflicts with an active single-valued value but gives NO sign that anything changed

Rules:
- Only facts about {speaker} themself that this turn states. Keep values short.
- For corrective, put the earlier value in the last field. Otherwise write none.
- If something is simply no longer true and there is no new value, write none as the value.
- Output one line per fact in exactly this format and nothing else:
FACT: <predicate> | <value> | <label> | <old value or none>
- If the turn states no such facts, output the single line NONE.

{EXAMPLES}

Now the real turn.
Speaker: {speaker}
Session date: {session_date}
Known facts:
{known}
Turn: {text}
"""

    def extract(self, speaker: str, text: str, session_date: str) -> List[ParsedFact]:
        prompt = self.build_prompt(speaker, text, session_date)
        self.last_raw = []
        for attempt in range(2):
            if attempt == 1:
                prompt += "\nYour previous answer did not follow the format. Output only FACT lines or NONE.\n"
            try:
                raw = self.llm.get_completion(prompt, temperature=0.0)
            except Exception as e:  # a failed call must not crash a run
                log.warning("fact extraction call failed: %s", e)
                raw = ""
            self.last_raw.append(raw)
            facts, said_none = parse_output(raw)
            if facts or said_none:
                return facts
        log.warning("could not parse fact extraction output: %r", self.last_raw[-1][:200])
        self.parse_failures += 1
        return []

    # ---- rules ----

    def process_turn(self, speaker: str, text: str, session_date: str,
                     source_note_id: Optional[str] = None) -> List[FactOp]:
        ops = []
        for pf in self.extract(speaker, text, session_date):
            op = self._apply(speaker, pf, session_date, source_note_id)
            if op is not None:
                ops.append(op)
        return ops

    def _apply(self, speaker: str, pf: ParsedFact, date: str,
               source_note_id: Optional[str]) -> Optional[FactOp]:
        active = self.store.active_facts(speaker, pf.predicate)
        single = is_single_valued(pf.predicate)
        matching_old = next((f for f in active if pf.old and norm(f.object) == norm(pf.old)), None)

        def op(label, fact=None, retired=None):
            return FactOp(predicate=pf.predicate, value=pf.value or "none", raw_label=pf.label,
                          label=label, fact_id=fact.id if fact else None,
                          retired_id=retired.id if retired else None,
                          conflict_with=fact.conflict_with if fact else None)

        # Pure negation: retire only the active fact equal to the old value.
        if pf.value is None:
            if not self.use_classifier or pf.label != "corrective" or matching_old is None:
                return None
            self.store.retire(matching_old.id, date)
            return op("corrective", retired=matching_old)

        def new_fact(**kw):
            return Fact(subject=speaker, predicate=pf.predicate, object=pf.value,
                        valid_from=date, source_note_id=source_note_id, **kw)

        # Safety net: an already active value is redundant whatever the model said.
        if any(norm(f.object) == norm(pf.value) for f in active):
            return op("redundant")

        label = pf.label if self.use_classifier else ("corrective" if single and active else "additive")

        if label == "redundant":
            return op("redundant")
        if label == "additive" and single and active:
            label = "contradictory"  # a second single value with no sign of change
        if label == "contradictory" and not (single and active):
            label = "additive"
        if label == "corrective":
            if single:
                old = matching_old or (active[-1] if active else None)  # else newest inserted
            else:
                old = matching_old  # never retire anything but the named value
            if old is None:
                label = "additive"
            else:
                fact = new_fact()
                self.store.supersede(old.id, fact)
                return op("corrective", fact, old)
        if label == "contradictory":
            other = matching_old or active[-1]
            fact = new_fact(conflict_with=other.id)
            self.store.add_fact(fact)
            return op("contradictory", fact)
        fact = new_fact()
        self.store.add_fact(fact)
        return op("additive", fact)
