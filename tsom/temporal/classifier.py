"""Decide how a newly extracted fact relates to what memory already holds."""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from ..contracts import Fact
from .dates import date_key, is_before
from .normalize import norm
from .parsers import parse_update_classification
from .prompts import UPDATE_CLASSIFY_PROMPT


class UpdateClassifier:
    def __init__(self, llm, store, temperature: float = 0.0):
        self.llm = llm  # anything with get_completion(prompt, temperature) -> str
        self.store = store
        self.temperature = temperature
        self.op_counts: Counter = Counter()

    def classify(self, sample_id: str, new: dict, observed_at: int) -> Tuple[str, List[Fact]]:
        """Return (op, old_facts) for `new`, a dict with entity, attribute, value,
        optional valid_from, and optional source_text (the sentence it came from).

        op is ADD, REAFFIRM, REPLACE, CORRECT or ADD_HISTORICAL. old_facts are
        the existing facts the op acts on: [] for ADD, the matching fact for
        REAFFIRM, all current facts for key for REPLACE, CORRECT, ADD_HISTORICAL.
        An LLM call is made only when the value differs and no date decides it.
        """
        op, old = self._classify(sample_id, new, observed_at)
        self.op_counts[op] += 1
        return op, old

    def _classify(self, sample_id, new, observed_at):
        old = self.store.active_facts(sample_id, new["entity"], new["attribute"])
        if not old:
            return "ADD", []
        for f in old:
            if norm(f.value) == norm(new["value"]):
                return "REAFFIRM", [f]
        # Out of order: new fact starts before every existing one (all dates known).
        new_from = new.get("valid_from")
        if (date_key(new_from) is not None
                and all(date_key(f.valid_from) is not None for f in old)
                and all(is_before(new_from, f.valid_from) for f in old)):
            return "ADD_HISTORICAL", old
        prompt = UPDATE_CLASSIFY_PROMPT.format(
            old_lines="\n  ".join(
                f"{f.entity} | {f.attribute} | {f.value} (learned at step {f.observed_at})"
                for f in old),
            entity=new["entity"], attribute=new["attribute"], new_value=new["value"],
            new_observed=observed_at, extra_context=new.get("source_text") or "")
        op, _reason = parse_update_classification(
            self.llm.get_completion(prompt, self.temperature))
        return op, ([] if op == "ADD" else old)
