"""Decide how a newly extracted fact relates to what memory already holds."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from ..contracts import Fact
from .dates import date_key, effective_start, is_before, key_is_before
from .normalize import norm
from .parsers import parse_update_classification
from .prompts import UPDATE_CLASSIFY_LLM_ONLY_PROMPT, UPDATE_CLASSIFY_PROMPT

log = logging.getLogger("tsom.classifier")

RULE_PAST_BEFORE_EXISTING = "rule: past-tense fact starting before the existing fact"
RULE_VALID_FROM_EARLIER = "rule: valid_from is earlier than every existing valid_from"

DEFAULT_MULTIVALUED = ("speaks", "languages", "children", "siblings", "hobbies", "member_of")
# The sentence reads as an addition, and nothing in it says the value changed.
_ADDITIVE = re.compile(r"\b(also|as well|too|in addition)\b", re.IGNORECASE)
_CHANGE = re.compile(r"\b(now|moved|switched|no longer|instead|changed|left|formerly)\b",
                     re.IGNORECASE)


def _stated(value: Optional[str]) -> str:
    return value if value else "not stated"


class UpdateClassifier:
    def __init__(self, llm, store, temperature: float = 0.0, seed_multivalued: bool = False,
                 additive_cue: bool = False,
                 multivalued_attributes: Sequence[str] = DEFAULT_MULTIVALUED,
                 mode: str = "rules_llm"):
        if mode not in ("rules_llm", "llm_only"):
            raise ValueError(f"unknown classifier mode: {mode!r}")
        self.mode = mode
        self.llm = llm  # anything with get_completion(prompt, temperature) -> str
        self.store = store
        self.temperature = temperature
        self.seed_multivalued = seed_multivalued
        self.additive_cue = additive_cue
        self.multivalued = {norm(a) for a in multivalued_attributes}
        self.op_counts: Counter = Counter()
        # How the latest classify() call was decided: {"op", "source", "reason", "raw"} where
        # source is the rule's name (no_existing_fact, same_value, valid_from_earlier,
        # past_before_existing, learned_multivalued, seed_multivalued, additive_cue) or "LLM",
        # and raw is the LLM response (None when a rule decided).
        self.last_decision: dict = {}

    def classify(self, sample_id: str, new: dict, observed_at: int,
                 observed_time: Optional[str] = None) -> Tuple[str, List[Fact]]:
        """Return (op, old_facts) for `new`, a dict with entity, attribute, value,
        optional valid_from, temporal_status and source_text (the sentence it came
        from). `observed_time` is the session date the new fact was learned in.

        op is ADD, REAFFIRM, REPLACE, CORRECT or ADD_HISTORICAL. old_facts are
        the existing facts the op touches: [] for a first ADD; for an ADD next to
        existing facts they are those facts, which stay active; the matching fact
        for REAFFIRM; all current facts for the key for REPLACE, CORRECT and
        ADD_HISTORICAL. Deterministic rules decide first (no LLM call); the LLM is
        asked only for a differing value that no rule settles.
        self.last_decision says which.
        """
        op, old, source, reason, raw = self._classify(sample_id, new, observed_at, observed_time)
        self.last_decision = {"op": op, "source": source, "reason": reason, "raw": raw}
        self.op_counts[op] += 1
        return op, old

    def _classify(self, sample_id, new, observed_at, observed_time):
        old = self.store.active_facts(sample_id, new["entity"], new["attribute"])
        if not old:
            return "ADD", [], "no_existing_fact", "rule: no existing fact for this entity and attribute", None
        for f in old:
            if norm(f.value) == norm(new["value"]):
                return "REAFFIRM", [f], "same_value", "rule: same value already stored", None
        if self.mode == "llm_only":
            result = self._ask_llm_only(old, new, observed_at, observed_time)
            if result is not None:
                return result
            # Two unparseable replies: fall back to the rule-based path rather than guess.
            op, old, source, reason, raw = self._classify_rules(old, new, observed_at, observed_time)
            return op, old, f"fallback:{source}", reason, raw
        return self._classify_rules(old, new, observed_at, observed_time)

    def _ask_llm_only(self, old, new, observed_at, observed_time):
        """Ask the model (retrying once on unparseable output). None if it never answers."""
        prompt = UPDATE_CLASSIFY_LLM_ONLY_PROMPT.format(
            **self._prompt_fields(old, new, observed_at, observed_time))
        for _ in range(2):
            raw = self.llm.get_completion(prompt, self.temperature)
            log.debug("classifier (llm_only) raw response for %s | %s | %s: %r",
                      new["entity"], new["attribute"], new["value"], raw)
            op, reason = parse_update_classification(
                raw, allowed=("REPLACE", "CORRECT", "ADD", "ADD_HISTORICAL"), default=None)
            if op is not None:
                return op, old, "LLM", reason or "(no reason given)", raw
        return None

    @staticmethod
    def _prompt_fields(old, new, observed_at, observed_time) -> dict:
        return dict(
            old_lines="\n  ".join(
                f"{f.entity} | {f.attribute} | {f.value} (learned at step {f.observed_at}, "
                f"session {f.observed_time or 'unknown'}, valid from: {_stated(f.valid_from)}, "
                f"valid until: {_stated(f.valid_until)})" for f in old),
            entity=new["entity"], attribute=new["attribute"], new_value=new["value"],
            new_observed=observed_at, new_session=observed_time or "unknown",
            new_from=_stated(new.get("valid_from")), new_until=_stated(new.get("valid_until")),
            new_tense=new.get("temporal_status") or "UNKNOWN",
            extra_context=new.get("source_text") or "")

    def _classify_rules(self, old, new, observed_at, observed_time):

        new_from = new.get("valid_from")
        new_key = date_key(new_from)
        # Both valid_from known: the new fact starts before every existing one.
        if (new_key is not None and all(date_key(f.valid_from) is not None for f in old)
                and all(is_before(new_from, f.valid_from) for f in old)):
            return "ADD_HISTORICAL", old, "valid_from_earlier", RULE_VALID_FROM_EARLIER, None
        # Past-tense wording with a stated start that precedes when the existing
        # fact is known to have begun (its valid_from, else its session date).
        # Tense alone never fires this; PRESENT and UNKNOWN never do.
        if new.get("temporal_status") == "PAST" and new_key is not None:
            starts = [effective_start(f) for f in old]
            if all(s is not None for s in starts) and all(key_is_before(new_key, s) for s in starts):
                return "ADD_HISTORICAL", old, "past_before_existing", RULE_PAST_BEFORE_EXISTING, None

        # Several values at once. The first rule needs no flag: it learns from the store.
        if len(old) >= 2:
            return "ADD", old, "learned_multivalued", "rule: key already holds several active values", None
        if self.seed_multivalued and norm(new["attribute"]) in self.multivalued:
            return "ADD", old, "seed_multivalued", \
                f"rule: {new['attribute']!r} is on the multi-valued list", None
        sentence = new.get("source_text") or ""
        if self.additive_cue and _ADDITIVE.search(sentence) and not _CHANGE.search(sentence):
            return "ADD", old, "additive_cue", \
                "rule: the sentence adds a value and says nothing changed", None

        prompt = UPDATE_CLASSIFY_PROMPT.format(
            **self._prompt_fields(old, new, observed_at, observed_time))
        raw = self.llm.get_completion(prompt, self.temperature)
        log.debug("classifier raw response for %s | %s | %s: %r",
                  new["entity"], new["attribute"], new["value"], raw)
        op, reason = parse_update_classification(raw)
        return op, old, "LLM", reason or "(no reason given)", raw
