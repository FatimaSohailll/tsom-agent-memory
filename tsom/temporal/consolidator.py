"""Apply a classified fact update to the store. Append-only: rows are never
deleted or overwritten; superseded/retracted rows stay and point to the new one."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..contracts import Fact

log = logging.getLogger("tsom.consolidator")


class Consolidator:
    def __init__(self, store):
        self.store = store

    def apply(self, sample_id: str, note_id: str, new: dict, op: str, old_facts: List[Fact],
              observed_at: int, observed_time: Optional[str]) -> Fact:
        """Apply `op` and return the fact now live for this key (ADD, REPLACE,
        CORRECT), the reaffirmed fact (REAFFIRM), or the inserted, already
        superseded fact (ADD_HISTORICAL)."""
        if op == "REAFFIRM":
            if not old_facts:
                raise ValueError("REAFFIRM needs the fact being reaffirmed")
            cur = self.store.get_fact(old_facts[0].fact_id) or old_facts[0]
            self.store.update_fact_status(cur.fact_id, cur.status,
                                          observation_count=cur.observation_count + 1)
            return self.store.get_fact(cur.fact_id)

        if op in ("REPLACE", "CORRECT") and not old_facts:
            log.warning("%s for %s | %s | %s found no old facts; falling back to ADD",
                        op, new["entity"], new["attribute"], new["value"])
            op = "ADD"  # nothing to replace or correct
        if op not in ("ADD", "REPLACE", "CORRECT", "ADD_HISTORICAL"):
            raise ValueError(f"unknown operation: {op!r}")
        if op == "ADD_HISTORICAL" and not old_facts:
            raise ValueError("ADD_HISTORICAL needs the existing fact it precedes")

        fields = dict(sample_id=sample_id, note_id=note_id, entity=new["entity"],
                      attribute=new["attribute"], value=new["value"],
                      observed_at=observed_at, observed_time=observed_time,
                      valid_from=new.get("valid_from"), valid_until=new.get("valid_until"))

        if op == "ADD":
            fact = Fact(**fields, operation="ADD")
            self.store.add_fact(fact)
            return fact

        if op == "ADD_HISTORICAL":
            first = old_facts[0]
            # a stated end date wins; otherwise it ends where the existing fact begins
            fields["valid_until"] = fields["valid_until"] or first.valid_from
            fact = Fact(**fields, operation="ADD", status="superseded",
                        superseded_by=first.fact_id)
            self.store.add_fact(fact)
            return fact

        fact = Fact(**fields, operation=op, supersedes=[f.fact_id for f in old_facts])
        self.store.add_fact(fact)
        for old in old_facts:
            if op == "REPLACE":
                # valid_until = new.valid_from; stays NULL if unstated, and a
                # date the old fact already states is kept
                self.store.update_fact_status(
                    old.fact_id, "superseded",
                    valid_until=None if old.valid_until else new.get("valid_from"),
                    superseded_by=fact.fact_id)
            else:
                self.store.update_fact_status(old.fact_id, "retracted",
                                              superseded_by=fact.fact_id)
        return fact
