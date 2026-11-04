"""The A-Mem memory system behind our common memory interface.

Runs A-Mem unchanged (the authors' code, via tsom.amem_bridge), mirrors its notes
into SQLite, and records cost and latency for every operation.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from .amem_bridge import RobustAgenticMemorySystem
from .contracts import MemoryOp, Note, RunConfig
from .llm import LLM
from .store import Store
from .temporal.classifier import UpdateClassifier
from .temporal.consolidator import Consolidator
from .temporal.attributes import AttributeCanonicalizer
from .temporal.extractor import FactExtractor
from .temporal.retriever import TemporalRetriever
from .temporal.serial import split_serial_facts

log = logging.getLogger("tsom.memory")
FACTS_HEADER = "Known facts (CURRENT = latest known value):"


class MemorySystem:
    def __init__(self, config: RunConfig, llm: LLM, db_path: str = ":memory:",
                 sample_id: Optional[str] = None):
        self.config = config
        self.llm = llm
        self.sample_id = sample_id
        self.store = Store(db_path)
        # Build A-Mem with a backend that needs no network at start-up,
        # then hand it our instrumented LLM so every call is counted.
        self.amem = RobustAgenticMemorySystem(model_name=config.embedder, llm_backend="ollama",
                                              llm_model=config.model)
        self.amem.llm_controller.llm = llm
        if config.system == "tsom":
            self.extractor = FactExtractor(llm)
            self.attributes = AttributeCanonicalizer(
                self.store, encode=lambda texts: self.amem.retriever.model.encode(texts))
            self.classifier = UpdateClassifier(
                llm, self.store, seed_multivalued=config.seed_multivalued,
                additive_cue=config.additive_cue,
                multivalued_attributes=config.multivalued_attributes)
            self.consolidator = Consolidator(self.store)
            self.observed_counter = 0
            self.temporal_retriever = TemporalRetriever(self)

    # ---- write path ----

    def write(self, content: str, time_str: str) -> str:
        temporal = self.config.system == "tsom"
        note_state = self._note_state() if temporal else None
        before = self.llm.usage.snapshot()
        evo_before = self.amem.evo_cnt
        start = time.perf_counter()
        note_id = self.amem.add_note(content, time=time_str)
        latency = time.perf_counter() - start
        used = self.llm.usage.since(before)
        self._sync_note(note_id)
        self.store.log_op(MemoryOp(
            op_type="write", system=self.config.system, sample_id=self.sample_id,
            ids_touched=[note_id], llm_calls=used.calls, prompt_tokens=used.prompt_tokens,
            completion_tokens=used.completion_tokens, latency_s=latency,
            evolved=self.amem.evo_cnt > evo_before,
        ))
        if temporal:
            self._temporal_write(note_id, content, time_str, note_state)
        return note_id

    # ---- read path ----

    def search(self, query: str, k: int) -> dict:
        start = time.perf_counter()
        if self.config.system == "tsom":
            found = self.temporal_retriever.search(query, k)
            notes = found["notes"]
            if found["facts"]:
                notes = (FACTS_HEADER + "\n" + found["facts"] + "\n\n" + notes)
            result = {"notes": notes, "facts": found["facts"], "query_type": found["query_type"]}
        else:
            result = {"notes": self.amem.find_related_memories_raw(query, k=k)}
        self.store.log_op(MemoryOp(op_type="search", system=self.config.system,
                                   sample_id=self.sample_id, latency_s=time.perf_counter() - start))
        return result

    # ---- temporal layer (system == "tsom" only) ----

    def _note_state(self) -> dict:
        """note_id -> (context, tags, links as note ids), copied so that
        A-Mem's in-place evolution cannot change the snapshot."""
        ids = list(self.amem.memories.keys())
        return {nid: (n.context, list(n.tags), self._link_ids(n, ids))
                for nid, n in self.amem.memories.items()}

    @staticmethod
    def _link_ids(note, ids: List[str]) -> List[str]:
        return [ids[i] for i in note.links if isinstance(i, int) and 0 <= i < len(ids)]

    def _temporal_write(self, note_id: str, content: str, time_str: str, note_state: dict):
        sid = self.sample_id or ""
        before = self.llm.usage.snapshot()
        counts_before = self.classifier.op_counts.copy()
        start = time.perf_counter()
        serials = split_serial_facts(content)
        failures = 0

        # Extract facts; numbered lines carry their own serial as observed_at.
        items = []  # (new fact dict, observed_at)
        try:
            hint = self.attributes.known_attributes(sid, content)  # names to reuse
            if serials:
                facts = self.extractor.extract_batch([text for _, text in serials], hint)
                for f in sorted(facts, key=lambda f: f["source"]):
                    serial, text = serials[f["source"]]
                    self.observed_counter = max(self.observed_counter, serial)
                    items.append(({**f, "source_text": text}, serial))
            else:
                for f in self.extractor.extract(content, hint):
                    self.observed_counter += 1
                    items.append(({**f, "source_text": content}, self.observed_counter))
        except Exception as e:
            failures += 1
            log.warning("temporal extraction failed for note %s (%r): %s",
                        note_id, content[:200], e, exc_info=True)

        # Snap drifted attribute names (speaks_language -> speaks) to the entity's own.
        for new, _ in items:
            try:
                attribute, why = self.attributes.canonicalize(sid, new["entity"], new["attribute"])
            except Exception as e:  # never let naming cleanup lose a fact
                log.warning("attribute canonicalization failed for %r: %s", new, e)
                continue
            if why:
                log.info("attribute %r snapped to %r for %r (%s)",
                         new["attribute"], attribute, new["entity"], why)
                new["raw_attribute"], new["attribute"] = new["attribute"], attribute

        # Keep the old state of every note A-Mem's evolution rewrote. A note
        # that yields no facts is stamped with the current observed_counter.
        change_at = items[0][1] if items else self.observed_counter
        ids = list(self.amem.memories.keys())
        for nid, (context, tags, links) in note_state.items():
            n = self.amem.memories[nid]
            if (n.context, list(n.tags), self._link_ids(n, ids)) != (context, tags, links):
                version = len(self.store.note_versions(nid)) + 1
                self.store.add_note_version(nid, version, change_at, context, tags, links,
                                            changed_by_note_id=note_id)

        # A failing fact is logged and skipped; the A-Mem note stays.
        fact_ids = []
        for new, observed_at in items:
            op = "classify"
            try:
                with self.store.atomic():
                    op, old = self.classifier.classify(sid, new, observed_at, time_str)
                    live = self.consolidator.apply(sid, note_id, new, op, old,
                                                   observed_at, time_str)
                    if self.consolidator.last_op != op:  # e.g. REPLACE guarded into ADD_HISTORICAL
                        counts = self.classifier.op_counts
                        counts[op] -= 1
                        if counts[op] <= 0:
                            del counts[op]
                        counts[self.consolidator.last_op] += 1
                fact_ids.append(live.fact_id)
            except Exception as e:
                failures += 1
                log.warning("temporal step failed (op=%s) for fact %r in note %s: %s",
                            op, new, note_id, e, exc_info=True)

        used = self.llm.usage.since(before)
        self.store.log_op(MemoryOp(
            op_type="temporal_write", system=self.config.system, sample_id=self.sample_id,
            ids_touched=[note_id] + fact_ids, llm_calls=used.calls,
            prompt_tokens=used.prompt_tokens, completion_tokens=used.completion_tokens,
            latency_s=time.perf_counter() - start,
            op_counts=dict(self.classifier.op_counts - counts_before), failures=failures,
        ))

    # ---- helpers ----

    def _sync_note(self, note_id: str):
        """Copy an A-Mem note (and any neighbours evolution changed) into SQLite."""
        ids = list(self.amem.memories.keys())
        targets = [note_id] + self._neighbour_ids(note_id, ids)
        for nid in targets:
            n = self.amem.memories[nid]
            links = [ids[i] for i in n.links if isinstance(i, int) and 0 <= i < len(ids)]
            self.store.upsert_note(Note(id=n.id, content=n.content, timestamp=n.timestamp,
                                        keywords=list(n.keywords), tags=list(n.tags),
                                        context=n.context, links=links))

    def _neighbour_ids(self, note_id: str, ids: List[str]) -> List[str]:
        n = self.amem.memories[note_id]
        return [ids[i] for i in n.links if isinstance(i, int) and 0 <= i < len(ids) and ids[i] != note_id]

    def sync_all(self):
        for nid in self.amem.memories:
            self._sync_note(nid)
        self.store.conn.commit()
