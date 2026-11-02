"""The memory module: A-Mem notes plus our fact layer, behind one class.

Phase 1 runs A-Mem unchanged (system="amem") and records cost and latency
for every operation. With system="tsom" every write also runs the fact layer
(tsom/facts.py); use_classifier picks its label mode. use_validity adds current
facts and OUTDATED markers to the retrieved context (tsom/context.py).
use_gate swaps in GatedAgenticMemory (tsom/gated_amem.py, Gap 2), which skips
A-Mem's evolution calls unless a fact changed or a neighbour is tau similar.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .amem_bridge import RobustAgenticMemorySystem
from .context import fact_context, render, retrieve_blocks
from .contracts import Fact, FactOp, MemoryOp, Note, RunConfig
from .facts import FactLayer
from .gated_amem import GatedAgenticMemory
from .llm import LLM
from .store import Store


class MemorySystem:
    def __init__(self, config: RunConfig, llm: LLM, db_path: str = ":memory:",
                 sample_id: Optional[str] = None):
        self.config = config
        self.llm = llm
        self.sample_id = sample_id
        self.current_session: Optional[int] = None  # set by the replay harness
        self.store = Store(db_path)
        # Build A-Mem with a backend that needs no network at start-up,
        # then hand it our instrumented LLM so every call is counted.
        if config.use_gate:
            self.amem = GatedAgenticMemory(model_name=config.embedder, llm_backend="ollama",
                                           llm_model=config.model, tau=config.tau,
                                           link_k=config.gate_link_k)
        else:
            self.amem = RobustAgenticMemorySystem(model_name=config.embedder, llm_backend="ollama",
                                                  llm_model=config.model)
        self.amem.llm_controller.llm = llm
        self.facts = FactLayer(self.store, llm, config.use_classifier) if config.system == "tsom" else None

    # ---- write path ----

    def write(self, content: str, time_str: str, speaker: Optional[str] = None) -> str:
        before = self.llm.usage.snapshot()  # before extraction, so it counts in this write
        evo_before = self.amem.evo_cnt
        start = time.perf_counter()
        fact_ops: List[FactOp] = []
        failures_before = self.facts.parse_failures if self.facts else 0
        if self.facts is not None and speaker is not None:
            fact_ops = self.facts.process_turn(speaker, self._turn_text(content, speaker), time_str)
        gated = isinstance(self.amem, GatedAgenticMemory)
        if gated:
            self.amem.set_write_context(op.label for op in fact_ops)
        note_id = self.amem.add_note(content, time=time_str)
        decision = self.amem.last_decision if gated else None
        new_ids = [op.fact_id for op in fact_ops if op.fact_id]
        self.store.set_source_note(new_ids, note_id)
        latency = time.perf_counter() - start
        used = self.llm.usage.since(before)
        self._sync_note(note_id)
        self.store.log_op(MemoryOp(
            op_type="write", system=self.config.system, sample_id=self.sample_id,
            ids_touched=[note_id] + new_ids + [op.retired_id for op in fact_ops if op.retired_id],
            llm_calls=used.calls, prompt_tokens=used.prompt_tokens,
            completion_tokens=used.completion_tokens, latency_s=latency,
            evolved=self.amem.evo_cnt > evo_before,
            labels=[op.label for op in fact_ops], raw_labels=[op.raw_label for op in fact_ops],
            parse_failures=(self.facts.parse_failures - failures_before) if self.facts else 0,
            session=self.current_session,
            gate=decision.reason if decision else None,
            neighbour_similarity=decision.max_similarity if decision else None,
        ))
        return note_id

    @staticmethod
    def _turn_text(content: str, speaker: str) -> str:
        """Undo Assistant.observe's prefix exactly; text without it is used as is."""
        prefix = "Speaker " + speaker + "says : "
        return content[len(prefix):] if content.startswith(prefix) else content

    # ---- read path ----

    def search(self, query: str, k: int, question: Optional[str] = None) -> dict:
        """Retrieve with A-Mem using query. question (if given) is only used to
        find people mentioned by name for the fact block."""
        start = time.perf_counter()
        facts: object = []
        if self.facts is None:
            context = self.amem.find_related_memories_raw(query, k=k)
        else:
            blocks = retrieve_blocks(self.amem, query, k)
            ctx = fact_context(self.store, [nid for nid, _ in blocks], question or query)
            facts = {"current": ctx["current"],
                     "outdated": [m for ms in ctx["outdated"].values() for m in ms]}
            context = render(blocks, ctx) if self.config.use_validity else "".join(t for _, t in blocks)
        self.store.log_op(MemoryOp(op_type="search", system=self.config.system,
                                   sample_id=self.sample_id, latency_s=time.perf_counter() - start))
        return {"notes": context, "facts": facts}

    def fact_history(self, entity: str, predicate: Optional[str] = None) -> List[Fact]:
        self.store.log_op(MemoryOp(op_type="fact_history", system=self.config.system,
                                   sample_id=self.sample_id))
        return self.store.fact_history(entity, predicate)

    def inspect(self, entity: str) -> dict:
        facts = self.store.fact_history(entity)
        return {
            "entity": entity,
            "current": [f.model_dump() for f in facts if f.status == "active"],
            "retired": [f.model_dump() for f in facts if f.status != "active"],
        }

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
