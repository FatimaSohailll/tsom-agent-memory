"""RAG baseline: store raw turns, retrieve the top-k most similar ones.

No LLM calls on write, no links, no evolution. It uses the same
embedder (all-MiniLM-L6-v2) and the same retriever class as A-Mem, and it
formats retrieved turns the way A-Mem does, so the answer prompts are
identical and the only difference between the two systems is the memory.
It has the same interface as MemorySystem, so the agent and tools are shared.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .amem_bridge import SimpleEmbeddingRetriever
from .contracts import MemoryOp, Note, RunConfig
from .llm import LLM
from .store import Store


class RagMemory:
    def __init__(self, config: RunConfig, llm: LLM, db_path: str = ":memory:",
                 sample_id: Optional[str] = None):
        self.config = config
        self.llm = llm  # unused on the write path; kept for the shared interface
        self.sample_id = sample_id
        self.store = Store(db_path)
        self.retriever = SimpleEmbeddingRetriever(config.embedder)
        self.notes: List[Note] = []

    # ---- write path ----

    def write(self, content: str, time_str: str) -> str:
        start = time.perf_counter()
        note = Note(content=content, timestamp=time_str)
        self.retriever.add_documents([content])
        self.notes.append(note)
        self.store.upsert_note(note)
        self.store.log_op(MemoryOp(
            op_type="write", system=self.config.system, sample_id=self.sample_id,
            ids_touched=[note.id], llm_calls=0, latency_s=time.perf_counter() - start,
            evolved=False,
        ))
        return note.id

    # ---- read path ----

    def search(self, query: str, k: int) -> dict:
        start = time.perf_counter()
        context = ""
        if self.notes:
            for i in self.retriever.search(query, k):
                n = self.notes[int(i)]
                context += "talk start time:" + n.timestamp + "memory content: " + n.content + "\n"
        self.store.log_op(MemoryOp(op_type="search", system=self.config.system,
                                   sample_id=self.sample_id, latency_s=time.perf_counter() - start))
        return {"notes": context}

    def sync_all(self):
        self.store.conn.commit()
