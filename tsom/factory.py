"""Build the memory system a RunConfig asks for."""

from __future__ import annotations

from typing import Optional, Union

from .contracts import RunConfig
from .llm import LLM
from .memory import MemorySystem
from .rag import RagMemory


def build_memory(config: RunConfig, llm: LLM, db_path: str = ":memory:",
                 sample_id: Optional[str] = None) -> Union[MemorySystem, RagMemory]:
    """RagMemory for system="rag", MemorySystem for "amem" and "tsom"."""
    memory_cls = RagMemory if config.system == "rag" else MemorySystem
    return memory_cls(config, llm, db_path=db_path, sample_id=sample_id)
