"""The four tools the agent can call, with JSON specs in OpenAI function format."""

from __future__ import annotations

from typing import Optional

from .memory import MemorySystem

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": "Store one conversation turn in long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "turn": {"type": "string", "description": "The turn text, with speaker."},
                    "time": {"type": "string", "description": "When the turn happened."},
                    "speaker": {"type": "string", "description": "Who said the turn (optional)."},
                },
                "required": ["turn", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Find current facts and linked notes relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 10},
                    "question": {"type": "string",
                                 "description": "The original question, used to spot people it names (optional)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fact_history",
            "description": "List current and retired facts about an entity, with their dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "predicate": {"type": "string"},
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_memory",
            "description": "For humans: show everything stored about an entity.",
            "parameters": {
                "type": "object",
                "properties": {"entity": {"type": "string"}},
                "required": ["entity"],
            },
        },
    },
]


class MemoryTools:
    def __init__(self, memory: MemorySystem):
        self.memory = memory

    def write_memory(self, turn: str, time: str, speaker: Optional[str] = None) -> dict:
        return {"note_id": self.memory.write(turn, time, speaker=speaker)}

    def search_memory(self, query: str, k: int = 10, question: Optional[str] = None) -> dict:
        return self.memory.search(query, k, question=question)

    def get_fact_history(self, entity: str, predicate: Optional[str] = None) -> dict:
        return {"facts": [f.model_dump() for f in self.memory.fact_history(entity, predicate)]}

    def inspect_memory(self, entity: str) -> dict:
        return self.memory.inspect(entity)

    def call(self, name: str, **kwargs) -> dict:
        if name not in {spec["function"]["name"] for spec in TOOL_SPECS}:
            raise ValueError(f"unknown tool {name}")
        return getattr(self, name)(**kwargs)
