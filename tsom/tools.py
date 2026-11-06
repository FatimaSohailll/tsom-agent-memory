"""The tools the agent can call, with JSON specs in OpenAI function format."""

from __future__ import annotations

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
                },
                "required": ["turn", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Find notes relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
]


class MemoryTools:
    def __init__(self, memory: MemorySystem):
        self.memory = memory

    def write_memory(self, turn: str, time: str) -> dict:
        return {"note_id": self.memory.write(turn, time)}

    def search_memory(self, query: str, k: int = 10) -> dict:
        return self.memory.search(query, k)

    def call(self, name: str, **kwargs) -> dict:
        if name not in {spec["function"]["name"] for spec in TOOL_SPECS}:
            raise ValueError(f"unknown tool {name}")
        return getattr(self, name)(**kwargs)
