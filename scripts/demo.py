"""Small end-to-end demo: two sessions, then questions.

    python scripts/demo.py                      # uses qwen2.5:3b through Ollama
    python scripts/demo.py --backend fake       # no model needed, checks the plumbing

Prints every tool call so you can see the agent loop working.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.assistant import Assistant  # noqa: E402
from tsom.contracts import RunConfig  # noqa: E402
from tsom.llm import LLM  # noqa: E402
from tsom.memory import MemorySystem  # noqa: E402
from tsom.tools import MemoryTools  # noqa: E402

SESSIONS = [
    ("10:00 am on 3 January, 2023", [
        ("Ali", "Hi! I just started a new job at a bank in Karachi."),
        ("Sara", "Congrats! Do you like living in Karachi?"),
        ("Ali", "Yes, I love the sea. I also have a cat called Biscuit."),
    ]),
    ("6:30 pm on 14 June, 2023", [
        ("Ali", "Big news, I moved to Lahore last week for a software job."),
        ("Sara", "Wow, so you left the bank?"),
        ("Ali", "Yes, I work at a startup now. Biscuit came with me."),
    ]),
]

QUESTIONS = [
    ("When did Ali start his job at the bank?", 2, "January 2023"),
    ("When did Ali move to Lahore?", 2, "June 2023"),
    ("What is the name of Ali's cat?", 4, "Biscuit"),
]


class LoggingTools(MemoryTools):
    def call(self, name, **kwargs):
        result = super().call(name, **kwargs)
        shown = {k: (v[:60] + "...") if isinstance(v, str) and len(v) > 60 else v for k, v in kwargs.items()}
        print(f"  tool {name}({json.dumps(shown)})")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()

    config = RunConfig(run_name="demo", backend=args.backend, model=args.model)
    llm = LLM(config.backend, config.model)
    llm.check()
    memory = MemorySystem(config, llm)
    agent = Assistant(LoggingTools(memory), llm, retrieve_k=5)

    for date, turns in SESSIONS:
        print(f"\nSession on {date}")
        for speaker, text in turns:
            print(f"{speaker}: {text}")
            agent.observe(speaker, text, date)

    print("\nQuestions")
    for question, category, gold in QUESTIONS:
        result = agent.answer(question, category, gold)
        print(f"Q: {question}\nA: {result['prediction']}   (expected: {gold})\n")

    writes = [op for op in memory.store.ops() if op.op_type == "write"]
    print(f"{len(writes)} writes, {sum(op.llm_calls for op in writes)} LLM calls, "
          f"{sum(op.latency_s for op in writes):.1f}s writing, "
          f"{sum(1 for op in writes if op.evolved)} writes triggered evolution")


if __name__ == "__main__":
    main()
