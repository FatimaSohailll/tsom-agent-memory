"""Print the exact context the answering LLM sees for "Where does Ali live now?".

Replays the two-session dialogue from show_facts.py through MemorySystem
(A-Mem plus the fact layer), then retrieves the way Assistant.answer does.

    python scripts/show_context.py --backend fake --set use_classifier=true use_validity=true
    python scripts/show_context.py --set use_classifier=true use_validity=false
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.assistant import Assistant  # noqa: E402
from show_facts import SESSIONS  # noqa: E402
from tsom.contracts import RunConfig  # noqa: E402
from tsom.facts import PROMPT_HEAD  # noqa: E402
from tsom.llm import LLM, fake_completion  # noqa: E402
from tsom.memory import MemorySystem  # noqa: E402
from tsom.tools import MemoryTools  # noqa: E402

QUESTION = "Where does Ali live now?"


class CannedLLM(LLM):
    """Canned extraction output per turn; A-Mem's own prompts go to fake_completion."""

    def __init__(self):
        super().__init__("fake", "canned")
        self.next_output = "NONE"

    def _call(self, prompt, temperature):
        text = self.next_output if prompt.startswith(PROMPT_HEAD) else fake_completion(prompt)
        return text, len(prompt.split()), len(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config fields, e.g. --set use_classifier=true use_validity=true")
    args = parser.parse_args()

    data = {"run_name": "show_context", "system": "tsom", "backend": args.backend, "model": args.model}
    for item in args.set:  # same parsing as harness/run_locomo.py
        key, value = item.split("=", 1)
        data[key] = json.loads(value) if value.replace(".", "", 1).isdigit() or value in ("true", "false", "null") else value
    config = RunConfig(**data)

    if config.backend == "fake":
        llm = CannedLLM()
    else:
        llm = LLM(config.backend, config.model, config.base_url)
        llm.check()
    memory = MemorySystem(config, llm)
    agent = Assistant(MemoryTools(memory), llm, retrieve_k=config.retrieve_k)
    for date, turns in SESSIONS:
        for speaker, text, canned in turns:
            llm.next_output = canned  # only CannedLLM reads this
            agent.observe(speaker, text, date)

    # Same two steps as Assistant.answer: keywords, then search with the question.
    keywords = agent._query_keywords(QUESTION)
    context = agent.tools.call("search_memory", query=keywords, k=agent.retrieve_k, question=QUESTION)["notes"]
    print(f"system={config.system} use_classifier={config.use_classifier} use_validity={config.use_validity}")
    print(f"question: {QUESTION}\nkeywords: {keywords}\n--- context ---")
    print(context, end="")


if __name__ == "__main__":
    main()
