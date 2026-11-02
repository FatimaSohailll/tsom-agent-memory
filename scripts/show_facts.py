"""Replay a short two-session dialogue through the fact layer only (no A-Mem).

    python scripts/show_facts.py                    # qwen2.5:3b through Ollama
    python scripts/show_facts.py --newest-wins      # same extraction, labels ignored
    python scripts/show_facts.py --backend fake     # canned model output, no model needed

Prints, per turn, the raw model output and the labels, then the final facts table.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsom.facts import FactLayer  # noqa: E402
from tsom.llm import LLM  # noqa: E402
from tsom.store import Store  # noqa: E402

# (session date, [(speaker, text, canned output for --backend fake)])
SESSIONS = [
    ("10:00 am on 8 May, 2023", [
        ("Ali", "Hi! I just started a job at a bank in Karachi.",
         "FACT: works_at | bank | additive | none\nFACT: lives_in | Karachi | additive | none"),
        ("Sara", "Nice! How do you spend your evenings?", "NONE"),
        ("Ali", "Mostly eating biryani with my cat, Biscuit.",
         "FACT: likes | biryani | additive | none\nFACT: has_pet | cat | additive | none"),
    ]),
    ("6:30 pm on 14 June, 2023", [
        ("Ali", "Big news: I moved to Lahore last week.",
         "FACT: lives_in | Lahore | corrective | Karachi"),
        ("Ali", "I also switched from the bank to a startup.",
         "FACT: works_at | startup | corrective | bank"),
        ("Ali", "Lahore food is great, I like nihari now. Still love biryani though.",
         "FACT: likes | nihari | additive | none\nFACT: likes | biryani | redundant | none"),
    ]),
]


class CannedLLM(LLM):
    """Returns the canned output for the current turn. Calls are still counted."""

    def __init__(self):
        super().__init__("fake", "canned")
        self.next_output = "NONE"

    def _call(self, prompt, temperature):
        return self.next_output, len(prompt.split()), len(self.next_output.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--newest-wins", action="store_true", help="use_classifier=False")
    args = parser.parse_args()

    if args.backend == "fake":
        llm = CannedLLM()
    else:
        llm = LLM(args.backend, args.model)
        llm.check()
    store = Store()
    facts = FactLayer(store, llm, use_classifier=not args.newest_wins)
    print(f"mode: {'newest value wins' if args.newest_wins else 'classifier labels'}\n")

    for date, turns in SESSIONS:
        print(f"=== session {date} ===")
        for speaker, text, canned in turns:
            llm.next_output = canned  # only CannedLLM reads this
            ops = facts.process_turn(speaker, text, date)
            print(f"{speaker}: {text}")
            for raw in facts.last_raw:
                print("  model> " + raw.strip().replace("\n", "\n  model> "))
            for op in ops:
                raw = "" if op.raw_label == op.label else f" (model said {op.raw_label})"
                print(f"  -> {op.label}{raw}: {op.predicate} = {op.value}")
            print()

    print(f"parse failures: {facts.parse_failures}, LLM calls: {llm.usage.calls}\n")
    rows = [(f.subject, f.predicate, f.object, f.status, f.valid_from, f.valid_to or "",
             "yes" if f.supersedes else "", "yes" if f.conflict_with else "")
            for s in store.subjects() for f in store.fact_history(s)]
    header = ("subject", "predicate", "object", "status", "valid_from", "valid_to", "supersedes", "conflict")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    main()
