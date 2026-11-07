"""Add a past-period question to every change and cancel scenario that can have one.

    python scripts/add_past_questions.py

Asks about the month the first value was stated, e.g. "Where did Ali live in May 2023?", with the first
value as the expected answer. Fills both the reviewed scenarios and the drafts; scenarios that already
have one are left as they are, and every other field is unchanged. Check a sample by hand afterwards.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.scenarios import DEFAULT_SCENARIOS, DRAFTS, read_jsonl, with_past_question, write_jsonl  # noqa: E402


def main():
    for path in (DEFAULT_SCENARIOS, DRAFTS):
        before = read_jsonl(path)
        after = [with_past_question(s) for s in before]
        added = Counter(s.kind for s, b in zip(after, before) if s.question_past and not b.question_past)
        write_jsonl(path, after)
        print(f"{path}: past-period questions added to {sum(added.values())} scenarios {dict(added)}")


if __name__ == "__main__":
    main()
