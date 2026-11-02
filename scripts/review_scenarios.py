"""Review draft scenarios one by one and keep the good ones.

    python scripts/review_scenarios.py

For each unreviewed draft: [a]ccept appends it to change_scenarios.jsonl with
reviewed=true, [s]kip leaves it for later, [q]uit saves and stops. Accepted
drafts are marked in drafts.jsonl too, so they are not shown again. To fix a
draft, edit drafts.jsonl by hand and review it again.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.scenarios import DEFAULT_SCENARIOS, DRAFTS, Scenario, read_jsonl, write_jsonl  # noqa: E402


def show(sc: Scenario) -> str:
    lines = [f"=== {sc.id} ({sc.kind}) ==="]
    for i, session in enumerate(sc.sessions):
        lines.append(f"  session {i} - {session.date}")
        lines += [f"    {t.speaker}: {t.text}" for t in session.turns]
    lines.append("  expected facts:")
    for f in sc.facts:
        rep = f" (replaces {f.replaces})" if f.replaces else ""
        lines.append(f"    s{f.session} {f.speaker} {f.predicate} = {f.value}  [{f.expected_label}]{rep}")
    lines.append(f"  Q:  {sc.question}\n  Q': {sc.question_alt}")
    lines.append(f"  current: {sc.current}   old: {sc.old}")
    return "\n".join(lines)


def review(drafts_path=DRAFTS, out_path=DEFAULT_SCENARIOS, input_fn=input, print_fn=print) -> int:
    drafts = read_jsonl(drafts_path)
    kept = read_jsonl(out_path)
    kept_ids = {s.id for s in kept}
    accepted = 0
    todo = [d for d in drafts if not d.reviewed]
    for n, draft in enumerate(todo, 1):
        print_fn(f"\n[{n}/{len(todo)}]\n" + show(draft))
        choice = input_fn("[a]ccept, [s]kip, [q]uit? ").strip().lower()[:1]
        if choice == "q":
            break
        if choice == "a":
            draft.reviewed = True
            if draft.id not in kept_ids:
                kept.append(draft.model_copy())
                kept_ids.add(draft.id)
            accepted += 1
    write_jsonl(out_path, kept)
    write_jsonl(drafts_path, drafts)
    print_fn(f"accepted {accepted}; {len(kept)} reviewed scenarios in {out_path}")
    return accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", default=DRAFTS)
    parser.add_argument("--out", default=DEFAULT_SCENARIOS)
    args = parser.parse_args()
    review(args.drafts, args.out)


if __name__ == "__main__":
    main()
