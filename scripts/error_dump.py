"""Print wrong answers (scenarios run) or wrong labels (actions run) for error analysis.

    python scripts/error_dump.py --run results/ablation/scenarios/tsom_both --n 15
    python scripts/error_dump.py --run results/ablation/actions/gap1_llm

For answers it shows the question, prediction, verdict, expected answers and the
context the model saw. For labels it shows the expected fact, the model's label
and its raw output line, plus false retirements and missed corrections.
"""

import argparse
import json
import sys
from pathlib import Path


def _cached(run: Path, sub: str):
    return [json.loads(p.read_text()) for p in sorted((run / sub).glob("*.json"))]


def dump_answers(run: Path, n: int, context_chars: int, out=print):
    shown = 0
    for r in _cached(run, "scenarios"):
        for a in r["answers"]:
            if a["verdict"] == "correct" or shown >= n:
                continue
            shown += 1
            out(f"=== {r['id']} ({r['kind']}) verdict={a['verdict']}")
            out(f"Q: {a['question']}\nA: {a['prediction']!r}")
            out(f"expected: {r.get('current')}   outdated: {r.get('old')}")
            ctx = a["context"]
            out("--- context ---\n" + (ctx if len(ctx) <= context_chars else ctx[:context_chars] + " [...]") + "\n")
    out(f"{shown} wrong answers shown")


def dump_labels(run: Path, n: int, out=print):
    shown = 0
    for r in _cached(run, "actions"):
        for f in r["facts"]:
            exp = f["expected"]
            if (f["found"] and f["got_label"] == exp["expected_label"]) or shown >= n:
                continue
            shown += 1
            got = f"label {f['got_label']} (applied {f['applied_label']})" if f["found"] else "NOT EXTRACTED"
            out(f"=== {r['id']} ({r['kind']}) s{exp['session']} {exp['speaker']} {exp['predicate']} = {exp['value']}"
                f"  expected {exp['expected_label']}" + (f", replaces {exp['replaces']}" if exp["replaces"] else ""))
            out(f"got: {got}\nraw: {f['raw_line']}\n")
        for key in ("false_retirements", "missed_corrections"):
            for item in r[key]["items"]:
                out(f"!!! {r['id']} {key.replace('_', ' ')[:-1]}: {item}")
    out(f"{shown} wrong or missing labels shown")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a run folder, e.g. results/ablation/scenarios/tsom_both")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--context-chars", type=int, default=2000)
    args = parser.parse_args(argv)
    run = Path(args.run)
    if (run / "actions").is_dir():
        dump_labels(run, args.n)
    elif (run / "scenarios").is_dir():
        dump_answers(run, args.n, args.context_chars)
    else:
        sys.exit(f"{run} has no actions/ or scenarios/ folder; is it an eval_actions or run_scenarios run?")


if __name__ == "__main__":
    main()
