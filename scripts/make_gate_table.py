"""Gap 2 table: ungated vs gated runs, and the H2 verdict.

Each argument is one or more run folders (repeated runs of one system), each
holding the summary.json that harness.run_locomo writes.

    python scripts/make_gate_table.py --ungated results/r1/tsom_both,results/r2/tsom_both \
        --gated results/r1/tsom_full_gated,results/r2/tsom_full_gated

Writes results/gate_table.md and results/gate_table.json (or under --out).
H2 is checked as three separate claims, each printed: (a) fewer LLM calls per
write, (b) lower p50 and p95 write latency, (c) overall F1 within 2 points
(a drop of at most 2 points; a gain always passes).
"""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

F1_MARGIN = 2.0
MIN_RUNS = 3
METRICS = [  # (key, label)
    ("calls_per_write", "LLM calls per write"),
    ("calls_per_session", "LLM calls per session"),
    ("latency_p50", "Write latency p50 (s)"),
    ("latency_p95", "Write latency p95 (s)"),
    ("answer_prompt_tokens", "Answer prompt tokens per question"),
    ("skip_rate", "Gate skip rate"),
    ("f1_Overall", "Overall F1"),
]
CATEGORIES = ("Single-hop", "Multi-hop", "Temporal", "Open-domain", "Adversarial")


def run_metrics(summary: dict) -> dict:
    cost, scores = summary.get("cost", {}), summary.get("scores", {})
    out = {
        "calls_per_write": cost.get("llm_calls_per_write"),
        "calls_per_session": cost.get("llm_calls_per_session"),
        "latency_p50": cost.get("write_latency_p50"),
        "latency_p95": cost.get("write_latency_p95"),
        "answer_prompt_tokens": cost.get("answer_prompt_tokens_per_question"),
        "skip_rate": (cost.get("gate") or {}).get("skip_rate"),
    }
    for cat, s in scores.items():
        out[f"f1_{cat}"] = s["f1"]
    return out


def aggregate(runs: list) -> dict:
    """runs: run_metrics dicts. Returns {key: {"mean", "std", "n"}}; std is None for n < 2."""
    keys = {k for r in runs for k in r}
    out = {}
    for key in sorted(keys):
        values = [r[key] for r in runs if r.get(key) is not None]
        if values:
            out[key] = {"mean": mean(values), "std": stdev(values) if len(values) > 1 else None,
                        "n": len(values)}
    return out


def _compare(ungated: float, gated: float) -> str:
    if gated < ungated:
        return "lower"
    return "no difference" if gated == ungated else "higher"


def h2_checks(ungated: dict, gated: dict) -> list:
    """Three checks, each {"id", "claim", "passed", "detail"}. Inputs are aggregate() outputs."""
    def m(agg, key):
        return agg.get(key, {}).get("mean")

    checks = []
    u, g = m(ungated, "calls_per_write"), m(gated, "calls_per_write")
    if u is None or g is None:
        checks.append({"id": "a", "claim": "fewer LLM calls per write", "passed": False,
                       "detail": "missing LLM calls per write"})
    else:
        checks.append({"id": "a", "claim": "fewer LLM calls per write", "passed": g < u,
                       "detail": f"gated {g:.2f} vs ungated {u:.2f} ({_compare(u, g)})"})

    parts, passed = [], True
    for key, label in (("latency_p50", "p50"), ("latency_p95", "p95")):
        u, g = m(ungated, key), m(gated, key)
        if u is None or g is None:
            parts.append(f"{label} missing")
            passed = False
        else:
            parts.append(f"{label} gated {g:.2f}s vs ungated {u:.2f}s ({_compare(u, g)})")
            passed = passed and g < u
    checks.append({"id": "b", "claim": "lower p50 and p95 write latency", "passed": passed,
                   "detail": "; ".join(parts)})

    u, g = m(ungated, "f1_Overall"), m(gated, "f1_Overall")
    if u is None or g is None:
        checks.append({"id": "c", "claim": "overall F1 within 2 points", "passed": False,
                       "detail": "missing overall F1"})
    else:
        change = g - u
        if change == 0:
            how = "no difference"
        elif change >= -F1_MARGIN:
            how = "within 2 points"
        else:
            how = "dropped by more than 2 points"
        checks.append({"id": "c", "claim": "overall F1 within 2 points", "passed": change >= -F1_MARGIN,
                       "detail": f"gated {g:.2f} vs ungated {u:.2f}, change {change:+.2f} ({how})"})
    return checks


def verdict(checks: list, n_ungated: int, n_gated: int) -> str:
    failed = [c for c in checks if not c["passed"]]
    if not failed:
        text = "H2 supported"
    elif len(failed) == len(checks):
        text = "H2 not supported"
    else:
        text = "H2 partly supported; failed: " + "; ".join(f"({c['id']}) {c['claim']}" for c in failed)
    if min(n_ungated, n_gated) < MIN_RUNS:
        text += " (fewer than 3 runs: indicative only)"
    return text


def _cell(stat):
    if stat is None:
        return "n/a"
    if stat["std"] is None:
        return f"{stat['mean']:.2f}"
    return f"{stat['mean']:.2f} ± {stat['std']:.2f}"


def _runs_note(n: int) -> str:
    return "n=1, no std" if n == 1 else f"n={n}, mean ± sample std"


def to_markdown(ungated: dict, gated: dict, n_ungated: int, n_gated: int, checks: list, text: str) -> str:
    lines = ["# Gap 2: conflict gate", "",
             f"Ungated: {_runs_note(n_ungated)}. Gated: {_runs_note(n_gated)}.", "",
             "| Metric | Ungated | Gated |", "|---|---|---|"]
    rows = METRICS + [(f"f1_{c}", f"{c} F1") for c in CATEGORIES]
    for key, label in rows:
        if key in ungated or key in gated:
            lines.append(f"| {label} | {_cell(ungated.get(key))} | {_cell(gated.get(key))} |")
    lines += ["", "H2 checks:"]
    for c in checks:
        lines.append(f"- ({c['id']}) {c['claim']}: {'pass' if c['passed'] else 'FAIL'}: {c['detail']}")
    lines += ["", f"Verdict: {text}"]
    return "\n".join(lines) + "\n"


def load_runs(arg: str) -> list:
    runs = []
    for folder in [p.strip() for p in arg.split(",") if p.strip()]:
        path = Path(folder) / "summary.json"
        if not path.exists():
            raise SystemExit(f"no summary.json in {folder}")
        runs.append(run_metrics(json.loads(path.read_text(encoding="utf-8"))))
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ungated", required=True, metavar="RESULT_DIR[,RESULT_DIR...]")
    parser.add_argument("--gated", required=True, metavar="RESULT_DIR[,RESULT_DIR...]")
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    ungated_runs, gated_runs = load_runs(args.ungated), load_runs(args.gated)
    ungated, gated = aggregate(ungated_runs), aggregate(gated_runs)
    checks = h2_checks(ungated, gated)
    text = verdict(checks, len(ungated_runs), len(gated_runs))
    md = to_markdown(ungated, gated, len(ungated_runs), len(gated_runs), checks, text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_table.md").write_text(md, encoding="utf-8")
    (out / "gate_table.json").write_text(json.dumps({
        "ungated": {"folders": args.ungated.split(","), "n": len(ungated_runs), "metrics": ungated},
        "gated": {"folders": args.gated.split(","), "n": len(gated_runs), "metrics": gated},
        "checks": checks, "verdict": text}, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
