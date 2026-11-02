"""Sweep the gate threshold tau on the development conversation (LoCoMo 0) only.

Runs configs/exp/tsom_both.json once (the ungated "-gate" ablation) and
configs/gate/tsom_full_gated.json once per tau, then writes
results/tau_sweep/sweep.md and sweep.json. Every run is resumable.

    python scripts/sweep_tau.py
    python scripts/sweep_tau.py --taus 0.5,0.6,0.7,0.8 --plot

There is deliberately no option to sweep another conversation: tau is chosen
on conversation 0 and never re-tuned on a test conversation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import run_locomo  # noqa: E402
from harness.run_locomo import load_config  # noqa: E402

DEV_CONVERSATION = 0
UNGATED_CONFIG = ROOT / "configs" / "exp" / "tsom_both.json"
GATED_CONFIG = ROOT / "configs" / "gate" / "tsom_full_gated.json"
F1_MARGIN = 2.0  # proposal H2: overall F1 within 2 points of the ungated run
FIXED_KEYS = ("conversation_ids", "max_samples", "run_name", "output_dir", "use_gate", "tau")
NOISE_NOTE = ("F1 differences of 2-3 points are inside the noise of a single run "
              "(one conversation, answer temperature 0.7). tau is chosen here, on the development "
              "conversation, and is never re-tuned on a test conversation.")


def run_name(tau: float) -> str:
    return f"tau_{tau:g}"


def summary_metrics(summary: dict) -> dict:
    cost, scores = summary.get("cost", {}), summary.get("scores", {})
    gate = cost.get("gate") or {}
    return {
        "skip_rate": gate.get("skip_rate"),
        "calls_per_write": cost.get("llm_calls_per_write"),
        "latency_p50": cost.get("write_latency_p50"),
        "latency_p95": cost.get("write_latency_p95"),
        "overall_f1": scores.get("Overall", {}).get("f1"),
        "temporal_f1": scores.get("Temporal", {}).get("f1"),
    }


def sweep_rows(ungated: dict, gated: dict) -> list:
    """ungated: a run summary; gated: {tau: run summary}. One row per run, ungated first."""
    base = summary_metrics(ungated)
    rows = [{"tau": None, **base, "calls_change": 0.0, "f1_change": 0.0, "within_margin": True}]
    for tau in sorted(gated):
        m = summary_metrics(gated[tau])
        calls_change = f1_change = None
        if m["calls_per_write"] is not None and base["calls_per_write"] is not None:
            calls_change = m["calls_per_write"] - base["calls_per_write"]
        if m["overall_f1"] is not None and base["overall_f1"] is not None:
            f1_change = m["overall_f1"] - base["overall_f1"]
        rows.append({"tau": tau, **m, "calls_change": calls_change, "f1_change": f1_change,
                     "within_margin": f1_change is not None and f1_change >= -F1_MARGIN})
    return rows


def suggest(rows: list):
    """Fewest LLM calls per write among gated runs whose F1 dropped by at most F1_MARGIN."""
    ok = [r for r in rows if r["tau"] is not None and r["within_margin"] and r["calls_per_write"] is not None]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["calls_per_write"], -(r["overall_f1"] or 0.0)))["tau"]


def similarity_distribution(sims: list) -> dict:
    if not sims:
        return {}
    return {f"p{q}": float(np.percentile(sims, q)) for q in range(0, 101, 10)}


def run_similarities(run_dir: Path) -> list:
    ops_file = run_dir / f"ops_sample_{DEV_CONVERSATION}.jsonl"
    if not ops_file.exists():
        return []
    ops = map(json.loads, ops_file.read_text(encoding="utf-8").splitlines())
    return [op["neighbour_similarity"] for op in ops
            if op.get("op_type") == "write" and op.get("neighbour_similarity") is not None]


def _fmt(value, spec=".2f", pct=False):
    if value is None:
        return "n/a"
    return f"{100 * value:{spec}}%" if pct else f"{value:{spec}}"


def _signed(value):
    return "n/a" if value is None else f"{value:+.2f}"


def to_markdown(rows: list, suggested, distribution: dict) -> str:
    lines = [f"# tau sweep on LoCoMo conversation {DEV_CONVERSATION} (development)", "",
             "| tau | Skip rate | LLM calls / write | Change vs ungated | Write p50 (s) | Write p95 (s) "
             "| Overall F1 | Temporal F1 | F1 change | Within 2 F1 points |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tau = "ungated" if r["tau"] is None else f"{r['tau']:g}"
        lines.append(f"| {tau} | {_fmt(r['skip_rate'], '.1f', pct=True)} | {_fmt(r['calls_per_write'])} "
                     f"| {_signed(r['calls_change'])} | {_fmt(r['latency_p50'])} | {_fmt(r['latency_p95'])} "
                     f"| {_fmt(r['overall_f1'])} | {_fmt(r['temporal_f1'])} | {_signed(r['f1_change'])} "
                     f"| {'yes' if r['within_margin'] else 'no'} |")
    lines.append("")
    if distribution:
        lines += ["Neighbour similarity over all gated writes: "
                  + ", ".join(f"{k} {v:.3f}" for k, v in distribution.items()), ""]
    if suggested is None:
        lines.append("No gated run kept overall F1 within 2 points of the ungated run. "
                     "The cost-accuracy curve above is the result.")
    else:
        lines.append(f"Suggested tau: {suggested:g} (fewest LLM calls per write among runs within "
                     f"2 F1 points of the ungated run).")
    lines += ["", NOISE_NOTE]
    return "\n".join(lines) + "\n"


def plot(rows: list, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gated = [r for r in rows if r["tau"] is not None]
    base = rows[0]
    taus = [r["tau"] for r in gated]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, label in ((axes[0], "calls_per_write", "LLM calls per write"),
                           (axes[1], "overall_f1", "Overall F1 (LoCoMo conv. 0)")):
        ax.plot(taus, [r[key] for r in gated], marker="o", color="black", label="gated")
        if base[key] is not None:
            ax.axhline(base[key], linestyle="--", color="gray", label="ungated")
        ax.set_xlabel("tau (gate similarity threshold)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taus", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--out", default="results/tau_sweep")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="overrides for every run, e.g. --set seed=1 (conversation and tau are fixed)")
    parser.add_argument("--plot", action="store_true", help="also save tau_sweep.png (needs matplotlib)")
    args = parser.parse_args(argv)
    fixed = [item for item in args.set if item.split("=", 1)[0] in FIXED_KEYS]
    if fixed:
        parser.error(f"these keys are set by the sweep and cannot be overridden: {fixed}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("amem_robust").setLevel(logging.WARNING)

    out = Path(args.out)
    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    common = [*args.set, f"output_dir={out}"]
    ungated = run_locomo.run(load_config(UNGATED_CONFIG, [*common, "run_name=ungated"], [DEV_CONVERSATION]))
    gated, sims = {}, []
    for tau in taus:
        name = run_name(tau)
        gated[tau] = run_locomo.run(load_config(GATED_CONFIG, [*common, f"run_name={name}", f"tau={tau}"],
                                                [DEV_CONVERSATION]))
        sims += run_similarities(out / name)

    rows = sweep_rows(ungated, gated)
    suggested = suggest(rows)
    distribution = similarity_distribution(sims)
    text = to_markdown(rows, suggested, distribution)
    (out / "sweep.md").write_text(text, encoding="utf-8")
    (out / "sweep.json").write_text(json.dumps({"conversation": DEV_CONVERSATION, "rows": rows,
                                                "suggested_tau": suggested,
                                                "similarity_distribution": distribution,
                                                "note": NOISE_NOTE}, indent=2), encoding="utf-8")
    print(text)
    if args.plot:
        plot(rows, out / "tau_sweep.png")
        print(f"plot written to {out / 'tau_sweep.png'}")


if __name__ == "__main__":
    main()
