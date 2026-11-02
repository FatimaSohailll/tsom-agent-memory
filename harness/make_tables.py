"""Build the result tables and verdicts from an ablation results directory.

    python -m harness.make_tables --out results/ablation

Reads <out>/actions/{gap1_llm,gap1_lww}/actions_summary.json,
<out>/scenarios/<row>/scenarios_summary.json and <out>/locomo/<row>/summary.json,
and writes <out>/tables.md. Missing runs are shown as missing, never guessed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

from harness.scenarios import fmt

ROWS = ["amem", "tsom_conflict", "tsom_time", "tsom_both", "rag"]
CATEGORIES = ["Multi-hop", "Temporal", "Open-domain", "Single-hop", "Adversarial", "Overall"]
ANSWER_COLS = [("stale", "Stale rate"), ("correct", "Correct rate"), ("hedged", "Hedged rate"),
               ("consistent", "Consistency"), ("retention", "Retention")]
SEED = 0
RESAMPLES = 1000
F1_MARGIN = 2.0  # LoCoMo Temporal F1 may drop by at most this many points
SMALL_N = 30  # below this many scenarios per row, overlapping CIs are flagged as an n problem


def _load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.exists() else None


def load_results(out: Path) -> dict:
    return {
        "actions": {name: _load(out / "actions" / name / "actions_summary.json") for name in ("gap1_llm", "gap1_lww")},
        "scenarios": {row: _load(out / "scenarios" / row / "scenarios_summary.json") for row in ROWS},
        "locomo": {row: _load(out / "locomo" / row / "summary.json") for row in ROWS},
    }


def bootstrap_ci(values: List[bool], seed: int = SEED, resamples: int = RESAMPLES) -> Optional[Tuple[float, float]]:
    """Percentile 95% CI (in percent) of the mean of 0/1 outcomes, resampling scenarios."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(100.0 * sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def _ci_text(ci) -> str:
    return f" [{ci[0]:.1f}–{ci[1]:.1f}]" if ci else ""


def stale_and_ci(summary: dict, metric: str = "stale"):
    vals = [bool(s[metric]) for s in summary.get("per_scenario", []) if s[metric] is not None]
    return summary["overall"][metric], bootstrap_ci(vals)


# ---- verdicts (pure functions on numbers) ----

def _rate(c: Optional[dict]) -> Optional[float]:
    return c["k"] / c["n"] if c and c["n"] else None


def gap1_verdict(llm: Optional[dict], lww: Optional[dict]) -> str:
    """llm / lww: overall blocks with false_retirements and missed_corrections as k/n dicts."""
    if not llm or not lww:
        return "Gap 1: not available (need both gap1_llm and gap1_lww)."
    fr_l, fr_w = _rate(llm["false_retirements"]), _rate(lww["false_retirements"])
    mc_l, mc_w = _rate(llm["missed_corrections"]), _rate(lww["missed_corrections"])
    counts = (f"false retirements {fmt(llm['false_retirements'])} (LLM labels) vs {fmt(lww['false_retirements'])} "
              f"(newest wins); missed corrections {fmt(llm['missed_corrections'])} vs {fmt(lww['missed_corrections'])}")
    if None in (fr_l, fr_w, mc_l, mc_w):
        return f"Gap 1: inconclusive (no facts to count). {counts}."
    if fr_l < fr_w and mc_l <= mc_w:
        verdict = "supported"
    else:
        verdict = "not supported"
    small = min(llm["false_retirements"]["n"], lww["false_retirements"]["n"]) < SMALL_N
    note = " Note: n is small, so this difference may not hold up." if small and verdict == "supported" else ""
    return f"Gap 1: {verdict}: {counts}.{note}"


def gap3_verdict(row: str, row_stale: Optional[dict], row_ci, amem_stale: Optional[dict], amem_ci,
                 row_temporal: Optional[float], amem_temporal: Optional[float]) -> str:
    if not row_stale or not amem_stale or not row_stale["n"] or not amem_stale["n"]:
        return f"Gap 3 ({row} vs amem): not available (missing scenario results)."
    stale = f"stale {fmt(row_stale)}{_ci_text(row_ci)} vs {fmt(amem_stale)}{_ci_text(amem_ci)}"
    if _rate(row_stale) >= _rate(amem_stale):
        return f"Gap 3 ({row} vs amem): not supported: {stale}."
    if row_ci is None or amem_ci is None or row_ci[1] >= amem_ci[0]:
        n = min(row_stale["n"], amem_stale["n"])
        small = f" n={n} scenarios per row is too small to separate the rows." if n < SMALL_N else ""
        return f"Gap 3 ({row} vs amem): inconclusive (CIs overlap): {stale}.{small}"
    if row_temporal is None or amem_temporal is None:
        return f"Gap 3 ({row} vs amem): inconclusive (no LoCoMo Temporal F1 for the no-harm check): {stale}."
    drop = amem_temporal - row_temporal
    harm = f"LoCoMo Temporal F1 {row_temporal:.1f} vs {amem_temporal:.1f}"
    if drop > F1_MARGIN:
        return f"Gap 3 ({row} vs amem): not supported: {stale}, but {harm} (drop {drop:.1f} > {F1_MARGIN:.0f} points)."
    return f"Gap 3 ({row} vs amem): supported: {stale}; {harm}."


def interaction_line(both: Optional[dict], both_ci, time: Optional[dict], time_ci) -> str:
    if not both or not time or not both["n"] or not time["n"]:
        return "Interaction (tsom_both vs tsom_time): not available."
    stale = f"stale {fmt(both)}{_ci_text(both_ci)} vs {fmt(time)}{_ci_text(time_ci)}"
    if _rate(both) >= _rate(time):
        return f"Interaction (tsom_both vs tsom_time): LLM labelling does not lower the stale rate once validity is on: {stale}."
    if both_ci and time_ci and both_ci[1] < time_ci[0]:
        return f"Interaction (tsom_both vs tsom_time): LLM labelling helps once validity is on: {stale}."
    return f"Interaction (tsom_both vs tsom_time): inconclusive (CIs overlap): {stale}."


# ---- tables ----

def table1(actions: dict) -> List[str]:
    lines = ["## Table 1. Gap 1, label level: LLM labels (gap1_llm) vs newest value wins (gap1_lww)", ""]
    runs = [(n, actions.get(n)) for n in ("gap1_llm", "gap1_lww")]
    if not any(s for _, s in runs):
        return lines + ["No actions results found.", ""]
    lines += ["| Kind | Run | Extraction recall | Label accuracy | False retirements | Missed corrections | Parse failures |",
              "|---|---|---|---|---|---|---|"]
    kinds = ["overall"] + sorted({k for _, s in runs if s for k in s["per_kind"]})
    for kind in kinds:
        for name, s in runs:
            if not s:
                lines.append(f"| {kind} | {name} | missing | | | | |")
                continue
            b = s["overall"] if kind == "overall" else s["per_kind"].get(kind)
            if not b:
                continue
            lines.append(f"| {kind} | {name} | {fmt(b['extraction_recall'])} | {fmt(b['label_accuracy'])} | "
                         f"{fmt(b['false_retirements'])} | {fmt(b['missed_corrections'])} | {b['parse_failures']} |")
    llm, lww = (s["overall"] if s else None for _, s in runs)
    return lines + ["", "**Verdict.** " + gap1_verdict(llm, lww), ""]


def table2(scen: dict, locomo: dict) -> List[str]:
    lines = ["## Table 2. Ablation, answer level (change scenarios)", "",
             "Each cell is k/n (percent); stale and correct rates also show a bootstrap 95% CI "
             f"({RESAMPLES} resamples over scenarios, seed {SEED}).", ""]
    rows = [r for r in ROWS if scen.get(r)]
    if not rows:
        return lines + ["No scenario results found.", ""]
    lines += ["| System | " + " | ".join(c for _, c in ANSWER_COLS) + " |", "|---" * (len(ANSWER_COLS) + 1) + "|"]
    cis = {}
    for r in rows:
        s = scen[r]
        cells = []
        for metric, _ in ANSWER_COLS:
            c = s["overall"][metric]
            if metric in ("stale", "correct"):
                _, ci = stale_and_ci(s, metric)
                cis[(r, metric)] = ci
                cells.append(fmt(c) + _ci_text(ci))
            else:
                cells.append(fmt(c))
        lines.append(f"| {r} | " + " | ".join(cells) + " |")

    def temporal(row):
        res = locomo.get(row)
        return res["scores"].get("Temporal", {}).get("f1") if res else None

    def stale(row):
        return scen[row]["overall"]["stale"] if scen.get(row) else None

    verdicts = [gap3_verdict(r, stale(r), cis.get((r, "stale")), stale("amem"), cis.get(("amem", "stale")),
                             temporal(r), temporal("amem")) for r in ("tsom_time", "tsom_both")]
    verdicts.append(interaction_line(stale("tsom_both"), cis.get(("tsom_both", "stale")),
                                     stale("tsom_time"), cis.get(("tsom_time", "stale"))))
    return lines + [""] + ["**Verdict.** " + v + "  " for v in verdicts] + [""]


def table3(locomo: dict) -> List[str]:
    lines = ["## Table 3. No-harm check on LoCoMo", ""]
    rows = [r for r in ROWS if locomo.get(r)]
    if not rows:
        return lines + ["No LoCoMo results found.", ""]
    sizes = {(locomo[r]["conversations"], locomo[r]["scores"].get("Overall", {}).get("n", 0)) for r in rows}
    if len(sizes) == 1:
        conv, q = sizes.pop()
        lines += [f"{conv} conversation(s), {q} questions per system. Cells are F1 / BLEU-1.", ""]
    else:
        lines += ["Rows were run on different amounts of data: " + ", ".join(
            f"{r}: {locomo[r]['conversations']} conv / {locomo[r]['scores'].get('Overall', {}).get('n', 0)} q"
            for r in rows) + ". Cells are F1 / BLEU-1.", ""]
    lines += ["| System | " + " | ".join(CATEGORIES) + " | LLM calls/write | Seconds/write |",
              "|---" * (len(CATEGORIES) + 3) + "|"]
    for r in rows:
        res = locomo[r]
        cells = [f"{res['scores'][c]['f1']:.1f} / {res['scores'][c]['bleu1']:.1f}" if c in res["scores"] else "–"
                 for c in CATEGORIES]
        cost = res.get("cost") or {}
        calls, secs = cost.get("llm_calls_per_write"), cost.get("seconds_per_write")
        cells.append(f"{calls:.2f}" if calls is not None else "–")
        cells.append(f"{secs:.2f}" if secs is not None else "–")
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    if "amem" in rows:
        base = locomo["amem"]["scores"]
        notes = []
        for r in rows:
            if r == "amem":
                continue
            diffs = []
            for c in ("Temporal", "Overall"):
                if c in base and c in locomo[r]["scores"]:
                    diffs.append(f"{c} F1 {locomo[r]['scores'][c]['f1'] - base[c]['f1']:+.1f}")
            temporal_ok = ("Temporal" in base and "Temporal" in locomo[r]["scores"] and
                           base["Temporal"]["f1"] - locomo[r]["scores"]["Temporal"]["f1"] <= F1_MARGIN)
            notes.append(f"{r} vs amem: {', '.join(diffs) or 'no comparable categories'} "
                         f"({'within' if temporal_ok else 'outside'} the {F1_MARGIN:.0f}-point Temporal margin)")
        lines += ["", "**Verdict.** No-harm: " + "; ".join(notes) + "."]
    return lines + [""]


def make_tables(out: Path) -> str:
    res = load_results(out)
    text = "\n".join(["# TSOM ablation results", "", f"Source: `{out}`", ""]
                     + table1(res["actions"]) + table2(res["scenarios"], res["locomo"]) + table3(res["locomo"]))
    (out / "tables.md").write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/ablation")
    args = parser.parse_args()
    print(make_tables(Path(args.out)))


if __name__ == "__main__":
    main()
