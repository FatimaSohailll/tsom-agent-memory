"""Result tables with 95% confidence intervals, built from finished run folders.

    python -m harness.make_tables --run amem=results/amem_s1,results/amem_s2 \\
        --run tsom=results/tsom_s1,results/tsom_s2 --out results/tables.md

Give one --run per system, with one folder per seed. A cell shows the mean across
seeds, the spread (sd) when there are several, and a bootstrap CI that resamples
questions or scenarios, each averaged across seeds. It reads the files each run
writes, including the sample_*.jsonl files that are not committed, so build the
tables where the runs finished. Missing runs are shown as missing, never guessed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

SEED = 0
RESAMPLES = 1000
SMALL_N = 30  # below this many items per system, overlapping CIs are an n problem
BASELINE = "amem"
Runs = Dict[str, List[Path]]
Interval = Tuple[float, float]


def _load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def bootstrap_ci(values: List[float], seed: int = SEED, resamples: int = RESAMPLES) -> Optional[Interval]:
    """Percentile 95% CI, in percent, of the mean of outcomes between 0 and 1."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(100.0 * sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def spread(values: List[float]) -> str:
    return f"{mean(values):.1f} ± {stdev(values):.1f}" if len(values) > 1 else f"{mean(values):.1f}"


def rate_cell(runs: List[List[bool]]) -> Tuple[str, Optional[Interval], int]:
    """Text, CI and item count for one metric of one system. `runs` has one list per seed."""
    runs = [outcomes for outcomes in runs if outcomes]
    if not runs:
        return "missing", None, 0
    if len({len(outcomes) for outcomes in runs}) > 1:
        raise ValueError("seeds of one system answered different numbers of questions")
    items = [mean(column) for column in zip(*runs)]
    ci = bootstrap_ci(items)
    return (f"{spread([100.0 * mean(outcomes) for outcomes in runs])}% "
            f"[{ci[0]:.1f}, {ci[1]:.1f}] (n={len(items)})"), ci, len(items)


def comparison(name: str, cis: Dict[str, Optional[Interval]], sizes: Dict[str, int]) -> List[str]:
    """How each system's interval sits against the baseline's. Lower is better."""
    base = cis.get(BASELINE)
    lines = []
    for system, ci in cis.items():
        if system == BASELINE or ci is None or base is None:
            continue
        if ci[1] < base[0]:
            result = "lower, and the intervals do not overlap"
        elif ci[0] > base[1]:
            result = "higher, and the intervals do not overlap"
        else:
            result = "inconclusive, the intervals overlap"
            if min(sizes[system], sizes[BASELINE]) < SMALL_N:
                result += f"; n below {SMALL_N} is too small to separate them"
        lines.append(f"{name}, {system} vs {BASELINE}: {result}.")
    return lines


def table(header: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|---" * len(header) + "|",
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def extraction_section(runs: Runs) -> List[str]:
    lines = ["## Extraction recall", ""]
    found = {s: [x for x in (_load(d / "extraction_summary.json") for d in dirs) if x]
             for s, dirs in runs.items()}
    if not any(found.values()):
        return lines + ["No extraction results found.", ""]
    rows = [[s, rate_cell([[f["found"] for f in x["per_fact"]] for x in xs])[0]]
            for s, xs in found.items() if xs]
    return lines + table(["System", "Recall"], rows)


def scenario_section(runs: Runs) -> List[str]:
    lines = ["## Change scenarios", ""]
    summaries = {s: [_load(d / "scenarios_summary.json") for d in dirs] for s, dirs in runs.items()}
    if not any(x for found in summaries.values() for x in found):
        return lines + ["No scenario results found.", ""]
    rows, cis, sizes = [], {}, {}
    for system, found in summaries.items():
        row = [system]
        for metric in ("correct", "stale", "hedged"):
            text, ci, n = rate_cell([[bool(s[metric]) for s in x["per_scenario"]] for x in found if x])
            row.append(text)
            if metric == "stale":
                cis[system], sizes[system] = ci, n
        rows.append(row)
    return lines + table(["System", "Correct", "Stale", "Hedged"], rows) + comparison("Stale", cis, sizes) + [""]


def _rows(folder: Path, tag: str) -> List[dict]:
    path = folder / f"sample_{tag}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _contradictions(dirs: List[Path], tag: str) -> str:
    found = [_load(d / f"summary_factcons_{tag}.json") for d in dirs]
    blocks = [x["contradictions"] for x in found if x and x.get("contradictions")]
    if not blocks:
        return "n/a"
    return (f"{spread([b['final_rate'] for b in blocks])} vs "
            f"{spread([b['final_rate_without_consolidation'] for b in blocks])}")


def factcons_section(runs: Runs) -> List[str]:
    lines = ["## FactConsolidation", ""]
    tags = sorted({p.stem[len("sample_"):] for dirs in runs.values() for d in dirs
                   for p in d.glob("sample_*.jsonl") if p.stem[len("sample_"):][:2] in ("sh", "mh")})
    if not tags:
        return lines + ["No FactConsolidation results found.", ""]
    for tag in tags:
        rows, cis, sizes = [], {}, {}
        for system, dirs in runs.items():
            per_seed = [_rows(d, tag) for d in dirs]
            accuracy = rate_cell([[bool(r["correct"]) for r in rs] for rs in per_seed])[0]
            stale, cis[system], sizes[system] = rate_cell(
                [[bool(r["stale"]) for r in rs if r.get("older")] for rs in per_seed])
            rows.append([system, accuracy, stale, _contradictions(dirs, tag)])
        lines += [f"### {tag}", ""]
        lines += table(["System", "Accuracy", "Stale", "Contradictions per 100 facts vs no consolidation"], rows)
        lines += comparison(f"Stale on {tag}", cis, sizes) + [""]
    if any(tag.startswith("mh") for tag in tags):
        lines += ["Multi-hop stale rates are estimates: they count stale values of the last fact only.", ""]
    return lines


def locomo_section(runs: Runs) -> List[str]:
    lines = ["## LoCoMo no-harm check", ""]
    found = {s: [x for x in (_load(d / "summary.json") for d in dirs) if x] for s, dirs in runs.items()}
    if not any(found.values()):
        return lines + ["No LoCoMo results found.", ""]

    def f1s(system: str, category: str) -> List[float]:
        return [x["scores"][category]["f1"] for x in found[system] if category in x["scores"]]

    rows = []
    for system in found:
        row = [system]
        for category in ("Overall", "Temporal"):
            values, base = f1s(system, category), f1s(BASELINE, category) if BASELINE in found else []
            text = spread(values) if values else "missing"
            if values and base and system != BASELINE:
                text += f" ({mean(values) - mean(base):+.1f})"
            row.append(text)
        rows.append(row)
    return lines + table(["System", f"Overall F1 (vs {BASELINE})", f"Temporal F1 (vs {BASELINE})"], rows)


def make_tables(runs: Runs) -> str:
    return "\n".join(["# Results", "", *extraction_section(runs), *scenario_section(runs),
                      *factcons_section(runs), *locomo_section(runs)])


def parse_run(text: str) -> Tuple[str, List[Path]]:
    system, _, folders = text.partition("=")
    if not system or not folders:
        raise argparse.ArgumentTypeError(f"expected SYSTEM=FOLDER[,FOLDER...], got {text!r}")
    return system, [Path(folder) for folder in folders.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, type=parse_run, metavar="SYSTEM=FOLDERS")
    parser.add_argument("--out", default="results/tables.md")
    args = parser.parse_args()
    text = make_tables(dict(args.run))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
