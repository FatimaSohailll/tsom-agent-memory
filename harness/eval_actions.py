"""Label-level evaluation of the fact layer (Gap 1).

For each scenario: a fresh store, every turn through FactLayer.process_turn
only (no A-Mem, so about one LLM call per turn), then the stored facts and
labels are compared with the scenario's expected facts.

    python -m harness.eval_actions --config configs/exp/tsom_both.json --set run_name=gap1_llm

Writes results/<run>/actions_summary.md/.json and actions_facts.jsonl (one line
per expected fact). Each finished scenario is cached in results/<run>/actions/,
so a re-run skips it.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from harness.run_locomo import load_config
from harness.scenarios import DEFAULT_SCENARIOS, Scenario, fmt, kn, load_scenarios, normalise, save_json
from tsom.contracts import Fact, RunConfig
from tsom.facts import FactLayer
from tsom.llm import LLM
from tsom.store import Store

log = logging.getLogger("eval_actions")
LABELS = ["additive", "redundant", "corrective", "contradictory"]


def same_value(a: str, b: str) -> bool:
    """Equal after normalising, or one contains the other as whole words ("a bank" ~ "bank")."""
    a, b = normalise(a), normalise(b)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return re.search(r"(?<!\w)" + re.escape(short) + r"(?!\w)", long_) is not None


def _raw_line(raw: str, predicate: str, value: str) -> str:
    for line in raw.splitlines():
        if predicate in line.lower() and normalise(value) in normalise(line):
            return line.strip()
    return raw.strip()


def evaluate_scenario(scenario: Scenario, llm: LLM, use_classifier: bool) -> dict:
    store = Store()
    layer = FactLayer(store, llm, use_classifier)
    before = llm.usage.snapshot()
    got = []  # (session, speaker, FactOp, raw output)
    turns = 0
    for si, session in enumerate(scenario.sessions):
        for turn in session.turns:
            turns += 1
            for op in layer.process_turn(turn.speaker, turn.text, session.date):
                got.append((si, turn.speaker, op, "\n".join(layer.last_raw)))

    # Match each expected fact to one extracted op from the same session.
    used, facts = set(), []
    for exp in scenario.facts:
        match = next((n for n, (si, spk, op, _) in enumerate(got)
                      if n not in used and si == exp.session and spk.lower() == exp.speaker.lower()
                      and op.predicate == exp.predicate and same_value(op.value, exp.value)), None)
        rec = {"scenario": scenario.id, "kind": scenario.kind, "expected": exp.model_dump(),
               "found": match is not None, "got_label": None, "applied_label": None, "raw_line": None}
        if match is not None:
            used.add(match)
            _, _, op, raw = got[match]
            rec.update(got_label=op.raw_label, applied_label=op.label,
                       raw_line=_raw_line(raw, op.predicate, op.value))
        facts.append(rec)

    # End state. Expected active: not redundant and not replaced by a later expected fact.
    stored: List[Fact] = [f for spk in {e.speaker for e in scenario.facts} for f in store.fact_history(spk)]

    def stored_matches(speaker, predicate, value):
        return [f for f in stored if f.subject.lower() == speaker.lower()
                and f.predicate == predicate and same_value(f.object, value)]

    false_ret, missed = [], []
    fr_n = mc_n = 0
    for exp in scenario.facts:
        if exp.expected_label == "redundant":
            continue
        replaced = any(later.session > exp.session and later.speaker == exp.speaker
                       and later.predicate == exp.predicate and later.replaces
                       and same_value(later.replaces, exp.value) for later in scenario.facts)
        rows = stored_matches(exp.speaker, exp.predicate, exp.value)
        if not replaced and rows:
            fr_n += 1
            if not any(f.status == "active" for f in rows):
                false_ret.append(f"{exp.speaker} {exp.predicate} {exp.value}")
        if exp.expected_label == "corrective" and exp.replaces:
            old_rows = stored_matches(exp.speaker, exp.predicate, exp.replaces)
            if old_rows:
                mc_n += 1
                if any(f.status == "active" for f in old_rows):
                    missed.append(f"{exp.speaker} {exp.predicate} {exp.replaces}")

    used_calls = llm.usage.since(before).calls
    return {
        "id": scenario.id, "kind": scenario.kind, "turns": turns, "llm_calls": used_calls,
        "parse_failures": layer.parse_failures, "facts": facts,
        "false_retirements": {"k": len(false_ret), "n": fr_n, "items": false_ret},
        "missed_corrections": {"k": len(missed), "n": mc_n, "items": missed},
        "spurious": len(got) - len(used),
        "store": [f.model_dump() for f in stored],
    }


def summarise_actions(results: List[dict]) -> dict:
    def block(rs):
        facts = [f for r in rs for f in r["facts"]]
        found = [f for f in facts if f["found"]]
        confusion = {e: {g: 0 for g in LABELS} for e in LABELS}
        for f in found:
            if f["got_label"] in LABELS:
                confusion[f["expected"]["expected_label"]][f["got_label"]] += 1
        turns = sum(r["turns"] for r in rs)
        return {
            "scenarios": len(rs),
            "extraction_recall": kn(len(found), len(facts)),
            "label_accuracy": kn(sum(f["got_label"] == f["expected"]["expected_label"] for f in found), len(found)),
            "confusion": confusion,
            "false_retirements": kn(sum(r["false_retirements"]["k"] for r in rs),
                                    sum(r["false_retirements"]["n"] for r in rs)),
            "missed_corrections": kn(sum(r["missed_corrections"]["k"] for r in rs),
                                     sum(r["missed_corrections"]["n"] for r in rs)),
            "parse_failures": sum(r["parse_failures"] for r in rs),
            "turns": turns,
            "llm_calls_per_turn": round(sum(r["llm_calls"] for r in rs) / turns, 3) if turns else None,
            "spurious_facts": sum(r["spurious"] for r in rs),
        }

    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]].append(r)
    return {"overall": block(results), "per_kind": {k: block(v) for k, v in sorted(by_kind.items())}}


def to_markdown(summary: dict) -> str:
    lines = [f"## {summary['run_name']}: label-level evaluation ({summary['model']}, "
             f"use_classifier={summary['use_classifier']})", "",
             "| Kind | Scenarios | Extraction recall | Label accuracy | False retirements | "
             "Missed corrections | Parse failures | LLM calls/turn | Spurious facts |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = [("overall", summary["overall"])] + list(summary["per_kind"].items())
    for name, b in rows:
        lines.append(f"| {name} | {b['scenarios']} | {fmt(b['extraction_recall'])} | {fmt(b['label_accuracy'])} | "
                     f"{fmt(b['false_retirements'])} | {fmt(b['missed_corrections'])} | {b['parse_failures']} | "
                     f"{b['llm_calls_per_turn']} | {b['spurious_facts']} |")
    lines += ["", "Label confusion (rows: expected, columns: model's raw label):", "",
              "| expected \\ got | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
    for e in LABELS:
        lines.append(f"| {e} | " + " | ".join(str(summary["overall"]["confusion"][e][g]) for g in LABELS) + " |")
    return "\n".join(lines) + "\n"


def run(config: RunConfig, scenarios_path: str = DEFAULT_SCENARIOS, limit: Optional[int] = None,
        include_unreviewed: bool = False) -> dict:
    out = Path(config.output_dir) / config.run_name
    cache = out / "actions"
    cache.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(config.model_dump_json(indent=2))
    scenarios = load_scenarios(scenarios_path, include_unreviewed)[:limit]

    llm = None
    results = []
    for sc in scenarios:
        path = cache / f"{sc.id}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        if llm is None:
            llm = LLM(config.backend, config.model, config.base_url)
            llm.check()
        log.info("scenario %s", sc.id)
        result = evaluate_scenario(sc, llm, config.use_classifier)
        save_json(path, result)
        results.append(result)

    summary = {"run_name": config.run_name, "system": config.system, "model": config.model,
               "backend": config.backend, "use_classifier": config.use_classifier,
               **summarise_actions(results)}
    save_json(out / "actions_summary.json", summary)
    (out / "actions_summary.md").write_text(to_markdown(summary))
    (out / "actions_facts.jsonl").write_text(
        "".join(json.dumps(f) + "\n" for r in results for f in r["facts"]))
    print(to_markdown(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--include-unreviewed", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(load_config(args.config, args.set), args.scenarios, args.limit, args.include_unreviewed)


if __name__ == "__main__":
    main()
