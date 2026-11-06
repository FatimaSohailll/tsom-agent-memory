"""Extraction recall: how many labelled scenario facts the extractor finds.

Each turn goes to the fact extractor alone, with no memory and no classifier, so
a miss here is the extractor's. A labelled fact is looked for in its own session:
found (right person and value), wrong_entity (right value, stored under someone
else) or missed. Attribute names are free-form, so they are not compared.

    python -m harness.eval_extraction --config configs/tsom_qwen2.5-3b.json

Writes results/<run>/extraction_summary.md/.json and extraction_results.jsonl.
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
from harness.scenarios import (DEFAULT_SCENARIOS, Scenario, ScenarioFact, fmt, kn, load_scenarios,
                               normalise, save_json)
from tsom.contracts import RunConfig
from tsom.llm import LLM
from tsom.temporal.extractor import FactExtractor

OUTCOMES = ("found", "wrong_entity", "missed")


def _contains(text: str, words: str) -> bool:
    return bool(words) and re.search(rf"(?<!\w){re.escape(words)}(?!\w)", text) is not None


def _same_value(expected: str, got: str) -> bool:
    expected, got = normalise(expected), normalise(got)
    return _contains(got, expected) or _contains(expected, got)


def match(fact: ScenarioFact, extracted: List[dict]) -> str:
    """found, wrong_entity or missed for one labelled fact."""
    values = [f for f in extracted if _same_value(fact.value, f["value"])]
    if any(_contains(normalise(f["entity"]), normalise(fact.speaker)) for f in values):
        return "found"
    return "wrong_entity" if values else "missed"


def evaluate_scenario(extractor: FactExtractor, scenario: Scenario) -> dict:
    sessions = [[f for turn in session.turns for f in extractor.extract(turn.text, speaker=turn.speaker)]
                for session in scenario.sessions]
    facts = [{"speaker": f.speaker, "predicate": f.predicate, "value": f.value, "session": f.session,
              "outcome": match(f, sessions[f.session])} for f in scenario.facts]
    return {"id": scenario.id, "kind": scenario.kind, "facts": facts, "extracted": sessions}


def _block(facts: List[dict]) -> dict:
    found = sum(f["outcome"] == "found" for f in facts)
    return {"recall": kn(found, len(facts)),
            **{name: sum(f["outcome"] == name for f in facts) for name in OUTCOMES[1:]}}


def summarise(results: List[dict]) -> dict:
    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]] += r["facts"]
    facts = [f for r in results for f in r["facts"]]
    return {"overall": _block(facts), "per_kind": {k: _block(v) for k, v in sorted(by_kind.items())},
            "per_fact": [{"found": f["outcome"] == "found"} for f in facts]}


def to_markdown(summary: dict) -> str:
    source = "reviewed labels" if summary["reviewed_only"] else "includes unreviewed labels"
    lines = [f"## {summary['run_name']}: extraction recall ({summary['model']}, {source})", "",
             "| Kind | Recall | Wrong person | Missed |", "|---|---|---|---|"]
    for name, b in [("overall", summary["overall"])] + list(summary["per_kind"].items()):
        lines.append(f"| {name} | {fmt(b['recall'])} | {b['wrong_entity']} | {b['missed']} |")
    return "\n".join(lines) + "\n"


def run(config: RunConfig, scenarios_path: str = DEFAULT_SCENARIOS, limit: Optional[int] = None,
        include_unreviewed: bool = False) -> dict:
    out = Path(config.output_dir) / config.run_name
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM.from_config(config)
    llm.check()
    extractor = FactExtractor(llm)
    scenarios = load_scenarios(scenarios_path, include_unreviewed)[:limit]
    results = [evaluate_scenario(extractor, s) for s in scenarios]
    summary = {"run_name": config.run_name, "model": config.model, "reviewed_only": not include_unreviewed,
               **summarise(results)}
    save_json(out / "extraction_summary.json", summary)
    (out / "extraction_summary.md").write_text(to_markdown(summary), encoding="utf-8")
    (out / "extraction_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results),
                                                  encoding="utf-8")
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
