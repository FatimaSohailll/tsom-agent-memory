"""Answer-level evaluation on the change scenarios.

For each scenario (conflict scenarios are skipped): fresh memory from the
config, every session replayed with Assistant.observe, then question and
question_alt asked with Assistant.answer(category=4). Every LLM call uses
temperature 0 unless the config sets another temperature.

    python -m harness.run_scenarios --config configs/tsom_qwen2.5-3b.json

Rates (correct, stale, hedged, retention) use the main question; consistency
compares the verdicts of both questions. Writes results/<run>/scenarios_summary.md/.json
and scenarios_results.jsonl. Finished scenarios are cached in results/<run>/scenarios/.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import List, Optional

from agent.assistant import Assistant
from harness.run_locomo import load_config
from harness.scenarios import DEFAULT_SCENARIOS, Scenario, fmt, kn, load_scenarios, save_json, score_answer
from tsom.contracts import RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.rag import RagMemory
from tsom.tools import MemoryTools

log = logging.getLogger("run_scenarios")
METRICS = ["correct", "stale", "hedged", "consistent", "retention"]


def run_scenario(config: RunConfig, llm: LLM, scenario: Scenario) -> dict:
    memory_cls = RagMemory if config.system == "rag" else MemorySystem
    memory = memory_cls(config, llm, sample_id=scenario.id)
    agent = Assistant(MemoryTools(memory), llm, config.retrieve_k, config.temperature_c5, seed=config.seed,
                      use_fact_history=config.use_fact_history)
    for session in scenario.sessions:
        for turn in session.turns:
            agent.observe(turn.speaker, turn.text, session.date)

    answers = []
    for q in (scenario.question, scenario.question_alt):
        result = agent.answer(q, 4, scenario.current[0])
        answers.append({"question": q, "prediction": result["prediction"], "context": result["context"],
                        "tools": result["tools"], **score_answer(result["prediction"], scenario)})
    writes = [op for op in memory.store.ops() if op.op_type == "write"]
    facts = final_facts(memory.store, scenario.id)
    memory.store.close()
    main = answers[0]
    return {
        "id": scenario.id, "kind": scenario.kind, "current": scenario.current, "old": scenario.old,
        "answers": answers,
        "correct": main["correct"], "stale": main["stale"], "hedged": main["hedged"],
        "retention": main["retention"], "consistent": answers[0]["verdict"] == answers[1]["verdict"],
        "writes": len(writes), "write_llm_calls": sum(op.llm_calls for op in writes),
        "write_seconds": sum(op.latency_s for op in writes),
        "facts": facts,
    }


def final_facts(store, sample_id: str) -> List[dict]:
    """Every fact the memory ended with, in the order learned: the state the answers came from."""
    facts = [f for key in store.entity_keys(sample_id) for f in store.entity_facts(sample_id, key)]
    return [{"entity": f.entity, "attribute": f.attribute, "value": f.value, "status": f.status,
             "operation": f.operation, "step": f.observed_at, "session": f.observed_time}
            for f in sorted(facts, key=lambda f: f.observed_at)]


def summarise_scenarios(results: List[dict]) -> dict:
    def block(rs):
        out = {"scenarios": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs if r[m] is not None]
            out[m] = kn(sum(vals), len(vals))
        return out

    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]].append(r)
    writes = sum(r["writes"] for r in results)
    return {
        "overall": block(results),
        "per_kind": {k: block(v) for k, v in sorted(by_kind.items())},
        "cost": {"writes": writes,
                 "llm_calls_per_write": round(sum(r["write_llm_calls"] for r in results) / writes, 3) if writes else None,
                 "seconds_per_write": round(sum(r["write_seconds"] for r in results) / writes, 3) if writes else None},
        # Per-scenario outcomes, kept small so make_tables can bootstrap from the summary alone.
        "per_scenario": [{"id": r["id"], "kind": r["kind"], **{m: r[m] for m in METRICS}} for r in results],
    }


def to_markdown(summary: dict) -> str:
    lines = [f"## {summary['run_name']}: change scenarios ({summary['system']}, {summary['model']}, "
             f"temperature {summary['temperature']})", "",
             "| Kind | Scenarios | Correct | Stale | Hedged | Consistent | Retention |",
             "|---|---|---|---|---|---|---|"]
    for name, b in [("overall", summary["overall"])] + list(summary["per_kind"].items()):
        lines.append(f"| {name} | {b['scenarios']} | " + " | ".join(fmt(b[m]) for m in METRICS) + " |")
    c = summary["cost"]
    lines += ["", f"Writes: {c['writes']}, LLM calls per write: {c['llm_calls_per_write']}, "
                  f"seconds per write: {c['seconds_per_write']}"]
    return "\n".join(lines) + "\n"


def run(config: RunConfig, scenarios_path: str = DEFAULT_SCENARIOS, limit: Optional[int] = None,
        include_unreviewed: bool = False) -> dict:
    if config.temperature is None:
        config = config.model_copy(update={"temperature": 0.0})
    out = Path(config.output_dir) / config.run_name
    cache = out / "scenarios"
    cache.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(config.model_dump_json(indent=2))
    scenarios = [s for s in load_scenarios(scenarios_path, include_unreviewed) if s.kind != "conflict"][:limit]

    llm = None
    results = []
    for sc in scenarios:
        path = cache / f"{sc.id}.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        if llm is None:
            llm = LLM.from_config(config)
            llm.check()
        log.info("scenario %s", sc.id)
        result = run_scenario(config, llm, sc)
        save_json(path, result)
        results.append(result)

    summary = {"run_name": config.run_name, "system": config.system, "model": config.model,
               "backend": config.backend, "temperature": config.temperature,
               **summarise_scenarios(results)}
    save_json(out / "scenarios_summary.json", summary)
    (out / "scenarios_summary.md").write_text(to_markdown(summary), encoding="utf-8")
    (out / "scenarios_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results),
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
    logging.getLogger("amem_robust").setLevel(logging.WARNING)
    run(load_config(args.config, args.set), args.scenarios, args.limit, args.include_unreviewed)


if __name__ == "__main__":
    main()
