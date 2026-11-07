"""LoCoMo replay harness.

For each conversation: start with empty memory, feed every turn session by
session in time order (as A-Mem does), then ask all the questions. Writes
predictions, a per-operation trace and a summary table to results/<run_name>/.

A conversation that already finished is skipped on re-run, so a Colab
disconnect only costs the conversation that was in progress.

    python -m harness.run_locomo --config configs/amem_qwen2.5-3b.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean

from agent.assistant import Assistant
from harness.contradictions import contradiction_line, read_curve, summarise_curves
from harness.metrics import bleu1, f1, summarise
from tsom.amem_bridge import load_locomo_dataset
from tsom.contracts import RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.rag import RagMemory
from tsom.tools import MemoryTools

log = logging.getLogger("harness")
REPO_ROOT = Path(__file__).resolve().parent.parent
WRITE_OPS = ("write", "temporal_write")  # temporal_write exists only for system "tsom"


def run(config: RunConfig) -> dict:
    out = Path(config.output_dir) / config.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(config.model_dump_json(indent=2))

    llm = LLM.from_config(config)
    llm.check()
    dataset = Path(config.dataset)
    if not dataset.is_absolute() and not dataset.exists():
        dataset = REPO_ROOT / dataset  # relative paths in configs are from the repo root
    everything = load_locomo_dataset(str(dataset))
    if config.conversations is not None:
        unknown = [i for i in config.conversations if not 0 <= i < len(everything)]
        if unknown:
            raise ValueError(f"no LoCoMo conversation {unknown}; there are {len(everything)} (0-{len(everything) - 1})")
        chosen = [(i, everything[i]) for i in config.conversations]
    else:
        chosen = list(enumerate(everything[: config.max_samples]))

    rows, write_ops, curves = [], [], {}
    for idx, sample in chosen:
        pred_file = out / f"sample_{idx}.jsonl"
        ops_file = out / f"ops_sample_{idx}.jsonl"
        if pred_file.exists() and ops_file.exists():
            log.info("conversation %d already done, loading", idx)
            rows += [json.loads(line) for line in pred_file.read_text().splitlines()]
            write_ops += [op for op in map(json.loads, ops_file.read_text().splitlines())
                          if op["op_type"] in WRITE_OPS]
        else:
            sample_rows, sample_ops = run_sample(config, llm, idx, sample, out)
            rows += sample_rows
            write_ops += [op for op in sample_ops if op["op_type"] in WRITE_OPS]
        if config.system == "tsom":
            curves[str(idx)] = read_curve(out / f"memory_sample_{idx}.sqlite", str(idx),
                                          config.multivalued_attributes)

    summary = {
        "run_name": config.run_name,
        "system": config.system,
        "backend": config.backend,
        "model": config.model,
        "conversations": len(chosen),
        "conversation_ids": [i for i, _ in chosen],
        "scores": summarise(rows),
        "cost": cost_summary(rows, write_ops),
        "contradictions": summarise_curves(curves),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(to_markdown(summary))
    print(to_markdown(summary))
    return summary


def run_sample(config: RunConfig, llm: LLM, idx: int, sample, out: Path):
    db = out / f"memory_sample_{idx}.sqlite"
    db.unlink(missing_ok=True)
    memory_cls = RagMemory if config.system == "rag" else MemorySystem
    memory = memory_cls(config, llm, db_path=str(db), sample_id=str(idx))
    agent = Assistant(MemoryTools(memory), llm, config.retrieve_k, config.temperature_c5,
                      seed=config.seed + idx, use_fact_history=config.use_fact_history)

    sessions = sorted(sample.conversation.sessions.items())[: config.max_sessions]
    n_turns = sum(len(s.turns) for _, s in sessions)
    log.info("conversation %d: writing %d turns from %d sessions", idx, n_turns, len(sessions))
    start = time.perf_counter()
    for _, session in sessions:
        for turn in session.turns:
            agent.observe(turn.speaker, turn.text, session.date_time)
    log.info("conversation %d: memory built in %.1fs", idx, time.perf_counter() - start)
    memory.sync_all()

    qas = [qa for qa in sample.qa if qa.category in (1, 2, 3, 4, 5)][: config.max_questions]
    rows = []
    for qa in qas:
        gold = qa.final_answer
        result = agent.answer(qa.question, qa.category, gold)
        rows.append({
            "sample": idx, "question": qa.question, "category": qa.category,
            "reference": gold, "prediction": result["prediction"],
            "f1": f1(result["prediction"], gold), "bleu1": bleu1(result["prediction"], gold),
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "tools": result["tools"], "context": result["context"],  # the trace for this answer
        })

    ops = [op.model_dump() for op in memory.store.ops()]
    memory.store.close()
    # Ops are written last: their file marks the conversation as finished.
    (out / f"sample_{idx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / f"ops_sample_{idx}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in ops))
    return rows, ops


def cost_summary(rows, ops) -> dict:
    """Cost of memory writes. `ops` may hold "write" and "temporal_write" ops.

    The plain keys describe A-Mem's own write (unchanged, so baselines stay
    comparable). When temporal ops exist, temporal_* and total_* keys are added;
    total_* is per memory write: (write + temporal) cost divided by writes.
    """
    write_ops = [op for op in ops if op["op_type"] == "write"]
    temporal_ops = [op for op in ops if op["op_type"] == "temporal_write"]
    if not write_ops:
        return {}
    cost = {
        "writes": len(write_ops),
        "llm_calls_per_write": mean(op["llm_calls"] for op in write_ops),
        "seconds_per_write": mean(op["latency_s"] for op in write_ops),
        "evolution_rate": mean(1.0 if op["evolved"] else 0.0 for op in write_ops),
        "answer_prompt_tokens_per_question": mean(r["prompt_tokens"] for r in rows) if rows else 0,
    }
    if temporal_ops:
        n = len(write_ops)
        fact_ops: dict = {}
        for op in temporal_ops:
            for name, count in (op.get("op_counts") or {}).items():
                fact_ops[name] = fact_ops.get(name, 0) + count
        cost.update({
            "temporal_writes": len(temporal_ops),
            "temporal_llm_calls_per_write": sum(op["llm_calls"] for op in temporal_ops) / n,
            "temporal_seconds_per_write": sum(op["latency_s"] for op in temporal_ops) / n,
            "total_llm_calls_per_write": sum(op["llm_calls"] for op in write_ops + temporal_ops) / n,
            "total_seconds_per_write": sum(op["latency_s"] for op in write_ops + temporal_ops) / n,
            "temporal_failures": sum(op.get("failures", 0) for op in temporal_ops),
            "fact_ops": fact_ops,
        })
    return cost


def to_markdown(summary: dict) -> str:
    lines = [f"## {summary['run_name']} ({summary['system']}, {summary['model']}, "
             f"{summary['conversations']} conversations)", "",
             "| Category | n | F1 | BLEU-1 |", "|---|---|---|---|"]
    for cat, s in summary["scores"].items():
        lines.append(f"| {cat} | {s['n']} | {s['f1']:.2f} | {s['bleu1']:.2f} |")
    cost = summary["cost"]
    if cost:
        lines += ["", "| Cost | Value |", "|---|---|",
                  f"| Memory writes | {cost['writes']} |",
                  f"| LLM calls per write | {cost['llm_calls_per_write']:.2f} |",
                  f"| Seconds per write | {cost['seconds_per_write']:.2f} |",
                  f"| Writes that triggered evolution | {100 * cost['evolution_rate']:.1f}% |",
                  f"| Prompt tokens per question | {cost['answer_prompt_tokens_per_question']:.0f} |"]
        if "temporal_writes" in cost:
            ops = ", ".join(f"{k} {v}" for k, v in sorted(cost["fact_ops"].items())) or "none"
            lines += ["", "| Per memory write | A-Mem write | Temporal | Total |", "|---|---|---|---|",
                      f"| LLM calls | {cost['llm_calls_per_write']:.2f} | "
                      f"{cost['temporal_llm_calls_per_write']:.2f} | {cost['total_llm_calls_per_write']:.2f} |",
                      f"| Seconds | {cost['seconds_per_write']:.2f} | "
                      f"{cost['temporal_seconds_per_write']:.2f} | {cost['total_seconds_per_write']:.2f} |",
                      "", f"Fact operations: {ops}. Temporal steps that failed and were skipped: "
                      f"{cost['temporal_failures']}."]
    if summary.get("contradictions"):
        lines += ["", contradiction_line(summary["contradictions"])]
    return "\n".join(lines) + "\n"


def load_config(path: str, overrides: list) -> RunConfig:
    """A config file with KEY=VALUE overrides applied."""
    data = json.loads(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        data[key] = json.loads(value) if (value.replace(".", "", 1).isdigit() or value in ("true", "false", "null")
                                          or value.startswith("[")) else value
    return RunConfig(**data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config fields, e.g. --set max_samples=1 model=llama3.2:3b")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("amem_robust").setLevel(logging.WARNING)

    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
