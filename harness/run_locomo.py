"""LoCoMo replay harness.

For each conversation: start with empty memory, feed every turn session by
session in time order (as A-Mem does), then ask all the questions. Writes
predictions, a per-operation trace and a summary table to results/<run_name>/.

A conversation that already finished is skipped on re-run, so a Colab
disconnect only costs the conversation that was in progress.

    python -m harness.run_locomo --config configs/amem_qwen2.5-3b.json
    python -m harness.run_locomo --config configs/gate/tsom_full_gated.json --conversation-ids 1,8
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from agent.assistant import Assistant
from harness.metrics import bleu1, f1, summarise
from tsom.amem_bridge import load_locomo_dataset
from tsom.contracts import RunConfig
from tsom.factory import build_memory
from tsom.llm import LLM
from tsom.tools import MemoryTools

log = logging.getLogger("harness")
REPO_ROOT = Path(__file__).resolve().parent.parent


def run(config: RunConfig) -> dict:
    out = Path(config.output_dir) / config.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(config.model_dump_json(indent=2))

    llm = LLM(config.backend, config.model, config.base_url)
    llm.check()
    dataset = Path(config.dataset)
    if not dataset.is_absolute() and not dataset.exists():
        dataset = REPO_ROOT / dataset  # relative paths in configs are from the repo root
    samples = select_samples(load_locomo_dataset(str(dataset)), config)

    rows, write_ops = [], []
    for idx, sample in samples:
        pred_file = out / f"sample_{idx}.jsonl"
        ops_file = out / f"ops_sample_{idx}.jsonl"
        if pred_file.exists() and ops_file.exists():
            log.info("conversation %d already done, loading", idx)
            rows += [json.loads(line) for line in pred_file.read_text().splitlines()]
            write_ops += [op for op in map(json.loads, ops_file.read_text().splitlines())
                          if op["op_type"] == "write"]
            continue
        sample_rows, sample_ops = run_sample(config, llm, idx, sample, out)
        rows += sample_rows
        write_ops += [op for op in sample_ops if op["op_type"] == "write"]

    summary = {
        "run_name": config.run_name,
        "system": config.system,
        "backend": config.backend,
        "model": config.model,
        "conversations": len(samples),
        "conversation_ids": [idx for idx, _ in samples],
        "seed": config.seed,
        "scores": summarise(rows),
        "cost": cost_summary(rows, write_ops),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(to_markdown(summary))
    print(to_markdown(summary))
    return summary


def select_samples(samples, config: RunConfig):
    """(LoCoMo index, sample) pairs: conversation_ids if given, else the first max_samples."""
    if config.conversation_ids is None:
        return list(enumerate(samples))[: config.max_samples]
    bad = [i for i in config.conversation_ids if not 0 <= i < len(samples)]
    if bad:
        raise ValueError(f"conversation ids {bad} out of range 0..{len(samples) - 1}")
    return [(i, samples[i]) for i in config.conversation_ids]


def run_sample(config: RunConfig, llm: LLM, idx: int, sample, out: Path):
    db = out / f"memory_sample_{idx}.sqlite"
    db.unlink(missing_ok=True)
    memory = build_memory(config, llm, db_path=str(db), sample_id=str(idx))
    agent = Assistant(MemoryTools(memory), llm, config.retrieve_k, config.temperature_c5,
                      seed=config.seed + idx)

    sessions = sorted(sample.conversation.sessions.items())[: config.max_sessions]
    n_turns = sum(len(s.turns) for _, s in sessions)
    log.info("conversation %d: writing %d turns from %d sessions", idx, n_turns, len(sessions))
    start = time.perf_counter()
    for session_id, session in sessions:
        memory.current_session = session_id
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
        })

    ops = [op.model_dump() for op in memory.store.ops()]
    memory.store.close()
    # Ops are written last: their file marks the conversation as finished.
    (out / f"sample_{idx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / f"ops_sample_{idx}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in ops))
    return rows, ops


def cost_summary(rows, write_ops) -> dict:
    if not write_ops:
        return {}
    latencies = [op["latency_s"] for op in write_ops]
    cost = {
        "writes": len(write_ops),
        "llm_calls_per_write": mean(op["llm_calls"] for op in write_ops),
        "seconds_per_write": mean(latencies),
        "evolution_rate": mean(1.0 if op["evolved"] else 0.0 for op in write_ops),
        "answer_prompt_tokens_per_question": mean(r["prompt_tokens"] for r in rows) if rows else 0,
        "write_latency_p50": float(np.percentile(latencies, 50)),
        "write_latency_p95": float(np.percentile(latencies, 95)),
    }
    sessions = session_rows(write_ops)
    cost["llm_calls_per_session"] = mean(s["llm_calls"] for s in sessions) if sessions else None
    cost["per_session"] = sessions
    gate = gate_summary(write_ops)
    if gate:
        cost["gate"] = gate
    return cost


def session_rows(write_ops) -> list:
    """Totals per (conversation, session). Ops logged before sessions were recorded are left out."""
    groups = defaultdict(list)
    for op in write_ops:
        if op.get("session") is not None:
            groups[(int(op["sample_id"]), op["session"])].append(op)
    out = []
    for (conv, session), ops in sorted(groups.items()):
        row = {"conversation": conv, "session": session, "writes": len(ops),
               "llm_calls": sum(op["llm_calls"] for op in ops),
               "write_seconds": sum(op["latency_s"] for op in ops)}
        if any(op.get("gate") for op in ops):
            row["skipped"] = sum(op.get("gate") == "skipped" for op in ops)
        out.append(row)
    return out


def gate_summary(write_ops) -> dict:
    """Decision counts, skip rate and the similarity distribution for gated runs."""
    gated = [op for op in write_ops if op.get("gate")]
    if not gated:
        return {}
    counts = Counter(op["gate"] for op in gated)
    sims = [op["neighbour_similarity"] for op in gated if op.get("neighbour_similarity") is not None]
    return {
        "decisions": {reason: counts.get(reason, 0) for reason in ("no_neighbours", "conflict", "neighbour", "skipped")},
        "skip_rate": counts.get("skipped", 0) / len(gated),
        "similarity": {f"p{q}": float(np.percentile(sims, q)) for q in (10, 50, 90)} if sims else None,
    }


def to_markdown(summary: dict) -> str:
    lines = [f"## {summary['run_name']} ({summary['system']}, {summary['model']}, "
             f"{summary['conversations']} conversations)", ""]
    if summary.get("conversation_ids") is not None:
        ids = ", ".join(map(str, summary["conversation_ids"]))
        lines += [f"Conversations: {ids} (seed {summary.get('seed')})", ""]
    lines += ["| Category | n | F1 | BLEU-1 |", "|---|---|---|---|"]
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
        if "write_latency_p50" in cost:
            lines += [f"| Write latency p50 / p95 (s) | {cost['write_latency_p50']:.2f} / {cost['write_latency_p95']:.2f} |"]
        if cost.get("llm_calls_per_session") is not None:
            lines += [f"| LLM calls per session | {cost['llm_calls_per_session']:.1f} |"]
        gate = cost.get("gate")
        if gate:
            d = gate["decisions"]
            lines += [f"| Gate skip rate | {100 * gate['skip_rate']:.1f}% |",
                      f"| Gate decisions (no_neighbours / conflict / neighbour / skipped) | "
                      f"{d['no_neighbours']} / {d['conflict']} / {d['neighbour']} / {d['skipped']} |"]
            if gate["similarity"]:
                s = gate["similarity"]
                lines += [f"| Neighbour similarity p10 / p50 / p90 | {s['p10']:.3f} / {s['p50']:.3f} / {s['p90']:.3f} |"]
        if cost.get("per_session"):
            gated = any("skipped" in r for r in cost["per_session"])
            lines += ["", "| Conversation | Session | Writes | LLM calls | Write seconds |" + (" Skipped |" if gated else ""),
                      "|---|---|---|---|---|" + ("---|" if gated else "")]
            for r in cost["per_session"]:
                lines.append(f"| {r['conversation']} | {r['session']} | {r['writes']} | {r['llm_calls']} | "
                             f"{r['write_seconds']:.2f} |" + (f" {r.get('skipped', 0)} |" if gated else ""))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config fields, e.g. --set max_samples=1 model=llama3.2:3b")
    parser.add_argument("--conversation-ids", type=parse_ids, metavar="IDS",
                        help="comma-separated LoCoMo indices to run, e.g. 1,8 (overrides max_samples)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("amem_robust").setLevel(logging.WARNING)

    run(load_config(args.config, args.set, args.conversation_ids))


def parse_ids(text: str) -> list:
    return [int(x) for x in text.split(",") if x.strip()]


def load_config(path: str, overrides=(), conversation_ids=None) -> RunConfig:
    """Read a config file and apply --set key=value overrides."""
    data = json.loads(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        data[key] = json.loads(value) if value.replace(".", "", 1).isdigit() or value in ("true", "false", "null") else value
    if conversation_ids is not None:
        data["conversation_ids"] = conversation_ids
    return RunConfig(**data)


if __name__ == "__main__":
    main()
