"""FactConsolidation replay (MemoryAgentBench, Conflict_Resolution split).

Each context is a numbered list of facts, some of which later facts contradict.
It is written to memory in chunks of --chunk-facts numbered lines (one
memory.write per chunk, numbering kept), then the questions are asked.

Output goes to results/<run_name>/, like run_locomo:
  sample_<tag>.jsonl, ops_sample_<tag>.jsonl, memory_sample_<tag>.sqlite,
  summary_factcons_<variant>_<size>.json / .md
where <tag> is "<variant>_<size>" (one context = one sample). A finished sample
(both jsonl files present) is skipped on re-run. Use a separate run_name per
experiment: the summary names carry variant and size, the sample files do not
carry the system.

    python -m harness.run_factcons --config configs/tsom_qwen2.5-3b.json --variant sh \\
        --max-samples 1 --max-questions 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional, Tuple

from harness.contradictions import contradiction_line, read_curve, summarise_curves
from harness.factcons_answer import answer_question, mark_stale, score, stale_counts
from harness.run_locomo import cost_summary
from tsom.contracts import RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.rag import RagMemory
from tsom.temporal.serial import split_serial_facts

log = logging.getLogger("harness")
DATASET = "ai-hyz/MemoryAgentBench"
SPLIT = "Conflict_Resolution"
SIZES = ("6k", "32k", "64k", "262k")
_SOURCE = re.compile(r"^factconsolidation_(sh|mh)_(\d+k)$")


# ---- data ----

def load_samples(variant: str, size: str = "6k", loader: Optional[Callable] = None) -> List[dict]:
    """Select FactConsolidation rows. Schema (checked on the real dataset): each row has
    context (str: "Here is a list of facts:" then one "N. fact" per line, N from 0),
    questions (100 strings), answers (100 lists of accepted strings) and
    metadata["source"] == "factconsolidation_<sh|mh>_<size>". `loader` returns the rows
    (default: HuggingFace); tests pass their own.
    """
    if loader is None:
        from datasets import load_dataset
        rows = load_dataset(DATASET, split=SPLIT)
    else:
        rows = loader()
    picked, seen = [], []
    for row in rows:
        source = (row.get("metadata") or {}).get("source", "")
        seen.append(source)
        m = _SOURCE.match(source)
        if m and m.group(1) == variant and size in ("all", m.group(2)):
            facts = split_serial_facts(row["context"])
            if not facts:
                raise ValueError(f"no numbered facts found in {source}")
            answers = [[a] if isinstance(a, str) else list(a) for a in row["answers"]]
            picked.append({"tag": f"{variant}_{m.group(2)}", "source": source, "size": m.group(2),
                           "facts": facts, "questions": list(row["questions"]), "answers": answers})
    if not picked:
        raise ValueError(f"no FactConsolidation rows for variant={variant} size={size}; "
                         f"sources in the data: {seen}")
    return sorted(picked, key=lambda s: int(s["size"][:-1]))


def chunk_facts(facts: List[Tuple[int, str]], n: int) -> List[Tuple[int, int, str]]:
    """Group numbered facts into chunks of n. Returns (first_serial, last_serial, text);
    the text keeps each fact's own "N. " prefix."""
    n = max(1, n)
    out = []
    for i in range(0, len(facts), n):
        part = facts[i:i + n]
        out.append((part[0][0], part[-1][0], "\n".join(f"{num}. {text}" for num, text in part)))
    return out


# ---- run ----

def run(config: RunConfig, variant: str, size: str = "6k", chunk: int = 20,
        samples: Optional[List[dict]] = None) -> dict:
    out = Path(config.output_dir) / config.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(config.model_dump_json(indent=2))

    if samples is None:
        samples = load_samples(variant, size)
    samples = samples[: config.max_samples]
    llm = LLM.from_config(config)
    llm.check()

    rows, ops, curves = [], [], {}
    for sample in samples:
        tag = sample["tag"]
        pred_file, ops_file = out / f"sample_{tag}.jsonl", out / f"ops_sample_{tag}.jsonl"
        if pred_file.exists() and ops_file.exists():
            log.info("%s already done, loading", tag)
            rows += [mark_stale(json.loads(line), sample)
                     for line in pred_file.read_text().splitlines()]
            ops += [json.loads(line) for line in ops_file.read_text().splitlines()]
        else:
            sample_rows, sample_ops = run_sample(config, llm, sample, chunk, out)
            rows += sample_rows
            ops += sample_ops
        if config.system == "tsom":
            curves[tag] = read_curve(out / f"memory_sample_{tag}.sqlite", tag,
                                     config.multivalued_attributes)

    summary = summarise_run(config, variant, size, chunk, rows, ops)
    summary["contradictions"] = summarise_curves(curves)
    stem = f"summary_factcons_{variant}_{size}"
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    (out / f"{stem}.md").write_text(to_markdown(summary))
    print(to_markdown(summary))
    return summary


def run_sample(config: RunConfig, llm: LLM, sample: dict, chunk: int, out: Path):
    tag = sample["tag"]
    db = out / f"memory_sample_{tag}.sqlite"
    db.unlink(missing_ok=True)
    memory_cls = RagMemory if config.system == "rag" else MemorySystem
    memory = memory_cls(config, llm, db_path=str(db), sample_id=tag)

    chunks = chunk_facts(sample["facts"], chunk)
    log.info("%s: writing %d facts in %d chunks", tag, len(sample["facts"]), len(chunks))
    start = time.perf_counter()
    for i, (first, last, text) in enumerate(chunks, 1):
        memory.write(text, f"facts {first}-{last}")
        if i % 10 == 0 or i == len(chunks):
            log.info("%s: chunk %d/%d (%.0fs)", tag, i, len(chunks), time.perf_counter() - start)
    memory.sync_all()

    rows = []
    pairs = list(zip(sample["questions"], sample["answers"]))[: config.max_questions]
    for qid, (question, golds) in enumerate(pairs):
        result = answer_question(memory, llm, question, config.retrieve_k)
        rows.append(mark_stale({
            "sample": tag, "qid": qid, "question": question, "gold": golds,
            "prediction": result["prediction"], **score(result["prediction"], golds),
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "tools": [{"tool": "search_memory", "query": question, "k": config.retrieve_k}],
            "context": result["context"],  # the trace for this answer
        }, sample))

    ops = [op.model_dump() for op in memory.store.ops()]
    memory.store.close()
    # Ops are written last: their file marks the sample as finished.
    (out / f"sample_{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / f"ops_sample_{tag}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in ops))
    return rows, ops


# ---- summary ----

def summarise_run(config: RunConfig, variant: str, size: str, chunk: int,
                  rows: List[dict], ops: List[dict]) -> dict:
    by_sample: Dict[str, List[dict]] = {}
    for r in rows:
        by_sample.setdefault(r["sample"], []).append(r)
    ingest = [op for op in ops if op["op_type"] in ("write", "temporal_write")]
    answers = [op for op in ops if op["op_type"] == "answer"]
    cost = cost_summary(rows, ingest)
    cost.update({
        "ingest_llm_calls": sum(op["llm_calls"] for op in ingest),
        "ingest_prompt_tokens": sum(op["prompt_tokens"] for op in ingest),
        "ingest_completion_tokens": sum(op["completion_tokens"] for op in ingest),
        "answer_llm_calls": sum(op["llm_calls"] for op in answers),
        "answer_prompt_tokens": sum(op["prompt_tokens"] for op in answers),
        "answer_completion_tokens": sum(op["completion_tokens"] for op in answers),
    })
    return {
        "run_name": config.run_name, "system": config.system, "backend": config.backend,
        "model": config.model, "variant": variant, "size": size, "chunk_facts": chunk,
        "retrieve_k": config.retrieve_k, "samples": len(by_sample), "questions": len(rows),
        "accuracy": mean(1.0 if r["correct"] else 0.0 for r in rows) if rows else 0.0,
        "f1": mean(r["f1"] for r in rows) if rows else 0.0,
        "stale": stale_counts(rows),
        "per_sample": {tag: {"n": len(rs),
                             "accuracy": mean(1.0 if r["correct"] else 0.0 for r in rs),
                             "f1": mean(r["f1"] for r in rs),
                             **stale_counts(rs)} for tag, rs in by_sample.items()},
        "cost": cost,
        # TSOM only: fact operation counts and temporal steps that failed and were skipped
        "tsom": {"fact_ops": cost.get("fact_ops", {}),
                 "failures": cost.get("temporal_failures", 0)} if "temporal_writes" in cost else None,
    }


def to_markdown(s: dict) -> str:
    lines = [f"## {s['run_name']} FactConsolidation-{s['variant']} {s['size']} "
             f"({s['system']}, {s['model']}, {s['samples']} samples, {s['questions']} questions)", "",
             "| Sample | n | Accuracy | F1 |", "|---|---|---|---|"]
    for tag, v in s["per_sample"].items():
        lines.append(f"| {tag} | {v['n']} | {v['accuracy']:.3f} | {v['f1']:.3f} |")
    c, stale = s["cost"], s["stale"]
    note = " Multi-hop estimate: only stale values of the last fact are counted." if s["variant"] == "mh" else ""
    lines += [f"| **All** | {s['questions']} | **{s['accuracy']:.3f}** | {s['f1']:.3f} |", "",
              f"Stale answers: {stale['stale_answers']} of {stale['changed_questions']} questions "
              f"whose fact changed ({stale['stale_rate']:.3f}).{note}", "",
              *([contradiction_line(s["contradictions"]), ""] if s["contradictions"] else []),
              "| Cost | Value |", "|---|---|",
              f"| Ingest LLM calls | {c['ingest_llm_calls']} |",
              f"| Ingest tokens (prompt / completion) | {c['ingest_prompt_tokens']} / {c['ingest_completion_tokens']} |",
              f"| Answer LLM calls | {c['answer_llm_calls']} |",
              f"| Answer tokens (prompt / completion) | {c['answer_prompt_tokens']} / {c['answer_completion_tokens']} |"]
    if s["tsom"] is not None:
        ops = ", ".join(f"{k} {v}" for k, v in sorted(s["tsom"]["fact_ops"].items())) or "none"
        lines += [f"| Fact operations | {ops} |", f"| Temporal failures | {s['tsom']['failures']} |"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True, choices=["sh", "mh"],
                        help="FactConsolidation single-hop or multi-hop")
    parser.add_argument("--size", default="6k", choices=[*SIZES, "all"], help="context length")
    parser.add_argument("--chunk-facts", type=int, default=20, help="numbered facts per memory write")
    parser.add_argument("--max-samples", type=int, help="override config max_samples")
    parser.add_argument("--max-questions", type=int, help="override config max_questions")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("amem_robust").setLevel(logging.WARNING)

    data = json.loads(Path(args.config).read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        data[key] = json.loads(value) if value.replace(".", "", 1).isdigit() or value in ("true", "false", "null") else value
    if args.max_samples is not None:
        data["max_samples"] = args.max_samples
    if args.max_questions is not None:
        data["max_questions"] = args.max_questions
    run(RunConfig(**data), args.variant, args.size, args.chunk_facts)


if __name__ == "__main__":
    main()
