# AGENTS.md

Local instructions for coding agents working on this repository. This file is gitignored.

## Project

Agent-memory systems over LoCoMo and FactConsolidation: `amem` (A-Mem, vendored and
unchanged), `rag` (raw-turn retrieval) and `tsom` (A-Mem plus an append-only temporal
fact layer), run through one shared agent, tool set and harness.

Python 3.10+. The only test framework is pytest. There is no lint or typecheck
command configured.

## Commands

Run everything from the repository root.

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b

# Tests (offline, CPU only, fake LLM)
python -m pytest -q tests
python -m pytest -q tests/test_store.py::test_note_round_trip

# Demo (use --backend fake to avoid a model)
python scripts/demo.py --backend fake
python scripts/demo.py

# Baselines
bash scripts/run_baseline.sh --set max_samples=1 max_questions=20
bash scripts/run_baseline.sh --config configs/rag_qwen2.5-3b.json

# Direct harness (Windows / no bash)
python -m harness.run_locomo --config configs/amem_qwen2.5-3b.json --set max_samples=1 max_questions=20

# TSOM (temporal layer)
python scripts/demo_temporal.py --backend fake
python scripts/demo_temporal.py --classifier llm_only
python -m harness.run_locomo --config configs/tsom_qwen2.5-3b.json --set max_samples=1 max_questions=20
python -m harness.run_factcons --config configs/tsom_qwen2.5-3b.json --variant sh --max-samples 1 --max-questions 20
python -m harness.run_locomo --config configs/tsom_llmonly_qwen2.5-3b.json --set max_samples=1

# Inspect a finished run
python scripts/inspect_memory.py results/<run>/memory_sample_0.sqlite --notes 5
```

`run_baseline.sh` defaults to `configs/amem_qwen2.5-3b.json` and passes extra
arguments through to `harness.run_locomo`.

## Layout

- `tsom/contracts.py` — Pydantic contracts: `Note`, `Fact`, `MemoryOp`, `RunConfig`.
  `python -m tsom.contracts` prints their JSON schemas.
- `tsom/store.py` — SQLite tables: notes, operation log, `facts` (append-only), `note_versions`.
- `tsom/memory.py` — `MemorySystem`, the A-Mem-backed implementation; also runs the temporal layer when `system == "tsom"`.
- `tsom/temporal/` — TSOM: `extractor`/`prompts`/`parsers` (fact extraction), `attributes` (attribute-name canonicalization), `classifier` (update operation; `rules_llm` or `llm_only`), `consolidator` (applies the operation append-only), `retriever` (notes + facts, current vs historical), `dates`, `serial`.
- `tsom/rag.py` — `RagMemory`, the RAG baseline (no LLM calls on write).
- `tsom/tools.py` — the agent tools and their OpenAI-style specs.
- `tsom/llm.py` — Ollama/OpenAI/vLLM/fake backend plus usage accounting.
- `tsom/amem_bridge.py` — the only module that imports vendored A-Mem code.
- `tsom/graph.py` — NetworkX view over the SQLite store.
- `agent/assistant.py` — the agent loop: store turns, retrieve, answer.
- `harness/run_locomo.py` — LoCoMo replay, output files and summary tables.
- `harness/run_factcons.py`, `harness/factcons_answer.py` — FactConsolidation (MemoryAgentBench) replay, answer prompt and scoring.
- `harness/metrics.py` — F1 and BLEU-1, copied from A-Mem.
- `scripts/` — `run_baseline.sh`, `demo.py`, `demo_temporal.py`, `inspect_memory.py`.
- `configs/` — one JSON `RunConfig` per run.
- `tests/` — offline tests; `conftest.py` stubs heavy deps when missing.
- `third_party/amem/` — vendored A-Mem at pinned commit `0c8039f` (MIT).
- `data/locomo10.json` — LoCoMo dataset (10 conversations, 1,986 questions).
- `results/<run_name>/` — run outputs.

## Conventions

- **Do not edit `third_party/amem/`.** It is vendored unchanged. Import it only
  through `tsom/amem_bridge.py`.
- Facts are append-only: never delete or overwrite fact rows. Only `status`, `valid_until`,
  `superseded_by` and `observation_count` may change (`Store.update_fact_status`).
- `classifier_mode` defaults to `rules_llm`; keep that default so existing runs are unchanged.
  In `llm_only` an unparseable reply is retried once, then falls back to the rules.
- Keep `MemorySystem` and `RagMemory` interface-compatible: `write`, `search`,
  `sync_all`. `agent/` and `tsom/tools.py` are shared
  between both systems.
- Shared data crosses component boundaries only as the Pydantic models in
  `tsom/contracts.py`. A-Mem note field names in `Note` are intentional.
- Tests must stay offline and CPU-only: use `backend="fake"` and rely on the
  stubs in `tests/conftest.py`. Do not add network or GPU requirements to tests.
- Add runs as JSON in `configs/` and override fields with `--set key=value`;
  avoid hard-coding run parameters.
- Keep evaluation definitions in `harness/metrics.py` identical to A-Mem so
  numbers stay comparable.
- Only `summary.md` and `summary.json` are committed from `results/`. Do not
  commit `sample_*.jsonl`, `ops_sample_*.jsonl` or `memory_sample_*.sqlite`.

## Gotchas

- The harness skips a conversation when both `sample_<i>.jsonl` and
  `ops_sample_<i>.jsonl` exist. Change `run_name` or delete those files to rerun.
- `ops_sample_<i>.jsonl` is written last and marks the conversation as finished.
- Relative `dataset` paths are resolved from the repo root, not the CWD.
- `harness/metrics.py` downloads NLTK `punkt`/`punkt_tab` on first import, which
  may need network access once.
- Use `.venv/bin/python`; there may be no `python` on PATH.
- `run_factcons` downloads `ai-hyz/MemoryAgentBench` from HuggingFace on first use (network once).
- The parser's default for an unparseable classifier reply is `REPLACE` (`rules_llm`), which can
  silently supersede facts when a small model misbehaves.
- `OPENAI_API_KEY` is read from the environment (`.env` is gitignored). Never
  commit secrets or keys.
