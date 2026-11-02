# Time-Scoped Ontological Memory (TSOM)

Conflict-aware self-organising memory for persistent LLM agents. AI4015 Agentic AI, Fall 2026, project G1-08.

We build on A-Mem (Xu et al., NeurIPS 2025). A-Mem stores every conversation turn as a linked note and lets new notes rewrite old ones ("evolution"). It never checks whether a new fact contradicts an old one, it runs evolution on every write, and it has no idea when a fact stopped being true. TSOM adds three things on top:

1. **Conflict classifier** (Gap 1): labels each new fact as additive, corrective, contradictory or redundant.
2. **Conflict gate** (Gap 2): runs A-Mem's evolution only when something conflicts or is very similar.
3. **Time-scoped facts** (Gap 3): old facts get a "true until" date instead of being overwritten.

**Phase 1 status:** two baselines run end to end on LoCoMo through our own agent, tools and harness:

- **A-Mem** (`system: amem`): the base paper's memory, using the authors' code unchanged.
- **RAG** (`system: rag`): stores raw turns and retrieves the top-k by embedding similarity. No LLM calls when storing, no links, no evolution. Same embedder, same answer prompts, same k as A-Mem, so the only difference is the memory.

The three additions live in `tsom/facts.py` (Gap 1), `tsom/gated_amem.py` (Gap 2) and `tsom/dates.py` / `tsom/context.py` (Gap 3); each is switched by its own flag. The MemGPT-style baseline is still to be built.

![Architecture](docs/architecture_diagram.png)

## Repository layout

```
tsom/            memory module
  contracts.py   Note, Fact, MemoryOp, RunConfig (JSON schemas: python -m tsom.contracts)
  schema.py      the 15 predicates, single- or multi-valued
  store.py       SQLite tables for notes, facts and the operation log
  memory.py      wraps A-Mem, logs cost and latency of every write; Phase 2 hooks
  rag.py         RAG baseline: raw turns, top-k retrieval, no LLM on write
  tools.py       write_memory, search_memory, get_fact_history, inspect_memory
  llm.py         provider setup: Ollama, OpenAI, vLLM, or a fake model for tests
  graph.py       NetworkX view of notes and facts
agent/           the assistant: stores turns, answers questions through the tools
harness/         LoCoMo replay, F1 and BLEU-1 (same definitions as A-Mem), cost tables
scripts/         run_baseline.sh, demo.py, inspect_memory.py
configs/         one JSON file per run
notebooks/       colab_baseline.ipynb, for running on a free Colab GPU
third_party/amem A-Mem's code, copied unchanged from a pinned commit (MIT)
data/            locomo10.json (10 conversations, 1,986 questions)
tests/           offline tests, no model or GPU needed
```

## Running it

### Option 1: Google Colab (free GPU, recommended)

Open `notebooks/colab_baseline.ipynb` in Colab, set the runtime to T4 GPU and run the cells. It installs Ollama, downloads `qwen2.5:3b`, runs the tests and the demo, then the baseline. Results are saved to Google Drive.

### Option 2: your own machine

Needs Python 3.10+ and [Ollama](https://ollama.com). A GPU helps a lot; on CPU the full run is very slow.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b

python -m pytest -q tests                                    # offline checks
python scripts/demo.py                                       # small two-session demo
bash scripts/run_baseline.sh --set max_samples=1 max_questions=20   # quick test
bash scripts/run_baseline.sh                                 # full A-Mem baseline, all 10 conversations
bash scripts/run_baseline.sh --config configs/rag_qwen2.5-3b.json   # full RAG baseline
```

On Windows without bash, call the harness directly (activate with `.venv\Scripts\activate`):

```
python -m harness.run_locomo --config configs/amem_qwen2.5-3b.json --set max_samples=1 max_questions=20
python -m harness.run_locomo --config configs/rag_qwen2.5-3b.json --set max_samples=1 max_questions=20
```

To use a different model, pick another config (`configs/amem_llama3.2-3b.json`, or `configs/amem_gpt-4o-mini.json` with `OPENAI_API_KEY` set) or override fields: `--set model=llama3.2:3b run_name=my_run`.

### What a run produces

Everything goes to `results/<run_name>/`:

- `summary.md` and `summary.json`: F1 and BLEU-1 per LoCoMo category, plus LLM calls and seconds per memory write, and prompt tokens per question.
- `sample_<i>.jsonl`: every question, the prediction, the reference answer and its scores.
- `ops_sample_<i>.jsonl`: a trace of every memory operation (type, LLM calls, tokens, latency).
- `memory_sample_<i>.sqlite`: the memory after the conversation. Look inside with `python scripts/inspect_memory.py results/<run>/memory_sample_0.sqlite --notes 5`.

If a run stops halfway, run the same command again. Finished conversations are skipped.

## Gap 2: conflict gate

A-Mem runs its evolution calls (decision, then strengthen and/or update neighbours) on every write. The gate (`tsom/gated_amem.py`) runs them only if

- a fact from this turn was labelled **corrective** or **contradictory** by the fact layer, or
- some stored note has cosine similarity **>= tau** with the new note's content (the text A-Mem uses to find neighbours).

Otherwise evolution is skipped and the note is linked to its `gate_link_k` most similar stored notes by embedding alone. The gate only removes A-Mem's evolution calls: the analysis call that builds the note and the fact extraction call always run, and the gate makes no LLM call itself, so it behaves the same on every backend (ollama, openai, vllm). Every write op records the gate's reason (`no_neighbours`, `conflict`, `neighbour`, `skipped`), the max similarity it saw and the LoCoMo session.

Flags (in the run config):

- `use_gate` (default `false`): turn the gate on. Needs `system: "tsom"`, because the gate reads the fact labels.
- `tau` (default `0.8`): similarity threshold.
- `gate_link_k` (default `3`): embedding links given to a note whose evolution was skipped.

`configs/gate/tsom_full_gated.json` is the full system (classifier, validity and gate). `configs/exp/tsom_both.json` is the same system without the gate (the "-gate" ablation).

**Choose tau on the development conversation (LoCoMo 0) only:**

```bash
python scripts/sweep_tau.py --taus 0.6,0.7,0.8,0.9 --plot
```

It runs the ungated system once and the gated system once per tau, all on conversation 0, and writes `results/tau_sweep/sweep.md`, `sweep.json` and `tau_sweep.png`. It also prints the similarity distribution so the grid can be moved. tau is never re-tuned on a test conversation.

**Ungated vs gated on chosen conversations** (repeat with different seeds for mean and std):

```bash
python -m harness.run_locomo --config configs/exp/tsom_both.json --conversation-ids 1,8 --set seed=1 run_name=tsom_both_s1
python -m harness.run_locomo --config configs/gate/tsom_full_gated.json --conversation-ids 1,8 --set seed=1 tau=0.8 run_name=gated_s1
```

`scripts/run_experiments.py --suite locomo` also accepts `--conversation-ids`.

**Table and H2 verdict:**

```bash
python scripts/make_gate_table.py --ungated results/tsom_both_s1,results/tsom_both_s2,results/tsom_both_s3 --gated results/gated_s1,results/gated_s2,results/gated_s3
```

It writes `results/gate_table.md` and `.json` with mean and sample std (or "n=1, no std"), and checks H2 as three separate claims, each printed: (a) fewer LLM calls per write, (b) lower p50 and p95 write latency, (c) overall F1 within 2 points (a drop of at most 2 points). With fewer than 3 runs per side the verdict is marked indicative only.

Metric definitions:

- **LLM calls per write**: every LLM call made during `write()`, including tsom's fact extraction and A-Mem's analysis call.
- **Write latency**: wall-clock time of `write()`, including extraction, embedding and the A-Mem call, excluding retrieval.
- **p50 / p95**: percentiles over all writes in the run.
- **Per-session rows**: totals per (conversation, session).
- **Skip rate**: share of writes whose evolution the gate skipped.

tsom's fact extraction adds about one LLM call per write compared with A-Mem, so the gate has to skip evolution on more than about half the writes before tsom is cheaper than A-Mem.

## Evaluation

We follow A-Mem's own LoCoMo evaluation: all 10 conversations, all five question categories, F1 and BLEU-1, the same answer prompts and the same retrieval k. The baseline uses A-Mem's released reproduction code without changes, so our A-Mem numbers can be checked against Table 1 of their paper for the same model.

## Results so far

Initial results live in `results/<run_name>/summary.md`. Each table says which model was used and how many conversations it covers. Runs on fewer than 10 conversations are partial and are labelled that way.

## Credits

`third_party/amem/` is from [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory) (MIT licence), commit `0c8039f`. LoCoMo is from Maharana et al., ACL 2024.
