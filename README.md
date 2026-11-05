# Agent memory: A-Mem, RAG and a temporal fact layer (TSOM)

Three memory systems for persistent LLM agents, evaluated through one shared agent, tool interface and harness (LoCoMo, and FactConsolidation for the temporal system):

- **A-Mem** (`system: amem`): Xu et al., NeurIPS 2025. Every conversation turn is stored as a linked note, and new notes can rewrite old ones ("evolution"). We use the authors' code unchanged.
- **RAG** (`system: rag`): stores raw turns and retrieves the top-k by embedding similarity. No LLM calls when storing, no links, no evolution. Same embedder, same answer prompts, same k as A-Mem, so the only difference is the memory.
- **TSOM** (`system: tsom`): A-Mem plus an append-only temporal fact layer. Each write also extracts `(entity, attribute, value)` facts, classifies how each relates to what is already stored (`ADD`, `REAFFIRM`, `REPLACE`, `CORRECT`, `ADD_HISTORICAL`) and records supersession without ever deleting rows. Retrieval returns A-Mem's notes plus the facts behind them, marked `[CURRENT]` or `[PAST]`.

## Repository layout

```
tsom/            memory module
  contracts.py   Note, Fact, MemoryOp, RunConfig (JSON schemas: python -m tsom.contracts)
  store.py       SQLite tables: notes, operation log, facts (append-only), note versions
  memory.py      wraps A-Mem, logs cost and latency of every write; runs the temporal layer when system=tsom
  temporal/      TSOM: extractor, attribute canonicalizer, update classifier, consolidator, retriever
  rag.py         RAG baseline: raw turns, top-k retrieval, no LLM on write
  tools.py       write_memory, search_memory
  llm.py         provider setup: Ollama, OpenAI, vLLM, or a fake model for tests
  graph.py       NetworkX view of notes and their links
agent/           the assistant: stores turns, answers questions through the tools
harness/         LoCoMo replay (run_locomo), FactConsolidation replay (run_factcons), F1 and BLEU-1 (same definitions as A-Mem), cost tables
scripts/         run_baseline.sh, demo.py, demo_temporal.py, inspect_memory.py
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

### TSOM and FactConsolidation

```bash
python scripts/demo_temporal.py                          # Ali demo with the real model
python scripts/demo_temporal.py --backend fake           # offline
python scripts/demo_temporal.py --classifier llm_only    # LLM decides every differing value

python -m harness.run_locomo --config configs/tsom_qwen2.5-3b.json --set max_samples=1 max_questions=20
python -m harness.run_factcons --config configs/tsom_qwen2.5-3b.json --variant sh --max-samples 1 --max-questions 20
```

FactConsolidation (MemoryAgentBench, `Conflict_Resolution` split) is loaded from HuggingFace on first use and needs network access once.

#### Update classifier modes

`classifier_mode` in the config (or `--set classifier_mode=llm_only`) controls how a new fact that differs from the stored value is classified:

- `rules_llm` (default): deterministic rules decide first (no existing fact, same value, earlier `valid_from`, past tense before the existing fact, several values already stored, and the optional `seed_multivalued` and `additive_cue` rules). The LLM is asked only for what the rules leave open.
- `llm_only`: the LLM decides every differing value, including `ADD_HISTORICAL`. An empty key (`ADD`) and an identical value (`REAFFIRM`) still need no call. Unparseable output is retried once, then falls back to the rule-based path. `configs/tsom_llmonly_qwen2.5-3b.json` is ready to use.

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

## Evaluation

FactConsolidation is scored by substring-match accuracy and F1 (`harness/factcons_answer.py`), with the same answer prompt and retrieval call for every memory system.

We follow A-Mem's own LoCoMo evaluation: all 10 conversations, all five question categories, F1 and BLEU-1, the same answer prompts and the same retrieval k. The baseline uses A-Mem's released reproduction code without changes, so our A-Mem numbers can be checked against Table 1 of their paper for the same model.

## Results

Initial results live in `results/<run_name>/summary.md`. Each table says which model was used and how many conversations it covers. Runs on fewer than 10 conversations are partial and are labelled that way.

## Credits

`third_party/amem/` is from [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory) (MIT licence), commit `0c8039f`. LoCoMo is from Maharana et al., ACL 2024.
