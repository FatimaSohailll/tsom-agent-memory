# Agent-memory baselines: A-Mem and RAG

Two memory systems for persistent LLM agents, evaluated on LoCoMo through one shared agent, tool interface and harness:

- **A-Mem** (`system: amem`): Xu et al., NeurIPS 2025. Every conversation turn is stored as a linked note, and new notes can rewrite old ones ("evolution"). We use the authors' code unchanged.
- **RAG** (`system: rag`): stores raw turns and retrieves the top-k by embedding similarity. No LLM calls when storing, no links, no evolution. Same embedder, same answer prompts, same k as A-Mem, so the only difference is the memory.

## Repository layout

```
tsom/            memory module
  contracts.py   Note, MemoryOp, RunConfig (JSON schemas: python -m tsom.contracts)
  store.py       SQLite tables for notes and the operation log
  memory.py      wraps A-Mem, logs cost and latency of every write
  rag.py         RAG baseline: raw turns, top-k retrieval, no LLM on write
  tools.py       write_memory, search_memory
  llm.py         provider setup: Ollama, OpenAI, vLLM, or a fake model for tests
  graph.py       NetworkX view of notes and their links
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

## Evaluation

We follow A-Mem's own LoCoMo evaluation: all 10 conversations, all five question categories, F1 and BLEU-1, the same answer prompts and the same retrieval k. The baseline uses A-Mem's released reproduction code without changes, so our A-Mem numbers can be checked against Table 1 of their paper for the same model.

## Results

Initial results live in `results/<run_name>/summary.md`. Each table says which model was used and how many conversations it covers. Runs on fewer than 10 conversations are partial and are labelled that way.

## Credits

`third_party/amem/` is from [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory) (MIT licence), commit `0c8039f`. LoCoMo is from Maharana et al., ACL 2024.
