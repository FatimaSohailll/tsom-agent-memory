"""End-to-end run on the first LoCoMo conversation with the fake LLM.

Checks the plumbing: replay, A-Mem writes, tool calls, answering, metrics,
traces and the summary table. Scores are meaningless with a fake model.
"""

import json

from harness.run_locomo import run
from tsom.contracts import RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.tools import TOOL_SPECS, MemoryTools


def test_locomo_smoke(tmp_path):
    config = RunConfig(run_name="smoke", backend="fake", model="fake", max_samples=1,
                       max_sessions=2, max_questions=10, output_dir=str(tmp_path))
    summary = run(config)

    out = tmp_path / "smoke"
    rows = [json.loads(line) for line in (out / "sample_0.jsonl").read_text().splitlines()]
    ops = [json.loads(line) for line in (out / "ops_sample_0.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    assert summary["scores"]["Overall"]["n"] == 10
    writes = [op for op in ops if op["op_type"] == "write"]
    assert writes and all(op["llm_calls"] >= 1 for op in writes)
    assert summary["cost"]["writes"] == len(writes)
    assert (out / "summary.md").exists()

    # Re-running skips the finished conversation and gives the same summary.
    again = run(config)
    assert again["scores"] == summary["scores"]


def test_tools_match_specs():
    memory = MemorySystem(RunConfig(run_name="t", backend="fake", model="fake"), LLM("fake", "fake"))
    tools = MemoryTools(memory)
    for spec in TOOL_SPECS:
        assert hasattr(tools, spec["function"]["name"])
    note_id = tools.call("write_memory", turn="Speaker Ali says : I live in Karachi", time="1 May 2023")["note_id"]
    assert memory.store.get_note(note_id) is not None
    assert "Karachi" in tools.call("search_memory", query="Karachi", k=1)["notes"]
