import importlib.util
from pathlib import Path

from harness.factcons_answer import ANSWER_PROMPT
from tsom.llm import fake_completion


def load_demo():
    path = Path(__file__).resolve().parent.parent / "scripts" / "demo_temporal.py"
    spec = importlib.util.spec_from_file_location("demo_temporal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_prompt_allows_lists_and_benchmark_prompt_is_unchanged():
    demo = load_demo()
    p = demo.LIST_ANSWER_PROMPT
    assert "list all of those values separated by commas" in p
    assert p.format(context="C", question="Q").endswith("Memory:\nC\nQuestion: Q\nAnswer:")
    assert fake_completion(p.format(context="C", question="Q")) == "fake answer"  # shared opening sentence
    # the FactConsolidation prompt still forces a single entity, exactly as specified for the benchmark
    assert "Answer with only the entity name, no explanation." in ANSWER_PROMPT
    assert "list" not in ANSWER_PROMPT
