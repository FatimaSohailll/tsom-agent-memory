import pytest

from agent.assistant import Assistant
from tsom.contracts import RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.temporal.extractor import FactExtractor, split_turn
from tsom.temporal.parsers import parse_facts
from tsom.temporal.serial import split_serial_facts
from tsom.tools import MemoryTools


class Canned:
    def __init__(self, *responses):
        self.responses, self.prompts = list(responses), []

    def get_completion(self, prompt, temperature=0.7):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_normal_parse():
    resp = """FACT 1:
ENTITY: France
ATTRIBUTE: capital_of
VALUE: Paris
VALID_FROM: 1900
VALID_UNTIL: NONE
TENSE: present

FACT 2:
ENTITY: Paris
ATTRIBUTE: located_in
VALUE: France
VALID_FROM: none
VALID_UNTIL: N/A"""
    facts = parse_facts(resp)
    assert facts == [
        {"entity": "France", "attribute": "capital_of", "value": "Paris",
         "valid_from": "1900", "valid_until": None, "temporal_status": "PRESENT"},
        {"entity": "Paris", "attribute": "located_in", "value": "France",
         "valid_from": None, "valid_until": None, "temporal_status": "UNKNOWN"},
    ]


def test_no_facts():
    assert parse_facts("NO_FACTS") == []
    assert FactExtractor(Canned("NO_FACTS")).extract("hello there") == []
    assert FactExtractor(Canned()).extract("   ") == []  # no LLM call


def test_malformed_output():
    assert parse_facts("") == []
    assert parse_facts(None) == []
    assert parse_facts("I could not find any facts, sorry.") == []
    # missing VALUE -> skipped; bullets, bold and extra whitespace tolerated;
    # missing FACT headers handled by the repeated ENTITY
    resp = """FACT 1:
ENTITY: Bob
ATTRIBUTE: works_at
- **ENTITY**:   Alice
  * ATTRIBUTE:  spouse
  * VALUE:   Bob
  * VALID_FROM:  NONE
ENTITY: Carol
ATTRIBUTE: citizen_of
VALUE: Canada  """
    assert [(f["entity"], f["attribute"], f["value"]) for f in parse_facts(resp)] == [
        ("Alice", "spouse", "Bob"), ("Carol", "citizen_of", "Canada")]


def test_extract_with_fake_backend():
    llm = LLM("fake", "fake")
    facts = FactExtractor(llm).extract("The capital of France is Paris.")
    assert facts[0]["entity"] == "The capital of France" and facts[0]["value"] == "Paris"
    assert llm.usage.calls == 1


def test_batch_mapping_and_chunking():
    llm = LLM("fake", "fake")
    statements = [f"Thing{i} is value{i}." if i % 3 else "No claim here" for i in range(25)]
    facts = FactExtractor(llm).extract_batch(statements)
    assert llm.usage.calls == 3  # 10 + 10 + 5
    got = {f["source"]: f["value"] for f in facts}
    assert got == {i: f"value{i}" for i in range(25) if i % 3}


def test_batch_drops_unmappable_source():
    resp = """FACT 1:
SOURCE: 2
ENTITY: A
ATTRIBUTE: is
VALUE: b
FACT 2:
SOURCE: 9
ENTITY: C
ATTRIBUTE: is
VALUE: d
FACT 3:
ENTITY: E
ATTRIBUTE: is
VALUE: f"""
    facts = FactExtractor(Canned(resp)).extract_batch(["s0", "s1"])
    assert [(f["entity"], f["source"]) for f in facts] == [("A", 1)]


def test_batch_skips_empty_statements_in_numbering():
    llm = Canned("FACT 1:\nSOURCE: 2\nENTITY: X\nATTRIBUTE: is\nVALUE: y")
    facts = FactExtractor(llm).extract_batch(["", "a", "", "b"])
    assert facts[0]["source"] == 3  # second usable statement is index 3
    assert "1. a\n2. b" in llm.prompts[0]


def test_serial_splitting():
    text = "12. The capital of France is Paris.\n  7.   Bob works at Acme.\nnot numbered\n3.x\n4. \n"
    assert split_serial_facts(text) == [
        (12, "The capital of France is Paris."), (7, "Bob works at Acme.")]
    assert split_serial_facts("") == []


class Recording(LLM):
    """Fake backend that keeps every prompt; extraction replies come from a script."""

    def __init__(self, *facts):
        super().__init__("fake", "fake")
        self.facts, self.prompts = list(facts), []

    def _call(self, prompt, temperature):
        self.prompts.append(prompt)
        if prompt.startswith("Extract the atomic facts") and self.facts:
            return self.facts.pop(0), 1, 1
        return super()._call(prompt, temperature)


def lives_in(value):
    return (f"FACT 1:\nENTITY: Ali\nATTRIBUTE: lives_in\nVALUE: {value}\n"
            "VALID_FROM: NONE\nVALID_UNTIL: NONE\nTENSE: PRESENT")


@pytest.mark.parametrize("content,expected", [
    ("Speaker Alisays : I moved", ("Ali", "I moved")),
    ("Speaker Ali says : I moved", ("Ali", "I moved")),
    ("Speaker Mary Annsays : line one\nline two", ("Mary Ann", "line one\nline two")),
    ("I moved to Lahore", (None, "I moved to Lahore")),
    ("1. Paris is in France.", (None, "1. Paris is in France.")),
])
def test_split_turn(content, expected):
    assert split_turn(content) == expected


def test_speaker_is_named_in_the_prompt():
    llm = Canned("NO_FACTS")
    FactExtractor(llm).extract("I moved to Lahore", speaker="Ali")
    assert 'The speaker is Ali. "I", "me" and "my" refer to Ali.' in llm.prompts[0]
    assert llm.prompts[0].endswith("Text:\nI moved to Lahore")


def test_no_speaker_leaves_the_prompt_unchanged():
    llm = Canned("NO_FACTS", "NO_FACTS")
    FactExtractor(llm).extract("Paris is in France.")
    FactExtractor(llm).extract("Paris is in France.", speaker=None)
    assert llm.prompts[0] == llm.prompts[1] and "The speaker is" not in llm.prompts[0]


def test_turn_prefix_never_reaches_the_fact_layer():
    llm = Recording(lives_in("Karachi"), lives_in("Lahore"))
    memory = MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake"),
                          llm, sample_id="s0")
    agent = Assistant(MemoryTools(memory), llm)
    agent.observe("Ali", "I live in Karachi", "1 May 2023")
    agent.observe("Ali", "I moved to Lahore", "2 June 2023")
    extraction = [p for p in llm.prompts if p.startswith("Extract the atomic facts")]
    classifier = [p for p in llm.prompts if p.startswith("A memory already contains")]
    assert len(extraction) == 2 and len(classifier) == 1
    assert not any("says :" in p for p in extraction + classifier)
    assert extraction[1].endswith("Text:\nI moved to Lahore")
    assert "already in use (reuse one of them if it means the same relation, otherwise use " \
           "a new short snake_case name): lives_in" in extraction[1]  # Ali's own names are offered
    assert "I moved to Lahore" in classifier[0]
    assert memory.store.get_note(list(memory.amem.memories)[0]).content == "Speaker Alisays : I live in Karachi"
