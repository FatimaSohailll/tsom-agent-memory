from tsom.llm import LLM
from tsom.temporal.extractor import FactExtractor
from tsom.temporal.parsers import parse_facts
from tsom.temporal.serial import split_serial_facts


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

FACT 2:
ENTITY: Paris
ATTRIBUTE: located_in
VALUE: France
VALID_FROM: none
VALID_UNTIL: N/A"""
    facts = parse_facts(resp)
    assert facts == [
        {"entity": "France", "attribute": "capital_of", "value": "Paris",
         "valid_from": "1900", "valid_until": None},
        {"entity": "Paris", "attribute": "located_in", "value": "France",
         "valid_from": None, "valid_until": None},
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
