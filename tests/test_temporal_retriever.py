import pytest

from tsom.contracts import Fact, RunConfig
from tsom.llm import LLM
from tsom.memory import FACTS_HEADER, MemorySystem
from tsom.temporal.retriever import TemporalRetriever, classify_query


class Scripted(LLM):
    def __init__(self, facts):
        super().__init__("fake", "fake")
        self.facts = list(facts)

    def get_completion(self, prompt, temperature=0.7):
        if self.facts and prompt.startswith("Extract the atomic facts stated in the text."):
            self.usage.calls += 1
            return self.facts.pop(0)
        return super().get_completion(prompt, temperature)


def ft(value, start="NONE", end="NONE", entity="Alice", attr="lives_in"):
    return (f"FACT 1:\nENTITY: {entity}\nATTRIBUTE: {attr}\nVALUE: {value}\n"
            f"VALID_FROM: {start}\nVALID_UNTIL: {end}")


def system(llm=None, name="tsom"):
    return MemorySystem(RunConfig(run_name="t", system=name, backend="fake", model="fake"),
                        llm or LLM("fake", "fake"), sample_id="s0")


@pytest.mark.parametrize("query,expected", [
    ("where does Alice live", ("current", None)),
    ("where did Alice live in 2022", ("historical", "2022")),
    ("Alice March 2021", ("historical", "2021-03")),
    ("Alice in May 2020", ("historical", "2020-05")),
    ("Alice 5th May 1999", ("historical", "1999-05")),
    ("Alice she may live in Paris", ("current", None)),
    ("Alice used to live in Rome", ("historical", None)),
    ("what was it originally", ("historical", None)),
    ("what happened in december", ("historical", None)),
    ("Alice previously", ("historical", None)),
    ("born in 1850", ("current", None)),  # not 19xx/20xx
])
def test_classify_query(query, expected):
    assert classify_query(query) == expected


def test_current_returns_newest_value():
    mem = system()
    mem.write("Alice is in Lahore.", "t1")
    mem.write("Alice is in Karachi.", "t2")
    r = mem.temporal_retriever.search("Alice", 5)
    assert r["query_type"] == "current"
    assert r["facts"] == "[CURRENT] Alice | is | in Karachi (step 2)"
    assert "Lahore" in r["notes"]  # A-Mem's notes are returned unchanged


def test_year_query_returns_older_value_when_dated():
    mem = system(Scripted([ft("Lahore", "2020"), ft("Karachi", "2023")]))
    mem.write("Alice lives in Lahore since 2020.", "t1")
    mem.write("Alice lives in Karachi since 2023.", "t2")
    old = mem.temporal_retriever.search("where did Alice live in 2022", 5)
    assert old["query_type"] == "historical"
    assert old["facts"] == "[PAST] Alice | lives_in | Lahore (step 1, replaced at step 2)"
    new = mem.temporal_retriever.search("where does Alice live in 2024", 5)
    assert new["facts"] == "[CURRENT] Alice | lives_in | Karachi (step 2)"
    assert mem.temporal_retriever.search("Alice", 5)["facts"].count("Lahore") == 0


def test_historical_without_dates_shows_whole_history():
    mem = system(Scripted([ft("Lahore"), ft("Karachi")]))
    mem.write("I live in Lahore.", "t1")
    mem.write("I live in Karachi.", "t2")
    facts = mem.temporal_retriever.search("Alice previously", 5)["facts"].splitlines()
    assert facts == ["[PAST] Alice | lives_in | Lahore (step 1, replaced at step 2)",
                     "[CURRENT] Alice | lives_in | Karachi (step 2)"]
    # a year with no dated facts also shows everything
    assert len(mem.temporal_retriever.search("Alice in 2022", 5)["facts"].splitlines()) == 2


JANUARY, JUNE, OCTOBER = ("1:56 pm on 8 January, 2023", "6:30 pm on 14 June, 2023",
                          "9:00 am on 2 October, 2023")


def moved_twice():
    mem = system(Scripted([ft("Karachi"), ft("Lahore"), ft("Dubai")]))
    mem.write("I live in Karachi.", JANUARY)
    mem.write("I moved to Lahore.", JUNE)
    mem.write("I moved to Dubai.", OCTOBER)
    return mem


def values(mem, query):
    return [line.split(" | ")[2].split(" (")[0] for line in
            mem.temporal_retriever.search(query, 5)["facts"].splitlines()]


@pytest.mark.parametrize("query,expected", [
    ("where did Alice live in March 2023", ["Karachi"]),
    ("where did Alice live in July 2023", ["Lahore"]),
    ("where did Alice live in November 2023", ["Dubai"]),
])
def test_session_dates_place_replaced_facts_in_time(query, expected):
    assert values(moved_twice(), query) == expected


def test_the_month_of_a_change_shows_both_values():
    assert values(moved_twice(), "where did Alice live in June 2023") == ["Karachi", "Lahore"]


def test_the_first_value_has_no_invented_start():
    # Learning something in 2023 does not mean it began then.
    assert values(moved_twice(), "where did Alice live in 2020") == ["Karachi"]


def test_session_dates_are_compared_but_never_written():
    mem = moved_twice()
    mem.temporal_retriever.search("where did Alice live in March 2023", 5)
    for fact in mem.store.fact_history("s0", "Alice", "lives_in"):
        assert fact.valid_from is None and fact.valid_until is None


def test_a_stated_end_date_wins_over_the_session_date():
    mem = system(Scripted([ft("Karachi", end="2022"), ft("Lahore")]))
    mem.write("I lived in Karachi until 2022.", JANUARY)
    mem.write("I moved to Lahore.", JUNE)
    assert values(mem, "where did Alice live in March 2021") == ["Karachi"]
    assert values(mem, "where did Alice live in March 2023") == []  # after 2022, before June


def test_retracted_never_appears():
    mem = system(Scripted([ft("Lahore"), ft("Karachi")]))
    mem.write("I live in Lahore.", "t1")
    mem.write("Actually, I live in Karachi.", "t2")  # fake classifier -> CORRECT
    (lahore, _) = mem.store.fact_history("s0", "alice", "lives_in")
    assert lahore.status == "retracted"
    for q in ("Alice", "Alice previously", "Alice in 2022", "Alice used to live"):
        r = mem.temporal_retriever.search(q, 5)
        assert "Lahore" not in r["facts"] and "RETRACTED" not in r["facts"], q
    assert "Karachi" in r["facts"]


def test_entity_named_in_query_pulls_in_facts_of_unretrieved_notes():
    mem = system()
    mem.write("Nothing to see here.", "t1")
    mem.store.add_fact(Fact(sample_id="s0", note_id="not-retrieved", entity="Zed",
                            attribute="works_at", value="Acme", observed_at=7))
    mem.store.add_fact(Fact(sample_id="s0", note_id="not-retrieved", entity="Bob",
                            attribute="works_at", value="Initech", observed_at=8))
    r = mem.temporal_retriever.search("What does zed do?", 5)
    assert r["facts"] == "[CURRENT] Zed | works_at | Acme (step 7)"


def test_multivalued_active_facts_all_kept():
    mem = system()
    for i, lang in enumerate(["French", "Urdu"], 1):
        mem.store.add_fact(Fact(sample_id="s0", note_id="n", entity="Alice",
                                attribute="speaks", value=lang, observed_at=i))
    facts = mem.temporal_retriever.search("Alice", 5)["facts"].splitlines()
    assert len(facts) == 2


def test_max_facts_cap():
    mem = system()
    mem.temporal_retriever.max_facts = 2
    for i in range(5):
        mem.store.add_fact(Fact(sample_id="s0", note_id="n", entity="Alice",
                                attribute=f"a{i}", value="v", observed_at=i))
    assert len(mem.temporal_retriever.search("Alice", 5)["facts"].splitlines()) == 2


def add(mem, entity, attribute, value, step, **fields):
    fact = Fact(sample_id="s0", note_id="n", entity=entity, attribute=attribute,
                value=value, observed_at=step, **fields)
    mem.store.add_fact(fact)
    return fact


def test_cap_keeps_the_fact_closest_to_the_query():
    mem = system()
    mem.temporal_retriever.max_facts = 1
    add(mem, "Bob", "likes", "tea", 1)
    add(mem, "Bob", "plays", "chess", 2)
    add(mem, "Zed", "born_in", "Paris", 3)
    facts = mem.temporal_retriever.search("where was zed born and not bob", 5)["facts"]
    assert facts == "[CURRENT] Zed | born_in | Paris (step 3)"


def test_cap_keeps_the_newest_line_of_a_history():
    mem = system()
    mem.temporal_retriever.max_facts = 1
    new = add(mem, "Alice", "lives_in", "Karachi", 2)
    add(mem, "Alice", "lives_in", "Lahore", 1, status="superseded", superseded_by=new.fact_id)
    facts = mem.temporal_retriever.search("Alice previously", 5)["facts"]
    assert facts == "[CURRENT] Alice | lives_in | Karachi (step 2)"


def test_select_breaks_score_ties_with_the_named_entity():
    groups = {("a", "x"): ["a1"], ("b", "x"): ["b1"]}
    scores = {("a", "x"): 0.5, ("b", "x"): 0.5}
    assert TemporalRetriever._select(groups, scores, {("b", "x")}, 1) == ["b1"]


def test_memory_search_hook():
    mem = system()
    mem.write("Alice is in Lahore.", "t1")
    out = mem.search("Alice", 5)
    assert out["notes"].startswith(FACTS_HEADER + "\n[CURRENT] Alice | is | in Lahore (step 1)\n\n")
    assert "memory content" in out["notes"] and out["query_type"] == "current"
    assert mem.store.ops()[-1].op_type == "search"
    empty = system()
    assert empty.search("anything", 5)["notes"] == ""  # no notes, no facts: no header


def test_amem_search_unchanged():
    mem = system(name="amem")
    mem.write("Alice is in Lahore.", "t1")
    out = mem.search("Alice", 5)
    assert set(out) == {"notes"} and FACTS_HEADER not in out["notes"]
    assert not hasattr(mem, "temporal_retriever")
