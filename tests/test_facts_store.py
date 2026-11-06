import pytest

from tsom.contracts import Fact, RunConfig
from tsom.store import Store
from tsom.temporal.normalize import norm


def mk(value="Lahore", observed_at=1, **kw):
    base = dict(sample_id="s0", note_id="n1", entity="Alice", attribute="Lives In",
                value=value, observed_at=observed_at)
    base.update(kw)
    return Fact(**base)


def test_norm():
    assert norm("  Lives   IN. ") == "lives in"
    assert norm("Hello?!") == "hello"
    assert norm("") == ""


def test_system_tsom_accepted():
    assert RunConfig(run_name="x", system="tsom").system == "tsom"
    assert RunConfig(run_name="x", system="rag").system == "rag"


def test_insert_round_trip():
    store = Store()
    f = mk(supersedes=["old"], observed_time="1 May 2023")
    store.add_fact(f)
    assert store.get_fact(f.fact_id) == f


def test_duplicate_fact_id_rejected():
    store = Store()
    f = mk()
    store.add_fact(f)
    with pytest.raises(Exception):
        store.add_fact(f.model_copy(update={"value": "Karachi"}))
    assert store.get_fact(f.fact_id).value == "Lahore"


def test_key_lookup_case_insensitive():
    store = Store()
    f = mk()
    store.add_fact(f)
    got = store.active_facts("s0", "  ALICE ", "lives   in.")
    assert [x.fact_id for x in got] == [f.fact_id]
    assert store.active_facts("other", "alice", "lives in") == []


def test_status_update():
    store = Store()
    a, b = mk("Lahore", 1), mk("Karachi", 2)
    store.add_fact(a)
    store.add_fact(b)
    store.update_fact_status(a.fact_id, "superseded", valid_until="2023-06",
                             superseded_by=b.fact_id)
    store.update_fact_status(b.fact_id, "active", observation_count=3)
    got = store.get_fact(a.fact_id)
    assert (got.status, got.valid_until, got.superseded_by) == ("superseded", "2023-06", b.fact_id)
    assert store.get_fact(b.fact_id).observation_count == 3
    assert [x.fact_id for x in store.active_facts("s0", "alice", "lives in")] == [b.fact_id]
    with pytest.raises(ValueError):
        store.update_fact_status(a.fact_id, "bogus")
    with pytest.raises(KeyError):
        store.update_fact_status("missing", "active")


def test_history_ordering():
    store = Store()
    for v, t in [("C", 3), ("A", 1), ("B", 2)]:
        store.add_fact(mk(v, t))
    store.add_fact(mk("X", 0, attribute="job"))
    assert [f.value for f in store.fact_history("s0", "ALICE", "lives in")] == ["A", "B", "C"]


def test_update_cannot_change_value():
    store = Store()
    f = mk()
    store.add_fact(f)
    with pytest.raises(TypeError):
        store.update_fact_status(f.fact_id, "active", value="Paris")
    store.update_fact_status(f.fact_id, "retracted")
    got = store.get_fact(f.fact_id)
    assert (got.entity, got.attribute, got.value) == ("Alice", "Lives In", "Lahore")


def test_facts_for_notes_and_versions():
    store = Store()
    a, b = mk(note_id="n1"), mk("Karachi", 2, note_id="n2")
    store.add_fact(a)
    store.add_fact(b)
    assert [f.fact_id for f in store.facts_for_notes("s0", ["n2"])] == [b.fact_id]
    assert store.facts_for_notes("s0", []) == []
    store.add_note_version("n1", 1, 1, "ctx", ["t"], ["n2"], "n2")
    rows = store.note_versions("n1")
    assert rows[0]["changed_by_note_id"] == "n2"
