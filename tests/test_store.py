from tsom.contracts import Fact, MemoryOp, Note
from tsom.store import Store


def test_note_round_trip():
    store = Store()
    note = Note(content="I moved to Lahore", timestamp="1 May 2023", keywords=["move"], links=["abc"])
    store.upsert_note(note)
    assert store.get_note(note.id) == note


def test_supersede_keeps_history():
    store = Store()
    old = Fact(subject="ali", predicate="lives_in", object="Karachi", valid_from="2023-01-01")
    store.add_fact(old)
    new = Fact(subject="ali", predicate="lives_in", object="Lahore", valid_from="2023-05-01")
    store.supersede(old.id, new)

    active = store.active_facts("ali", "lives_in")
    assert [f.object for f in active] == ["Lahore"]
    history = store.fact_history("ali", "lives_in")
    assert [f.object for f in history] == ["Karachi", "Lahore"]
    assert history[0].status == "retired" and history[0].valid_to == "2023-05-01"
    assert history[1].supersedes == old.id


def test_revert_supersession():
    store = Store()
    old = Fact(subject="ali", predicate="lives_in", object="Karachi", valid_from="2023-01-01")
    store.add_fact(old)
    new = Fact(subject="ali", predicate="lives_in", object="Lahore", valid_from="2023-05-01")
    store.supersede(old.id, new)

    restored = store.revert_supersession(new.id)
    assert restored.object == "Karachi" and restored.status == "active" and restored.valid_to is None
    assert store.get_fact(new.id).status == "rejected"


def test_op_log():
    store = Store()
    store.log_op(MemoryOp(op_type="write", system="amem", llm_calls=3, latency_s=1.5))
    ops = store.ops()
    assert len(ops) == 1 and ops[0].llm_calls == 3


def test_conflict_with_round_trip():
    store = Store()
    old = Fact(subject="ali", predicate="lives_in", object="Karachi", valid_from="2023-01-01")
    new = Fact(subject="ali", predicate="lives_in", object="Lahore", valid_from="2023-05-01",
               conflict_with=old.id)
    store.add_fact(old)
    store.add_fact(new)
    assert store.get_fact(new.id) == new
    assert {f.object for f in store.active_facts("ali", "lives_in")} == {"Karachi", "Lahore"}


def test_subject_lookups_ignore_case():
    store = Store()
    store.add_fact(Fact(subject="Ali", predicate="likes", object="biryani", valid_from="d1"))
    store.add_fact(Fact(subject="ali", predicate="likes", object="nihari", valid_from="d2"))
    assert [f.object for f in store.active_facts("ALI")] == ["biryani", "nihari"]
    assert [f.object for f in store.fact_history("ali", "likes")] == ["biryani", "nihari"]
    assert store.active_facts("Ali")[0].subject == "Ali"  # stored as given
    assert len(store.subjects()) == 1


def test_history_uses_insertion_order_not_date_strings():
    store = Store()
    # As text, "9 May" sorts after "10 June"; insertion order must win.
    old = Fact(subject="ali", predicate="lives_in", object="Karachi", valid_from="9 May, 2023")
    store.add_fact(old)
    store.supersede(old.id, Fact(subject="ali", predicate="lives_in", object="Lahore",
                                 valid_from="10 June, 2023"))
    assert [f.object for f in store.fact_history("ali")] == ["Karachi", "Lahore"]


def test_retire_and_set_source_note():
    store = Store()
    fact = Fact(subject="ali", predicate="likes", object="tea", valid_from="d1")
    store.add_fact(fact)
    store.set_source_note([fact.id], "note1")
    store.retire(fact.id, "d2")
    got = store.get_fact(fact.id)
    assert got.status == "retired" and got.valid_to == "d2" and got.source_note_id == "note1"
