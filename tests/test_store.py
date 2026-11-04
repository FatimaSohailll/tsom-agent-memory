from tsom.contracts import MemoryOp, Note
from tsom.store import Store


def test_note_round_trip():
    store = Store()
    note = Note(content="I moved to Lahore", timestamp="1 May 2023", keywords=["move"], links=["abc"])
    store.upsert_note(note)
    assert store.get_note(note.id) == note


def test_op_log():
    store = Store()
    store.log_op(MemoryOp(op_type="write", system="amem", llm_calls=3, latency_s=1.5))
    ops = store.ops()
    assert len(ops) == 1 and ops[0].llm_calls == 3
