"""Undoing a supersession, and the inspection script around it."""

import sys
from pathlib import Path

import pytest

from harness.contradictions import curve
from tsom.contracts import Fact, Note
from tsom.store import Store
from tsom.temporal.consolidator import Consolidator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import inspect_memory  # noqa: E402


def apply(store, value, op, step, valid_from=None, attribute="lives_in"):
    new = {"entity": "Ali", "attribute": attribute, "value": value, "valid_from": valid_from}
    old = store.active_facts("s", "Ali", attribute)
    return Consolidator(store).apply("s", "n", new, op, old, step, f"t{step}")


def replaced(op="REPLACE", valid_from=None):
    store = Store()
    old = apply(store, "Lahore", "ADD", 1)
    return store, old, apply(store, "Karachi", op, 2, valid_from)


@pytest.mark.parametrize("op", ["REPLACE", "CORRECT"])
def test_revert_restores_the_old_fact_and_retracts_the_new(op):
    store, old, new = replaced(op)
    restored = store.revert_supersession(new.fact_id)
    assert [f.fact_id for f in restored] == [old.fact_id]
    assert [f.value for f in store.active_facts("s", "Ali", "lives_in")] == ["Lahore"]
    assert store.get_fact(old.fact_id).superseded_by is None
    assert store.get_fact(new.fact_id).status == "retracted"


def test_revert_keeps_every_row():
    store, _, new = replaced()
    store.revert_supersession(new.fact_id)
    assert len(store.fact_history("s", "Ali", "lives_in")) == 2


def test_revert_restores_every_value_a_replacement_retired():
    store = Store()
    apply(store, "Lahore", "ADD", 1)
    apply(store, "Dubai", "ADD", 2)
    new = apply(store, "Karachi", "REPLACE", 3)
    store.revert_supersession(new.fact_id)
    assert {f.value for f in store.active_facts("s", "Ali", "lives_in")} == {"Lahore", "Dubai"}


def test_revert_removes_an_end_date_the_replacement_set():
    store, old, new = replaced(valid_from="2023")
    assert store.get_fact(old.fact_id).valid_until == "2023"
    store.revert_supersession(new.fact_id)
    assert store.get_fact(old.fact_id).valid_until is None


def test_revert_keeps_an_end_date_the_old_fact_stated():
    store = Store()
    old = Fact(sample_id="s", note_id="n", entity="Ali", attribute="lives_in", value="Lahore",
               observed_at=1, valid_until="2021")
    store.add_fact(old)
    new = apply(store, "Karachi", "REPLACE", 2, "2023")
    store.revert_supersession(new.fact_id)
    assert store.get_fact(old.fact_id).valid_until == "2021"


def test_a_fact_that_replaced_nothing_cannot_be_reverted():
    store = Store()
    first = apply(store, "Lahore", "ADD", 1)
    with pytest.raises(ValueError, match="replaced nothing"):
        store.revert_supersession(first.fact_id)


def test_only_the_newest_step_of_a_chain_can_be_reverted():
    store, _, middle = replaced()
    apply(store, "Dubai", "REPLACE", 3)
    with pytest.raises(ValueError, match="no longer active"):
        store.revert_supersession(middle.fact_id)


def test_an_unknown_fact_cannot_be_reverted():
    with pytest.raises(KeyError):
        Store().revert_supersession("nope")


def test_a_reverted_value_does_not_count_as_a_contradiction():
    store, _, new = replaced()
    store.revert_supersession(new.fact_id)
    assert curve(store, "s", ())[-1]["contradictions"] == 0


def database(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = Store(str(path))
    store.upsert_note(Note(content="Ali lives in Lahore.", timestamp="t1"))
    apply(store, "Lahore", "ADD", 1)
    new = apply(store, "Karachi", "REPLACE", 2)
    store.close()
    return path, new


def run_script(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["inspect_memory.py", *args])
    inspect_memory.main()


def test_the_script_lists_an_entitys_facts(tmp_path, monkeypatch, capsys):
    path, new = database(tmp_path)
    run_script(monkeypatch, str(path), "--entity", "ali")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and "superseded" in lines[0] and "Lahore" in lines[0]
    assert "active" in lines[1] and new.fact_id in lines[1]


def test_the_script_reverts_and_saves(tmp_path, monkeypatch, capsys):
    path, new = database(tmp_path)
    run_script(monkeypatch, str(path), "--revert", new.fact_id)
    assert "Lahore" in capsys.readouterr().out
    store = Store(str(path))
    assert [f.value for f in store.active_facts("s", "Ali", "lives_in")] == ["Lahore"]
    store.close()


def test_the_script_reports_a_refused_revert(tmp_path, monkeypatch, capsys):
    path, _ = database(tmp_path)
    with pytest.raises(SystemExit):
        run_script(monkeypatch, str(path), "--revert", "nope")
    assert "nope" in capsys.readouterr().err


def test_the_script_still_shows_notes_by_default(tmp_path, monkeypatch, capsys):
    path, _ = database(tmp_path)
    run_script(monkeypatch, str(path))
    assert "Ali lives in Lahore." in capsys.readouterr().out
