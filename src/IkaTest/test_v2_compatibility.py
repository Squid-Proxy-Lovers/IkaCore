"""Keep existing callers and checkpoint databases compatible during v2 ports."""

import importlib
import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from IkaCore.checkpoint import CheckpointStore

CONTRACT = json.loads((Path(__file__).parent / "fixtures" / "v1_public_api.json").read_text())


def _resolve(path):
    module_name, *parts = path.split(".")
    value = importlib.import_module(module_name)
    for part in parts:
        try:
            value = getattr(value, part)
        except AttributeError:
            if not inspect.ismodule(value):
                raise
            value = importlib.import_module(f"{value.__name__}.{part}")
    return value


@pytest.mark.parametrize("module_name,names", sorted(CONTRACT["exports"].items()))
def test_existing_public_exports_remain_available(module_name, names):
    module = importlib.import_module(module_name)
    assert set(names) <= set(module.__all__)
    for name in names:
        assert getattr(module, name) is not None


@pytest.mark.parametrize("path,expected", sorted(CONTRACT["signatures"].items()))
def test_existing_call_shapes_and_defaults_remain_compatible(path, expected):
    current = inspect.signature(_resolve(path)).parameters
    positions = list(current)
    original_names = {item["name"] for item in expected}
    for index, item in enumerate(expected):
        assert item["name"] in current, f"{path} removed {item['name']}"
        parameter = current[item["name"]]
        assert parameter.kind.name == item["kind"], f"{path} changed how {parameter.name} is passed"
        default = None if parameter.default is inspect.Parameter.empty else repr(parameter.default)
        assert default == item["default"], f"{path} changed the default for {parameter.name}"
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            assert positions.index(parameter.name) == index, f"{path} shifted positional argument {parameter.name}"
    for name, parameter in current.items():
        if name not in original_names and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            assert parameter.default is not inspect.Parameter.empty, f"{path} added required argument {name}"


@pytest.mark.parametrize("scope", ["agent", "stage"])
def test_v1_checkpoint_database_remains_readable_and_writable(tmp_path, scope):
    db_path = tmp_path / "v1-checkpoints.db"
    uid = "previous-project-checkpoint"
    payload = {"scope": scope, "agent_name": "ExistingAgent", "remaining_steps": 4, "custom_field": {"keep": True}}
    # Create the released schema independently of the current store implementation.
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE checkpoints (id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT NOT NULL UNIQUE, "
            "scope TEXT NOT NULL, payload_json TEXT NOT NULL, created_at DATETIME NOT NULL)"
        )
        connection.execute(
            "INSERT INTO checkpoints (uid, scope, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (uid, scope, json.dumps(payload), "2026-09-08T00:00:00+00:00"),
        )
    store = CheckpointStore(str(db_path))
    assert store.load_checkpoint(uid) == payload
    updated = {**payload, "remaining_steps": 3}
    assert store.save_checkpoint(scope, updated, uid=uid) == uid
    assert store.load_checkpoint(uid) == updated
    store.delete_checkpoint(uid)
    assert store.load_checkpoint(uid) is None
