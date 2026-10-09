import sys
from types import ModuleType

import pytest

from IkaModel.resource_loader import read_embedded_resource


def test_embedding_application_reader_is_used(monkeypatch):
    module = ModuleType("src.resources")
    module.read_text = lambda path: f"embedded:{path}"
    monkeypatch.setitem(sys.modules, "src.resources", module)
    assert read_embedded_resource("model/prompt") == "embedded:model/prompt"


def test_missing_embedding_application_preserves_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.resources", None)
    with pytest.raises(ImportError):
        read_embedded_resource("model/prompt")


def test_missing_reader_preserves_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.resources", ModuleType("src.resources"))
    with pytest.raises(ImportError, match="read_text"):
        read_embedded_resource("model/prompt")
