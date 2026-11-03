import importlib
import importlib.metadata
from unittest.mock import MagicMock

import pytest

import IkaCore
import IkaMem
import IkaModel
from IkaModel import base
from IkaModel.base import AgentEndException, BareBoneModel, HumanInputRequired


class DummyCli:
    def __init__(self):
        self.emitted = []

    def get_step(self, _agent_name):
        return 7

    def emit(self, *args, **kwargs):
        self.emitted.append((args, kwargs))


def test_barebone_model_validates_required_fields_and_temperature():
    with pytest.raises(ValueError, match="api_key"):
        BareBoneModel(model_id="gpt-4o", api_key="", api_url="https://example.test")
    with pytest.raises(ValueError, match="model_id"):
        BareBoneModel(model_id="", api_key="key", api_url="https://example.test")
    with pytest.raises(ValueError, match="api_url"):
        BareBoneModel(model_id="gpt-4o", api_key="key", api_url="")
    with pytest.raises(ValueError, match="Temperature"):
        BareBoneModel(model_id="gpt-4o", api_key="key", api_url="https://example.test", temperature=2)


def test_barebone_model_emits_init_prompts_when_not_suppressed(monkeypatch):
    cli = DummyCli()
    monkeypatch.setattr(base, "get_cli_output", lambda: cli)

    model = BareBoneModel(
        model_id="gpt-4o",
        api_key="key",
        api_url="https://example.test",
        system_prompt="system",
        content_prompt="content",
        agent_name="agent",
        suppress_init_output=False,
    )

    assert model.agent_hierarchy == []
    assert [entry[0][1] for entry in cli.emitted] == [
        "System Prompt:\nsystem",
        "Content Prompt:\ncontent",
    ]
    assert all(entry[1]["step"] == 7 for entry in cli.emitted)


def test_base_exceptions_default_payloads_and_final_text():
    end = AgentEndException(response={"content": "done"})
    assert end.final_text == "done"
    assert end.response == {"content": "done"}
    explicit = AgentEndException(response={"content": "ignored"}, final_text="final")
    assert explicit.final_text == "final"
    human = HumanInputRequired()
    assert human.payload == {}


def test_memory_backends_are_explicit_and_not_global(monkeypatch):
    from IkaMem.agent_tools import MemoryTools
    backend = MagicMock()
    first = MemoryTools('first', long_term=backend)
    second = MemoryTools('second')
    assert first.operations.long_term_memory is backend
    assert second.operations.long_term_memory is None
    assert not hasattr(base, 'get_global_long_term_memory')
    assert not hasattr(base, 'init_global_long_term_memory')


def test_lazy_package_exports_cache_values_and_reject_unknowns():
    assert IkaModel.__getattr__("BareBoneModel") is BareBoneModel
    assert IkaModel.BareBoneModel is BareBoneModel
    assert "BareBoneModel" in IkaModel.__dir__()
    with pytest.raises(AttributeError):
        IkaModel.__getattr__("missing")

    ika_stage = IkaCore.__getattr__("IkaStage")
    assert IkaCore.IkaStage is ika_stage
    assert "IkaStage" in IkaCore.__dir__()
    with pytest.raises(AttributeError):
        IkaCore.__getattr__("missing")


@pytest.mark.parametrize("package", [IkaCore, IkaModel, IkaMem])
def test_package_version_uses_distribution_metadata_and_source_fallback(package, monkeypatch):
    assert package.__version__ == importlib.metadata.version("ikacore")

    with monkeypatch.context() as patch:
        def missing_distribution(name):
            raise importlib.metadata.PackageNotFoundError(name)

        patch.setattr(importlib.metadata, "version", missing_distribution)
        importlib.reload(package)
        assert package.__version__ == "2.0.0"
    importlib.reload(package)
    assert package.__version__ == importlib.metadata.version("ikacore")
