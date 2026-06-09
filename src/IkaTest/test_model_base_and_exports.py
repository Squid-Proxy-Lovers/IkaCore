import importlib

import pytest

import IkaCore
import IkaMem
import IkaModel
from IkaModel import base
from IkaModel.base import AgentEndException, BareBoneModel, HumanInputRequired, load_gemini_payload


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


def test_global_long_term_memory_initializes_once(monkeypatch):
    class DummyLTMemory:
        instances = []

        def __init__(self, embedder_config):
            self.embedder_config = embedder_config
            DummyLTMemory.instances.append(self)

    monkeypatch.setattr(base, "_GLOBAL_LONG_TERM_MEMORY", None)
    monkeypatch.setattr(IkaMem, "LTMemory", DummyLTMemory)

    memory = base.init_global_long_term_memory({"provider": "fake"}, search_limit=3)
    assert memory.embedder_config == {"provider": "fake"}
    assert memory._search_limit == 3
    assert memory._filter_func([1, 2, 3, 4]) == [1, 2, 3, 4][:5]

    assert base.init_global_long_term_memory({"provider": "other"}) is memory


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


def test_load_gemini_payload_success_path():
    assert load_gemini_payload() is importlib.import_module("IkaModel.gemini.google").gemini_fill_payload
