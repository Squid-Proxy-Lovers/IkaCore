"""Presentation adapters tolerate optional events and preserve output contracts."""

from unittest.mock import Mock

from IkaCore import OutputConfig, cli_output
from IkaCore.events import OutputType, RuntimeEvent, _shared_cli
from IkaCore.output_adapters import CLIAdapter, FileLogAdapter, UsageAdapter, make_output


def test_legacy_stdout_sink_and_empty_wrapping(monkeypatch, capsys):
    monkeypatch.setattr(cli_output, "_stdout_enabled", True)
    cli_output._write_stdout("visible")
    assert capsys.readouterr().out == "visible"
    assert cli_output.is_stdout_enabled()
    monkeypatch.setattr(cli_output, "_output_sink", None)
    cli_output._route_to_sink(None, "ignored")
    renderer = cli_output.BoxRenderer(use_colors=False)
    assert renderer._wrap_line("", 20) == [""]
    assert renderer._wrap_paragraph("   ", 20) == [""]
    assert renderer._colorize("plain", OutputType.AGENT_RESPONSE) == "plain"
    obj = cli_output.CLIOutput.owned()
    assert obj._tool_result_prefix(True, False) == "ERROR"
    circular = {}
    circular["self"] = circular
    assert "self" in obj._format_tool_args(circular)


def test_cli_unknown_events_and_shared_adapter_dispatch():
    adapter = CLIAdapter()
    event = RuntimeEvent("unknown", {}, 0)
    adapter(event)
    shared = Mock()
    token = _shared_cli.set(shared)
    try:
        adapter(event)
        shared.assert_called_once_with(event)
    finally:
        _shared_cli.reset(token)
    adapter.flush()
    assert not adapter.is_buffering()


def test_file_and_usage_adapters_ignore_unrelated_events_and_flush(tmp_path):
    config = OutputConfig(cli=False, logging_level=2, logging_file=str(tmp_path / "log.txt"))
    adapter = FileLogAdapter(config)
    adapter(RuntimeEvent("unknown", {}, 0))
    adapter.flush()
    adapter.shutdown()
    written = []
    usage = UsageAdapter(written.append)
    usage(RuntimeEvent("unknown", {}, 0))
    usage(RuntimeEvent("step", {"stage_name": "stage", "usage": {"tokens": 2}, "cost": {}}, 0))
    assert len(written) == 1 and "tokens" in written[0]
    output = make_output(OutputConfig(cli=False))
    assert output.subscribers == ()
