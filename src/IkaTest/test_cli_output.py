import time

from IkaCore.cli_output import (
    BoxRenderer,
    CLIOutput,
    OutputBuffer,
    OutputContext,
    OutputType,
    get_cli_output,
    set_output_sink,
    set_stdout_enabled,
)


def make_context(content: str, timestamp: float = 1.0, output_type: OutputType = OutputType.AGENT_RESPONSE) -> OutputContext:
    return OutputContext(
        output_type=output_type,
        step_number=1,
        hierarchy_chain=["agent"],
        thread_id=123,
        instance_id=0,
        content=content,
        timestamp=timestamp,
    )


def test_box_renderer_clamps_tiny_width_and_truncates_tool_content():
    renderer = BoxRenderer(use_colors=False, width=3)
    renderer.MAX_CONTENT_LENGTH = 12

    assert renderer._truncate("abcdefghijklmnopqrstuvwxyz", OutputType.TOOL_RESULT) == "abcdefghi..."
    assert renderer._truncate("abcdefghijklmnopqrstuvwxyz", OutputType.SUMMARIZATION) == "abcdefghijklmnopqrstuvwxyz"

    rendered = renderer.render(make_context("abcdef", output_type=OutputType.TOOL_RESULT))
    lines = rendered.splitlines()
    assert lines[0].startswith(renderer.TOP_LEFT)
    assert lines[-1].startswith(renderer.BOTTOM_LEFT)
    assert all(len(line) >= renderer.MIN_BOX_WIDTH for line in lines)


def test_output_buffer_flush_sorts_buffered_output_and_routes_sink_when_stdout_disabled():
    captured = []

    def sink(ctx: OutputContext, rendered: str) -> None:
        captured.append((ctx.content, rendered))

    set_stdout_enabled(False)
    set_output_sink(sink)
    try:
        buffer = OutputBuffer(BoxRenderer(use_colors=False, width=60))
        buffer.enable_buffering()
        buffer.add(make_context("second", timestamp=2.0))
        buffer.add(make_context("first", timestamp=1.0))
        buffer.flush()
    finally:
        set_output_sink(None)
        set_stdout_enabled(True)

    assert [content for content, _ in captured] == ["first", "second"]
    assert all(rendered.endswith("\n") for _, rendered in captured)
    assert not buffer.is_buffering()


def test_cli_output_formats_events_tracks_steps_and_uses_singleton_sink():
    captured = []

    def sink(ctx: OutputContext, rendered: str) -> None:
        captured.append((ctx, rendered))

    cli = CLIOutput()
    set_stdout_enabled(False)
    set_output_sink(sink)
    try:
        cli.configure(use_colors=False, width=80)
        cli._buffer.disable_buffering()
        cli.reset_steps()
        cli.set_instance_id(7)
        cli.set_step("agent", 1)

        assert get_cli_output() is cli
        assert cli.get_step("agent") == 1
        assert cli.increment_step("agent") == 2

        cli.tool_call("search", {"query": "x"}, ["agent"], cli.get_step("agent"))
        cli.tool_result("search", "late", ["agent"], 3, is_timeout=True)
        cli.agent_response("agent", "done", ["agent"], 4, is_final=True)
        cli.workflow_status("workflow", "running", step=5)
    finally:
        cli.set_instance_id(0)
        cli.reset_steps()
        set_output_sink(None)
        set_stdout_enabled(True)

    rendered = "\n".join(item[1] for item in captured)
    assert len(captured) == 4
    assert captured[0][0].instance_id == 7
    assert '"query": "x"' in rendered
    assert "TIMEOUT:" in rendered
    assert "Agent: agent (Final)" in rendered
    assert "Workflow: workflow" in rendered


def test_output_buffer_sink_failures_do_not_block_stdout_suppressed_path():
    def failing_sink(ctx: OutputContext, rendered: str) -> None:
        raise RuntimeError("sink failed")

    set_stdout_enabled(False)
    set_output_sink(failing_sink)
    try:
        buffer = OutputBuffer(BoxRenderer(use_colors=False))
        buffer.add(make_context("content", timestamp=time.time()))
    finally:
        set_output_sink(None)
        set_stdout_enabled(True)
