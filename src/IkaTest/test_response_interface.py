import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from IkaModel.base import AgentEndException, HumanInputRequired
from IkaModel.chat_interface import response_interface as ri


class _CompletedFuture:
    def __init__(self, value=None, exc=None):
        self.value = value
        self.exc = exc
        self.cancel = MagicMock()

    def result(self, timeout=None):
        if self.exc:
            raise self.exc
        return self.value


class _FakePool:
    def __init__(self, future):
        self.future = future
        self.submit_calls = []

    def submit(self, *args, **kwargs):
        self.submit_calls.append((args, kwargs))
        return self.future


def _tool_call(name, arguments="{}", call_id="call_1"):
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def test_validate_tool_args_coerces_scalars_and_rejects_invalid_payloads():
    out = ri.validate_tool_args(
        "search",
        {"enabled": "true", "count": "3", "ratio": "1.5", "query": "text"},
    )

    assert out == {"enabled": True, "count": 3, "ratio": 1.5, "query": "text"}

    with pytest.raises(ValueError, match="got None"):
        ri.validate_tool_args("search", None)
    with pytest.raises(ValueError, match="JSON object"):
        ri.validate_tool_args("search", ["not", "a", "dict"])
    with pytest.raises(ValueError, match="JSON-serializable"):
        ri.validate_tool_args("search", {"bad": object()})
    with pytest.raises(ValueError, match="non-empty 'input'"):
        ri.validate_tool_args("agent_end", {"input": ""})


def test_formatters_handle_missing_unknown_and_gemini_edge_results():
    calls = [_tool_call("lookup", call_id="call_a"), {"name": "gemini_tool"}]

    assert ri.format_provider_tool_results("unknown", calls, ["result"]) == []
    assert json.loads(ri.format_openai_results(calls, [])[0]["content"]) == {"error": "No result"}
    assert ri.format_openai_responses_results(calls, ["ok"])[0]["tool_call_id"] == "call_a"
    anthropic = ri.format_anthropic_results(calls, [])
    assert anthropic[0]["content"][0]["tool_use_id"] == "call_a"
    assert json.loads(anthropic[0]["content"][0]["content"]) == {"error": "No result"}

    formatted = ri.format_gemini_results(
        [{"name": "none"}, {"name": "blank"}, {"name": "empty"}, {"name": "list"}],
        [None, "", "{}", "[1, 2]"],
    )
    assert formatted[0]["functionResponse"]["response"] == {"error": "No result"}
    assert formatted[1]["functionResponse"]["response"] == {"result": ""}
    assert formatted[2]["functionResponse"]["response"] == {}
    assert formatted[3]["functionResponse"]["response"] == {"result": [1, 2]}


def test_execute_tool_reports_missing_validation_timeout_and_executor_errors():
    cli = MagicMock()
    with patch("IkaModel.chat_interface.tool_execution_sync.get_cli_output", return_value=cli):
        missing = json.loads(ri.execute_tool("missing", {}, {}, agent_hierarchy=["A"], step=2))
        invalid = json.loads(ri.execute_tool("search", None, {"search": lambda _args: "ok"}))

    assert "not found" in missing["error"]
    assert "Validation error" in invalid["error"]
    assert cli.tool_result.call_count == 2

    timeout_future = _CompletedFuture(exc=ri.FutureTimeoutError())
    with patch("IkaModel.chat_interface.tool_execution_sync.get_cli_output", return_value=MagicMock()) as cli_factory:
        with patch("IkaModel.chat_interface.tool_execution_sync._ensure_tool_executor_pool", return_value=_FakePool(timeout_future)):
            timed_out = json.loads(ri.execute_tool("slow", {}, {"slow": lambda _args: "ok"}, timeout=0.01))

    assert "timed out" in timed_out["error"]
    assert timeout_future.cancel.called
    assert cli_factory.return_value.tool_result.call_args.kwargs["is_timeout"] is True

    error_future = _CompletedFuture(exc=RuntimeError("executor failed"))
    with patch("IkaModel.chat_interface.tool_execution_sync.get_cli_output", return_value=MagicMock()):
        with patch("IkaModel.chat_interface.tool_execution_sync._ensure_tool_executor_pool", return_value=_FakePool(error_future)):
            errored = json.loads(ri.execute_tool("boom", {}, {"boom": lambda _args: "ok"}))

    assert "executor failed" in errored["error"]


def test_execute_tool_recreates_shutdown_pool_once_and_reraises_control_exceptions():
    cli = MagicMock()
    first_pool = MagicMock()
    first_pool.submit.side_effect = RuntimeError("pool closed")
    retry_pool = _FakePool(_CompletedFuture({"ok": True}))

    with patch("IkaModel.chat_interface.tool_execution_sync.get_cli_output", return_value=cli):
        with patch("IkaModel.chat_interface.tool_execution_sync._ensure_tool_executor_pool", return_value=first_pool):
            with patch("IkaModel.chat_interface.tool_execution_sync._reset_tool_executor_pool", return_value=retry_pool):
                out = ri.execute_tool("lookup", {"q": "x"}, {"lookup": lambda _args: {"unused": True}})

    assert json.loads(out) == {"ok": True}
    assert first_pool.submit.called
    assert retry_pool.submit_calls
    cli.tool_result.assert_called_with("lookup", '{"ok": true}', ["lookup"], 0)

    with patch("IkaModel.chat_interface.tool_execution_sync.get_cli_output", return_value=MagicMock()):
        with pytest.raises(AgentEndException):
            ri.execute_tool(
                "finish",
                {},
                {"finish": lambda _args: (_ for _ in ()).throw(AgentEndException(response={"content": "done"}))},
            )
        with pytest.raises(HumanInputRequired):
            ri.execute_tool(
                "ask",
                {},
                {"ask": lambda _args: (_ for _ in ()).throw(HumanInputRequired({"question": "Q?"}))},
            )


def test_tool_call_planning_handles_schema_fallback_limits_duplicates_and_ordering():
    def executor(_args):
        return "ok"

    executor.__tool_schema__ = {
        "properties": {"query": {"type": "string", "description": "Query text"}},
        "required": ["query"],
    }
    counts = {"limited": 1}
    plan = ri._prepare_tool_call_plan(
        [
            _tool_call("needs_query", "{}", "empty"),
            _tool_call("limited", '{"q":"x"}', "limited"),
            _tool_call("sequential", '{"q":"x"}', "seq"),
            _tool_call("parallel", '{"q":"x"}', "par"),
            _tool_call("parallel", '{"q":"x"}', "duplicate"),
        ],
        {"needs_query": executor, "limited": executor, "sequential": executor, "parallel": executor},
        {
            "limited": {"limit_calls": 1},
            "sequential": {"parallel": False},
            "parallel": {"parallel": True},
        },
        counts,
    )

    assert "Expected schema" in plan.tool_call_id_to_result["empty"]
    assert "call limit" in plan.tool_call_id_to_result["limited"]
    assert "Duplicate tool call" in plan.tool_call_id_to_result["duplicate"]
    assert plan.sequential_calls == [("sequential", {"q": "x"}, "seq")]
    assert plan.parallel_calls == [("parallel", {"q": "x"}, "par")]
    assert counts["needs_query"] == 1
    assert ri._ordered_tool_results(plan)[0] == plan.tool_call_id_to_result["empty"]


def test_execute_tool_calls_surfaces_parallel_and_sequential_interrupts():
    parallel_plan = ri.ToolCallPlan(
        tool_call_order=[_tool_call("ask", "{}", "ask")],
        parallel_calls=[("ask", {}, "ask")],
    )
    with patch(
        "IkaModel.chat_interface.tool_execution_sync.execute_tool",
        side_effect=HumanInputRequired({"question": "Parallel?"}),
    ):
        interrupt = ri._execute_parallel_tool_plan(
            parallel_plan,
            {"ask": lambda _args: "unused"},
            timeout=1,
            agent_hierarchy=["A"],
            step=1,
            tool_call_counts={},
        )

    assert interrupt == {"question": "Parallel?"}
    assert json.loads(parallel_plan.tool_call_id_to_result["ask"])["__ika_interrupt__"] is True

    sequential_plan = ri.ToolCallPlan(
        tool_call_order=[_tool_call("ask", "{}", "ask")],
        sequential_calls=[("ask", {}, "ask")],
    )
    counts = {}
    with patch(
        "IkaModel.chat_interface.tool_execution_sync.execute_tool",
        side_effect=HumanInputRequired({"question": "Sequential?"}),
    ):
        interrupt = ri._execute_sequential_tool_plan(
            sequential_plan,
            {"ask": lambda _args: "unused"},
            timeout=1,
            agent_hierarchy=["A"],
            step=1,
            tool_call_counts=counts,
        )

    assert interrupt == {"question": "Sequential?"}
    assert counts == {"ask": 1}
    assert json.loads(sequential_plan.tool_call_id_to_result["ask"])["__ika_interrupt__"] is True

    assert ri.execute_tool_calls([], {"tool": lambda _args: "ok"}, "openai", tool_call_counts={"tool": 1}) == (
        [],
        [],
        {"tool": 1},
        [],
        None,
    )


def test_execute_tool_calls_keeps_parallel_errors_ordered_with_successful_results():
    def ok(args):
        return {"ok": args["q"]}

    with patch("IkaModel.chat_interface.tool_execution_sync.get_cli_output", return_value=MagicMock()):
        formatted, tool_results, counts, executed, interrupt = ri.execute_tool_calls(
            [_tool_call("ok", '{"q":"x"}', "ok"), _tool_call("boom", "{}", "boom")],
            {"ok": ok, "boom": lambda _args: (_ for _ in ()).throw(RuntimeError("bad"))},
            "openai",
        )

    assert interrupt is None
    assert [call["id"] for call in executed] == ["ok", "boom"]
    assert json.loads(tool_results[0]) == {"ok": "x"}
    assert "bad" in json.loads(tool_results[1])["error"]
    assert [message["tool_call_id"] for message in formatted] == ["ok", "boom"]
    assert counts == {"ok": 1, "boom": 1}


def test_execute_tool_calls_preserves_successful_parallel_result_without_timeout():
    calls = []

    def save_result(args):
        calls.append(args)
        return {"saved": args["finding_id"]}

    with patch("IkaModel.chat_interface.tool_execution_sync.get_cli_output", return_value=MagicMock()):
        formatted, tool_results, counts, executed, interrupt = ri.execute_tool_calls(
            [_tool_call("save_result", '{"finding_id":"finding-1"}', "save")],
            {"save_result": save_result},
            "openai",
            timeout=None,
        )

    assert calls == [{"finding_id": "finding-1"}]
    assert json.loads(tool_results[0]) == {"saved": "finding-1"}
    assert json.loads(formatted[0]["content"]) == {"saved": "finding-1"}
    assert counts == {"save_result": 1}
    assert [call["id"] for call in executed] == ["save"]
    assert interrupt is None


def test_async_execute_tool_reports_common_errors_and_runs_sync_or_async_executors():
    async def run_case():
        cli = MagicMock()

        async def async_executor(args):
            return {"async": args["value"]}

        async def slow(_args):
            await asyncio.sleep(0.05)
            return "late"

        with patch("IkaModel.chat_interface.tool_execution_async.get_cli_output", return_value=cli):
            missing = json.loads(await ri.async_execute_tool("missing", {}, {}))
            invalid = json.loads(await ri.async_execute_tool("bad", None, {"bad": lambda _args: "ok"}))
            sync_out = json.loads(await ri.async_execute_tool("sync", {"value": 1}, {"sync": lambda args: {"sync": args["value"]}}))
            async_out = json.loads(await ri.async_execute_tool("async", {"value": 2}, {"async": async_executor}))
            timed_out = json.loads(await ri.async_execute_tool("slow", {}, {"slow": slow}, timeout=0.001))
            errored = json.loads(
                await ri.async_execute_tool("boom", {}, {"boom": lambda _args: (_ for _ in ()).throw(RuntimeError("bad"))})
            )

        return missing, invalid, sync_out, async_out, timed_out, errored, cli

    missing, invalid, sync_out, async_out, timed_out, errored, cli = asyncio.run(run_case())

    assert "not found" in missing["error"]
    assert "Validation error" in invalid["error"]
    assert sync_out == {"sync": 1}
    assert async_out == {"async": 2}
    assert "timed out" in timed_out["error"]
    assert "bad" in errored["error"]
    assert cli.tool_result.call_count >= 6


def test_async_execute_tool_and_call_plans_propagate_control_interrupts():
    async def run_case():
        with patch("IkaModel.chat_interface.tool_execution_async.get_cli_output", return_value=MagicMock()):
            with pytest.raises(AgentEndException):
                await ri.async_execute_tool(
                    "finish",
                    {},
                    {
                        "finish": lambda _args: (_ for _ in ()).throw(
                            AgentEndException(response={"content": "done"})
                        )
                    },
                )
            with pytest.raises(HumanInputRequired):
                await ri.async_execute_tool(
                    "ask",
                    {},
                    {"ask": lambda _args: (_ for _ in ()).throw(HumanInputRequired({"question": "Q?"}))},
                )

        formatted, results, counts, executed, interrupt = await ri.async_execute_tool_calls(
            [_tool_call("ask", "{}", "ask")],
            {"ask": lambda _args: (_ for _ in ()).throw(HumanInputRequired({"question": "Async?"}))},
            "openai",
        )
        return formatted, results, counts, executed, interrupt

    formatted, results, counts, executed, interrupt = asyncio.run(run_case())

    assert interrupt == {"question": "Async?"}
    assert counts == {"ask": 1}
    assert executed[0]["id"] == "ask"
    assert json.loads(results[0])["__ika_interrupt__"] is True
    assert formatted[0]["tool_call_id"] == "ask"


def test_async_execute_tool_calls_sequential_path_and_empty_short_circuit():
    async def run_case():
        async def sequential(args):
            return {"ok": args["q"]}

        empty = await ri.async_execute_tool_calls([], {"tool": sequential}, "openai", tool_call_counts={"tool": 2})
        formatted, results, counts, executed, interrupt = await ri.async_execute_tool_calls(
            [_tool_call("sequential", '{"q":"x"}', "seq")],
            {"sequential": sequential},
            "openai",
            tool_metadata={"sequential": {"parallel": False}},
        )
        return empty, formatted, results, counts, executed, interrupt

    empty, formatted, results, counts, executed, interrupt = asyncio.run(run_case())

    assert empty == ([], [], {"tool": 2}, [], None)
    assert json.loads(results[0]) == {"ok": "x"}
    assert counts == {"sequential": 1}
    assert executed[0]["id"] == "seq"
    assert formatted[0]["tool_call_id"] == "seq"
    assert interrupt is None
