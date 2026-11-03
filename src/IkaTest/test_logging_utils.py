import copy
import json

from IkaCore.logging_utils import IkaLogger


def test_level_one_logger_writes_human_readable_lines(tmp_path):
    log_path = tmp_path / "ika.log"
    logger = IkaLogger(level=1, log_file=str(log_path), use_colors=False, log_input_enabled=True)

    logger.log_input([{"role": "user", "content": "hello"}])
    logger.log_tool_results(
        [{"function": {"name": "search", "arguments": "{}"}}],
        ["result"],
    )
    logger.log_output("done", usage={"total_tokens": 3}, cost={"total_cost": 0.01})
    logger.log_summary("summary")
    logger.log_action("action")
    logger.log_stage_start("Stage", hitl=True, remaining_steps=4, step_limit=2)
    logger.log_stage_end("Stage", used_steps=2)
    logger.log_step(
        stage_name="Stage",
        step_idx=1,
        output="line one\nline two",
        tool_calls=[{"function": {"name": "lookup"}}],
        usage={"total_tokens": 7},
        cost={"total_cost": 0.02},
        elapsed=0.25,
    )
    logger.log_hitl_prompt("Stage")
    logger.log_hitl_input("Stage", "answer")
    logger.log_hitl_question("Stage", "question?")
    logger.log_hitl_answer("Stage", "answer")

    assert copy.copy(logger) is logger
    assert copy.deepcopy(logger) is logger

    logger.flush()
    logger.shutdown()

    text = log_path.read_text(encoding="utf-8")
    assert "[INPUT] role=user content=hello" in text
    assert "[TOOL] name=search result=result" in text
    assert "[OUTPUT] content=done" in text
    assert "[SUMMARY] summary" in text
    assert "[ACTION] action" in text
    assert "[STAGE START] Stage hitl=True remaining=4 limit=2" in text
    assert "[STAGE END] Stage used_steps=2" in text
    assert "tools=['lookup']" in text
    assert "line one line two" in text
    assert "[HITL INPUT] stage=Stage user='answer'" in text


def test_level_two_logger_writes_structured_json_events(tmp_path):
    log_path = tmp_path / "ika.jsonl"
    logger = IkaLogger(level=2, log_file=str(log_path), use_colors=False, log_input_enabled=True)

    logger.write_line("plain")
    logger.log_input([{"role": "user", "content": "hello"}])
    logger.log_tool_results([{"name": "search"}], ["result"])
    logger.log_output("done", usage={"total_tokens": 3}, cost={"total_cost": 0.01}, message_history={"messages": {}})
    logger.log_summary("summary")
    logger.log_action("action")
    logger.log_stage_start("Stage", hitl=False, remaining_steps=5, step_limit=3)
    logger.log_stage_end("Stage", used_steps=2)
    logger.log_step("Stage", 1, "output", [{"name": "tool"}], {"total_tokens": 1}, {}, 0.1)
    logger.log_hitl_prompt("Stage")
    logger.log_hitl_input("Stage", "answer")
    logger.log_hitl_question("Stage", "question?")
    logger.log_hitl_answer("Stage", "answer")

    logger.flush()
    logger.shutdown()

    payloads = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    events = {payload.get("event") for payload in payloads}

    assert {"text": "plain"}.items() <= payloads[0].items()
    assert {
        "input",
        "tool_results",
        "output",
        "summary",
        "action",
        "stage_start",
        "stage_end",
        "step",
        "hitl_prompt",
        "hitl_input",
        "hitl_question",
        "hitl_answer",
    }.issubset(events)
    assert all("ts" in payload for payload in payloads)


def test_logger_cost_calculation_handles_known_and_unknown_models():
    logger = IkaLogger(level=0)

    assert logger.compute_cost("unknown-model", {"total_tokens": 100}) == {
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
    }

    input_cost_per_m, cached_input_cost_per_m, output_cost_per_m = IkaLogger.get_model_cost("gpt-4o")
    expected_input = (0.5 * input_cost_per_m) + (0.5 * cached_input_cost_per_m)
    expected_output = 1.0 * output_cost_per_m

    assert logger.compute_cost(
        "gpt-4o",
        {
            "input_tokens": 1_000_000,
            "input_cached_tokens": 500_000,
            "output_tokens": 1_000_000,
        },
    ) == {
        "input_cost": round(expected_input, 6),
        "output_cost": round(expected_output, 6),
        "total_cost": round(expected_input + expected_output, 6),
    }


def test_level_zero_logger_is_noop_without_writer(tmp_path):
    log_path = tmp_path / "unused.log"
    logger = IkaLogger(level=0, log_file=str(log_path), use_colors=False, log_input_enabled=True)

    logger.write_line("plain")
    logger.log_json({"event": "ignored"})
    logger.log_input([{"role": "user", "content": "hello"}])
    logger.log_output("done")
    logger.log_step("Stage", 1, "output", [], {}, {}, 0.1)
    logger.flush()
    logger.shutdown()

    assert logger._writer_thread is None
    assert not log_path.exists()


def test_logger_shutdown_drains_queued_lines(tmp_path):
    log_path = tmp_path / "ika.log"
    logger = IkaLogger(level=1, log_file=str(log_path), use_colors=False)

    for idx in range(25):
        logger.write_line(f"line {idx}")
    logger.shutdown()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines == [f"line {idx}" for idx in range(25)]
