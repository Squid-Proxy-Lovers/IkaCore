import sys
from pathlib import Path

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from IkaCore.cli_output import OutputContext, OutputType
from IkaGeneral.execution_env import IkaExecutionEnvironment
from IkaGeneral.execution_trace import ExecutionTraceWriter


def test_execution_trace_writer_splits_root_and_subagent_logs(tmp_path):
    writer = ExecutionTraceWriter(tmp_path / "run", root_name="root")

    root_ctx = OutputContext(
        output_type=OutputType.TOOL_CALL,
        step_number=0,
        hierarchy_chain=["root", "make_subagent"],
        thread_id=1,
        instance_id=0,
        content="root tool",
    )
    sub_ctx = OutputContext(
        output_type=OutputType.TOOL_CALL,
        step_number=1,
        hierarchy_chain=["root", "recon_subagent", "web_search"],
        thread_id=1,
        instance_id=0,
        content="subagent tool",
    )

    writer._sink(root_ctx, "ROOT\n")
    writer._sink(sub_ctx, "SUB\n")

    root_text = writer.root_log_path.read_text(encoding="utf-8")
    assert "ROOT" in root_text
    assert "SUB" in root_text

    subagent_logs = list((tmp_path / "run").glob("recon_subagent-*.log"))
    assert len(subagent_logs) == 1
    assert subagent_logs[0].read_text(encoding="utf-8") == "SUB\n"


def test_subagent_hierarchy_is_propagated_into_model():
    env = IkaExecutionEnvironment(
        model_id="deepseek-chat",
        api_key="k",
        system_prompt="sys",
        initial_prompt="task",
        _execution_name="recon_subagent",
        _agent_hierarchy=["root", "recon_subagent"],
    )

    env._setup()

    assert env._barebone is not None
    assert env._barebone.agent_name == "recon_subagent"
    assert env._barebone.agent_hierarchy == ["root", "recon_subagent"]
