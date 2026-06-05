from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


FILES_WITHOUT_SYS_PATH_MUTATION = [
    REPO_ROOT / "src" / "IkaCore" / "agents.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_memory.py",
    REPO_ROOT / "src" / "IkaModel" / "base.py",
    REPO_ROOT / "examples" / "example_fileagent.py",
    REPO_ROOT / "examples" / "example_hitl.py",
    REPO_ROOT / "examples" / "example_stages.py",
    REPO_ROOT / "examples" / "example_workflow.py",
    REPO_ROOT / "examples" / "example_explain.py",
    REPO_ROOT / "examples" / "example_workflow_advanced.py",
]


def test_runtime_and_examples_do_not_mutate_sys_path():
    for path in FILES_WITHOUT_SYS_PATH_MUTATION:
        content = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in content, f"unexpected sys.path mutation in {path}"
