import ast
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    REPO_ROOT / "src" / "IkaCore",
    REPO_ROOT / "src" / "IkaModel",
    REPO_ROOT / "src" / "IkaMem",
)
QUALITY_RUNNER = REPO_ROOT / "scripts" / "quality.py"

MAX_DEFINITION_LINES = 79
HOTSPOT_LINE_THRESHOLD = 70
MAX_MODULE_LINES = 600
STRICT_TYPE_ROOTS = (REPO_ROOT / "src" / "IkaMem",)
STRICT_TYPE_FILES = (
    REPO_ROOT / "src" / "IkaCore" / "__init__.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_execution.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_execution_helpers.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_helper_support.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_helpers.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_initialization.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_memory.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_runtime.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_runtime_dispatch.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_runtime_foundation.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_runtime_payloads.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_runtime_simple_execution.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_runtime_stage_execution.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_tools.py",
    REPO_ROOT / "src" / "IkaCore" / "agent_tool_conversion.py",
    REPO_ROOT / "src" / "IkaCore" / "checkpoint.py",
    REPO_ROOT / "src" / "IkaCore" / "execution_types.py",
    REPO_ROOT / "src" / "IkaCore" / "logging_utils.py",
    REPO_ROOT / "src" / "IkaCore" / "prompts.py",
    REPO_ROOT / "src" / "IkaCore" / "stages.py",
    REPO_ROOT / "src" / "IkaCore" / "tools.py",
    REPO_ROOT / "src" / "IkaCore" / "workflow.py",
    REPO_ROOT / "src" / "IkaCore" / "workflow_types.py",
    REPO_ROOT / "src" / "IkaModel" / "anthropic" / "chat_helpers_anthropic.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "chat_runtime.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "chat_request.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "chat_response.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "chat_tool_common.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "chat_tool_loop_async.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "chat_tool_loop_sync.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "types.py",
    REPO_ROOT / "src" / "IkaModel" / "codex" / "chat_helpers_codex.py",
    REPO_ROOT / "src" / "IkaModel" / "codex" / "codex_responses.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "response_usage.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "tool_execution_async.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "tool_execution_sync.py",
    REPO_ROOT / "src" / "IkaModel" / "chat_interface" / "tool_result_formatters.py",
    REPO_ROOT / "src" / "IkaModel" / "codex_constants.py",
    REPO_ROOT / "src" / "IkaModel" / "deepseek" / "chat_helpers_deepseek.py",
    REPO_ROOT / "src" / "IkaModel" / "gemini" / "chat_helpers_gemini.py",
    REPO_ROOT / "src" / "IkaModel" / "model_metadata.py",
    REPO_ROOT / "src" / "IkaModel" / "openai" / "chat_helpers_openai.py",
    REPO_ROOT / "src" / "IkaModel" / "openai" / "openai.py",
    REPO_ROOT / "src" / "IkaModel" / "openrouter" / "chat_helpers_openrouter.py",
    REPO_ROOT / "src" / "IkaModel" / "openrouter" / "openrouter.py",
    REPO_ROOT / "src" / "IkaModel" / "request_interface.py",
    REPO_ROOT / "src" / "IkaModel" / "runtime_errors.py",
    REPO_ROOT / "src" / "IkaModel" / "summarization.py",
    REPO_ROOT / "src" / "IkaModel" / "tool_schema.py",
)
STRICT_TYPE_DIRECTIVE = "# pyright: strict"
STRICT_HEADER_LINES = 20

KNOWN_BROAD_EXCEPTION_BOUNDARIES = Counter({
    ("src/IkaCore/agent_memory.py", "LongTermAgentMemorySaveMixin._save_to_long_term"): 1,
    ("src/IkaCore/agent_memory.py", "LongTermAgentMemorySearchMixin._search_long_term"): 2,
    ("src/IkaCore/agent_memory.py", "ShortTermAgentMemoryMixin._save_to_short_term"): 1,
    ("src/IkaCore/agent_memory.py", "ShortTermAgentMemoryMixin._search_short_term"): 1,
    ("src/IkaCore/agent_tool_execution.py", "SubagentToolExecutorMixin._build_subagent_executor.subagent_executor"): 1,
    ("src/IkaCore/cli_output.py", "_route_to_sink"): 1,
    ("src/IkaCore/workflow_async_executor.py", "AsyncWorkflowExecutor.wait_for_completion"): 1,
    ("src/IkaCore/workflow_async_executor.py", "_run_async_workflow_agent"): 1,
    ("src/IkaModel/chat_interface/tool_execution_async.py", "async_execute_tool"): 1,
    ("src/IkaModel/chat_interface/tool_execution_sync.py", "_execute_parallel_tool_plan"): 1,
    ("src/IkaModel/chat_interface/tool_execution_sync.py", "execute_tool"): 1,
    ("src/IkaModel/codex/chat_helpers_codex.py", "_request_codex_once"): 1,
    ("src/IkaModel/request_interface.py", "request_cancelled"): 1,
})


def _source_files():
    for source_root in SOURCE_ROOTS:
        yield from source_root.rglob("*.py")


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


class _DefinitionVisitor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.stack: list[str] = []
        self.records: list[tuple[str, str, str, int]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, "class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node, "func")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _record(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        size = node.end_lineno - node.lineno + 1
        qualified_name = ".".join([*self.stack, node.name])
        self.records.append((_relative(self.path), kind, qualified_name, size))


class _BroadExceptionVisitor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.stack: list[str] = []
        self.records: Counter[tuple[str, str]] = Counter()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if self._is_broad_exception(node):
            qualified_name = ".".join(self.stack) or "<module>"
            self.records[(_relative(self.path), qualified_name)] += 1
        self.generic_visit(node)

    @staticmethod
    def _is_broad_exception(node: ast.ExceptHandler) -> bool:
        return (
            node.type is None
            or isinstance(node.type, ast.Name)
            and node.type.id in {"Exception", "BaseException"}
        )


def _definition_records() -> list[tuple[str, str, str, int]]:
    records = []
    for path in _source_files():
        visitor = _DefinitionVisitor(path)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        records.extend(visitor.records)
    return records


def _broad_exception_boundaries() -> Counter[tuple[str, str]]:
    records: Counter[tuple[str, str]] = Counter()
    for path in _source_files():
        visitor = _BroadExceptionVisitor(path)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        records.update(visitor.records)
    return records


def test_no_large_functions_or_classes_regress():
    oversized = [
        record
        for record in _definition_records()
        if record[3] > MAX_DEFINITION_LINES
    ]

    assert oversized == []


def test_no_structural_hotspots_over_seventy_lines():
    hotspots = {
        (path, kind, qualified_name)
        for path, kind, qualified_name, size in _definition_records()
        if size >= HOTSPOT_LINE_THRESHOLD
    }

    assert hotspots == set()


def test_no_oversized_production_modules():
    oversized = []
    for path in _source_files():
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_MODULE_LINES:
            oversized.append((_relative(path), line_count))

    assert oversized == []


def test_no_new_broad_exception_boundaries():
    current = _broad_exception_boundaries()
    unexpected = current - KNOWN_BROAD_EXCEPTION_BOUNDARIES
    missing_or_reduced = KNOWN_BROAD_EXCEPTION_BOUNDARIES - current

    assert unexpected == Counter()
    assert missing_or_reduced == Counter()


def test_strict_type_islands_remain_marked():
    missing = []
    for source_root in STRICT_TYPE_ROOTS:
        for path in source_root.rglob("*.py"):
            header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:STRICT_HEADER_LINES])
            if STRICT_TYPE_DIRECTIVE not in header:
                missing.append(_relative(path))
    for path in STRICT_TYPE_FILES:
        header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:STRICT_HEADER_LINES])
        if STRICT_TYPE_DIRECTIVE not in header:
            missing.append(_relative(path))

    assert missing == []


def test_quality_runner_is_documented_and_used_by_ci():
    ci_workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    contributing = REPO_ROOT / "CONTRIBUTING.md"

    assert "python scripts/quality.py" in ci_workflow.read_text(encoding="utf-8")
    assert "python scripts/quality.py" in contributing.read_text(encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(QUALITY_RUNNER), "--list"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    commands = result.stdout
    assert "ruff check ." in commands
    assert "-m py_compile" in commands
    assert "pyright" in commands
    assert "pytest --cov --cov-report=term-missing:skip-covered" in commands
    assert "pip wheel . --no-deps --no-build-isolation" in commands
    assert "verify wheel contains required package data" in commands
    assert "verify built wheel imports outside source tree" in commands
