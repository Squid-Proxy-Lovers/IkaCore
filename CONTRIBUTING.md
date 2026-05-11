# Contributing

## Local Setup

```bash
pip install -e .[test]
```

## Test Commands

Run the main suite:

```bash
pytest
```

Run the focused provider and workflow regressions:

```bash
pytest src/IkaTest/test_ika_base_agent.py
pytest src/IkaTest/test_workflow_semantics.py
pytest src/IkaTest/test_hitl_interrupt_resume.py
```

## Project Conventions

- keep runtime behavior aligned with tests
- prefer narrowly scoped changes over broad refactors
- update `docs/USAGE.md` when the public API changes
- preserve current workflow semantics:
  - `next` edges propagate context
  - `child` edges are stage wiring only
- treat subagent delegated input as untrusted content

## Current Cleanup Priorities

- remove remaining `sys.path` mutation from runtime modules
- keep packaging metadata accurate
- add regression tests for behavior changes before modifying provider logic
