# IkaCore runtime development draft

This draft establishes the compatibility requirements for future core-runtime
work. It contains no imported feature patches or copied experimental code.
The production runtime remains unchanged while candidate improvements are
implemented and reviewed individually.

## Compatibility requirements

- Preserve existing public exports, import paths, positional arguments, and defaults.
- Add new configuration as optional keyword arguments and keep new policies opt-in.
- Preserve existing execution entry points, result shapes, tool/HITL behavior,
  and documented workflow semantics.
- Keep existing checkpoint databases and UIDs readable and writable. New snapshot
  storage must be additive or separate and must not destructively migrate old files.
- Preserve current provider routing, timeout/retry defaults, environment proxies,
  authentication opt-in, and client ownership/cleanup.
- Document observable behavior changes from bug fixes and validate downstream users.

The compatibility fixture and tests under `src/IkaTest` check existing public
call shapes/defaults and independently created legacy checkpoint databases.
They supplement runtime tests; representative downstream applications still
need smoke tests before release.

## Development order

1. Restore the current quality baseline.
2. Implement cancellation and deadline controls with deterministic lifecycle tests.
3. Add explicit tool-result limits and reconcile completion handling.
4. Add opt-in snapshots and compaction with legacy-storage compatibility coverage.
5. Review provider and workflow optimizations for behavior preservation.
6. Run the complete quality suite, package checks, and downstream smoke tests.

Each feature requires separate review and regression coverage before merging.
No significant change to existing default behavior is accepted without an
explicit maintainer decision.

## Checks

```bash
pip install -e ".[test,dev]"
pytest src/IkaTest/test_v2_compatibility.py
python scripts/quality.py
```
