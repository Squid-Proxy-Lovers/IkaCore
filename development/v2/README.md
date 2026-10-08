# Secret-scanned runtime source proposals

This directory captures candidate framework implementation and regression-test
source for incremental integration. Sources are stored as text, so they are not
imported, executed, linted as production Python, or included in the installed
IkaCore package. Their presence is not a claim that they have been ported or that
their behavior is backward-compatible.

`source/inventory.json` lists the feature groups, files, review status, and content
digests. All 98 Python/reference files were selected from framework/test paths;
Python sources parse successfully and the captured directory has zero Gitleaks
v8.30.1 default-rule candidates. A scanner cannot certify the absence of every
secret or decide whether code is confidential.

Included core proposals:
- Shared cancellation, bounded retry controls, and stream deadlines.
- General/Codex tool-output limits.
- Run/frame snapshots, pause/resume, restart/fork, and compaction.
- Parent-chain checkpoints, entry checkpoints, tool handling, and diagnostics.
- Provider/history replay, transport, retries, and workflow scheduling proposals.

Generic general-execution library source is isolated under
`experimental_general_execution` as review material. It is outside the initial
core runtime integration scope and is not enabled or installed by this draft.

Excluded material includes credential/configuration files, experiment logs,
checkpoint databases, unrelated application/research runners, and executable-bit
only changes. Existing CI/quality tooling is already present in the supported
runtime; obsolete copies of that tooling are not retained here.

## Review and integration

1. Review the inventory before removing any remaining source branches.
2. Accept, adapt, or reject each proposal individually.
3. Port selected behavior into the current runtime with compatibility tests.
4. Preserve defaults; enable new execution policies through explicit settings.
5. Run the full quality suite and representative downstream projects before merge.

The current public API fixture, existing checkpoint-format checks, and runtime
regressions remain authoritative during porting. Captured candidate tests require
adaptation to current APIs and are not part of the active test suite yet.
