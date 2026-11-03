# IkaCore 2.0.0 release validation

Version 2.0.0 remains unreleased and intentionally breaks v1 compatibility.
Validated runtime revision: 97ddb62879f66f913223054b32696e0567a13f3b.
Measured on 2026-10-18 with CPython 3.13.15, HTTPX 0.28.1, httpcore 1.0.9,
and Pyright 1.1.414 on macOS. Runtime Python-content SHA-256:
fa0574ce1286249e16bdcdae3cdf0e19ed325e4a13a626394bf37c0f7134462d.

## Reproduced release gates

- pytest: 1,825 passed, 2 opt-in skips in 40.52 seconds.
- python scripts/quality.py: one final uninterrupted passing run, including Ruff,
  all-source compilation, standard Pyright, 1,825 passed / 2 skipped under coverage
  in 65.03 seconds, the coverage matrix, sdist/wheel build, twine metadata validation,
  installed-wheel public consumers, typing floors and offline adoption checks.
- Coverage: 10,133/10,133 executable statements (100%),
  2,811/2,888 branches (97.3338%), 99.4086% combined; 217 excluded lines.
  All 129 runtime modules and 16 area floors pass. The current manifest contains
  149 required area/check entries, 146 unique; no matrix gate failures.
- Retained validation from runtime revision 230e2d4: HTTPX 0.27.0 with httpcore 1.0.9
  passed 262 targeted transport/config tests in 14.03 seconds. CI defines all six Python 3.11–3.13 × minimum/current HTTPX
  combinations, using pinned Actions. Linux and the other Python cells were not
  executed locally; the minimum-HTTPX check was not repeated for this revision.
- python -m build --outdir artifacts/verification/release: isolated sdist and wheel-from-sdist
  builds pass. python -m twine check artifacts/verification/release/* passes for both.
- A clean noneditable wheel runtime passes pip check and the two-direct-dependency
  footprint budget. Its runtime Python bytes exactly match source. The separately
  retained installed-wheel seven-route reasoning replay applies to 230e2d4; current
  source-suite provider replay regressions and installed-wheel adoption checks pass.

The wheel is 239,417 bytes compressed / 780,157 bytes unpacked, with 146 entries and
129 Python modules. The sdist is 181,441 bytes. All runtime Python bytes in the final
build match both source and the quality gate's installed wheel. All three py.typed
markers, embedded prompts/model metadata and LICENSE/NOTICE are present; tests,
caches and validation reports are absent. Examples/scripts require the Git checkout.

## Interruption regression coverage

Ninety-three added cases exercise interruption handling at public runtime boundaries.
Workflow dependencies and chained agents stop while approval is pending, with or
without snapshots; workflow results expose status, per-instance interruptions and
resume guidance. Resuming a workflow requires snapshots enabled before execution.
Later empty truncated responses remain interrupted and do not trigger a success summary.

Delegated continuation identity is stored separately from capped provider text.
Resume preserves approved child output under small caps, fresh-agent restoration,
forks and repeated questions across OpenAI Responses, Anthropic, Gemini and DeepSeek,
using sync and async requests. Completed child output and repeated-pause text obey
both request-scoped and environment tool-output caps. Older snapshots retain their
legacy marker-based continuation support.

## Public typing

Owned pyright --verifytypes PACKAGE --ignoreexternal improves the measured
b2e4f35 baseline as follows:

- IkaCore: 83.42% to 100% (959 known, zero ambiguous/unknown exports).
- IkaModel: 93.02% to 100% (619 known, zero ambiguous/unknown exports).
- IkaMem: 84.11% to 100% (107 known, zero ambiguous/unknown exports).
- IkaCore.testing: 100% to 100% (10 known exports).

CI enforces 100% for these owned types. External HTTPX private types are deliberately
excluded from completeness scoring. A strict consumer outside the repository imports
the installed wheel successfully; BareBoneModel(unknown_typo=True), with required
arguments supplied, produces exactly one expected type error. A separate pytest wheel
regression enforces that contract. Runtime get_type_hints(AgentConfig) succeeds.

## Installed-wheel adoption checks

All 25 runnable Python blocks pass offline: README (3), USAGE (15), migration (2),
and the optional memory guide (5). Blocks execute in document order, with documented
prerequisites supplied. API's 53 Python fences display signatures rather than runnable
scripts; the complete generated reference matches installed-wheel introspection.

All seven example mains pass; advanced workflow also runs its sequential path.
All six application --offline modes pass, including fresh-process resume, plus the
parallel cancellation demonstration. MockProvider substitutes native model responses;
the optional Mem0 documentation substitutes its client boundary. Network sockets,
credential refresh and interactive input are blocked or stubbed. Repository-relative
documentation links and anchors, including archived reports, pass their pytest check.

## Single live Codex matrix

Retained validation at runtime revision 230e2d4, reproduced once with:

    python scripts/live_matrix.py --provider codex --use-local-login --model gpt-5.6-sol

Result: 14 passed, 0 failed, 1 skipped; 37.57 summed case seconds.
The reasoning_tools skip is reasoning_not_exposed. Connectivity, sync/async tools,
simple/staged agents, fresh-process resume, cancellation, SSE, usage, deadlines,
synthetic transient retry, compaction and workflow HITL pass. Each case retains its
60-second bound and one live attempt per request.

The live matrix was not repeated for the current revision. This establishes only
that earlier Codex model/account combination. The skip does not verify
live reasoning replay. Seven provider reasoning contracts pass offline; other live
providers, sustained soak, real HTTP/2 and proxy handshakes remain outside this run.
Blocking DNS/custom code can outlive a controlled caller; see
[request limits](../README.md#retained-controls) and
[provider reasoning](../../providers/reasoning.md).

## Evidence and reproduction

Local outputs are retained under artifacts/quality (JUnit, coverage, matrix, typing and
offline JSON), artifacts/verification (final logs and isolated wheel/sdist),
artifacts/live (retained sanitized Codex report),
artifacts/typing-before and artifacts/benchmarks. Build timestamps can change archive
hashes without changing runtime bytes.

Run the commands above from a checkout after installing the test/dev extras in a venv.
The quality command needs loopback access for local transport fixtures and package-index
access to install bounded dependencies in its clean wheel runtime. Ordinary model calls
remain mocked. Historical results are retained in the [archive](archive/merge-validation.md)
and apply only to their labeled revisions.
