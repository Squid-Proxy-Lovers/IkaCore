# IkaCore 2.0.0 release validation

Version 2.0.0 remains unreleased and intentionally breaks v1 compatibility.
Validated runtime revision: 230e2d401901259bb3df825d662c6a9661a171c8.
Measured on 2026-10-18 with CPython 3.13.15, HTTPX 0.28.1, httpcore 1.0.9,
and Pyright 1.1.414 on macOS. Runtime Python-content SHA-256:
f1ff785a4b50bd8bbb5205d72dc95c8aa91004d784f83e2486520c3b47d04c7f.

## Reproduced release gates

- pytest: 1,732 passed, 2 opt-in skips in 37.84 seconds.
- python scripts/quality.py: one final uninterrupted passing run, including Ruff,
  all-source compilation, standard Pyright, 1,732 passed / 2 skipped under coverage
  in 56.46 seconds, the coverage matrix, sdist/wheel build, twine metadata validation,
  installed-wheel public consumers, typing floors and offline adoption checks.
- Coverage: 10,074/10,074 executable statements (100%),
  2,788/2,864 branches (97.3464%), 99.4126% combined; 216 excluded lines.
  All 129 runtime modules and 16 area floors pass. The current manifest contains
  149 required area/check entries, 146 unique; no matrix gate failures.
- HTTPX 0.27.0 with httpcore 1.0.9: 262 targeted transport/config tests passed
  in 14.03 seconds. CI defines all six Python 3.11–3.13 × minimum/current HTTPX
  combinations, using pinned Actions. Linux and the other Python cells were not
  executed locally.
- python -m build --outdir artifacts/release: isolated sdist and wheel-from-sdist
  builds pass. python -m twine check artifacts/release/* passes for both.
- A clean noneditable wheel runtime passes pip check, seven-route offline reasoning
  replay, and the updated two-direct-dependency footprint budget.

The wheel is 238,098 bytes compressed / 775,060 bytes unpacked, with 146 entries and
129 Python modules. The sdist is 179,530 bytes. All runtime Python bytes in the final
build match both source and the quality gate's installed wheel. All three py.typed
markers, embedded prompts/model metadata and LICENSE/NOTICE are present; tests,
caches and validation reports are absent. Examples/scripts require the Git checkout.

## Public typing

Installed-wheel pyright --verifytypes PACKAGE --ignoreexternal improves the measured
b2e4f35 baseline as follows:

- IkaCore: 83.42% to 100% (952 known, zero ambiguous/unknown exports).
- IkaModel: 93.02% to 100% (618 known, zero ambiguous/unknown exports).
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
prerequisites supplied. API's 52 Python fences display signatures rather than runnable
scripts; the complete generated reference matches installed-wheel introspection.

All seven example mains pass; advanced workflow also runs its sequential path.
All six application --offline modes pass, including fresh-process resume, plus the
parallel cancellation demonstration. MockProvider substitutes native model responses;
the optional Mem0 documentation substitutes its client boundary. Network sockets,
credential refresh and interactive input are blocked or stubbed. Repository-relative
documentation links and anchors, including archived reports, pass their pytest check.

## Single live Codex matrix

Reproduced once with:

    python scripts/live_matrix.py --provider codex --use-local-login --model gpt-5.6-sol

Result: 14 passed, 0 failed, 1 skipped; 37.57 summed case seconds.
The reasoning_tools skip is reasoning_not_exposed. Connectivity, sync/async tools,
simple/staged agents, fresh-process resume, cancellation, SSE, usage, deadlines,
synthetic transient retry, compaction and workflow HITL pass. Each case retains its
60-second bound and one live attempt per request.

This establishes only this Codex model/account combination. The skip does not verify
live reasoning replay. Seven provider reasoning contracts pass offline; other live
providers, sustained soak, real HTTP/2 and proxy handshakes remain outside this run.
Blocking DNS/custom code can outlive a controlled caller; see
[request limits](../README.md#retained-controls) and
[provider reasoning](../../providers/reasoning.md).

## Evidence and reproduction

Local outputs are retained under artifacts/quality (JUnit, coverage, matrix, typing and
offline JSON), artifacts/release (wheel/sdist), artifacts/live (sanitized Codex report),
artifacts/typing-before and artifacts/benchmarks. Build timestamps can change archive
hashes without changing runtime bytes.

Run the commands above from a checkout after installing the test/dev extras in a venv.
The quality command needs loopback access for local transport fixtures and package-index
access to install bounded dependencies in its clean wheel runtime. Ordinary model calls
remain mocked. Historical results are retained in the [archive](archive/merge-validation.md)
and apply only to their labeled revisions.
