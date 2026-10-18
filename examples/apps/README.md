# IkaCore v2 through small applications

These applications use the installed package without sys.path changes. All data and tools are synthetic or disposable. Run from the repository root:

```bash
python3 -m venv .venv
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"
python -m pip install build
python -m build --wheel
python -m pip install 'dist/ikacore-2.0.0-py3-none-any.whl[test,dev]'
python -m examples.apps.research_assistant.main --offline
pytest -q src/IkaTest/test_dogfood_apps.py src/IkaTest/dogfood_findings
```

Use a fresh venv for wheel validation. Do not install editable or set PYTHONPATH to src. `python -c "import IkaCore; print(IkaCore.__file__)"` should point into `.venv/.../site-packages`. Development pytest intentionally checks source too; each app subprocess clears PYTHONPATH and uses the wheel.

- [research_assistant](research_assistant/): staged tools, extensions and final-answer validation.
- [code_reviewer](code_reviewer/): temporary repository, structured review, timeout and output caps.
- [approval_workflow](approval_workflow/): dependencies, persisted HITL, new-process resume and alternative fork.
- [long_chat](long_chat/): twelve turns, IkaMem tools, controlled compaction and new-session resume.
- [parallel_pipeline](parallel_pipeline/): parallel fan-out/fan-in, deadlines, Ctrl-C and caller-owned client.
- [provider_switch](provider_switch/): seven routes with the grouped configuration API.

Every module defaults to offline mode; `--live --use-local-login --model gpt-5.6-sol` opts into Codex. The provider switch keeps other providers mocked. No service receives real repository contents. Snapshot demos use temporary directories by default. To retain state and answer manually, use the approval module's `--phase start|resume|fork --state DIRECTORY` or the chat module's `--phase first|second --state DIRECTORY` options.

The shared `common.py` contains bounded construction, credential opt-in and offline HTTP fixtures. Fixtures use the supported `IkaCore.testing.MockProvider` adapter; agent execution and tool callbacks run normally. Apps use `send()`, staged fail-closed validation, grouped token budgets, workflow fork-and-answer, and `ProviderConfig(client=...)`. Live timing and sanitized results are in [LIVE_RESULTS.md](LIVE_RESULTS.md); resolved findings and passing regression reproductions are in [the report](../../docs/v2/dogfood-report.md).
