# Staged synthetic research

Searches two synthetic web documents, fetches evidence, extends a one-step discovery stage, then gives the evidence to a simple agent with a cited-answer check. The application also checks the returned answer because validation retries do not fail closed. Staged checks currently bypass validation; see the strict repros.

From the repository root, after following [wheel setup](../README.md):

```bash
python -m examples.apps.research_assistant.main --offline
python -m examples.apps.research_assistant.main --live --use-local-login --model gpt-5.6-sol
```

Both tools execute and the final recommendation cites [A] and [B] and recommends a pilot. Offline mode also asserts an extension summary request occurred. Live extension count is not exposed by AgentRunState and is reported as null.

Offline mode runs real agent/tool/workflow/snapshot code against scripted provider-native HTTP response envelopes. It blocks accidental HTTP transport access and needs no credentials. Live mode opts in to the local Codex CLI login (or CODEX_BEARER); prompts and files are synthetic. Request scopes allow one attempt and expire after 90 seconds; per-request agent timeouts are at most 45 seconds. Each subprocess session owns its own scope. The apps print only check results, never model answers or credentials.

See [live outcomes](../LIVE_RESULTS.md) and [friction and findings](../../../docs/v2/dogfood-report.md).
