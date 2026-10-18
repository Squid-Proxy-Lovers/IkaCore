# Configuration-only provider switching

Uses the same agent task and control flow for Codex, OpenAI Responses/Chat, Anthropic, Gemini, DeepSeek and OpenRouter. ProviderConfig selects model routing; ExecutionLimits, ExecutionPolicy and OutputConfig bound the agent; RuntimeControl pauses at post_model for a saved-response resume.

From the repository root, after following [wheel setup](../README.md):

```bash
python -m examples.apps.provider_switch.main --offline
python -m examples.apps.provider_switch.main --live --use-local-login --model gpt-5.6-sol
```

Every route returns READY and resumes its saved turn. --live defaults to all routes but only Codex is live; the six remaining routes stay mocked. --provider selects a single route; requesting a non-Codex live route is rejected.

Offline mode runs real agent/tool/workflow/snapshot code against scripted provider-native HTTP response envelopes. It blocks accidental HTTP transport access and needs no credentials. Live mode opts in to the local Codex CLI login (or CODEX_BEARER); prompts and files are synthetic. Request scopes allow one attempt and expire after 90 seconds; per-request agent timeouts are at most 45 seconds. Each subprocess session owns its own scope. The apps print only check results, never model answers or credentials.

See [live outcomes](../LIVE_RESULTS.md) and [friction and findings](../../../docs/v2/dogfood-report.md).
