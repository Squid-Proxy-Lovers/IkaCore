"""Run one agent contract on provider-specific configuration only."""
from __future__ import annotations

import tempfile
from pathlib import Path

from examples.apps.common import credentials, emit, end, parser, session
from IkaCore import ExecutionLimits, ExecutionPolicy, IkaBaseAgent, OutputConfig, ProviderConfig, RuntimeControl
from IkaModel.codex import CODEX_API_URL

PROVIDERS = {
    "codex": ("gpt-5.6-sol", CODEX_API_URL, True),
    "openai-responses": ("gpt-4o-mini", "https://api.openai.com/v1/responses", True),
    "openai-chat": ("gpt-4o-mini", "https://api.openai.com/v1/chat/completions", False),
    "anthropic": ("claude-sonnet-4-5", "https://api.anthropic.com/v1/messages", True),
    "gemini": ("gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent", True),
    "deepseek": ("deepseek-chat", "https://api.deepseek.com/chat/completions", True),
    "openrouter": ("openrouter/auto", "https://openrouter.ai/api/v1/chat/completions", True),
}


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument("--provider", choices=[*PROVIDERS, "all"], default="all")
    args = cli.parse_args()
    if args.live and args.provider not in {"all", "codex"}:
        cli.error("Only Codex is authorized live; use --offline for other routes")
    routes = list(PROVIDERS) if args.provider == "all" else [args.provider]
    outcomes = {}
    for route in routes:
        model, url, responses = PROVIDERS[route]
        # --live --provider all keeps the remaining providers mocked, with no keys needed.
        local = type(args)(**vars(args))
        local.live = args.live and route == "codex"
        if route == "codex":
            model = args.model
        helper = IkaBaseAgent(
            "portable", prompt="TASK:portable\nCall agent_end with exactly READY.", model_id=model,
            api_key=credentials(local), provider=ProviderConfig(api_url=url, max_tokens=400,
                              use_responses_api=responses, reasoning_effort="low" if route == "codex" else None),
            limits=ExecutionLimits(maxsteps=2, step_timeout=40, max_tool_rounds=2, max_tool_calls=2),
            policy=ExecutionPolicy(max_step_extensions=0, max_stage_extensions=0), output=OutputConfig(cli=False))
        with tempfile.TemporaryDirectory(prefix="ika-provider-") as directory, session(local, {"portable": [end("READY")]}):
            helper.enable_snapshots(str(Path(directory) / "route.db"), RuntimeControl(pause_points={"post_model"}))
            try:
                paused = helper.execution()
                assert paused["status"] == "paused"
                helper.clear_pause_requests()
                result = helper.resume_from_snapshot(paused["snapshot_id"])
                assert result["final_message"] == "READY"
                outcomes[route] = "live" if local.live else "mocked"
            finally:
                helper.shutdown()
    emit("provider_switch", routes=outcomes, grouped_config=True, runtime_control_resume=True)


if __name__ == "__main__":
    main()
