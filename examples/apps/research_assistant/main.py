"""Research a synthetic energy pilot, then validate its cited recommendation."""
from __future__ import annotations

from typing import Any

from examples.apps.common import agent, call, emit, end, parser, session
from IkaCore import ExecutionLimits, ExecutionPolicy, IkaStage, IkaTools

SOURCES = {
    "A": "Synthetic field study: LEDs reduce lighting energy by 20%; small sample.",
    "B": "Synthetic facilities guide: pilot one floor and meter results before rollout.",
}


def cited_answer(result: dict[str, Any]) -> bool:
    """Include citations [A] and [B] and recommend a pilot, not guaranteed savings."""
    text = result.get("final_message", "")
    return "[A]" in text and "[B]" in text and "pilot" in text.lower()


def main() -> None:
    args = parser(__doc__).parse_args()
    seen: list[str] = []

    def search(data):
        seen.append("search")
        return [{"id": key, "snippet": value} for key, value in SOURCES.items() if data["query"]]

    def fetch(data):
        seen.append("fetch")
        return SOURCES[data["id"]]

    tools = [IkaTools("search_web", "Search synthetic public documents", {"query": {"type": "string", "required": True}},
                      execute_function=search, replay_policy="allow", side_effect_type="read"),
             IkaTools("fetch_page", "Fetch source A or B", {"id": {"type": "string", "required": True}},
                      execute_function=fetch, replay_policy="allow", side_effect_type="read")]
    stages = [IkaStage("discover", "Call search_web once for LED pilot, then stage_end. Do not finish the whole agent.",
                       tools[:1], stage_max_step=1),
              IkaStage("verify", "Call fetch_page for B, then agent_end with a brief cited recommendation using [A] and [B].",
                       tools[1:], stage_max_step=3)]
    researcher = agent(args, "research", "Investigate whether a small office should pilot LED lighting. Sources are synthetic.",
                       Stages=stages, limits=ExecutionLimits(maxsteps=7, max_tool_rounds=1, step_timeout=45),
                       policy=ExecutionPolicy(max_stage_extensions=1, extend_stage_steps_by=2, max_step_extensions=0))
    answer = "Pilot one floor and measure savings [B]; the small study reports 20%, with uncertainty [A]."
    with session(args, {"research": [call("search_web", query="LED pilot"), call("stage_end", input="Found A and B"),
                                     call("fetch_page", id="B"), end(answer)],
                        "writer": [end(answer)]}) as mock:
        try:
            research = researcher.execution()
            assert {"search", "fetch"} <= set(seen), "Both research tools must actually execute"
            # Staged final checks currently bypass validation (dogfood repro). Use a simple writer
            # and also reject a failed final answer at the application boundary.
            writer = agent(args, "writer", "Use the evidence below. Call agent_end with a concise pilot recommendation "
                           "citing [A] and [B].\nEvidence: " + research["final_message"],
                           policy=ExecutionPolicy(final_answer_checks=(cited_answer,), validation_retries=1,
                                                  validation_extra_steps=1, max_step_extensions=0))
            try:
                result = writer.execution()
                assert cited_answer(result), "Provider exhausted validation retries without meeting the contract"
            finally:
                writer.shutdown()
            if mock:
                assert mock.summary_calls >= 1, "The one-step discovery stage must exercise its extension"
            emit("research_assistant", tools_used=seen, citations_valid=True,
                 extension_observed=(mock.summary_calls >= 1) if mock else None)
        finally:
            researcher.shutdown()


if __name__ == "__main__":
    main()
