AGENT_END_INSTRUCTION = """
CRITICAL: When you call the agent_end tool, you MUST provide your final answer/output in the tool arguments.
The agent_end tool REQUIRES a non-empty response. You must pass your final answer using the "input" parameter.

DO NOT call agent_end with empty arguments {}. This will cause an error.
Your final answer must be based on your initial prompt and any context you have gathered.
The output format is given by the rest of the prompt.
When you call agent_end, it must be the ONLY tool call in that response. Never batch agent_end with any other tool calls.
If you still need tool outputs, call those tools first. After you have reviewed their results, send a separate response containing only agent_end.

CRITICAL RULE:
NEVER SEND MORE THEN ONE AGENT_END TOOL CALL IN A SINGLE RESPONSE, THIS WILL END EXECUTION OF THE AGENT.
DO NOT BATCH AGENT_END TOOL CALLS WITH OTHER TOOLS. AGENT_END MUST BE A STANDALONE TOOL CALL.
"""

STAGE_MOVEMENT_INSTRUCTION = """
STAGE MOVEMENT:
Use the stage_end tool when the current stage is complete to advance to the next stage.
If the change_stage tool is available, use it with stage_index (the stage to change to) and reason (why you need to go back). stage_index must be in that stage's allowed_back_to list.
"""

HITL_INSTRUCTION = """
HITL: You can use the ask_user tool when you need the user to answer something. Use it as often as needed. The question must be clear and direct.
"""

PARALLEL_TOOL_INSTRUCTION = """
PARALLEL TOOLS:
<use_parallel_tool_calls>
For maximum efficiency, whenever you perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially. Prioritize calling tools in parallel whenever possible. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. When running multiple read-only commands like `ls` or `list_dir`, always run all of the commands in parallel. Err on the side of maximizing parallel tool calls rather than running too many tools sequentially.
Exception: never batch `agent_end` with any other tool. If you are finished, send a response that contains only `agent_end`.
</use_parallel_tool_calls>
"""
