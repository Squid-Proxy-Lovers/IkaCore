# Remaining hardening branch coverage

Historical inventory at `dba0e05`, before reasoning/dogfood integration. For current
figures and artifact locations, see [merge-validation.md](merge-validation.md).

The final run covers every one of the 9,305 measured runtime statements. There are **no remaining uncovered lines**. The 194 excluded lines predate this work; no `no cover` or `no branch` pragmas were added.

These **81 branch arcs** remain uncovered. They are explicit testable edge-path gaps, not assertions of unreachability or offline impossibility. Negative destinations denote a function exit. The behavioral tests exercise every statement on each alternative; they do not exercise every combination of conditions or loop exit. The global branch floor is 95%; measured coverage is 96.8629%.

| Module | Uncovered arcs (source → destination) | Source conditions |
| --- | --- | --- |
| `src/IkaCore/agent_chat_support.py` | 192 → 194, 236 → 240, 249 → 251, 259 → 261 | 192: if logger:; 236: if elapsed < interval:; 249: if hasattr(agent, "per_tool_rate_limit"):; 259: if agent.logger: |
| `src/IkaCore/agent_execution_loop.py` | 40 → 42, 53 → exit, 76 → exit, 178 → 180 | 40: elif agent.logger:; 53: elif agent.logger:; 76: if agent.logger:; 178: if agent.logger: |
| `src/IkaCore/agent_initialization.py` | 128 → 132, 133 → exit | 128: if callable(aclose):; 133: if logger: |
| `src/IkaCore/agent_parse.py` | 71 → 64, 150 → 152 | 71: elif value and (not isinstance(value, (dict, list))) and (str(value).strip() not in [".", ""]):; 150: if logger: |
| `src/IkaCore/agent_policy.py` | 46 → 48, 144 → 148, 153 → 151 | 46: if force_answer and force_answer.strip():; 144: if agent.logger:; 153: if func_name in failed_checks: |
| `src/IkaCore/agent_runtime_dispatch.py` | 29 → 32, 57 → 59, 201 → 203 | 29: if isinstance(raw_interrupt_data, dict):; 57: if isinstance(raw_message_history, dict):; 201: if isinstance(raw_message_history, dict): |
| `src/IkaCore/agent_runtime_foundation.py` | 53 → 55, 61 → 63, 81 → 83, 175 → 177 | 53: if agent.logger:; 61: if isinstance(raw_tool_call_counts, dict):; 81: if agent.logger:; 175: if agent.logger: |
| `src/IkaCore/agent_snapshots.py` | 87 → 89 | 87: if run is not None and run["root_frame_id"] == root: |
| `src/IkaCore/agent_tools.py` | 155 → 157, 162 → 164, 184 → 186, 253 → 255 | 155: if logger:; 162: if logger:; 184: if logger:; 253: if stage_end_index is not None: |
| `src/IkaCore/cli_output.py` | 170 → 172, 335 → exit, 422 → 424, 516 → 519 | 170: if current_line:; 335: if rendered_outputs:; 422: if description:; 516: if instance is None: |
| `src/IkaCore/events.py` | 96 → 94 | 96: if flush is not None: |
| `src/IkaCore/logging_utils_support.py` | 63 → exit, 81 → exit, 116 → 118, 118 → 120, 122 → exit | 63: elif item_type == "json" and isinstance(data, dict) and self.level == 2:; 81: if not self._atexit_registered:; 116: if self._queue is not None:; 118: if self._writer_thread.is_alive() and self._writer_thread != threading.current_thread():; 122: if self._atexit_registered: |
| `src/IkaCore/snapshot_agent_hooks.py` | 27 → 33, 83 → exit | 27: if frame is not None and frame["status"] in {"completed", "cancelled", "failed"}:; 83: if info is not None and info["parent_frame_id"]: |
| `src/IkaCore/snapshot_agent_state.py` | 77 → 79, 79 → 86 | 77: if self.logger:; 79: if resume_input.strip(): |
| `src/IkaCore/workflow_async_executor.py` | 85 → 78, 98 → exit | 85: if task_id in self.pending_tasks:; 98: if pending: |
| `src/IkaCore/workflow_prompt_context.py` | 10 → 14 | 10: if previous != agent.prompt or not hasattr(agent, "_workflow_base_input"): |
| `src/IkaCore/workflow_snapshots.py` | 39 → 32, 42 → 32 | 39: elif frame is not None and frame["status"] == "paused":; 42: if latest: |
| `src/IkaMem/storage/mem0_storage.py` | 200 → 206 | 200: if self.memory_type in memory_type_map: |
| `src/IkaModel/anthropic/claude.py` | 83 → 76, 103 → 105 | 83: elif msg["role"] == "assistant":; 103: if _PARALLEL_TOOL_PROMPT not in system_prompt: |
| `src/IkaModel/base.py` | 148 → exit | 148: if self.content_prompt: |
| `src/IkaModel/chat_interface/chat_request.py` | 168 → 171 | 168: if usage_info: |
| `src/IkaModel/chat_interface/chat_response.py` | 132 → exit | 132: if hasattr(barebone_model, "forced_tool_name"): |
| `src/IkaModel/chat_interface/chat_tool_common.py` | 39 → 41 | 39: if remember_signatures: |
| `src/IkaModel/chat_interface/chat_tool_loop_async.py` | 60 → 55, 182 → 184, 379 → 387 | 60: if recent_count >= 5:; 182: if usage_info:; 379: if not stage_ended: |
| `src/IkaModel/chat_interface/chat_tool_loop_sync.py` | 171 → 173 | 171: if usage_info: |
| `src/IkaModel/chat_interface/tool_execution_sync.py` | 166 → 168, 170 → 172, 464 → 473 | 166: if future is not None:; 170: if future is not None:; 464: if not interrupt_data: |
| `src/IkaModel/codex/auth.py` | 138 → exit | 138: if fd is not None: |
| `src/IkaModel/codex/chat_helpers_codex.py` | 364 → 362, 381 → 356, 384 → 356 | 364: if block_data.get("type") == "output_text":; 381: if summary:; 384: if joined: |
| `src/IkaModel/codex/stream_transport.py` | 23 → 25 | 23: if client is not None: |
| `src/IkaModel/deepseek/chat_helpers_deepseek.py` | 93 → 81 | 93: if isinstance(inner, (dict, list)): |
| `src/IkaModel/deepseek/deepseek.py` | 85 → 87, 88 → 90 | 85: if message_history["system"]["message"]:; 88: if plan.emit_first_input: |
| `src/IkaModel/gemini/google.py` | 100 → 103 | 100: if plan.emit_first_input: |
| `src/IkaModel/model_metadata.py` | 273 → 276 | 273: if "openai.com" in api_url_lower: |
| `src/IkaModel/openai/chat_helpers_openai.py` | 154 → 152 | 154: if block_data.get("type") == "output_text": |
| `src/IkaModel/openrouter/chat_helpers_openrouter.py` | 72 → 74, 74 → 76, 102 → 112 | 72: if provider:; 74: if raw:; 102: if log_path: |
| `src/IkaModel/request_interface.py` | 235 → 240 | 235: if "rate limit" in error_text.lower() or "rate_limit" in error_text.lower(): |
| `src/IkaModel/request_lifecycle.py` | 153 → exit, 158 → exit | 153: if active and task is not None and not task.done():; 158: if active: |
| `src/IkaModel/summarization.py` | 265 → 263, 270 → 268, 384 → 386, 509 → 512 | 265: if item_data.get("type") == "message":; 270: if block_data.get("type") == "output_text":; 384: if message:; 509: if write_to_history: |
