"""Static live probe inventory and an allowlist-only report boundary (no output text)."""

import math
import re

PROVIDERS = {
    'codex': {'key': 'CODEX_BEARER', 'model_env': 'IKACORE_LIVE_CODEX_MODEL',
              'adapter': 'codex', 'url': 'https://chatgpt.com/backend-api/codex/responses'},
    'openai-responses': {'key': 'OPENAI_API_KEY', 'model_env': 'IKACORE_LIVE_OPENAI_RESPONSES_MODEL',
                         'adapter': 'openai_responses', 'url': 'https://api.openai.com/v1/responses'},
    'openai-chat': {'key': 'OPENAI_API_KEY', 'model_env': 'IKACORE_LIVE_OPENAI_CHAT_MODEL',
                    'adapter': 'openai', 'url': 'https://api.openai.com/v1/chat/completions'},
    'anthropic': {'key': 'ANTHROPIC_API_KEY', 'model_env': 'IKACORE_LIVE_ANTHROPIC_MODEL',
                  'adapter': 'anthropic', 'url': 'https://api.anthropic.com/v1/messages'},
    'deepseek': {'key': 'DEEPSEEK_API_KEY', 'model_env': 'IKACORE_LIVE_DEEPSEEK_MODEL',
                 'adapter': 'deepseek', 'url': 'https://api.deepseek.com/chat/completions'},
    'gemini': {'key': 'GEMINI_API_KEY', 'model_env': 'IKACORE_LIVE_GEMINI_MODEL',
               'adapter': 'gemini', 'url': 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'},
    'openrouter': {'key': 'OPENROUTER_API_KEY', 'model_env': 'IKACORE_LIVE_OPENROUTER_MODEL',
                   'adapter': 'openrouter', 'url': 'https://openrouter.ai/api/v1/chat/completions'},
    'zai': {'key': 'ZAI_API_KEY', 'model_env': 'IKACORE_LIVE_ZAI_MODEL',
            'adapter': 'openai', 'url': 'https://api.z.ai/api/coding/paas/v4/chat/completions'},
}
CASES = ('sync_text', 'async_text', 'sync_tools', 'async_tools', 'simple_agent', 'staged_agent',
         'fresh_process_resume', 'async_cancellation', 'reasoning_tools', 'streaming_sse',
         'usage_accounting', 'deadline', 'transient_retry', 'compaction', 'workflow_hitl')
USAGE_KEYS = ('input_tokens', 'output_tokens', 'total_tokens', 'input_cached_tokens')
BOOL_FIELDS = {'usage_reported', 'tool_result_replayed', 'completed', 'fresh_process',
               'network_blocked_during_resume', 'usage_preserved', 'live_response_closed',
               'caller_client_open', 'reasoning_preserved', 'sse_parsed', 'deadline_enforced',
               'dependency_answer_missing', 'dependency_answer_contains_fixture', 'dependency_answer_empty',
               'synthetic_transport', 'compacted', 'coherent_answer', 'dependency_preserved', 'hitl_resumed'}
INT_FIELDS = {'tool_executions', 'stages_executed', 'attempts', 'compactions'}
REASONS = {'not_configured', 'model_required', 'runtime_no_sse', 'runtime_no_extended_thinking',
           'model_not_reasoning', 'reasoning_not_exposed', 'connectivity_failed'}
CHECKS = {'workflow_initial_pause', 'workflow_interrupted_instance', 'workflow_upstream_answer',
          'workflow_dependency_answer', 'workflow_dependency_context', 'workflow_resumed', 'compaction_answer', 'compaction_summary',
          'compaction_snapshot', 'reasoning_replay'}
ERRORS = {'AssertionError', 'RuntimeError', 'ValueError', 'TypeError', 'KeyError', 'TimeoutExpired',
          'HTTPStatusError', 'ConnectError', 'ReadTimeout', 'IkaAPIError', 'IkaAuthenticationError',
          'IkaRequestDeadlineExceeded', 'IkaProviderResponseError', 'IkaProviderPayloadError',
          'IkaContextWindowError', 'IkaRequestCancelled', 'ChildProcessError'}


def clean_usage(value):
    value = value if isinstance(value, dict) else {}
    return {key: value[key] for key in USAGE_KEYS
            if type(value.get(key)) is int and 0 <= value[key] < 10**12}


def safe_details(value):
    """Never trust exception prose, stdout, model strings, or arbitrary result fields."""
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in BOOL_FIELDS if type(value.get(key)) is bool}
    result.update({key: value[key] for key in INT_FIELDS
                   if type(value.get(key)) is int and 0 <= value[key] < 10**6})
    result['usage'] = clean_usage(value.get('usage'))
    if isinstance(value.get('check'), str) and value['check'] in CHECKS:
        result['check'] = value['check']
    if isinstance(value.get('reason'), str) and value['reason'] in REASONS:
        result['reason'] = value['reason']
    if isinstance(value.get('error_type'), str) and value['error_type'] in ERRORS:
        result['error_type'] = value['error_type']
    if type(value.get('http_status')) is int and 100 <= value['http_status'] <= 599:
        result['http_status'] = value['http_status']
    return result


def validate_model(value, env):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/@-]{0,159}', value):
        raise ValueError('invalid model identifier')
    secrets = [env.get(config['key'], '') for config in PROVIDERS.values()]
    if any(secret and secret in value for secret in secrets):
        raise ValueError('model identifier contains a credential')
    return value


def safe_seconds(value):
    return round(value, 2) if type(value) in (float, int) and math.isfinite(value) and value >= 0 else 0
