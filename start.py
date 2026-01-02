import os 
from openai import OpenAI
from anthropic import Anthropic
import requests
import json



## DeepSeek


url = "https://api.deepseek.com/chat/completions"

payload = json.dumps({
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "Hi",
      "role": "user"
    }
  ],
  "model": "deepseek-chat",
  "thinking": {
    "type": "disabled"
  },
  "frequency_penalty": 0,
  "max_tokens": 4096,
  "presence_penalty": 0,
  "response_format": {
    "type": "text"
  },
  "stop": None,
  "stream": False,
  "stream_options": None,
  "temperature": 1,
  "top_p": 1,
  "tools": None,
  "tool_choice": "none",
  "logprobs": False,
  "top_logprobs": None
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Authorization': 'Bearer <TOKEN>'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
## response format: 
# {
#   "id": "string",
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "message": {
#         "content": "string",
#         "reasoning_content": "string",
#         "tool_calls": [
#           {
#             "id": "string",
#             "type": "function",
#             "function": {
#               "name": "string",
#               "arguments": "string"
#             }
#           }
#         ],
#         "role": "assistant"
#       },
#       "logprobs": {
#         "content": [
#           {
#             "token": "string",
#             "logprob": 0,
#             "bytes": [
#               0
#             ],
#             "top_logprobs": [
#               {
#                 "token": "string",
#                 "logprob": 0,
#                 "bytes": [
#                   0
#                 ]
#               }
#             ]
#           }
#         ],
#         "reasoning_content": [
#           {
#             "token": "string",
#             "logprob": 0,
#             "bytes": [
#               0
#             ],
#             "top_logprobs": [
#               {
#                 "token": "string",
#                 "logprob": 0,
#                 "bytes": [
#                   0
#                 ]
#               }
#             ]
#           }
#         ]
#       }
#     }
#   ],
#   "created": 0,
#   "model": "string",
#   "system_fingerprint": "string",
#   "object": "chat.completion",
#   "usage": {
#     "completion_tokens": 0,
#     "prompt_tokens": 0,
#     "prompt_cache_hit_tokens": 0,
#     "prompt_cache_miss_tokens": 0,
#     "total_tokens": 0,
#     "completion_tokens_details": {
#       "reasoning_tokens": 0
#     }
#   }
# }









## OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5-nano",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)

## Anthropic
client = Anthropic()  # Reads ANTHROPIC_API_KEY from environment
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024, # this is the max output tokens
    messages=[{"role": "user", "content": "Hello, Claude"}]
)

# response format: 
# {
#   "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
#   "type": "message",
#   "role": "assistant",
#   "content": [
#     {
#       "type": "text",
#       "text": "Hello! How can I assist you today?"
#     }
#   ],
#   "model": "claude-sonnet-4-5",
#   "stop_reason": "end_turn",
#   "usage": {
#     "input_tokens": 12,
#     "output_tokens": 8
#   }
# }