#!/usr/bin/env python3
"""
Summarization module for preventing context window overflow.
Based on Para-Core's summarization approach.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add odin-main to Python path
_ODIN_MAIN_PATH = Path(__file__).parent.parent / "odin-main"
sys.path.insert(0, str(_ODIN_MAIN_PATH))

from odin.model import Message, MessageRole, Model  # noqa: E402

_LOG = logging.getLogger(__name__)

SUMMARY_PROMPT = """You are an expert conversation summarization assistant. Your task is to create a comprehensive, well-structured summary of a conversation history that preserves critical information while condensing the content.

Your summary must capture and organize the following elements:

1. CONTEXT AND BACKGROUND
   - Initial situation or problem statement
   - Relevant background information that sets the stage
   - Any constraints, requirements, or parameters established early in the conversation

2. KEY TOPICS AND THEMES
   - Main subjects discussed throughout the conversation
   - Important subtopics and their relationships
   - Any recurring themes or patterns that emerged

3. DECISIONS AND AGREEMENTS
   - All decisions that were made, explicitly or implicitly
   - Agreements reached between participants
   - Commitments or promises made
   - Any consensus or disagreement points

4. ACTIONS TAKEN AND PLANS
   - Specific actions that were executed or initiated
   - Step-by-step plans that were outlined
   - Tasks assigned or responsibilities delegated
   - Any workflows or processes established

5. CRITICAL INFORMATION EXCHANGED
   - Important facts, data, or statistics shared
   - Key insights, observations, or realizations
   - Technical details, specifications, or requirements
   - Dates, deadlines, or time-sensitive information
   - Names, identifiers, or references to external resources

6. CURRENT STATE AND STATUS
   - Where things stand at the end of the conversation
   - What has been completed versus what remains
   - Current blockers, challenges, or open questions
   - Next steps or immediate action items

7. RELATIONSHIPS AND DEPENDENCIES
   - How different topics or actions relate to each other
   - Dependencies between tasks or decisions
   - Cause-and-effect relationships identified
   - Any conditional logic or branching scenarios

STRUCTURAL REQUIREMENTS:
- Use clear section headers to organize information
- Maintain chronological flow when relevant
- Preserve important details that might be needed for future reference
- Use bullet points or numbered lists for clarity
- Include specific examples or quotes when they are particularly important
- Note any uncertainty, ambiguity, or areas that need clarification

TONE AND STYLE:
- Be objective and factual
- Avoid redundancy while ensuring completeness
- Write in clear, concise language
- Use professional terminology appropriate to the domain
- Maintain consistency in naming conventions and terminology

Your summary should be detailed enough that someone reading it can understand the full context of the conversation without needing to review the original history, while being concise enough to be practical for future reference.
"""


CONTEXT_WINDOW_SIZES = {
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    "claude-sonnet-4": 200000,
    "claude-opus-4": 200000,
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
}


def get_context_window_size(model_id: str) -> int:
    """Get context window size for a model."""
    for key, size in CONTEXT_WINDOW_SIZES.items():
        if key.lower() in model_id.lower():
            return size
    return 200000


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token."""
    return len(text) // 4


def get_total_tokens(messages: List[Message]) -> int:
    """Estimate total tokens in message history."""
    total = 0
    for msg in messages:
        if msg.content:
            total += estimate_tokens(msg.content)
        if msg.reasoning:
            for r in msg.reasoning:
                total += estimate_tokens(r)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total += estimate_tokens(tc.name)
                total += estimate_tokens(json.dumps(tc.arguments))
        if msg.token_usage:
            total += msg.token_usage.input_tokens + msg.token_usage.output_tokens
    return total


def summarize_messages(model: Model, messages: List[Message], system_prompt: str, first_input: Optional[str] = None) -> str:
    """
    Summarize message history using the model.
    
    Args:
        model: Model instance to use for summarization
        messages: List of messages to summarize
        system_prompt: Original system prompt to preserve
        first_input: First user input to preserve
    
    Returns:
        Summary string
    """
    try:
        conversation_text = ""
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue
            role = msg.role.value
            content = msg.content or ""
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content += f"\n[Tool Call: {tc.name} with args {json.dumps(tc.arguments)}]"
            conversation_text += f"{role}: {content}\n\n"
        
        summary_messages = [
            Message(role=MessageRole.SYSTEM, content=SUMMARY_PROMPT),
            Message(role=MessageRole.USER, content=f"Please summarize the following conversation history:\n\n{conversation_text}")
        ]
        
        _LOG.info("Summarizing message history (estimated %d tokens)", estimate_tokens(conversation_text))
        response = model.generate(messages=summary_messages)
        
        summary = response.content or ""
        _LOG.info("Summary generated (%d tokens)", estimate_tokens(summary))
        
        return summary
        
    except Exception as e:
        _LOG.error(f"Failed to summarize message history: {e}")
        return ""


def should_summarize(messages: List[Message], model_id: str, threshold: float = 0.8) -> bool:
    """
    Check if message history should be summarized.
    
    Args:
        messages: List of messages
        model_id: Model identifier
        threshold: Fraction of context window to trigger summarization (default: 0.8)
    
    Returns:
        True if summarization is needed
    """
    max_tokens = get_context_window_size(model_id)
    current_tokens = get_total_tokens(messages)
    threshold_tokens = int(max_tokens * threshold)
    
    should = current_tokens > threshold_tokens
    if should:
        _LOG.info(f"Token count ({current_tokens}) exceeds threshold ({threshold_tokens}/{max_tokens}). Summarization needed.")
    
    return should


def compress_messages(messages: List[Message], model: Model, model_id: str) -> List[Message]:
    """
    Compress message history by summarizing when needed.
    Preserves system prompt, first input, and recent messages.
    
    Args:
        messages: List of messages to compress
        model: Model instance for summarization
        model_id: Model identifier
    
    Returns:
        Compressed message list
    """
    if not should_summarize(messages, model_id):
        return messages
    
    if len(messages) < 5:
        return messages
    
    system_msg = None
    first_user_msg = None
    recent_messages = []
    
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.SYSTEM:
            system_msg = msg
        elif msg.role == MessageRole.USER and first_user_msg is None:
            first_user_msg = msg
        elif i >= len(messages) - 3:
            recent_messages.append(msg)
    
    messages_to_summarize = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            continue
        if msg == first_user_msg:
            continue
        if msg in recent_messages:
            continue
        messages_to_summarize.append(msg)
    
    if not messages_to_summarize:
        return messages
    
    summary_text = summarize_messages(
        model,
        messages_to_summarize,
        system_msg.content if system_msg else "",
        first_user_msg.content if first_user_msg else None
    )
    
    if not summary_text:
        _LOG.warning("Summarization failed, keeping original messages")
        return messages
    
    compressed = []
    if system_msg:
        compressed.append(system_msg)
    
    if first_user_msg:
        compressed.append(first_user_msg)
    
    compressed.append(Message(
        role=MessageRole.ASSISTANT,
        content=f"[SUMMARY]\n{summary_text}"
    ))
    
    compressed.extend(recent_messages)
    
    _LOG.info(f"Compressed {len(messages)} messages to {len(compressed)} messages")
    
    return compressed
