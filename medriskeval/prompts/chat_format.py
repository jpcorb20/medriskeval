"""Chat format normalization utilities.

This module provides utilities for normalizing various input formats to
OpenAI-style chat messages. It handles conversion from:
- Raw strings to user messages
- Dictionary-style messages
- Provider-specific formats
"""

from __future__ import annotations

from typing import Any, Sequence, Union

from medriskeval.core.types import ChatMessage, Role


# Type for inputs that can be normalized to chat messages
MessageInput = Union[
    str,
    dict[str, Any],
    ChatMessage,
    Sequence[Union[str, dict[str, Any], ChatMessage]],
]


def normalize_message(msg: str | dict[str, Any] | ChatMessage) -> ChatMessage:
    """Normalize a single message to ChatMessage format.
    
    Handles various input formats:
    - String: converted to user message
    - Dict: expected to have 'role' and 'content' keys
    - ChatMessage: returned as-is
    
    Args:
        msg: Input message in any supported format.
        
    Returns:
        Normalized ChatMessage object.
        
    Raises:
        ValueError: If dict format is missing required keys.
    """
    if isinstance(msg, ChatMessage):
        return msg
    
    if isinstance(msg, str):
        return ChatMessage(role=Role.USER.value, content=msg)
    
    if isinstance(msg, dict):
        if "role" not in msg or "content" not in msg:
            raise ValueError(
                f"Message dict must have 'role' and 'content' keys, got: {msg.keys()}"
            )
        return ChatMessage(
            role=msg["role"],
            content=msg["content"],
            name=msg.get("name"),
        )
    
    raise TypeError(f"Cannot normalize message of type {type(msg)}")


def normalize_messages(
    messages: MessageInput,
    default_role: str = "user",
) -> list[ChatMessage]:
    """Normalize various input formats to a list of ChatMessages.
    
    Handles:
    - Single string → [user message]
    - Single message → [message]
    - List of strings → [user messages]
    - List of messages → normalized list
    - Mixed lists → normalized list
    
    Args:
        messages: Input messages in any supported format.
        default_role: Role to use when converting raw strings.
        
    Returns:
        List of normalized ChatMessage objects.
        
    Raises:
        ValueError: If input cannot be normalized.
    """
    if isinstance(messages, str):
        return [ChatMessage(role=default_role, content=messages)]
    
    if isinstance(messages, ChatMessage):
        return [messages]
    
    if isinstance(messages, dict):
        return [normalize_message(messages)]
    
    # Sequence of messages
    result: list[ChatMessage] = []
    for msg in messages:
        result.append(normalize_message(msg))
    return result


def to_openai_format(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessages to OpenAI API format.
    
    Args:
        messages: Sequence of ChatMessage objects.
        
    Returns:
        List of dicts in OpenAI chat completion format.
    """
    return [msg.to_dict() for msg in messages]


def from_openai_format(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    """Convert OpenAI API format to ChatMessages.
    
    Args:
        messages: List of dicts from OpenAI API.
        
    Returns:
        List of ChatMessage objects.
    """
    return [ChatMessage.from_dict(msg) for msg in messages]


def prepend_system_message(
    messages: list[ChatMessage],
    system_content: str,
) -> list[ChatMessage]:
    """Prepend a system message to a conversation.
    
    If the first message is already a system message, it is replaced.
    Otherwise, a new system message is prepended.
    
    Args:
        messages: Existing conversation messages.
        system_content: Content for the system message.
        
    Returns:
        New list with system message at the start.
    """
    system_msg = ChatMessage(role=Role.SYSTEM.value, content=system_content)
    
    if messages and messages[0].role == Role.SYSTEM.value:
        return [system_msg] + messages[1:]
    return [system_msg] + messages


def append_assistant_message(
    messages: list[ChatMessage],
    content: str,
) -> list[ChatMessage]:
    """Append an assistant message to a conversation.
    
    Args:
        messages: Existing conversation messages.
        content: Content for the assistant message.
        
    Returns:
        New list with assistant message at the end.
    """
    return messages + [ChatMessage(role=Role.ASSISTANT.value, content=content)]


def format_conversation(
    user_message: str,
    system_message: str | None = None,
    assistant_prefix: str | None = None,
) -> list[ChatMessage]:
    """Build a standard single-turn conversation.
    
    Convenience function for the common case of system + user + optional prefix.
    
    Args:
        user_message: The user's input message.
        system_message: Optional system instruction.
        assistant_prefix: Optional prefix to constrain assistant response.
        
    Returns:
        List of ChatMessage objects forming the conversation.
    """
    messages: list[ChatMessage] = []
    
    if system_message:
        messages.append(ChatMessage(role=Role.SYSTEM.value, content=system_message))
    
    messages.append(ChatMessage(role=Role.USER.value, content=user_message))
    
    if assistant_prefix:
        messages.append(ChatMessage(role=Role.ASSISTANT.value, content=assistant_prefix))
    
    return messages


def merge_system_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Merge consecutive system messages into one.
    
    Some prompt construction flows may produce multiple system messages.
    This function combines them for providers that only support one.
    
    Args:
        messages: Input messages that may have multiple system messages.
        
    Returns:
        Messages with at most one system message at the start.
    """
    if not messages:
        return []
    
    # Collect all system content
    system_parts: list[str] = []
    other_messages: list[ChatMessage] = []
    
    for msg in messages:
        if msg.role == Role.SYSTEM.value:
            system_parts.append(msg.content)
        else:
            other_messages.append(msg)
    
    result: list[ChatMessage] = []
    if system_parts:
        merged_system = "\n\n".join(system_parts)
        result.append(ChatMessage(role=Role.SYSTEM.value, content=merged_system))
    
    result.extend(other_messages)
    return result
