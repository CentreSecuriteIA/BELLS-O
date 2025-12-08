"""Implement the result mapping function for Llama Guard 4 via REST API (Together AI)."""

import re
from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Llama Guard 4 REST API to `Result`.

    Llama Guard 4 outputs "safe" or "unsafe" in the OpenAI-compatible chat completion format.
    The response structure is: output["choices"][0]["message"]["content"]

    Args:
        output: A dict that is the raw response of a supervisor (OpenAI-compatible format).
        usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: A Result object with content_moderation flag set based on parsed output.
    """
    # Extract text from OpenAI-compatible API response format
    # Structure: output["choices"][0]["message"]["content"]
    text = ""
    choices = output.get("choices", [])
    if choices and len(choices) > 0:
        choice = choices[0]
        message = choice.get("message", {})
        if message:
            text = message.get("content", "")

    # Parse the text to find "safe" or "unsafe"
    # Llama Guard 4 typically outputs "unsafe\nS9" or just "safe"
    text_lower = text.lower().strip()
    
    # Search for "unsafe" or "safe" as standalone words
    unsafe_match = re.search(r'\bunsafe\b', text_lower)
    safe_match = re.search(r'\bsafe\b', text_lower)
    
    # If we find "unsafe", it's harmful
    if unsafe_match:
        is_harmful = True
    elif safe_match:
        is_harmful = False
    else:
        # Fallback: if we can't find either keyword, check for category codes (S1-S14)
        # If category codes are present, it's likely unsafe
        category_match = re.search(r'\bS(1[0-4]|[1-9])\b', text_lower)
        is_harmful = category_match is not None
    
    return Result(content_moderation=is_harmful)

