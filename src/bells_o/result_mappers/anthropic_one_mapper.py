"""Implement the result mapping function for Anthropic classification output."""

import re
from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Anthropic API to `Result` for classification.

    Parses text output["content"][0]["text"] from Anthropic API looking for "1" (harmful) or "0" (benign).
    The model is instructed to output 1 if harmful, 0 if benign.

    Args:
        output: A dict that is the raw response of a supervisor (Anthropic format).
        usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: A Result object with content_moderation flag set based on parsed output.

    """
    # Extract text from Anthropic API response format
    # Structure: output["content"][0]["text"]
    text = ""
    content = output.get("content", [])
    if output.get("stop_reason") == "refusal" and isinstance(content, list) and len(content) == 0:
        # Refusal to answer; treat as non-harmful (benign)
        return Result(content_moderation=True)

    # Otherwise, extract text from content as usual
    text = ""
    if content and len(content) > 0:
        text_item = content[0]
        if isinstance(text_item, dict):
            text = text_item.get("text", "")

    text_clean = text.strip()

    # Try to find explicit "1" or "0" as standalone or at start
    match = re.search(r"\b([01])\b", text_clean)
    if match:
        digit = match.group(1)
        is_harmful = digit == "1"
    else:
        # Fallback: look for "1" or "0" at the start
        first_char = text_clean[0] if text_clean else ""
        if first_char == "1":
            is_harmful = True
        elif first_char == "0":
            is_harmful = False
        else:
            # Check for keywords and fallback
            text_lower = text_clean.lower()
            if "harmful" in text_lower or "1" in text_clean:
                is_harmful = True
            elif "benign" in text_lower or "safe" in text_lower or "0" in text_clean:
                is_harmful = False
            else:
                is_harmful = False

    return Result(content_moderation=is_harmful)
