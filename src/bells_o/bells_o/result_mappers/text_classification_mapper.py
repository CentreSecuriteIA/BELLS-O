"""Implement the result mapping function for text-based classification output from OpenAI-compatible APIs."""

import re
from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of OpenAI-compatible APIs to `Result` for classification.

    Parses text output from APIs like Mistral, X-AI, OpenAI looking for "1" (harmful) or "0" (benign).
    The model is instructed to output 1 if harmful, 0 if benign.

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

    # Parse the text to find "1" or "0"
    # Look for standalone "1" or "0", or at the start of the text
    # Handle cases like "1", "0", "1\n", "0\n", "The answer is 1", etc.
    text_clean = text.strip()
    
    # Try to find "1" or "0" as standalone or at the start
    # First, try to find a standalone digit
    match = re.search(r'\b([01])\b', text_clean)
    if match:
        digit = match.group(1)
        is_harmful = digit == "1"
    else:
        # Fallback: look for "1" or "0" anywhere in the first few characters
        # This handles cases where model outputs just "1" or "0"
        first_char = text_clean[0] if text_clean else ""
        if first_char == "1":
            is_harmful = True
        elif first_char == "0":
            is_harmful = False
        else:
            # If we can't find a clear "1" or "0", check for keywords
            text_lower = text_clean.lower()
            if "harmful" in text_lower or "1" in text_clean:
                is_harmful = True
            elif "benign" in text_lower or "safe" in text_lower or "0" in text_clean:
                is_harmful = False
            else:
                # Default to False (benign) if we can't determine
                is_harmful = False

    return Result(content_moderation=is_harmful)

