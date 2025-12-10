"""Implement the result mapping function for Gemini classification output."""

import re
from typing import Any

from bells_o.common import Result, Usage

def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Gemini classification to `Result`.

    Parses Gemini's text output looking for "1" (harmful) or "0" (benign).
    The model is instructed to output 1 if harmful, 0 if benign.
    Additionally, handles scenario where output contains a blockReason (treat as "1"/harmful).

    Args:
        output: A dict that is the raw response of a supervisor.
        usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: A Result object with content_moderation flag set based on parsed output.
    """
    # The output parameter is already output_raw (passed from supervisor.__call__)
    # Extract text from candidates[0].content.parts[0].text
    # Structure: output["candidates"][0]["content"]["parts"][0]["text"]
    text = ""
    candidates = output.get("candidates", [])
    if candidates and len(candidates) > 0:
        candidate = candidates[0]
        content = candidate.get("content", {})
        if content:
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                text = parts[0].get("text", "")

    # If there is no text, check for 'promptFeedback' and 'blockReason'
    if not text.strip() and "promptFeedback" in output:
        prompt_feedback = output["promptFeedback"]
        if (
            isinstance(prompt_feedback, dict) and
            "blockReason" in prompt_feedback and
            prompt_feedback["blockReason"]
        ):
            return Result(content_moderation=True)

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


    return Result(content_moderation=is_harmful)

