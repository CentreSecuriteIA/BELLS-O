"""Implement the result mapping function for google/shieldgemma models on HF."""

import re
from typing import Any

from bells_o.common import Result, Usage


def mapper(output: str | dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of ShieldGemma to `Result`.

    ShieldGemma outputs text that starts with "Yes" or "No" to indicate
    whether content violates safety policies. "Yes" means harmful, "No" means benign.

    Args:
        output: A string (raw model output) or dict that should be mapped.
        usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: The result that the output maps to.

    """
    # Handle both string and dict outputs
    if isinstance(output, dict):
        # If it's a dict, try to extract text from common fields
        text = output.get("output_raw", output.get("text", str(output)))
        if not isinstance(text, str):
            text = str(text)
    else:
        text = str(output)

    # Clean the text
    text_clean = text.strip()

    # ShieldGemma outputs should start with "Yes" or "No"
    # Look for "Yes" or "No" at the start of the text (case-insensitive)
    text_lower = text_clean.lower()
    
    # Check if it starts with "yes" or "no"
    if text_lower.startswith("yes"):
        is_harmful = True
    elif text_lower.startswith("no"):
        is_harmful = False
    else:
        # Try to find "Yes" or "No" anywhere in the first few words
        # This handles cases where there might be whitespace or formatting
        match = re.search(r'\b(yes|no)\b', text_lower, re.IGNORECASE)
        if match:
            is_harmful = match.group(1).lower() == "yes"
        else:
            # Fallback: check for keywords that indicate harm
            if "violate" in text_lower or "harmful" in text_lower or "unsafe" in text_lower:
                is_harmful = True
            elif "not violate" in text_lower or "safe" in text_lower or "benign" in text_lower:
                is_harmful = False
            else:
                # Default to False (benign) if we can't determine
                is_harmful = False

    return Result(content_moderation=is_harmful)

