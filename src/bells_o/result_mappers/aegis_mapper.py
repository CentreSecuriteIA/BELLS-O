"""Implement the result mapping function for nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0 on HF."""

import re
from typing import Any

from bells_o.common import Result, Usage


def mapper(output: str | dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Aegis to `Result`.

    Aegis outputs "safe" or "unsafe" on the first line, and if unsafe,
    category IDs (O1-O13) on the second line.

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

    # Aegis outputs should start with "safe" or "unsafe" on the first line
    # Look for "safe" or "unsafe" at the start of the text (case-insensitive)
    text_lower = text_clean.lower()
    
    # Split by newlines to get the first line
    first_line = text_lower.split('\n')[0].strip()
    
    # Check if it starts with "safe" or "unsafe"
    if first_line.startswith("unsafe"):
        is_harmful = True
    elif first_line.startswith("safe"):
        is_harmful = False
    else:
        # Try to find "safe" or "unsafe" anywhere in the first few words
        # This handles cases where there might be whitespace or formatting
        match = re.search(r'\b(safe|unsafe)\b', first_line, re.IGNORECASE)
        if match:
            is_harmful = match.group(1).lower() == "unsafe"
        else:
            # Fallback: check for category codes (O1-O13) which indicate unsafe
            # If category codes are present, it's likely unsafe
            category_match = re.search(r'\bO(1[0-3]|[1-9])\b', text_lower)
            is_harmful = category_match is not None

    return Result(content_moderation=is_harmful)

