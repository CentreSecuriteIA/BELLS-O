"""Implement the result mapping function for ThinkGuard."""

import re
from typing import Any

from bells_o.common import Result, Usage


def extract_label_and_categories(content: str) -> tuple[str | None, list[str]]:
    """Extract safety label and categories from ThinkGuard output.

    Args:
        content (str): The decoded text output from ThinkGuard model.

    Returns:
        tuple[str | None, list[str]]: A tuple containing the safety label and list of categories.

    """
    # ThinkGuard output format:
    # - Safety Assessment: safe/unsafe
    # - Categories: category1,category2,...
    
    safe_pattern = r"Safety Assessment:\s*(safe|unsafe)"
    category_pattern = r"Categories:\s*([^\n-]+)"
    
    safe_label_match = re.search(safe_pattern, content, re.IGNORECASE)
    label = safe_label_match.group(1).lower() if safe_label_match else None
    
    # Extract categories if present
    categories = []
    category_match = re.search(category_pattern, content, re.IGNORECASE)
    if category_match:
        category_str = category_match.group(1).strip()
        # Split by comma and clean up whitespace
        categories = [cat.strip() for cat in category_str.split(',') if cat.strip()]
    
    return label, categories


def mapper(output: str | dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of ThinkGuard to `Result`.

    ThinkGuard outputs text that contains a safety assessment (safe or unsafe)
    and potentially a list of categories. "unsafe" means harmful.

    Args:
        output: A string (raw model output) or dict that should be mapped.
        usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: The result that the output maps to.

    """
    # Handle both str and dict inputs
    if isinstance(output, str):
        text = output
    else:
        text = output.get("output_raw", output.get("text", output.get("content", str(output))))

    # Extract the safety label and categories
    label, categories = extract_label_and_categories(text) 
    
    # Determine if content is harmful
    # "safe" = False (not harmful), "unsafe" = True (harmful)
    if label == "safe":
        is_harmful = False
    elif label == "unsafe":
        is_harmful = True
    else:
        # Fallback: if we can't extract a clear label, check for keywords
        text_lower = text.lower()
        if "unsafe" in text_lower:
            is_harmful = True
        elif "safe" in text_lower:
            is_harmful = False
        else:
            # Default to False if we can't determine
            is_harmful = False

    return Result(content_moderation=is_harmful)
