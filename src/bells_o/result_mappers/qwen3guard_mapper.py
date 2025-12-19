"""Implement the result mapping function for Qwen3Guard."""

import re
from typing import Any

from bells_o.common import Result, Usage


def extract_label_and_categories(content: str) -> tuple[str | None, list[str]]:
    """Extract safety label and categories from Qwen3Guard output.

    Args:
        content (str): The decoded text output from Qwen3Guard model.

    Returns:
        tuple[str | None, list[str]]: A tuple containing the safety label and list of categories.

    """
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"

    safe_label_match = re.search(safe_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else None
    categories = re.findall(category_pattern, content)

    return label, categories


def mapper(output: str | dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Qwen3Guard to `Result`.

    Qwen3Guard outputs text that contains a safety label (Safe, Unsafe, or Controversial)
    and potentially a list of categories. "Unsafe" or "Controversial" means harmful.

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
    # "Safe" = False (not harmful), "Unsafe" or "Controversial" = True (harmful)
    if label == "Safe":
        is_harmful = False
    else:
        if "Jailbreak" in categories and len(categories) == 1:
            is_harmful = False  # TODO: make this work
        is_harmful = True

    return Result(content_moderation=is_harmful)
