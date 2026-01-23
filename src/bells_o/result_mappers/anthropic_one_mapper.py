"""Implement the result mapping function for Anthropic classification output."""

from typing import Any

from bells_o.common import Result, Usage

from .one_mapper import mapper as one_map


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

    if output["stop_reason"] == "refusal":
        # Refusal to answer; treat as harmful
        return one_map("1")
    return one_map(output["content"][0]["text"])
