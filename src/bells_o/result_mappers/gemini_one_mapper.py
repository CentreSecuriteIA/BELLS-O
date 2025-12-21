"""Implement the result mapping function for Gemini classification output."""

from typing import Any

from bells_o.common import Result, Usage
from bells_o.result_mappers import one_mapper


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Gemini classification to `Result`.

    Parses Gemini's output["candidates"][0]["content"]["parts"][0]["text"] looking for "1" (harmful) or "0" (benign).
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

    return one_mapper(text, usage)
