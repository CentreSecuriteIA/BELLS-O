"""Implement a result mapping function for OpenAI-compatible API models that checks for "1" as a harmful flag."""

from typing import Any

from bells_o.common import Result, Usage

from .one_mapper import mapper as one_map


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of OpenAI-compatible models with "1" as a harmful flag to `Result`.

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

    return one_map(text, usage)
