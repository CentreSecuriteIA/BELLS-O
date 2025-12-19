"""Implement the result mapping function for OpenAI-compatible API models."""

from typing import Any

from bells_o.common import Result, Usage
from bells_o.result_mappers import unsafe_mapper


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of OpenAI-compatible models with "unsafe" as a harmful flag to `Result`.

    Maps output "safe" or "unsafe" in the OpenAI-compatible chat completion format.
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

    return unsafe_mapper(text, usage)
