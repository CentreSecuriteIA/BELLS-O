"""Implement the result mapping function for Vertex AI Moderation."""

from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Vertex Gemini moderation to `Result`.

    Only maps to `content_moderation`.

    Args:
        output: A dict that is the raw response of a supervisor.
        usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: A Result object with content_moderation flag set based on block decision.
    """
    # Vertex Moderation: when blocked, output_raw contains "promptFeedback" with "blockReason";
    # when passing, "promptFeedback" is not present, or does not contain "blockReason".
    # "candidates" may be present for non-blocked outputs, but not always for blocked ones.

    output_raw = output.get("output_raw", output)
    blocked = False

    prompt_feedback = output_raw.get("promptFeedback")
    if prompt_feedback is not None and prompt_feedback.get("blockReason"):
        blocked = True
    else:
        # If not blocked via promptFeedback, then not blocked (as seen in "not blocked" samples)
        blocked = False

    return Result(content_moderation=blocked)
