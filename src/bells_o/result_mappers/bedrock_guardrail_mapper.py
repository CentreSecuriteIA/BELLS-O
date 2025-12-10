"""Implement the result mapping function for AWS Bedrock Guardrail."""

from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of AWS Bedrock Guardrail to `Result`.

    AWS Bedrock Guardrail returns an action field that indicates if content was blocked.
    The action can be "GUARDRAIL_INTERVENED", "INTERVENED", "1" (blocked) or "NONE", "0" (allowed).

    Args:
        output (dict[str, Any]): A dict that is the raw response of a supervisor.
        usage (Usage): The usage types to map to. Defaults to all usage types.

    Returns:
        Result: A Result object with the mapped values.

    """
    # Check if the guardrail intervened (blocked the content)
    action = output.get("action", "NONE")
    
    # Handle different possible action formats
    is_blocked = (
        action == "GUARDRAIL_INTERVENED"
        or action == "INTERVENED"
        or action == "1"
        or str(action).upper() == "TRUE"
    )

    result = Result()
    # Map to all usage types specified
    for usage_type in usage:
        result[usage_type] = is_blocked

    return result

