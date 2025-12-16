"""Implement the result mapping function for google/shieldgemma models on HF."""

from bells_o.common import Result, Usage


def mapper(output: str, usage: Usage = Usage()) -> Result:
    """Map the output format of ShieldGemma to `Result`.

    ShieldGemma outputs text that starts with "Yes" or "No" to indicate
    whether content violates safety policies. "Yes" means harmful, "No" means benign.

    Args:
        output: A string (raw model output) or dict that should be mapped.
        usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: The result that the output maps to.

    """
    # ShieldGemma outputs should start with "Yes" or "No"
    # add case insensitivity for robustness
    # yes = harmful, no = not harmful
    return Result(content_moderation="yes" in output.lower())
