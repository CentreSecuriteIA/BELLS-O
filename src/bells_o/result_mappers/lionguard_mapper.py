"""Implement the result mapping function for saillab/x-guard on HF."""

from bells_o.common import Result, Usage


def mapper(output: dict[str, float], usage: Usage = Usage()) -> Result:
    """Map the output format of saillab/x-guard to `Result`.

    Only maps to `content_moderation`. Checks for content between <label> flags.

    Args:
        output (str): A string or list of strings that should be mapped.
        usage (Usage): Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: The result that the output maps to.

    """
    probability = output["binary"]
    assert isinstance(probability, float)
    return Result(content_moderation=probability >= 0.85)
