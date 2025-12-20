"""Implement the result mapping function for saillab/x-guard on HF."""

from re import findall

from bells_o.common import Result, Usage


def mapper(output: str, usage: Usage = Usage()) -> Result:
    """Map the output format of saillab/x-guard to `Result`.

    Only maps to `content_moderation`. Checks for content between <label> flags.

    Args:
        output (str): A string or list of strings that should be mapped.
        usage (Usage): Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: The result that the output maps to.

    """
    labels = findall(r"<label>(.*)</label>", output)
    if len(labels) == 0:
        return Result(content_moderation=False)
    return Result(content_moderation=True)
