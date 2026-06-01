"""Implement the result mapping function for ibm-granite/granite-guardian-3.3-{variant} on HF."""

from re import findall

from bells_o.common import Result, Usage

from .yes_mapper import mapper as yes_map


def mapper(output: str, usage: Usage = Usage()) -> Result:
    """Map the output format of ibm-granite/granite-guardian-3.3-{variant} to `Result`.

    Only maps to `content_moderation`. Checks for content between <score> flags.

    Args:
        output (str): A string or list of strings that should be mapped.
        usage (Usage): Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: The result that the output maps to.

    """
    labels = findall(r"<score>(.*)</score>", output)
    if len(labels) == 0:
        return yes_map("1", usage)  # if output format breaks, assume harmful

    return yes_map(labels[0], usage)
