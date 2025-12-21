"""Implement the result mapping function for ibm-granite/granite-guardian-3.3-{variant} on HF."""

from re import findall

from bells_o.common import Result, Usage

from .yes_mapper import mapper as yes_map


def mapper(output: str, usage: Usage = Usage()) -> Result:
    """Map the output format of ibm-granite/granite-guardian-3.3-{variant} to `Result`.

    Only maps to `content_moderation`. Checks for content between <label> flags.

    Args:
        output (str): A string or list of strings that should be mapped.
        usage (Usage): Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: The result that the output maps to.

    """
    input_labels = findall(r"Harmful request: (\w+)", output)
    if len(input_labels) == 0:
        input_result = yes_map("", usage)
    else:
        input_result = yes_map(input_labels[0], usage)

    output_labels = findall(r"Harmful response: (\w+)", output)
    if len(output_labels) == 0:
        output_result = yes_map("", usage)
    else:
        output_result = yes_map(output_labels[0], usage)

    return input_result or output_result  # returns a flagged result if any of them is true
