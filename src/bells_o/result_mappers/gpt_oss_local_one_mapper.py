"""Implement the one-mapper function for locally hosted GPT-OSS models."""

from re import findall

from bells_o.common import Result, Usage

from .one_mapper import mapper as one_map


def mapper(output: str, usage: Usage = Usage()) -> Result:
    """Map the output format of ibm-granite/granite-guardian-3.3-{variant} to `Result`.

    Only maps to `content_moderation`. Checks for content after <score> flags.

    Args:
        output (str): A string or list of strings that should be mapped.
        usage (Usage): Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: The result that the output maps to.

    """
    output_strings = findall(r"assistantfinal(.*)", output)
    if len(output_strings) == 0:
        return one_map("1", usage)  # if output format breaks, assume harmful

    output_string = output_strings[0]  # there should only be one output string
    return one_map(output_string, usage)
