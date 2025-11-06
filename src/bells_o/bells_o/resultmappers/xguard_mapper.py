"""Implement the result mapping function for saillab/x-guard on HF."""

from re import findall

from bells_o.common import Result


def mapper(output: str) -> Result:
    """Map the output format of saillab/x-guard to `Result`.

    Only maps to `content_moderation`.

    Args:
        output (str): A string or list of strings that should be mapped.

    Returns:
        Result: The result that the output maps to.

    """
    label = findall(r"<label>(.*)</label>", output)[0]
    return Result(content_moderation=not label == "safe")
