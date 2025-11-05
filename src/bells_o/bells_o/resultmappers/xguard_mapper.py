"""Implement the result mapping function for saillab/x-guard on HF."""

from re import findall
from typing import overload

from bells_o.common import Result, Usage


@overload
def mapper(output: str, usage: Usage) -> Result: ...


@overload
def mapper(output: list[str], usage: Usage) -> list[Result]: ...


def mapper(
    output: str | list[str], usage: Usage = Usage("content_moderation")
) -> Result | list[Result]:
    """Map the output format of saillab/x-guard to `Result`.

    Only maps to `content_moderation`.

    Args:
        output (str|list[str]): A string or list of strings that should be mapped.
        usage: unused.

    Returns:
        Result: The result that the output maps to.

    """
    if isinstance(output, list):
        labels = [findall(r"<label>(.*)</label>", string)[0] for string in output]
        return [Result(content_moderation=not label == "safe") for label in labels]
    else:
        label = findall(r"<label>(.*)</label>", output)[0]
        return Result(content_moderation=not label == "safe")
