"""Implement the result mapping function for saillab/x-guard on HF."""

from re import findall
from typing import overload

from ..common import Result


@overload
def mapper(output: str) -> Result: ...


@overload
def mapper(output: list[str]) -> list[Result]: ...


def mapper(output: str | list[str]) -> Result | list[Result]:
    """Map the output format of saillab/x-guard to `Result`.

    Only maps to `content_moderation`.

    Args:
        output (str|list[str]): A string or list of strings that should be mapped.

    """
    if isinstance(output, list):
        labels = [findall(r"<label>(.*)</label>", string)[0] for string in output]
        return [Result(content_moderation=1.0 if label == "safe" else 0.0) for label in labels]
    else:
        label = findall(r"<label>(.*)</label>", output)[0]
        return Result(content_moderation=1.0 if label == "safe" else 0.0)
