"""Implement the result mapping function for saillab/x-guard on HF."""

from typing import Any, overload

from bells_o.common import Result, Usage


@overload
def mapper(output: dict[str, Any], usage: Usage) -> Result: ...


@overload
def mapper(output: list[dict[str, Any]], usage: Usage) -> list[Result]: ...


def mapper(output: dict[str, str] | list[dict[str, Any]], usage: Usage) -> Result | list[Result]:
    """Map the output format of LakeraGuard to `Result`.

    Maps to usage types of passed `usage`.

    Args:
        output (dict[str, str]|list[dict[str, str]]): A dict or list of dicts that are the raw response of a supervisor.
        usage (Usage): The usage types to map to.

    """
    if isinstance(output, list):
        labels = [out["flagged"] for out in output]
        results = []
        for flagged in labels:
            result = Result()
            for usage_type in usage:
                result[usage_type] = flagged
        return results
    else:
        result = Result()
        for usage_type in usage:
            result[usage_type] = output["flagged"]
        return result
