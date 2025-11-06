"""Implement the result mapping function for saillab/x-guard on HF."""

from bells_o.common import Result, Usage


def mapper(output: dict[str, str], usage: Usage) -> Result:
    """Map the output format of LakeraGuard to `Result`.

    LakeraGuard supports different policies, so `usage` needs to be passed

    Args:
        output (dict[str, str]): A dict that is the raw response of a supervisor.
        usage (Usage): The usage types to map to.

    """
    result = Result()
    for usage_type in usage:
        result[usage_type] = output["flagged"]
    return result
