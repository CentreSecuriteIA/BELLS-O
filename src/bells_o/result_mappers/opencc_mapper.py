"""Implement the result mapping function for OpenCC."""

from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of OpenCC to `Result`.

    OpenCC returns a single terminal `decision`. Any decision other than "allow"
    (i.e. "block" or "annotate") is treated as a detection. The resulting flag is
    applied to every declared usage type.

    Args:
        output (dict[str, Any]): A dict that is the raw response of the supervisor.
        usage (Usage): The usage types to map to.

    """
    flagged = output["decision"] != "allow"

    result = Result()
    for usage_type in usage:
        result[usage_type] = flagged
    return result
