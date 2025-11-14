"""Implement the result mapping function for saillab/x-guard on HF."""

from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of OpenAI Moderation to `Result`.

    Only maps to `content_moderation`.

    Args:
        output (dict[str, str]): A dict that is the raw response of a supervisor.
        usage (Usage): Conformity with ResultMapper type. Argument is ignored as usage type is static.

    """
    result = Result(content_moderation=output["results"][0]["flagged"])
    return result
