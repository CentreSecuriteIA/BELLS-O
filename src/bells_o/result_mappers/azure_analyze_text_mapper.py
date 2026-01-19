"""Implement the result mapping function for Azure's Analyze Text."""

from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Azure's Analyze Text to `Result`.

    Only maps to `content_moderation`. Assumes that 6 is the severity threshold.
    See https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-text?pivots=programming-language-rest.

    Args:
        output (dict[str, str]): A dict that is the raw response of a supervisor.
        usage (Usage): Conformity with ResultMapper type. Argument is ignored as usage type is static.

    """
    flagged = any(category["severity"] >= 2 for category in output["categoriesAnalysis"])
    result = Result(content_moderation=flagged)
    return result
