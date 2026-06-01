"""Implement the result mapping function for Azure's Prompt Shield."""

from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Azure's Prompt Shield to `Result`.

    Only maps to `jailbreak`.
    See https://ai.azure.com/doc/azure/ai-services/content-safety/quickstart-jailbreak.

    Args:
        output (dict[str, str]): A dict that is the raw response of a supervisor.
        usage (Usage): Conformity with ResultMapper type. Argument is ignored as usage type is static.

    """
    try:
        flagged = output["userPromptAnalysis"]["attackDetected"]
    except KeyError:
        try:
            error_code = output["error"]["code"]
            if error_code == "InvalidRequestBody":  # handles prompts that are too long
                flagged = False  # result is unflagged
            else:
                print(f"DEBUG: value error: {output}")
        except KeyError as e:
            print(f"DEBUGGING: output: {output}")
            raise KeyError from e

    result = Result(jailbreak=flagged)
    return result