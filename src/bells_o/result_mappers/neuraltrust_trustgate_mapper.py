"""Implement the result mapping function for NeuralTrust Jailbreak detection."""

from typing import Any

from bells_o.common import Result, Usage


# TODO: think about making this universal, so for any plugin, check if blocked is True
def mapper(output: dict[str, Any], usage: Usage) -> Result:
    """Map the output format of a NeuralTrustTrustGateSupervisor to `Result`.

    NeuralTrust supports different policies, so `usage` needs to be passed.

    Args:
        output (dict[str, str]): A dict that is the raw response of a supervisor.
        usage (Usage): The usage types to map to.

    """
    # find jailbreak flag in output
    for plugin in output["metadata"]:
        if plugin["plugin_name"] == "neuraltrust_jailbreak":
            flag = plugin["data"]["blocked"]
            break

    result = Result()
    for usage_type in usage:
        result[usage_type] = flag
    return result
