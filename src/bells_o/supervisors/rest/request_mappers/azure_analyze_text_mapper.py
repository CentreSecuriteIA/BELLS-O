"""Implement the payload mapper for Azure's Analyze Text."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..azure import AzureAnalyzeTextSupervisor


def mapper(
    supervisor: "AzureAnalyzeTextSupervisor",
    prompt: str,
) -> dict[str, str]:
    """Generate a json payload for an API request to the AzureAnalyzeTextSupervisor.

    Maps to {"text": `prompt`, "categories": `supervisor.categories`, "haltOnBlocklistHit": False, "outputType": `supervisor.output_type`}.

    Args:
        prompt: The prompt to evaluate.
        supervisor (AzureAnalyzeTextSupervisor): The AzureAnalyzeTextSupervisor for which to generate the authentication payload.

    Returns:
        The mapped authentication payload.

    """
    json_repr = {
        "text": prompt,
        "categories": supervisor.categories,
        "haltOnBlocklistHit": False,
        "outputType": supervisor.output_type,
    }
    return json_repr
