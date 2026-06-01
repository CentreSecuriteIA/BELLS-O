"""Implement the payload mapper for Azure's Prompt Shield."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..azure import AzurePromptShieldSupervisor


def mapper(
    supervisor: "AzurePromptShieldSupervisor",
    prompt: str,
) -> dict[str, str]:
    """Generate a json payload for an API request to the AzurePromptShieldSupervisor.

    Maps to {"userPrompt": `prompt`}.

    Args:
        prompt: The prompt to evaluate.
        supervisor (AzureAnalyzeTextSupervisor): The AzureAnalyzeTextSupervisor for which to generate the authentication payload.

    Returns:
        The mapped authentication payload.

    """
    json_repr = {
        "userPrompt": prompt,
    }
    return json_repr
