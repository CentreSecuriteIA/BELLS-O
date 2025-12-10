"""Implement the authentication mapper for Anthropic API."""

from ..custom_endpoint import RestSupervisor


def mapper(supervisor: RestSupervisor) -> dict[str, str]:
    """Generate an authentication payload for Anthropic API.

    Maps to {"x-api-key": `API_KEY`, "anthropic-version": "2023-06-01"}.

    Args:
        supervisor (RestSupervisor): The supervisor for which to generate the authentication payload.

    Returns:
        dict[str, str]: The mapped authentication payload.

    """
    json_repr = {
        "x-api-key": supervisor.api_key,
        "anthropic-version": "2023-06-01",
    }
    return json_repr

