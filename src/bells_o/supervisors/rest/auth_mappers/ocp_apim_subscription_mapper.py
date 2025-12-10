"""Implement the authentication mapper for the `Ocp-Apim-Subscription-Key` header."""

from ..custom_endpoint import RestSupervisor


def mapper(supervisor: RestSupervisor) -> dict[str, str]:
    """Generate an authentication payload for the `Ocp-Apim-Subscription-Key` header commonly used by Azure.

    Maps to {"Ocp-Apim-Subscription-Key": f"{`API_KEY`}"}.

    Args:
        supervisor (RestSupervisor): The supervisor for which to generate the authentication payload.

    Returns:
        dict[str, str]: The mapped authentication payload.

    """
    json_repr = {"Ocp-Apim-Subscription-Key": f"{supervisor.api_key}"}
    return json_repr
