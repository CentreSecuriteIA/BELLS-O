"""Implement the authentication mapper for the X-TG method. Employed by NeuralTrust."""

from ..rest_supervisor import RestSupervisor


def mapper(supervisor: RestSupervisor) -> dict[str, str]:
    """Generate an authentication payload for the X-TG method..

    Maps to {"X-TG-API-Key": `API_KEY`}.

    Args:
        supervisor (RestSupervisor): The supervisor for which to generate the authentication payload.

    Returns:
        dict[str, str]: The mapped authentication payload.

    """
    json_repr = {"X-TG-API-Key": supervisor.api_key}
    return json_repr
