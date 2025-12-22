"""Implement the authentication mapper for the bearer method."""

from ..rest_supervisor import RestSupervisor


def mapper(supervisor: RestSupervisor) -> dict[str, str]:
    """Generate an authentication payload for the bearer method.

    Maps to {"Authorization": f"Bearer {`API_KEY`}"}.

    Args:
        supervisor (RestSupervisor): The supervisor for which to generate the authentication payload.

    Returns:
        dict[str, str]: The mapped authentication payload.

    """
    json_repr = {"Authorization": f"Bearer {supervisor.api_key}"}
    return json_repr
