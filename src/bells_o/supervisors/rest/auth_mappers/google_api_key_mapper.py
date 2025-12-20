"""Implement the authentication mapper for Google AI Studio API key."""

from ..rest_supervisor import RestSupervisor


def mapper(supervisor: RestSupervisor) -> dict[str, str]:
    """Generate an authentication payload for Google AI Studio API.

    Maps to {"x-goog-api-key": f"{`API_KEY`}"}.

    Args:
        supervisor (RestSupervisor): The supervisor for which to generate the authentication payload.

    Returns:
        dict[str, str]: The mapped authentication payload with x-goog-api-key header.

    """
    api_key = supervisor.api_key
    if not api_key:
        # Debug: check if API key is missing
        print(f"WARNING: API key is empty. api_variable={supervisor.api_variable}")
    # Add default Google API "Content-Type" header as well (see @custom_endpoint.py 94-95)
    json_repr = {"x-goog-api-key": f"{api_key}"}
    return json_repr
