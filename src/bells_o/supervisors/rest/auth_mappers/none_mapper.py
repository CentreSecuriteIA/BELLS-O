"""Implement a no-op authentication mapper for unauthenticated endpoints."""

from ..rest_supervisor import RestSupervisor


def mapper(supervisor: RestSupervisor) -> dict[str, str]:
    """Generate an empty authentication payload for endpoints that need no authentication.

    Maps to {}.

    Args:
        supervisor (RestSupervisor): The supervisor for which to generate the authentication payload.

    Returns:
        dict[str, str]: An empty header payload.

    """
    return {}
