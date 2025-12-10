"""Implement the authentication mapper for AWS (placeholder - uses boto3 for actual auth)."""

from ..custom_endpoint import RestSupervisor


def mapper(supervisor: RestSupervisor) -> dict[str, str]:
    """Generate an authentication payload for AWS (placeholder).

    Note: AWS authentication is handled by boto3 using AWS credentials (environment variables,
    IAM roles, or credentials file). This mapper is a placeholder and won't be used when
    boto3 is used for API calls.

    Args:
        supervisor (RestSupervisor): The supervisor for which to generate the authentication payload.

    Returns:
        dict[str, str]: Empty dict as boto3 handles authentication.

    """
    # AWS authentication is handled by boto3, not via headers
    return {}

