"""Implement the payload mapper for LakeraGuard."""

from ..custom_endpoint import RestSupervisor


def mapper(prompt: str, supervisor: RestSupervisor) -> dict[str, str]:
    """Generate an authentication payload for the bearer method.

    Maps to {"messages": [{"content": prompt, "role": "user"}],"project_id": project-id},

    Args:
        prompt: The prompt to evaluate.
        supervisor (LakeraGuardSupervisor): The LakeraGuardSupervisor for which to generate the authentication payload.

    Returns:
        dict[str, str]: The mapped authentication payload.

    """
    json_repr = {
        "messages": [{"content": prompt, "role": "user"}],
        "project_id": f"{supervisor.project_id}",
    }
    return json_repr
