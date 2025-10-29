"""Implement the payload mapper for LakeraGuard."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..lakeraguard import LakeraGuardSupervisor


def mapper(prompt: str, supervisor: "LakeraGuardSupervisor") -> dict[str, str]:
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
