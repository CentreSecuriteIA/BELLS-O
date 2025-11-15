"""Implement the payload mapper for LakeraGuard."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..lakeraguard import LakeraGuardSupervisor


def mapper(
    supervisor: "LakeraGuardSupervisor",
    prompt: str,
) -> dict[str, str]:
    """Generate a json payload for an API request to the LakeraGuardSupervisor.

    Maps to {"messages": [{"content": `prompt`, "role": "user"}], "project_id": `project-id`}.

    Args:
        prompt: The prompt to evaluate.
        supervisor (LakeraGuardSupervisor): The LakeraGuardSupervisor for which to generate the authentication payload.

    Returns:
        The mapped authentication payload.

    """
    json_repr = {
        "messages": [{"content": prompt, "role": "user"}],
        "project_id": f"{supervisor.project_id}",
    }
    return json_repr
