"""Implement the payload mapper for LakeraGuard."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..openai import OpenAiSupervisor


def mapper(
    supervisor: "OpenAiSupervisor",
    prompt: str,
) -> dict[str, str]:
    """Generate an authentication payload for the bearer method.

    Maps to {"messages": [{"content": prompt, "role": "user"}],"project_id": project-id},

    Args:
        prompt: The prompt to evaluate.
        supervisor (LakeraGuardSupervisor): The LakeraGuardSupervisor for which to generate the authentication payload.

    Returns:
        The mapped authentication payload.

    """
    json_repr = {
        "model": f"{supervisor.name}",
        "input": prompt,
    }

    if supervisor.system_prompt:
        json_repr["instructions"] = supervisor.system_prompt

    return json_repr
