"""Generate a json payload for an API request to the OpenAiSupervisor."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..openai import OpenAiSupervisor


def mapper(
    supervisor: "OpenAiSupervisor",
    prompt: str,
) -> dict[str, str]:
    """Generate a json payload for an API request to the OpenAiSupervisor.

    Maps to {"model": `supervisor.name`, "input": `prompt`, "instructions": `supervisor.instructions`}.

    Args:
        prompt: The prompt to evaluate.
        supervisor (OpenAiSupervisor): The OpenAiSupervisor for which to generate the authentication payload.

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
