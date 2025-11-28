"""Generate a json payload for an API request to the MistralSupervisor."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..mistral import MistralSupervisor


def mapper(
    supervisor: "MistralSupervisor",
    prompt: str,
) -> dict[str, str]:
    """Generate a json payload for an API request to the MistralSupervisor.

    Maps to Mistral API chat completions format with model and messages.

    Args:
        prompt: The prompt to evaluate.
        supervisor (MistralSupervisor): The MistralSupervisor for which to generate the request payload.

    Returns:
        The mapped request payload.

    """
    messages = [{"role": "user", "content": prompt}]
    
    json_repr: dict[str, str | list] = {
        "model": supervisor.name,
        "messages": messages,
    }

    # Add system prompt if provided
    if getattr(supervisor, "system_prompt", ""):
        # Mistral API supports system role in messages
        messages.insert(0, {"role": "system", "content": supervisor.system_prompt})

    return json_repr

