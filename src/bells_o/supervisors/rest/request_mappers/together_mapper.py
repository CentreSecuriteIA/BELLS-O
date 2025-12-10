"""Generate a json payload for an API request to the TogetherAISupervisor."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..together import TogetherAISupervisor


def mapper(
    supervisor: "TogetherAISupervisor",
    prompt: str,
) -> dict[str, Any]:
    """Generate a json payload for an API request to the TogetherAISupervisor.

    Maps to Together AI API chat completions format with model and messages.

    Args:
        prompt: The prompt to evaluate.
        supervisor (TogetherAISupervisor): The TogetherAISupervisor for which to generate the request payload.

    Returns:
        The mapped request payload.

    """
    messages = [{"role": "user", "content": prompt}]
    
    json_repr: dict[str, Any] = {
        "model": supervisor.name,
        "messages": messages,
    }

    # Add system prompt if provided
    system_prompt = getattr(supervisor, "system_prompt", "")
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    return json_repr

