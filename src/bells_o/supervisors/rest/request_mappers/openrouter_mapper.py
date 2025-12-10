"""Generate a json payload for an API request to the OpenRouterSupervisor."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..openrouter import OpenRouterSupervisor


def mapper(
    supervisor: "OpenRouterSupervisor",
    prompt: str,
) -> dict[str, Any]:
    """Generate a json payload for an API request to the OpenRouterSupervisor.

    Maps to OpenRouter API chat completions format with model, messages, and reasoning.

    Args:
        prompt: The prompt to evaluate.
        supervisor (OpenRouterSupervisor): The OpenRouterSupervisor for which to generate the request payload.

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

    # Add reasoning parameter if enabled
    reasoning_enabled = getattr(supervisor, "reasoning_enabled", False)
    if reasoning_enabled:
        json_repr["reasoning"] = {"enabled": True}

    return json_repr

