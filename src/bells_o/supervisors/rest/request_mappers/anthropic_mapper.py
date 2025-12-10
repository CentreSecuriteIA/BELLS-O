"""Generate a json payload for an API request to the AnthropicSupervisor."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..anthropic import AnthropicSupervisor


def mapper(
    supervisor: "AnthropicSupervisor",
    prompt: str,
) -> dict[str, str]:
    """Generate a json payload for an API request to the AnthropicSupervisor.

    Maps to Anthropic API messages format with model, max_tokens, and messages.

    Args:
        prompt: The prompt to evaluate.
        supervisor (AnthropicSupervisor): The AnthropicSupervisor for which to generate the request payload.

    Returns:
        The mapped request payload.

    """
    messages = [{"role": "user", "content": prompt}]
    
    json_repr: dict = {
        "model": supervisor.name,
        "max_tokens": getattr(supervisor, "max_tokens", 1024),
        "messages": messages,
    }

    # Add system prompt if provided
    if getattr(supervisor, "system_prompt", ""):
        json_repr["system"] = supervisor.system_prompt

    return json_repr

