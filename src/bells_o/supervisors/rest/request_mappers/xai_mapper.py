"""Generate a json payload for an API request to the XAiSupervisor."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..xai import XAiSupervisor


def mapper(
    supervisor: "XAiSupervisor",
    prompt: str,
) -> dict[str, Any]:
    """Generate a json payload for an API request to the XAiSupervisor.

    Maps to X-AI API chat completions format with model and messages.

    Args:
        prompt: The prompt to evaluate.
        supervisor (XAiSupervisor): The XAiSupervisor for which to generate the request payload.

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
        # X-AI API supports system role in messages
        messages.insert(0, {"role": "system", "content": supervisor.system_prompt})

    return json_repr
