"""Generate a json payload for an API request to the HuggingFace Router API."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..huggingface_api import HuggingFaceApiSupervisor


def mapper(
    supervisor: "HuggingFaceApiSupervisor",
    prompt: str | list[dict[str, str]],
) -> dict[str, Any]:
    """Generate a json payload for HuggingFace Router API chat completions.

    Args:
        prompt: The prompt to evaluate (string or message list from RoleWrapper).
        supervisor: The HuggingFaceApiSupervisor for which to generate the request payload.

    Returns:
        The mapped request payload.
    """
    # Handle both string prompts and message lists
    if isinstance(prompt, list):
        # Use message list directly (from RoleWrapper)
        messages = prompt
    else:
        # Convert string to message format
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    # Chat completions format
    json_repr: dict[str, Any] = {
        "model": supervisor.model_id,
        "messages": messages
    }
    
    # Add generation parameters if provided (map to OpenAI-compatible format)
    if supervisor.generation_kwargs:
        # Map common generation kwargs to OpenAI format
        if "max_new_tokens" in supervisor.generation_kwargs:
            json_repr["max_tokens"] = supervisor.generation_kwargs["max_new_tokens"]
        if "temperature" in supervisor.generation_kwargs:
            json_repr["temperature"] = supervisor.generation_kwargs["temperature"]
        if "top_p" in supervisor.generation_kwargs:
            json_repr["top_p"] = supervisor.generation_kwargs["top_p"]
        # Add other parameters that are OpenAI-compatible
        for key in ["stop", "presence_penalty", "frequency_penalty"]:
            if key in supervisor.generation_kwargs:
                json_repr[key] = supervisor.generation_kwargs[key]
    
    return json_repr

