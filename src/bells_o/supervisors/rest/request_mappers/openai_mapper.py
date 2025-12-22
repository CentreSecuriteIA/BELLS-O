"""Generate a json payload for an API request to the OpenAiSupervisor."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..openai import OpenAiSupervisor


def mapper(
    supervisor: "OpenAiSupervisor",
    prompt: str,
) -> dict[str, Any]:
    """Generate a json payload for an API request to the OpenAiSupervisor.

    Supports both moderation API format (input/instructions) and chat completions format (messages).
    Determines format based on the base_url.

    Args:
        prompt: The prompt to evaluate.
        supervisor (OpenAiSupervisor): The OpenAiSupervisor for which to generate the request payload.

    Returns:
        The mapped request payload.

    """
    base_url = getattr(supervisor, "base_url", "")
    
    # Check if this is a moderation endpoint
    if "/moderations" in base_url:
        # Moderation API format
        json_repr: dict[str, Any] = {
            "model": supervisor.name,
            "input": prompt,
        }
        if supervisor.system_prompt:
            json_repr["instructions"] = supervisor.system_prompt
    else:
        # Chat completions API format - all models use "messages" array
        messages = [{"role": "user", "content": prompt}]
        json_repr: dict[str, Any] = {
            "model": supervisor.name,
            "messages": messages,
        }
        
        # Add system prompt if provided
        system_prompt = getattr(supervisor, "system_prompt", "")
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Add max_tokens or max_completion_tokens
        max_tokens = getattr(supervisor, "max_tokens", None)
        model_name = getattr(supervisor, "name", "")
        if max_tokens is not None:
            # Use "max_completion_tokens" for reasoning models (GPT-5, o-series)
            # Use "max_tokens" for other models
            if "gpt-5" in model_name or model_name.startswith("o"):
                json_repr["max_completion_tokens"] = max_tokens
            else:
                json_repr["max_tokens"] = max_tokens
        
        # Add reasoning_effort for reasoning models (top-level parameter)
        if "gpt-5" in model_name or model_name.startswith("o"):
            reasoning_effort = getattr(supervisor, "reasoning_effort", None)
            if reasoning_effort is not None:
                json_repr["reasoning_effort"] = reasoning_effort
        
        # Add verbosity for all models (top-level parameter)
        verbosity = getattr(supervisor, "text_verbosity", None)
        if verbosity is not None:
            json_repr["verbosity"] = verbosity

    return json_repr
