"""Generate a json payload for an API request to the GeminiSupervisor."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..google.gemini import GeminiSupervisor


def mapper(
    supervisor: "GeminiSupervisor",
    prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a json payload for an API request to the GeminiSupervisor.

    Maps to Google AI Studio Gemini generateContent format with contents, system_instruction,
    and safetySettings.

    Args:
        supervisor: The GeminiSupervisor for which to generate the request payload.
        prompt: The prompt to evaluate.
        **kwargs: Additional arguments (e.g., generation_config).

    Returns:
        The mapped request payload as a dictionary.
    """
    # Google AI Studio API format - no "role" field needed
    contents = [
        {
            "parts": [{"text": prompt}],
        }
    ]

    body: dict[str, Any] = {"contents": contents}

    # System instruction (Gemini's "system_prompt" equivalent)
    if getattr(supervisor, "system_prompt", ""):
        body["system_instruction"] = {
            "parts": [{"text": supervisor.system_prompt}]
        }

    # Safety settings (the "system settings" you want to evaluate)
    safety_settings = getattr(supervisor, "safety_settings", None)
    if safety_settings:
        body["safetySettings"] = safety_settings

    # You can also add generationConfig if needed (maxOutputTokens, etc.)
    gen_config = kwargs.get("generation_config")
    if gen_config:
        body["generationConfig"] = gen_config

    return body

