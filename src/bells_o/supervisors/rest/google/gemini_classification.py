"""Implement the Gemini Classification Supervisor for content moderation."""

from typing import Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import gemini_one as gemini_classification_result_map

from .. import default_prompts
from .gemini import GeminiSupervisor


class GeminiClassificationSupervisor(GeminiSupervisor):
    """Gemini supervisor configured for classification with a system prompt.

    Uses Gemini with a classification system prompt to determine if content is harmful.
    The model outputs "1" for harmful or "0" for benign, which is parsed by the result mapper.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "GEMINI_API_KEY",
        used_for: Literal["input", "output"] = "input",
    ):
        """Initialize the GeminiClassificationSupervisor.

        Args:
            model: Gemini model id. Defaults to "gemini-1.5-pro".
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: Google AI Studio API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "GEMINI_API_KEY".
            used_for (Literal["input", "output"]): The type of strings that are checked. Defaults to "input".

        """
        if used_for == "input":
            system_prompt = default_prompts.DEFAULT_INPUT
        if used_for == "output":
            system_prompt = default_prompts.DEFAULT_OUTPUT

        super().__init__(
            model=model,
            usage=Usage("content_moderation"),
            result_mapper=gemini_classification_result_map,
            system_prompt=system_prompt,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
