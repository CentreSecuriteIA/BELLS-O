"""Implement the Mistral Classification Supervisor for content moderation."""

from typing import Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import openai_compatible_one as text_classification_result_map

from ... import default_prompts
from .mistral import MistralSupervisor


class MistralClassificationSupervisor(MistralSupervisor):
    """Mistral supervisor configured for classification with a system prompt.

    Uses Mistral with a classification system prompt to determine if content is harmful.
    The model outputs "1" for harmful or "0" for benign, which is parsed by the result mapper.
    """

    def __init__(
        self,
        model: str = "ministral-3b-2512",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "MISTRAL_API_KEY",
        used_for: Literal["input", "output"] = "input",
    ):
        """Initialize the MistralClassificationSupervisor.

        Args:
            model: Mistral model id. Defaults to "mistral-large-latest".
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: Mistral API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "MISTRAL_API_KEY".
            used_for (Literal["input", "output"]): The type of strings that are checked. Defaults to "input".

        """
        if used_for == "input":
            system_prompt = default_prompts.DEFAULT_INPUT
        if used_for == "output":
            system_prompt = default_prompts.DEFAULT_OUTPUT

        super().__init__(
            model=model,
            usage=Usage("content_moderation"),
            result_mapper=text_classification_result_map,
            system_prompt=system_prompt,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
