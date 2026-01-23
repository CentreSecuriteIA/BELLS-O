"""Implement the GPT OSS Safeguard 20 Supervisor for content moderation via OpenRouter."""

from typing import Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import openai_compatible_one as text_classification_result_map

from ... import default_prompts
from .openrouter import OpenRouterSupervisor


class GptOssSafeguard20Supervisor(OpenRouterSupervisor):
    """OpenRouter supervisor for GPT OSS Safeguard 20 model configured for classification.

    Uses OpenRouter's GPT OSS Safeguard 20 model (openai/gpt-oss-safeguard-20b) with a classification
    system prompt to determine if content is harmful. The model outputs "1" for harmful or "0" for benign,
    which is parsed by the result mapper.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-safeguard-20b",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "OPENROUTER_API_KEY",
        reasoning_enabled: bool = True,
        used_for: Literal["input", "output"] = "input",
    ):
        """Initialize the GptOssSafeguard20Supervisor.

        Args:
            model: OpenRouter model id. Defaults to "openai/gpt-oss-safeguard-20b".
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: OpenRouter API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "OPENROUTER_API_KEY".
            reasoning_enabled: Whether to enable reasoning for the model. Defaults to True.
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
            reasoning_enabled=reasoning_enabled,
        )
