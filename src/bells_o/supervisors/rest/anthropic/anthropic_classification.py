"""Implement the Anthropic Classification Supervisor for content moderation."""

from typing import Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import anthropic_one as anthropic_classification_result_map

from .. import default_prompts
from .anthropic import AnthropicSupervisor


class AnthropicClassificationSupervisor(AnthropicSupervisor):
    """Anthropic supervisor configured for classification with a system prompt.

    Uses Anthropic (Claude) with a classification system prompt to determine if content is harmful.
    The model outputs "1" for harmful or "0" for benign, which is parsed by the result mapper.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 10,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "ANTHROPIC_API_KEY",
        used_for: Literal["input", "output"] = "input",
    ):
        """Initialize the AnthropicClassificationSupervisor.

        Args:
            model: Anthropic model id. Defaults to "claude-3-5-sonnet-20241022".
            max_tokens: Maximum tokens to generate. Defaults to 10 (just need "1" or "0").
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: Anthropic API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "ANTHROPIC_API_KEY".
            used_for (Literal["input", "output"]): The type of strings that are checked. Defaults to "input".

        """
        if used_for == "input":
            system_prompt = default_prompts.DEFAULT_INPUT
        if used_for == "output":
            system_prompt = default_prompts.DEFAULT_OUTPUT

        super().__init__(
            model=model,
            usage=Usage("content_moderation"),
            result_mapper=anthropic_classification_result_map,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
