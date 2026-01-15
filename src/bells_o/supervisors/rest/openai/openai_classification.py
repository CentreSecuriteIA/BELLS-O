"""Implement the OpenAI Classification Supervisor for content moderation."""

from typing import Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import openai_compatible_one as text_classification_result_map

from .. import default_prompts
from .openai import OpenAiSupervisor


class OpenAIClassificationSupervisor(OpenAiSupervisor):
    """OpenAI supervisor configured for classification with a system prompt.

    Uses OpenAI with a classification system prompt to determine if content is harmful.
    The model outputs "1" for harmful or "0" for benign, which is parsed by the result mapper.
    """

    def __init__(
        self,
        model: str = "gpt-5-nano-2025-08-07",
        max_tokens: int = 500,
        reasoning_effort: str = "low",
        text_verbosity: str = "low",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "OPENAI_API_KEY",
        used_for: Literal["input", "output"] = "input",
    ):
        """Initialize the OpenAIClassificationSupervisor.

        Args:
            model: OpenAI model id. Defaults to "gpt-5-nano-2025-08-07".
            max_tokens: Maximum completion tokens (includes reasoning + output for GPT-5).
                Defaults to 200 to account for reasoning tokens in GPT-5 models.
            reasoning_effort: Reasoning effort level for GPT-5 models ("low", "medium", "high").
                Defaults to "low" to minimize reasoning tokens and ensure output tokens are available.
            text_verbosity: Text verbosity level for GPT-5 models ("low", "medium", "high").
                Defaults to "low" to keep responses concise.
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: OpenAI API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "OPENAI_API_KEY".
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
            base_url="https://api.openai.com/v1/chat/completions",
            system_prompt=system_prompt,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
        # Store max_tokens, reasoning_effort, and text_verbosity for potential use in request mapper
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
