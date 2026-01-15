"""Implement the X-AI Classification Supervisor for content moderation."""

from typing import Any, Literal

from bells_o.common import Result, Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import openai_compatible_one as text_classification_result_map

from .. import default_prompts
from .xai import XAiSupervisor


def xai_map(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Handle XAI specific failure mode. Otherwise identical to OpenAI compatible One mapper."""
    try:
        return text_classification_result_map(output, usage)
    except KeyError:  # XAI API has safety measures that change the response body
        flag = "Content violates usage guidelines" in output["error"]
        result = Result()
        for usage_type in usage:
            result[usage_type] = flag
        return result


class XAiClassificationSupervisor(XAiSupervisor):
    """X-AI supervisor configured for classification with a system prompt.

    Uses X-AI (Grok) with a classification system prompt to determine if content is harmful.
    The model outputs "1" for harmful or "0" for benign, which is parsed by the result mapper.
    """

    def __init__(
        self,
        model: str = "grok-4-latest",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "XAI_API_KEY",
        used_for: Literal["input", "output"] = "input",
    ):
        """Initialize the XAiClassificationSupervisor.

        Args:
            model: X-AI model id. Defaults to "grok-4-latest".
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: X-AI API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "XAI_API_KEY".
            used_for (Literal["input", "output"]): The type of strings that are checked. Defaults to "input".

        """
        if used_for == "input":
            system_prompt = default_prompts.DEFAULT_INPUT
        if used_for == "output":
            system_prompt = default_prompts.DEFAULT_OUTPUT

        super().__init__(
            model=model,
            usage=Usage("content_moderation"),
            result_mapper=xai_map,
            system_prompt=system_prompt,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
