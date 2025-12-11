"""Implement the configuration for Qwen/Qwen3Guard-Gen-8B supervisor from HuggingFace."""

from time import time
from typing import Any

import torch

from bells_o.common import OutputDict, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing, RoleWrapper
from bells_o.result_mappers import qwen3guard as qwen3guard_result_map

from ..custom_model import HuggingFaceSupervisor
from transformers import BatchEncoding


class Qwen3GuardSupervisor(HuggingFaceSupervisor):
    """Implement the pre-configured Qwen/Qwen3Guard-Gen-8B supervisor from HuggingFace.
    
    Qwen3Guard is a content moderation model that outputs safety labels (Safe, Unsafe, or Controversial)
    along with categories of detected violations.
    """

    def __init__(
        self,
        variant: str = "8B",
        pre_processing: list[PreProcessing] = [],
        model_kwargs: dict[str, Any] = {},
        tokenizer_kwargs: dict[str, Any] = {},
        generation_kwargs: dict[str, Any] = {},
    ):
        """Initialize the supervisor.

        Args:
            variant (str, optional): The model variant to use (e.g., "8B", "4B", "0.6B"). Defaults to "8B".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to []
            model_kwargs (dict[str, Any], optional):  Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs (dict[str, Any], optional):  Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs (dict[str, Any], optional): Keyword arguments to configure generation. Defaults to {}.

        """
        self.name: str = f"Qwen/Qwen3Guard-Gen-{variant}"
        self.usage: Usage = Usage("content_moderation")
        self.res_map_fn: ResultMapper = qwen3guard_result_map
        pre_processing.append(RoleWrapper("user"))
        self.pre_processing = pre_processing
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.generation_kwargs = generation_kwargs
        super().__post_init__()

