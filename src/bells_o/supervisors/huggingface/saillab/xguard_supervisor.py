"""Implement the pre-configured saillab/x-guard supervisor from HuggingFace."""

from typing import Any

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing, RoleWrapper
from bells_o.result_mappers import xguard as xguard_result_map

from ..custom_model import HuggingFaceSupervisor


class XGuardSupervisor(HuggingFaceSupervisor):
    """Implement the pre-configured saillab/x-guard supervisor from HuggingFace."""

    def __init__(
        self,
        pre_processing: list[PreProcessing] = [],
        model_kwargs: dict[str, Any] = {},
        tokenizer_kwargs: dict[str, Any] = {},
        generation_kwargs: dict[str, Any] = {},
    ):
        """Initialize the supervisor.

        Args:
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to []
            model_kwargs (dict[str, Any], optional):  Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs (dict[str, Any], optional):  Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs (dict[str, Any], optional): Keyword arguments to configure generation. Defaults to {}.

        """
        self.name: str = "saillab/x-guard"
        self.usage: Usage = Usage("content_moderation")
        self.res_map_fn: ResultMapper = xguard_result_map
        pre_processing.append(RoleWrapper("user", opposite_prompt="\n <think>"))
        self.pre_processing = pre_processing
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        base_kwargs = {"temperature": 0.0000001, "do_sample": True}
        for key in base_kwargs:
            if key in generation_kwargs:
                print(f"INFO: ignoring set generation kwarg {key}. This kwarg should not be set for this supervisor.")
        self.generation_kwargs = generation_kwargs | base_kwargs
        super().__post_init__()
