"""Implement the pre-configured saillab/x-guard supervisor from HuggingFace."""

from typing import Any

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing, RoleWrapper
from bells_o.resultmappers import xguard as xguard_result_map

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
        self.name: str = "saillab/x-guard"
        self.usage: Usage = Usage("content_moderation")
        self.res_map_fn: ResultMapper = xguard_result_map
        pre_processing.append(RoleWrapper("user", opposite_prompt="\n <think>"))
        self.pre_processing = pre_processing
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.generation_kwargs = generation_kwargs
        super().__post_init__()
