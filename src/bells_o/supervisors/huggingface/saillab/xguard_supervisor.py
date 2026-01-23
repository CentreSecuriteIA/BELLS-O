"""Implement the pre-configured saillab/x-guard supervisor from HuggingFace."""

from typing import Any, Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing, RoleWrapper, TemplateWrapper
from bells_o.result_mappers import xguard as xguard_result_map

from ..hf_supervisor import HuggingFaceSupervisor


inference_format = "<USER TEXT STARTS>\n{prompt}\n<USER TEXT ENDS>"


class XGuardSupervisor(HuggingFaceSupervisor):
    """Implement the pre-configured saillab/x-guard supervisor from HuggingFace."""

    def __init__(
        self,
        pre_processing: list[PreProcessing] = [],
        model_kwargs: dict[str, Any] = {},
        tokenizer_kwargs: dict[str, Any] = {},
        generation_kwargs: dict[str, Any] = {},
        backend: Literal["transformers", "vllm"] = "transformers",
    ):
        """Initialize the supervisor.

        Args:
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to []
            model_kwargs (dict[str, Any], optional):  Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs (dict[str, Any], optional):  Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs (dict[str, Any], optional): Keyword arguments to configure generation. Defaults to {}.
            backend (Literal["transformers", "vllm"]): The inference backend to use. Defaults to "transformers".

        """
        pre_processing.append(TemplateWrapper(inference_format))
        pre_processing.append(RoleWrapper("user", opposite_prompt="\n <think>"))

        if backend == "transformers":
            custom_generation_kwargs = {"temperature": 0.0000001, "do_sample": True, "max_new_tokens": 512}
            generation_kwargs |= custom_generation_kwargs
        elif backend == "vllm":
            custom_generation_kwargs = {"temperature": 0.01, "max_tokens": 512}
            generation_kwargs |= custom_generation_kwargs

        super().__init__(
            name="saillab/x-guard",
            usage=Usage("content_moderation"),
            res_map_fn=xguard_result_map,
            pre_processing=pre_processing,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            generation_kwargs=generation_kwargs,
            provider_name="SAIL Lab",
            backend=backend,
        )
