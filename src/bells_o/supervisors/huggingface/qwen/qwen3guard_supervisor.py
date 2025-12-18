"""Implement the configuration for Qwen/Qwen3Guard-Gen-8B supervisor from HuggingFace."""

from typing import Any, Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing, RoleWrapper
from bells_o.result_mappers import qwen3guard as qwen3guard_result_map

from ..hf_supervisor import HuggingFaceSupervisor


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
        backend: Literal["transformers", "vllm"] = "transformers",
    ):
        """Initialize the supervisor.

        Args:
            variant (str, optional): The model variant to use (e.g., "8B", "4B", "0.6B"). Defaults to "8B".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to []
            model_kwargs (dict[str, Any], optional):  Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs (dict[str, Any], optional):  Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs (dict[str, Any], optional): Keyword arguments to configure generation. Defaults to {}.
            backend (Literal["transformers", "vllm"]): The inference backend to use. Defaults to "transformers".

        """
        pre_processing.append(RoleWrapper("user"))

        super().__init__(
            name=f"Qwen/Qwen3Guard-Gen-{variant}",
            usage=Usage("content_moderation"),
            res_map_fn=qwen3guard_result_map,
            pre_processing=pre_processing,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            generation_kwargs=generation_kwargs,
            provider_name="Qwen",
            backend=backend,
        )
