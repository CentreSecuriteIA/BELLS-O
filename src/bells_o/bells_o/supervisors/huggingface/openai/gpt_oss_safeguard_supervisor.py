"""Implement the pre-configured saillab/x-guard supervisor from HuggingFace."""

from typing import Any, Literal

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing, RoleWrapper

from ..custom_model import HuggingFaceSupervisor


class GptOssSafeguardSupervisor(HuggingFaceSupervisor):
    """Implement the configured openai/gpt-oss-safeguard-{"20b", "120b"} supervisor from HuggingFace."""

    def __init__(
        self,
        policy: str,
        usage: Usage,
        result_mapper: ResultMapper,
        variant: Literal["20b", "120b"] = "20b",
        pre_processing: list[PreProcessing] = [],
        model_kwargs: dict[str, Any] = {},
        tokenizer_kwargs: dict[str, Any] = {},
        generation_kwargs: dict[str, Any] = {},
    ):
        """Initialize the supervisor.

        Args:
            policy (str): A string that describes the policy and taxonomy that the model should classify for.
            usage (Usage): The usage type of the model, which is defined by the policy.
            result_mapper (ResultMapper): A ResultMapper that can map the output format specified in the policy to a `Result` object.
            variant (Literal["20b", "120b"], optional): _description_. Defaults to "20b".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to []
            model_kwargs (dict[str, Any], optional):  Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs (dict[str, Any], optional):  Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs (dict[str, Any], optional): Keyword arguments to configure generation. Defaults to {}.

        """
        self.name: str = f"openai/gpt-oss-safeguard-{variant}"
        self.usage = usage
        self.res_map_fn = result_mapper
        pre_processing.append(RoleWrapper("user", system_prompt=policy))
        self.pre_processing = pre_processing
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.generation_kwargs = generation_kwargs
        super().__post_init__()
