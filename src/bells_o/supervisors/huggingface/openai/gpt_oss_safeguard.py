"""openai/gpt-oss-safeguard-{"20b", "120b"} from HuggingFace."""

from typing import Any, Literal

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing, RoleWrapper
from bells_o.result_mappers import one_map

from ... import default_prompts
from ..hf_supervisor import HuggingFaceSupervisor


class GptOssSafeguardSupervisor(HuggingFaceSupervisor):
    """Implement the configured openai/gpt-oss-safeguard-{"20b", "120b"} supervisor from HuggingFace."""

    def __init__(
        self,
        usage: Usage | None = None,
        policy: str | None = None,
        result_mapper: ResultMapper = one_map,
        variant: Literal["20b", "120b"] = "20b",
        used_for: Literal["input", "output"] = "input",
        pre_processing: list[PreProcessing] = [],
        model_kwargs: dict[str, Any] = {},
        tokenizer_kwargs: dict[str, Any] = {},
        generation_kwargs: dict[str, Any] = {},
        backend: Literal["transformers", "vllm"] = "transformers",
    ):
        """Initialize the supervisor.

        Args:
            usage (Usage): The usage type of the model, which is defined by the system prompt. Defaults to the appropriate usage type for `used_for` if applicable.
            policy (str, optional): A string that describes the policy and taxonomy that the model should classify for. Defaults to the default prompts for `used_for`.
            result_mapper (ResultMapper, optional): A ResultMapper that can map the output format specified in the policy to a `Result` object. Defaults to `bells_o.result_mappers.one_map`
            variant (Literal["20b", "120b"], optional): _description_. Defaults to "20b".
            used_for (Literal["input", "output"]): If `policy` is not set, use this value to determine the default prompt to use. Defaults to "input".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to []
            model_kwargs (dict[str, Any], optional):  Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs (dict[str, Any], optional):  Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs (dict[str, Any], optional): Keyword arguments to configure generation. Defaults to {}.
            used_for (Literal["input", "output"]): If `policy` is not set, use this value to determine the default prompt to use. Defaults to "input".
            backend (Literal["transformers", "vllm"]): The inference backend to use. Defaults to "transformers".

        """
        # TODO: deal with usage properly
        if policy is None:
            if used_for == "input":
                policy = default_prompts.DEFAULT_INPUT
                policy += "\nReasoning:low\n"
                usage = Usage("content_moderation")
            if used_for == "output":
                policy = default_prompts.DEFAULT_OUTPUT
                policy += "\nReasoning:low\n"
                usage = Usage("content_moderation")
            if not result_mapper == one_map:
                print(
                    "WARNING: Overriding set `result_mapper` because `policy` was not set and is falling back to the default prompts."
                )
        else:
            if result_mapper == one_map:
                print(
                    "WARNING: `result_mapper` was not set. Using default `bells_o.result_mappers.one_map` which could be inappropriate for the custom policy."
                )

        if backend == "transformers":
            custom_generation_kwargs = {"max_new_tokens": 2048}
            generation_kwargs |= custom_generation_kwargs
        elif backend == "vllm":
            custom_generation_kwargs = {"max_tokens": 2048}
            generation_kwargs |= custom_generation_kwargs

        pre_processing.append(RoleWrapper("user", system_prompt=policy))

        assert isinstance(usage, Usage), (
            f"Expected `usage` to be of type `Usage` at this point. Got type `{type(usage)}` instead."
        )

        super().__init__(
            name=f"openai/gpt-oss-safeguard-{variant}",
            usage=usage,
            res_map_fn=result_mapper,
            pre_processing=pre_processing,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            generation_kwargs=generation_kwargs,
            provider_name="OpenAI",
            backend=backend,
        )
