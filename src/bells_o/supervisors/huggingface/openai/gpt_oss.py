"""openai/gpt-oss-{"20b", "120b"} from HuggingFace."""

from typing import Any, Literal

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing, RoleWrapper
from bells_o.result_mappers import one_map

from ... import default_prompts
from ..hf_supervisor import HuggingFaceSupervisor


class GptOssSupervisor(HuggingFaceSupervisor):
    """Implement the configured openai/gpt-oss-{"20b", "120b"} supervisor from HuggingFace."""

    def __init__(
        self,
        usage: Usage | None = None,
        system_prompt: str | None = None,
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
            system_prompt (str, optional): A string that describes how to classify text.
            result_mapper (ResultMapper, optional): A ResultMapper that can map the output format specified in the policy to a `Result` object. Defaults to `bells_o.result_mappers.one_map`
            variant (Literal["20b", "120b"], optional): _description_. Defaults to "20b".
            used_for (Literal["input", "output"]): If `system_prompt` is not set, use this value to determine the default prompt to use. Defaults to "input".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to []
            model_kwargs (dict[str, Any], optional):  Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs (dict[str, Any], optional):  Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs (dict[str, Any], optional): Keyword arguments to configure generation. Defaults to {}.
            used_for (Literal["input", "output"]): If `system_prompt` is not set, use this value to determine the default prompt to use. Defaults to "input".
            backend (Literal["transformers", "vllm"]): The inference backend to use. Defaults to "transformers".

        """
        # TODO: deal with usage properly
        if system_prompt is None:
            if used_for == "input":
                usage = Usage("content_moderation")
                system_prompt = default_prompts.DEFAULT_INPUT
                system_prompt += "\nReasoning: low\n"
            if used_for == "output":
                usage = Usage("content_moderation")
                system_prompt = default_prompts.DEFAULT_OUTPUT
                system_prompt += "\nReasoning: low\n"
            if not result_mapper == one_map:
                print(
                    "WARNING: Overriding set `result_mapper` because `system_prompt` was not set and is falling back to the default prompts."
                )
        else:
            if result_mapper == one_map:
                print(
                    "WARNING: `result_mapper` was not set. Using default `bells_o.result_mappers.one_map` which could be inappropriate for the custom system prompt."
                )

        pre_processing.append(RoleWrapper("user", system_prompt=system_prompt))

        assert isinstance(usage, Usage), (
            f"Expected `usage` to be of type `Usage` at this point. Got type `{type(usage)}` instead."
        )

        super().__init__(
            name=f"openai/gpt-oss-{variant}",
            usage=usage,
            res_map_fn=result_mapper,
            pre_processing=pre_processing,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            generation_kwargs=generation_kwargs,
            provider_name="OpenAI",
            backend=backend,
        )
