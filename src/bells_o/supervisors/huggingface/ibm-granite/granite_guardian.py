"""Implement the pre-configured IBM Granite Guardian supervisors from HuggingFace."""

from typing import Any, Literal

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing, RoleWrapper

from ..hf_supervisor import HuggingFaceSupervisor


def _get_result_mapper(model_id: str) -> ResultMapper:
    if "3.3" in model_id:
        from bells_o.result_mappers import granite_33 as result_mapper
    else:
        from bells_o.result_mappers import yes_mapper as result_mapper

    return result_mapper


class GraniteGuardianSupervisor(HuggingFaceSupervisor):
    """Implement the pre-configured google/shieldgemma-27b supervisor from HuggingFace.

    ShieldGemma is a safety content moderation model that outputs "Yes" or "No"
    to indicate whether content violates safety policies.
    """

    def __init__(
        self,
        model_id: str = "ibm-granite/granite-guardian-3.3-8b",
        criteria: Literal[
            "harm",
            "social_bias",
            "profanity",
            "sexual_content",
            "unethical_behavior",
            "violence",
            "harm_engagement",
            "evasiveness",
            "jailbreal",
            "groundedness",
            "relevance",
            "answer_relevance",
            "function_call",
        ]
        | None = None,
        think: bool = True,
        pre_processing: list[PreProcessing] = [],
        model_kwargs: dict[str, Any] = {},
        tokenizer_kwargs: dict[str, Any] = {},
        generation_kwargs: dict[str, Any] = {},
        backend: Literal["transformers", "vllm"] = "transformers",
    ):
        """Initialize the supervisor.

        Args:
            model_id (str, optional): The id of the exact model to use. This class supports different GraniteGuardian models.
                Defaults to "ibm-granite/granite-guardian-3.3-8b".
            criteria (Literal["harm", "social_bias", "profanity", "sexual_content", "unethical_behavior", "violence", "harm_engagement", "evasiveness", "jailbreal", "groundedness", "relevance", "answer_relevance", "function_call"], optional):
                The classification criteria to use. See https://www.ibm.com/granite/docs/models/guardian for more info. Availability might differ for different models.
            think (bool, optional): If the supervisor should utilize thinking. Only available in "ibm-granite/granite-guardian-3.3-8b". Defaults to True.
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            model_kwargs: Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs: Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs: Keyword arguments to configure generation. Defaults to {}.
            backend (Literal["transformers", "vllm"]): The inference backend to use. Defaults to "transformers".

        """
        pre_processing.append(RoleWrapper("user"))

        self._supported_backends = ["transformers", "vllm"]
        if think and not model_id == "ibm-granite/granite-guardian-3.3-8b":
            self.think = False
        else:
            self.think = think

        self.criteria = criteria

        super().__init__(
            name=model_id,
            usage=Usage("content_moderation"),
            res_map_fn=_get_result_mapper(model_id),
            pre_processing=pre_processing,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            generation_kwargs=generation_kwargs,
            provider_name="Google",
            backend=backend,
        )

    def _apply_chat_template(self, inputs: list[str]) -> list[str]:
        """Identical to super()._apply_chat_template(), just that we also pass the guardian_config parameter here."""
        if getattr(self._tokenizer, "chat_template", None) is not None:
            assert isinstance(inputs, list), (
                "If `tokenizer.chat_template` is not None, then use a `RoleWrapper` as the last pre-processor."
            )

            # gotta deal with think kwarg differently because not all tokenizers support it
            think_kwarg = {}
            if self.think:
                think_kwarg["think"] = True

            inputs = self._tokenizer.apply_chat_template(
                inputs,  # type: ignore
                tokenize=False,
                guardian_config={"criteria_id": self.criteria},
                add_generation_prompt=True,
                **think_kwarg,
            )  # TODO customize the kwargs of apply_chat_template?
        return inputs
