"""Implement the OpenAI Classification Supervisor for content moderation."""

from typing import Any, Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing

from ..auth_mappers import x_tg_api
from ..rest_supervisor import RestSupervisor


class NeuralTrustTrustGateSupervisor(RestSupervisor):
    """OpenAI supervisor configured for classification with a system prompt.

    Uses OpenAI with a classification system prompt to determine if content is harmful.
    The model outputs "1" for harmful or "0" for benign, which is parsed by the result mapper.
    """

    def __init__(
        self,
        policy_id: str,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "NEURALTRUST_API_KEY",
        base_url="https://actions.neuraltrust.ai/v1/actions",
        used_for: Literal["jailbreak"] = "jailbreak",
    ):
        """Initialize the OpenAIClassificationSupervisor.

        Args:
            model: OpenAI model id. Defaults to "gpt-5-nano-2025-08-07".
            max_tokens: Maximum completion tokens (includes reasoning + output for GPT-5).
                Defaults to 200 to account for reasoning tokens in GPT-5 models.
            reasoning_effort: Reasoning effort level for GPT-5 models ("low", "medium", "high").
                Defaults to "low" to minimize reasoning tokens and ensure output tokens are available.
            text_verbosity: Text verbosity level for GPT-5 models ("low", "medium", "high").
                Defaults to "low" to keep responses concise.
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: OpenAI API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "OPENAI_API_KEY".
            used_for (Literal["input", "output"]): The type of strings that are checked. Defaults to "input".

        """
        usage = None
        if used_for == "jailbreak":
            from bells_o.result_mappers import neuraltrust as neuraltrust_res

            from ..request_mappers import neuraltrust as neuraltrust_req

            usage = Usage("jailbreak")
            result_mapper = neuraltrust_res
            request_mapper = neuraltrust_req

        self.policy_id = policy_id

        assert isinstance(usage, Usage)
        super().__init__(
            name="TrustGate Prompt Guard",
            usage=usage,
            res_map_fn=result_mapper,
            req_map_fn=request_mapper,
            auth_map_fn=x_tg_api,
            base_url=base_url,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
            provider_name="NeuralTrust",
        )

    @classmethod
    def _get_token_counts(cls, output_raw: dict[str, Any]) -> dict[str, Any]:
        try:
            input_tokens = output_raw["metadata"][0]["data"]["input_length"]
            output_tokens = 1
        except KeyError as e:
            print("DEBUGGING: output_raw dict:")
            print(output_raw)
            raise KeyError from e

        return {"input_tokens": input_tokens, "output_tokens": output_tokens}
