"""Implementation of NeuralTrustTrustGateSupervisor."""

from typing import Any, Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing

from ..auth_mappers import x_tg_api
from ..rest_supervisor import RestSupervisor


class NeuralTrustTrustGateSupervisor(RestSupervisor):
    """Implementation of NeuralTrustTrustGateSupervisor."""

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
            policy_id: The id of the created policy, which should be used for classification.
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: NeuralTrust API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "NEURALTRUST_API_KEY".
            base_url: The base url of the REST endpoint to target. Can differ for self-hosted applications. Defaults to "https://actions.neuraltrust.ai/v1/actions".
            used_for (Literal["jailbreak"]): The type of strings that are checked. Defaults to "jailbreak".

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
