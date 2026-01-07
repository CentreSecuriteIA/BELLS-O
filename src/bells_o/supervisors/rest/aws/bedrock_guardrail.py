"""Implement the AWS Bedrock Guardrail supervisor via boto3."""

from typing import Any

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import bedrock_guardrail as bedrock_guardrail_result_map

from .aws import AwsSupervisor


# Example usage:
# GUARDRAIL_ID = "h2nlgerrlgip"
# GUARDRAIL_VERSION = "1"  # Use "DRAFT" for newly created guardrails, or a version number if published
# REGION = "us-east-1"


class BedrockGuardrailSupervisor(AwsSupervisor):
    """Implement the AWS Bedrock Guardrail API via boto3."""

    def __init__(
        self,
        usage: Usage = Usage("content_moderation"),
        guardrail_identifier: str = "h2nlgerrlgip",
        guardrail_version: str = "1",
        region: str = "us-east-1",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "AWS_ACCESS_KEY_ID",
        source: str = "INPUT",
    ):
        """Initialize the BedrockGuardrailSupervisor.

        Args:
            guardrail_identifier (str): The identifier of the guardrail to use.
            guardrail_version (str): The version of the guardrail (e.g., "DRAFT" or a version number).
            usage (Usage): The usage type of the supervisor.
            region (str): AWS region. Defaults to "us-east-1".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): AWS Access Key ID (optional, can use IAM role or credentials file). Defaults to None.
            api_variable (str, optional): Environment variable name for AWS Access Key ID. Defaults to "AWS_ACCESS_KEY_ID".
            source (str, optional): Content source, either "INPUT" or "OUTPUT". Defaults to "INPUT".

        """
        super().__init__(
            name=f"bedrock-guardrail-{guardrail_identifier}",
            usage=usage,
            result_mapper=bedrock_guardrail_result_map,
            base_url="",  # Not used with boto3
            region=region,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
            source=source,
        )

        self.guardrail_identifier = guardrail_identifier
        self.guardrail_version = guardrail_version

    def _call_bedrock_api(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Call the AWS Bedrock Guardrail ApplyGuardrail API.

        Args:
            request_payload: The request payload containing content and source.

        Returns:
            dict[str, Any]: The API response.

        """
        response = self.bedrock_client.apply_guardrail(
            guardrailIdentifier=self.guardrail_identifier,
            guardrailVersion=self.guardrail_version,
            content=request_payload["content"],
            source=request_payload["source"],
        )
        return response

    @classmethod
    def _get_token_counts(cls, output_raw: dict[str, Any]) -> dict[str, Any]:
        """For Bedrock Guardrails AWS counts usage in text units, where each text unit may contain up to 1000 characters."""
        input_tokens = int(output_raw["assessments"][0]["guardrailCoverage"]["textCharacters"]["total"]) // 1000 + 1
        output_tokens = 0

        return {"input_tokens": input_tokens, "output_tokens": output_tokens}
