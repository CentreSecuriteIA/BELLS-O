"""Generate a json payload for an API request to the AWS Bedrock Guardrail Supervisor."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..aws import AwsSupervisor


def mapper(
    supervisor: "AwsSupervisor",
    prompt: str,
) -> dict[str, Any]:
    """Generate a json payload for an API request to the AWS Bedrock Guardrail Supervisor.

    Maps to AWS Bedrock Guardrail ApplyGuardrail API format.

    Args:
        prompt: The prompt to evaluate.
        supervisor (AwsSupervisor): The AwsSupervisor for which to generate the request payload.

    Returns:
        The mapped request payload.

    """
    json_repr: dict[str, Any] = {
        "content": [
            {
                "text": {
                    "text": prompt,
                }
            }
        ],
        "source": getattr(supervisor, "source", "INPUT"),
    }
    return json_repr

