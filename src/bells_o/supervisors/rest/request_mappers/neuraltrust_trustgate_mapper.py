"""Implement the payload mapper for Neuraltrust TrustGate."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..neuraltrust import NeuralTrustTrustGateSupervisor


def mapper(
    supervisor: "NeuralTrustTrustGateSupervisor",
    prompt: str,
) -> dict[str, str]:
    """Generate a json payload for an API request to the NeuralTrustTrustGateSupervisor.

    Maps to {"policy_id": `policy_id`, "payload": {"content": `prompt`}}.

    Args:
        prompt: The prompt to evaluate.
        supervisor (NeuralTrustTrustGateSupervisor): The NeuralTrustTrustGateSupervisor for which to generate the authentication payload.

    Returns:
        The mapped authentication payload.

    """
    json_repr = {
        "payload": {"content": prompt},
        "policy_id": f"{supervisor.policy_id}",
    }
    return json_repr
