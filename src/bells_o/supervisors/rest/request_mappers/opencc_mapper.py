"""Implement the payload mapper for OpenCC."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..opencc import OpenCCSupervisor


def mapper(
    supervisor: "OpenCCSupervisor",
    prompt: str,
) -> dict[str, str]:
    """Generate a json payload for an API request to the OpenCCSupervisor.

    Maps to {"text": `prompt`} and adds {"mode": `mode`} only when a mode override is set.

    Args:
        supervisor (OpenCCSupervisor): The OpenCCSupervisor for which to generate the payload.
        prompt: The prompt to evaluate.

    Returns:
        The mapped request payload.

    """
    json_repr = {"text": prompt}
    if getattr(supervisor, "mode", None) is not None:
        json_repr["mode"] = supervisor.mode
    return json_repr
