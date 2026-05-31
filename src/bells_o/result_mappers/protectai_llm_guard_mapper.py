"""Implement the result mapping function for ProtectAI LLM Guard."""

from bells_o.common import Result, Usage


def mapper(output: tuple[str, bool, float], usage: Usage = Usage()) -> Result:
    """Map the output format of ProtectAI LLM Guard to `Result`.

    Takes the inverse of the passed boolean in the tuple. Only maps to jailbreak usage.

    Args:
        output (tuple[str, bool, float]): The output tuple of the supervisor.
        usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: A Result object with jailbreak flag set based on parsed output.

    """
    _, is_harmless, _ = output
    return Result(jailbreak=not is_harmless)
