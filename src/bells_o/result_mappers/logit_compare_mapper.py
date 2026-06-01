"""Implement the result mapping function for leolee99/PIGuard on HF."""

from bells_o.common import Result, Usage


def mapper(output: list[float], usage: Usage = Usage()) -> Result:
    """Map the output format of leolee99/PIGuard to `Result`.

    Only maps to `jailbreak`. Compares passed logits of format [<benign>, <injection>]

    Args:
        output (list[float]): A string or list of strings that should be mapped.
        usage (Usage): Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: The result that the output maps to.

    """
    benign, harmful = output
    flagged = harmful > benign
    return Result(jailbreak=flagged)
