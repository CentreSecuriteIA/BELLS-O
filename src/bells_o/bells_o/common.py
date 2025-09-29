"""Define common datatypes and classes."""

from typing import TypedDict


USAGE_TYPES = ["JAILBREAK", "PROMPT_INJECTION", "CONTENT_MODERATION", "GENERAL"]


class UsageType(TypedDict):
    """Unifying class that acts as a usage type switch."""

    jailbreak: bool
    content_moderation: bool
    prompt_injection: bool

    # TODO: agree on usage types (e.g. General, jailbreak, etc.)


def usage_type(*args) -> UsageType:
    """Initialize a usage type.

    To be used as: usage_type = usage_type('Jailbreak', 'content_mOdEration', 'prompt_injection')
    """
    usage: UsageType = {"jailbreak": False, "content_moderation": False, "prompt_injection": False}
    if not args:
        for key in usage:
            usage[key] = True
    for arg in args:
        key = arg.lower()
        if key not in usage.keys():
            print(f"WARNING: did not find {key} in {usage.keys()}, skipping.")
            continue
        usage[key] = True
    return usage


class Result(TypedDict):
    """Unifying class that holds a result."""

    # just example types, have to agree on final list
    jailbreak: float | None
    content_moderation: float | None
    prompt_injection: float | None
