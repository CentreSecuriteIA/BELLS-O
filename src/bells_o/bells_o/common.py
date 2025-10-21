"""Define common datatypes and classes."""

from typing import Any, Callable, NotRequired, Self, TypedDict


### Usage Type definitions
USAGE_TYPES = ["JAILBREAK", "PROMPT_INJECTION", "CONTENT_MODERATION", "GENERAL"]
# TODO: agree on usage types (e.g. General, jailbreak, etc.)


class Usage:
    """Unifying class that acts as a usage type switch."""

    def __init__(self, *args: str) -> None:
        """Initialize a usage type object.

        All usage types that were listed during initialization will be set to true, the rest to False. All `<usage_type>` values can
        be accessed via `Usage()[<usage_type>]`.
        To be instantiated as: usage_type = Usage('Jailbreak', 'content_mOdEration', 'prompt_injection')

        Args:
            *args (str): Usage types that are supported

        """
        lower_args = [arg.lower() for arg in args]
        # if no types are passed, we assume it is a general type
        if not args or "general" in lower_args:
            self._usage_types: dict[str, bool] = {
                usage_type.lower(): True for usage_type in USAGE_TYPES
            }
        else:
            self._usage_types: dict[str, bool] = {
                usage_type.lower(): False for usage_type in USAGE_TYPES
            }
            for arg in lower_args:
                if arg.upper() not in USAGE_TYPES:
                    print(f"WARNING: did not find {arg} in {USAGE_TYPES}, skipping.")
                    continue
                self._usage_types[arg] = True

    def values(self):
        """Return a dict_values object of the usage types."""
        return self._usage_types.values()

    def keys(self):
        """Return a list of the supported usage types."""
        return [k for k, v in self._usage_types.items() if v]

    def __iter__(self):
        """Return iterator over supported usage types."""
        return iter([k for k, v in self._usage_types.items() if v])

    def __getitem__(self, key: str) -> bool:
        """Return the boolean value representing if a certain usage is part of this UsageType.

        Args:
            key (str): the name of the usage to be checked.

        """
        return self._usage_types[key]

    def __eq__(self, other: Self) -> bool:
        """Implement usage1 == usage2.

        Currently fails if `other` is not of type `Usage`

        Args:
            other (Usage): Instance of `Usage` for which equivalence is checked.

        """
        return all(
            self._usage_types[key.lower()] == other._usage_types[key.lower()] for key in USAGE_TYPES
        )

    def __ne__(self, other: Self):
        """Inverse of Usage.__eg__."""
        return not self.__eq__(other)


### Typed dictionary definitions
class Result(TypedDict):
    """Unifying class that holds a result."""

    # just example types, have to agree on final list
    jailbreak: NotRequired[float]
    content_moderation: NotRequired[float]
    prompt_injection: NotRequired[float]


class OutputDict(TypedDict):
    """Structured dictionary for type hinting `judge` outputs."""

    output_result: NotRequired[Result]
    target_result: NotRequired[Result]
    output_raw: str | dict[str, str]
    metadata: dict[str, Any]


### Mapper type definitions
ResultMapper = Callable[[str | dict[str, str]], Result]
JsonMapper = Callable[[Any], dict[str, str]]
