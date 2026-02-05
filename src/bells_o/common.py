"""Define common datatypes and classes."""

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Self,
    TypedDict,
    TypeVar,
    Unpack,
    get_args,
)


if TYPE_CHECKING:
    from .supervisors.rest import RestSupervisor


### Usage Type definitions for type hinting, based on USAGE_TYPES defined above
UsageType = Literal[
    "jailbreak",
    # "prompt_injection",
    "content_moderation",
]

UsageTypes = TypedDict(
    "UsageTypes",
    {
        "jailbreak": NotRequired[bool],
        "prompt_injection": NotRequired[bool],
        "content_moderation": NotRequired[bool],
    },
)

USAGE_TYPES = get_args(UsageType)


class Usage(dict):
    """Class that represents the usage type of a dataset or supervisor."""

    def __init__(self, *args: UsageType) -> None:
        """Initialize a usage type object.

        All usage types that were listed during initialization will be set to true, the rest to False. All `<usage_type>` values can
        be accessed via `Usage()[<usage_type>]`.
        To be instantiated as: usage_type = Usage('Jailbreak', 'content_mOdEration', 'prompt_injection')

        Args:
            *args (str): Usage types that are supported

        """
        # if no args are passed, we assume it is a general type
        if not args:
            kwargs = dict.fromkeys(USAGE_TYPES, True)
        else:
            kwargs = {}
            for arg in args:
                if arg not in USAGE_TYPES:
                    print(f"WARNING: '{arg}' is not in {USAGE_TYPES}, skipping '{arg}'.")
                    continue
                kwargs[arg] = True
        super().__init__(**kwargs)

    def __getitem__(self, key) -> bool:
        """Return `True` if Usage supports this type, else `False`."""
        if key in USAGE_TYPES:
            try:
                return super().__getitem__(key)
            except KeyError:
                return False
        else:
            raise ValueError(f"{key} if not a valid usage type. Expected one of {USAGE_TYPES}.")


### Typed dictionary definitions
class Result(dict):
    """Unifying class that holds a result."""

    def __init__(self, **kwargs: Unpack[UsageTypes]):
        """Initialize Result object."""
        super().__init__(**kwargs)

    def __eq__(self, other: Self):
        if not isinstance(other, type(self)):
            raise NotImplementedError(
                "Currently, comparison is only implemented for two `Result` objects. If you are comparing to a dict, try wrapping it as follows: `Result(**dict_var)`."
            )
        # for proper comparison, one has to be a subset of the other
        keys_self = list(self.keys())
        keys_other = list(other.keys())
        is_subset = all(key in keys_self for key in keys_other) or all(key in keys_other for key in keys_self)
        if is_subset:
            smallest_key_set = min(keys_self, keys_other, key=len)
            return all(self[key] == other[key] for key in smallest_key_set)
        return False

    def __bool__(self):
        """Return boolean value of this result. True if any of the usage_types are True."""
        return any(self.values())


# TODO: maybe add a .json() method to deal with indentation issues (dict(output_result))
class OutputDict(TypedDict):
    """Structured dictionary for type hinting `judge` outputs."""

    output_raw: str | dict[str, Any]
    metadata: dict[str, Any]
    output_result: NotRequired[Result]
    target_result: NotRequired[Result]
    is_correct: NotRequired[bool]


T = TypeVar("T", bound="RestSupervisor")

RequestMapper = Callable[[T, str], dict[str, str]]  # for json output
type AuthMapper = Callable[["RestSupervisor"], dict[str, str]]  # for json output
type ResultMapper = Callable[[str, Usage], Result] | Callable[[dict[str, Any], Usage], Result]
