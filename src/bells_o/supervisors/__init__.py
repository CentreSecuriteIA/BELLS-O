"""Impose structure on Class structure."""

from .custom import AutoCustomSupervisor, CustomSupervisor
from .huggingface import AutoHuggingFaceSupervisor, HuggingFaceSupervisor
from .rest import (
    AutoRestSupervisor,
    RestSupervisor,
)
from .supervisor import Supervisor


__all__ = [
    "Supervisor",
    "HuggingFaceSupervisor",
    "AutoHuggingFaceSupervisor",
    "RestSupervisor",
    "AutoRestSupervisor",
    "CustomSupervisor",
    "AutoCustomSupervisor",
]
