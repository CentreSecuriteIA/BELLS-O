"""Impose structure on Class structure."""

from .huggingface import AutoHuggingFaceSupervisor, HuggingFaceSupervisor
from .rest import RestSupervisor, jsonmappers
from .supervisor import Supervisor


__all__ = [
    "Supervisor",
    "RestSupervisor",
    "HuggingFaceSupervisor",
    "AutoHuggingFaceSupervisor",
    "RestSupervisor",
    "jsonmappers",
]
