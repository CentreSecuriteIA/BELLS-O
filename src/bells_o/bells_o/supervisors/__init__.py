"""Impose structure on Class structure."""

from .huggingface import AutoHuggingFaceSupervisor, HuggingFaceSupervisor
from .rest import RestSupervisor
from .supervisor import Supervisor


__all__ = ["Supervisor", "RestSupervisor", "HuggingFaceSupervisor", "AutoHuggingFaceSupervisor"]
