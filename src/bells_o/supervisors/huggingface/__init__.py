"""Module structure."""

from .auto_model import AutoHuggingFaceSupervisor
from .hf_supervisor import HuggingFaceSupervisor


# TODO: Figure out a way to lazy load all the other concrete model implementations


__all__ = ["HuggingFaceSupervisor", "AutoHuggingFaceSupervisor"]
