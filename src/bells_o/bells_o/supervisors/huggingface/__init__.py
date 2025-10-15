"""Module structure."""

from .auto_model import AutoHuggingFaceSupervisor
from .custom_model import HuggingFaceSupervisor


# TODO: Figure out a way to lazy load all the other concrete model implementations


__all__ = ["HuggingFaceSupervisor", "AutoHuggingFaceSupervisor"]
