"""Impose structure on Class structure."""

from .huggingface import AutoHuggingFaceSupervisor, HuggingFaceSupervisor
from .rest import (
    AnthropicClassificationSupervisor,
    AnthropicSupervisor,
    AutoRestSupervisor,
    LakeraGuardSupervisor,
    MistralClassificationSupervisor,
    MistralSupervisor,
    OpenAiModerationSupervisor,
    OpenAiSupervisor,
    RestSupervisor,
    XAiClassificationSupervisor,
    XAiSupervisor,
)
from .supervisor import Supervisor


__all__ = [
    "Supervisor",
    "RestSupervisor",
    "HuggingFaceSupervisor",
    "AutoHuggingFaceSupervisor",
    "RestSupervisor",
    "AutoRestSupervisor",
    "LakeraGuardSupervisor",
    "OpenAiSupervisor",
    "OpenAiModerationSupervisor",
    "MistralSupervisor",
    "MistralClassificationSupervisor",
    "XAiSupervisor",
    "XAiClassificationSupervisor",
    "AnthropicSupervisor",
    "AnthropicClassificationSupervisor",
]
