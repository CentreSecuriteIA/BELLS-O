"""Impose structure on Class structure."""

from .huggingface import AutoHuggingFaceSupervisor, HuggingFaceSupervisor
from .rest import (
    AnthropicClassificationSupervisor,
    AnthropicSupervisor,
    AutoRestSupervisor,
    GptOssSupervisor,
    LakeraGuardSupervisor,
    LlamaGuard4BModerationSupervisor,
    MistralClassificationSupervisor,
    MistralSupervisor,
    OpenAiModerationSupervisor,
    OpenAiSupervisor,
    RestSupervisor,
    TogetherAISupervisor,
    VirtueGuardTextLiteModerationSupervisor,
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
    "TogetherAISupervisor",
    "GptOssSupervisor",
    "GptOssSafeguardSupervisor",
    "LlamaGuard4BModerationSupervisor",
    "VirtueGuardTextLiteModerationSupervisor",
    "MistralSupervisor",
    "MistralClassificationSupervisor",
    "XAiSupervisor",
    "XAiClassificationSupervisor",
    "AnthropicSupervisor",
    "AnthropicClassificationSupervisor",
]
