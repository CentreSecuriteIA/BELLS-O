"""Module structure for mappers."""
# TODO: figure out lazy loading

from .anthropic_one_mapper import mapper as anthropic_one
from .azure_analyze_text_mapper import mapper as azure_analyze_text
from .azure_prompt_shield_mapper import mapper as azure_prompt_shield
from .bedrock_guardrail_mapper import mapper as bedrock_guardrail
from .gemini_moderation_mapper import mapper as gemini_moderation
from .gemini_one_mapper import mapper as gemini_one
from .gpt_oss_local_one_mapper import mapper as gpt_oss_local_one
from .granite_33_mapper import mapper as granite_33
from .harmful_request_mapper import mapper as harmful_request
from .lakeraguard_mapper import mapper as lakeraguard
from .lionguard_mapper import mapper as lionguard
from .logit_compare_mapper import mapper as logit_compare
from .nemotron_mapper import mapper as nemotron
from .neuraltrust_trustgate_mapper import mapper as neuraltrust
from .one_mapper import mapper as one_map
from .openai_compatible_one_mapper import mapper as openai_compatible_one
from .openai_compatible_unsafe_mapper import mapper as openai_compatible_unsafe
from .openai_moderation_mapper import mapper as openai_moderation
from .protectai_llm_guard_mapper import mapper as protectai
from .qwen3guard_mapper import mapper as qwen3guard
from .unsafe_mapper import mapper as unsafe_map
from .xguard_mapper import mapper as xguard
from .yes_mapper import mapper as yes_map


# TODO: fix the module structure, delete all files keys so that only the function keys are left over
__all__ = [
    "xguard",
    "lakeraguard",
    "openai_moderation",
    "azure_analyze_text",
    "gemini_moderation",
    "gemini_one",
    "openai_compatible_unsafe",
    "openai_compatible_one",
    "anthropic_one",
    "unsafe_map",
    "yes_map",
    "one_map",
    "bedrock_guardrail",
    "qwen3guard",
    "granite_33",
    "harmful_request",
    "lionguard",
    "nemotron",
    "gpt_oss_local_one",
    "azure_prompt_shield",
    "logit_compare",
    "protectai",
    "neuraltrust",
]
