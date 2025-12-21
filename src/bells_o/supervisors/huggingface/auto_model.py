"""Implement the AutoHuggingFaceSupervisor class."""

from importlib import import_module


MODEL_MAPPING = {
    "saillab/xguard": ("saillab", "XGuardSupervisor", {}),
    "openai/gptossafeguard-20b": ("openai", "GptOssSafeguardSupervisor", {"variant": "20b"}),
    "openai/gptossafeguard-120b": ("openai", "GptOssSafeguardSupervisor", {"variant": "120b"}),
    "google/shieldgemma-2b": ("google", "ShieldGemmaSupervisor", {"variant": "2b"}),
    "google/shieldgemma-9b": ("google", "ShieldGemmaSupervisor", {"variant": "9b"}),
    "google/shieldgemma-27b": ("google", "ShieldGemmaSupervisor", {"variant": "27b"}),
    "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0": ("nvidia", "AegisSupervisor", {}),
    "qwen/qwen3guard-gen-8b": ("qwen", "Qwen3GuardSupervisor", {"variant": "8B"}),
    "qwen/qwen3guard-gen-4b": ("qwen", "Qwen3GuardSupervisor", {"variant": "4B"}),
    "qwen/qwen3guard-gen-0.6b": ("qwen", "Qwen3GuardSupervisor", {"variant": "0.6B"}),
    "rakancorle1/thinkguard": ("rakancorle1", "ThinkGuardSupervisor", {}),
    "allenai/wildguard": ("allenai", "WildGuardSupervisor", {}),
    "ibm-granite/granite-guardian-3.3-8b": (
        "ibm-granite",
        "GraniteGuardianSupervisor",
        {"model_id": "ibm-granite/granite-guardian-3.3-8b"},
    ),
    "ibm-granite/granite-guardian-3.0-2b": (
        "ibm-granite",
        "GraniteGuardianSupervisor",
        {"model_id": "ibm-granite/granite-guardian-3.0-2b"},
    ),
    "ibm-granite/granite-guardian-3.0-8b": (
        "ibm-granite",
        "GraniteGuardianSupervisor",
        {"model_id": "ibm-granite/granite-guardian-3.0-8b"},
    ),
    "ibm-granite/granite-guardian-3.1-2b": (
        "ibm-granite",
        "GraniteGuardianSupervisor",
        {"model_id": "ibm-granite/granite-guardian-3.1-2b"},
    ),
    "ibm-granite/granite-guardian-3.1-8b": (
        "ibm-granite",
        "GraniteGuardianSupervisor",
        {"model_id": "ibm-granite/granite-guardian-3.1-8b"},
    ),
    "ibm-granite/granite-guardian-3.2-5b": (
        "ibm-granite",
        "GraniteGuardianSupervisor",
        {"model_id": "ibm-granite/granite-guardian-3.2-5b"},
    ),
    "ibm-granite/granite-guardian-3.2-3b-a800m": (
        "ibm-granite",
        "GraniteGuardianSupervisor",
        {"model_id": "ibm-granite/granite-guardian-3.2-3b-a800m"},
    ),
}


class AutoHuggingFaceSupervisor:
    """Class that implements automatic loading of previously implemented supervisor models from HuggingFace."""

    @classmethod
    def load(cls, model_id: str, **kwargs):
        """Load a HuggingFaceSupervisor automatically from pre-configured supervisors.

        Args:
            model_id (str): Model ID of implemented supervisor model on Huggingface.
            **kwargs: Optional keyword arguments to override default parameters.

        """
        module_name, class_attribute, special_kwargs = MODEL_MAPPING[model_id.lower()]
        model_module = import_module(f".{module_name}", "bells_o.supervisors.huggingface")

        if hasattr(model_module, class_attribute):
            model_class = getattr(model_module, class_attribute)
        else:
            raise NotImplementedError(
                f"Did not find attribute `{class_attribute}` in module `bells_o.supervisors.huggingface.{module_name}`."
            )

        # merge kwargs, special kwargs take priority because they are determined by the model_id
        kwargs |= special_kwargs

        return model_class(**kwargs)
