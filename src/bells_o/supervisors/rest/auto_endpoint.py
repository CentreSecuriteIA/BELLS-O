"""Implement the AutoHuggingFaceSupervisor class."""

from importlib import import_module


# TODO: remove the _classification classes and incorporate them as defaults in the normal supervisors
# TODO: make naming better
MODEL_MAPPING = {
    "lakeraguard": ("lakeraguard", "LakeraGuardSupervisor"),
    "lakeraguard-default": ("lakeraguard", "LakeraGuardDefaultSupervisor"),
    "openai": ("openai", "OpenAiSupervisor"),
    "openai-moderation": ("openai", "OpenAiModerationSupervisor"),
    "openai-classification": ("openai", "OpenAIClassificationSupervisor"),
    "azure-analyze-text": ("azure", "AzureAnalyzeTextSupervisor"),
    "google": ("google", "GeminiSupervisor"),
    "google-moderation": ("google", "GeminiModerationSupervisor"),
    "google-classification": ("google", "GeminiClassificationSupervisor"),
    "mistral": ("mistral", "MistralSupervisor"),
    "mistral-classification": ("mistral", "MistralClassificationSupervisor"),
    "xai": ("xai", "XAiSupervisor"),
    "xai-classification": ("xai", "XAiClassificationSupervisor"),
    "anthropic": ("anthropic", "AnthropicSupervisor"),
    "anthropic-classification": ("anthropic", "AnthropicClassificationSupervisor"),
    "together-gpt-oss": ("together", "GptOssSupervisor"),
    "together-llama-guard-4b": ("together", "LlamaGuard4BModerationSupervisor"),
    "together-virtueguard-text-lite": ("together", "VirtueGuardTextLiteModerationSupervisor"),
    "openrouter-gpt-oss-safeguard-20b": ("openrouter", "GptOssSafeguard20Supervisor"),
    "bedrock-guardrail": ("aws", "BedrockGuardrailSupervisor"),
}


class AutoRestSupervisor:
    """Class that implements automatic loading of previously implemented supervisor REST APIs."""

    @classmethod
    def load(cls, endpoint_name, *args, **kwargs):
        """Load a RestSupervisor automatically from pre-configured APIs.

        Args:
            endpoint_name (str): name of the endpoint, consult documentation for implemented APIs.
            *args: Positional arguments for the class type.
            **kwargs: Optional keyword arguments for the instantiated class.

        """
        module_name, class_attribute = MODEL_MAPPING[endpoint_name]
        model_module = import_module(f".{module_name}", "bells_o.supervisors.rest")

        if hasattr(model_module, class_attribute):
            model_class = getattr(model_module, class_attribute)
        else:
            raise NotImplementedError(
                f"Did not find attribute `{class_attribute}` in module `bells_o.supervisors.rest.{module_name}`."
            )
        return model_class(*args, **kwargs)
