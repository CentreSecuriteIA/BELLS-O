"""Implement the AutoCustomSupervisor class."""

from importlib import import_module


MODEL_MAPPING = {
    "protectai/llm-guard": ("protectai", "ProtectAiLlmGuard", {}),
}


class AutoCustomSupervisor:
    """Class that implements automatic loading of previously implemented supervisor models from HuggingFace."""

    @classmethod
    def load(cls, model_id: str, **kwargs):
        """Load a CustomSupervisor automatically from pre-configured supervisors.

        Args:
            model_id (str): Model identifier for implemented supervisor model.
            **kwargs: Optional keyword arguments to override default parameters.

        """
        module_name, class_attribute, special_kwargs = MODEL_MAPPING[model_id.lower()]
        model_module = import_module(f".{module_name}", "bells_o.supervisors.custom")

        if hasattr(model_module, class_attribute):
            model_class = getattr(model_module, class_attribute)
        else:
            raise NotImplementedError(
                f"Did not find model `{class_attribute}` in module `bells_o.supervisors.custom.{module_name}`."
            )

        # merge kwargs, special kwargs take priority because they are determined by the model_id
        kwargs |= special_kwargs

        return model_class(**kwargs)
