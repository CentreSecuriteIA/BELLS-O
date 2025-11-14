"""Implement the AutoHuggingFaceSupervisor class."""

from importlib import import_module


MODEL_MAPPING = {
    "saillab/x-guard": ("saillab", "XGuardSupervisor", {}),
    "openai/gpt-oss-safeguard-20b": ("openai", "GptOssSafeGuardSupervisor", {"variant": "20b"}),
    "openai/gpt-oss-safeguard-120b": ("openai", "GptOssSafeGuardSupervisor", {"variant": "120b"}),
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
        module_name, class_attribute, special_kwargs = MODEL_MAPPING[model_id]
        model_module = import_module(f".{module_name}", "bells_o.supervisors.huggingface")

        if hasattr(model_module, class_attribute):
            model_class = getattr(model_module, class_attribute)
        else:
            raise NotImplementedError(
                f"Did not find attribute `{class_attribute}` in module `bells_o.supervisors.huggingface.{module_name}`."
            )

        # clean kwargs - passed kwargs take priority
        keys_to_delete = []
        for key in special_kwargs:
            if key in kwargs:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del special_kwargs[key]

        return model_class(**kwargs, **special_kwargs)
