"""Implement the base class for HuggingFace-accessible supersivor models."""

from dataclasses import dataclass, field
from time import time
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from bells_o.common import OutputDict

from ..supervisor import Supervisor


@dataclass
class HuggingFaceSupervisor(Supervisor):
    """A concrete class that enables loading any HuggingFace model as a supervisor."""

    model_kwargs: dict[str, Any] | None = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] | None = field(default_factory=dict)
    generation_kwargs: dict[str, Any] | None = field(default_factory=dict)
    provider_name: str | None = None

    def __post_init__(self):
        """Load the model and tokenizer from HuggingFace."""
        assert isinstance(self.model_kwargs, dict) and isinstance(self.tokenizer_kwargs, dict), (
            "Expected arguments to not be None at this stage."
        )
        self._model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.name, **self.tokenizer_kwargs)

    def metadata(self) -> dict[str, Any]:
        """Return metadata dictionary for this Supervisor.

        Returns:
            dict: Dictionary with metadata.

        """
        metadata = super().metadata()
        if self.generation_kwargs is not None:
            metadata["generation_kwargs"] = self.generation_kwargs
        if self.model_kwargs is not None:
            metadata["model_kwargs"] = self.model_kwargs
        if self.tokenizer_kwargs is not None:
            metadata["tokenizer_kwargs"] = self.tokenizer_kwargs
        return metadata

    def pre_process(self, message: str):
        """Apply all preprocessing steps.

        Concrete classes will likely need a tokenization equivalent implemented.
        """  # TODO: improve this docstring
        if self.pre_processing:
            for pre_processor in self.pre_processing:
                message = pre_processor(message)
        if self._tokenizer.chat_template is not None:
            assert isinstance(message, list), (
                "If `tokenizer.chat_template` is not None, then use a `RoleWrapper` as the last pre-processor."
            )
            message = self._tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )  # TODO customize the kwargs of apply_chat_template?
        return self._tokenizer(message, return_tensors="pt")

    def judge(self, input_ids: BatchEncoding) -> list[OutputDict]:
        """Run one evaluation on the supervisor model.

        Expects tokenized inputs

        Args:
            input_ids (Tensor): The input that is to be judged, tokenized appropriately.

        Returns:
            str: Output of the supversior model of the classifier

        """
        assert isinstance(self.generation_kwargs, dict), (
            "Expected argument to not be None at this stage."
        )
        input_ids = input_ids.to(device=self._model.device)
        start_time = time()
        outputs = self._model.generate(**input_ids, **self.generation_kwargs)
        decoded_outputs: list[str] = self._tokenizer.batch_decode(outputs)
        generation_time = start_time - time()
        batch_size = len(input_ids)
        return [
            OutputDict(
                output_raw=output,
                metadata={"latency": generation_time, "batch_size": batch_size},
            )
            for output in decoded_outputs
        ]
