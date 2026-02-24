"""Implement the configuration for meta-llama/Llama-Prompt-Guard-2-86M supervisor from HuggingFace."""

from time import time
from typing import Any, Literal, cast

from torch import Tensor

from bells_o.common import OutputDict, Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import logit_compare as result_map

from ..hf_supervisor import HuggingFaceSupervisor


class LLamaPromptGuardV2Supervisor(HuggingFaceSupervisor):
    """Implement the pre-configured meta-llama/Llama-Prompt-Guard-2-86M supervisor from HuggingFace.

    PiGuard is a jailbreak detection model.
    """

    def __init__(
        self,
        variant: Literal["22m", "86m"] = "86m",
        pre_processing: list[PreProcessing] = [],
        model_kwargs: dict[str, Any] = {},
        tokenizer_kwargs: dict[str, Any] = {},
        generation_kwargs: dict[str, Any] = {},
        backend: Literal["transformers"] = "transformers",
    ):
        """Initialize the supervisor.

        Args:
            variant (Literal["22m", "86m"]): The model size to be used. Can be 86M or 22M parameters. Defaults to "86m".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            model_kwargs (dict[str, Any], optional):  Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs (dict[str, Any], optional):  Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs (dict[str, Any], optional): Keyword arguments to configure generation. Defaults to {}.
            backend (Literal["transformers", "vllm"]): The inference backend to use. Defaults to "transformers".

        """
        self._supported_backends = ["transformers"]

        super().__init__(
            name=f"meta-llama/Llama-Prompt-Guard-2-{variant}",
            usage=Usage("jailbreak"),
            res_map_fn=result_map,  # pyright: ignore[reportArgumentType]
            pre_processing=pre_processing,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            generation_kwargs=generation_kwargs,
            provider_name="Meta",
            backend=backend,
        )

    def _load_model_tokenizer(self):
        # loading model and tokenizer for different backend implementations
        if self.backend not in self._supported_backends:
            raise NotImplementedError(
                f"The requested backend `{self.backend}` is not supported. Choose one of {self._supported_backends}."
            )

        if self.backend == "transformers":
            from transformers import (  # noqa: F401
                AutoModelForSequenceClassification,
                AutoTokenizer,
                PreTrainedTokenizerBase,  # pre-caching
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.name, **self.tokenizer_kwargs)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.name, trust_remote_code=True, **self.model_kwargs
            )

    def _judge_sample(self, inputs: list[str]) -> tuple[float, Any, Any]:
        """Run tokenization and forward pass.

        Outsourced for clarity.
        """
        import torch
        from transformers import PreTrainedTokenizerBase

        with torch.no_grad():
            assert isinstance(self._tokenizer, PreTrainedTokenizerBase), f"Got {type(self._tokenizer)}"
            encoded_batch = self._tokenizer(inputs, return_tensors="pt", truncation=True, padding=True).to(
                device=getattr(self._model, "device")
            )

            start_time = time()
            outputs = self._model(**encoded_batch, **self.generation_kwargs)
            generation_time = time() - start_time
        return generation_time, outputs, encoded_batch

    def _judge_transformers(self, inputs: list[str]) -> list[OutputDict]:
        """Reimplementation of `super._judge_transformers` with minor changes to accomodate `AutoModelForSequenceClassification`."""
        assert self.backend == "transformers", (
            f'Backend should be "transformers" at this point, but got "{self.backend}".'
        )
        import torch

        try:
            generation_time, outputs, encoded_batch = self._judge_sample(inputs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            generation_time, outputs, encoded_batch = self._judge_sample(inputs)

        scores = cast(Tensor, outputs["logits"]).tolist()
        batch_size = len(inputs)
        input_tokens = cast(Tensor, encoded_batch["attention_mask"]).sum(dim=1).tolist()

        return [
            OutputDict(
                output_raw=score,
                metadata={
                    "latency": generation_time / batch_size,
                    "batch_size": batch_size,
                    "input_tokens": input_t,
                    "output_tokens": 1,
                },
            )
            for score, input_t in zip(scores, input_tokens)
        ]
