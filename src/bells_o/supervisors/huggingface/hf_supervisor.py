"""Implement the base class for HuggingFace-accessible supersivor models."""

from time import time
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from bells_o.common import OutputDict, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing

from ..supervisor import Supervisor


if TYPE_CHECKING:
    pass


SUPPORTED_BACKENDS = ["transformers", "vllm"]


class HuggingFaceSupervisor(Supervisor):
    """A concrete class that enables loading any HuggingFace model as a supervisor.

    Args:
        Supervisor (_type_): _description_

    Returns:
        _type_: _description_

    """

    # TODO: doc strings
    def __init__(
        self,
        name: str,
        usage: Usage,
        res_map_fn: ResultMapper,
        pre_processing: list[PreProcessing] = [],
        model_kwargs: dict[str, Any] = {},
        tokenizer_kwargs: dict[str, Any] = {},
        generation_kwargs: dict[str, Any] = {},
        provider_name: str | None = None,
        backend: Literal["transformers", "vllm"] = "transformers",
    ):
        super().__init__(name, usage, res_map_fn, pre_processing)
        self._model_kwargs = model_kwargs
        self._tokenizer_kwargs = tokenizer_kwargs
        self.generation_kwargs = generation_kwargs
        self._provider_name = provider_name

        if getattr(self, "_supported_backends", None) is None:
            self._supported_backends = ["transformers", "vllm"]

        self._backend = backend
        self._load_model_tokenizer()

    def _load_model_tokenizer(self):
        # loading model and tokenizer for different backend implementations
        # making this a separate method such that it can be easily changed by supervisor implementations (e.g. for LORA)
        if self.backend not in self._supported_backends:
            raise NotImplementedError(
                f"The requested backend `{self.backend}` is not supported. Choose one of {self._supported_backends}."
            )
        if self.backend == "transformers":
            import transformers

            self._transformers = transformers

            self._model = self._transformers.AutoModelForCausalLM.from_pretrained(self.name, **self.model_kwargs)
            self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(self.name, **self.tokenizer_kwargs)
        elif self.backend == "vllm":
            import vllm

            self._vllm = vllm
            self._model = self._vllm.LLM(self.name, **self.model_kwargs, tensor_parallel_size=torch.cuda.device_count())
            self._tokenizer = self._model.get_tokenizer()

    @property
    def backend(self) -> str:  # noqa: D102
        return self._backend

    @property
    def model_kwargs(self) -> dict[str, Any]:  # noqa: D102
        return self._model_kwargs

    @property
    def tokenizer_kwargs(self) -> dict[str, Any]:  # noqa: D102
        return self._tokenizer_kwargs

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

    def pre_process(self, inputs: str | list[str]) -> list[str]:
        """Apply all preprocessing steps.

        Concrete classes will likely need a tokenization equivalent implemented.
        """  # TODO: improve this docstring
        if isinstance(inputs, str):
            inputs = [inputs]

        if self.pre_processing:
            for pre_processor in self.pre_processing:
                inputs = [pre_processor(input) for input in inputs]

        inputs = self._apply_chat_template(inputs)
        return inputs

    def _apply_chat_template(self, inputs: list[str]) -> list[str]:
        if getattr(self._tokenizer, "chat_template", None) is not None:
            assert isinstance(inputs, list), (
                "If `tokenizer.chat_template` is not None, then use a `RoleWrapper` as the last pre-processor."
            )
            inputs = self._tokenizer.apply_chat_template(
                inputs,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
            )  # TODO customize the kwargs of apply_chat_template?
        return inputs

    def judge(self, inputs: str | list[str]) -> list[OutputDict]:
        """Evaluate samples with model.

        Args:
            inputs (str | list[str]): The sample or batch of samples to be evaluated.

        Returns:
            list[str]: The outputs of the supervisor as a list.

        """
        if isinstance(inputs, str):
            inputs = [inputs]
        if self.backend == "transformers":
            return self._judge_transformers(inputs)
        if self.backend == "vllm":
            return self._judge_vllm(inputs)

        raise NotImplementedError(
            f"The requested backend `{self.backend}` is not supported. Choose one of {SUPPORTED_BACKENDS}."
        )

    # judge() implementations for different backends
    def _judge_transformers(self, inputs: list[str]) -> list[OutputDict]:
        assert self.backend == "transformers", (
            f'Backend should be "transformers" at this point, but got "{self.backend}".'
        )
        assert isinstance(self._tokenizer, self._transformers.PreTrainedTokenizerBase), f"got {type(self._tokenizer)}"

        encoded_batch = self._tokenizer(inputs, return_tensors="pt", padding=True).to(
            device=getattr(self._model, "device")
        )
        start_time = time()
        outputs = cast(torch.Tensor, self._model.generate(**encoded_batch, **self.generation_kwargs))
        generation_time = time() - start_time

        # cut outputs to only include generated tokens, assume that all samples were padded to the same length
        input_ids = encoded_batch["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        sequence_length = input_ids.size(1)  # padded sequence length

        outputs = outputs[:, sequence_length:, ...]

        decoded_outputs: list[str] = self._tokenizer.batch_decode(outputs)
        batch_size = len(inputs)
        input_tokens = cast(torch.Tensor, encoded_batch["attention_mask"]).sum().item()  # only count non-padding tokens
        output_tokens = (
            cast(torch.Tensor, outputs != self._tokenizer.pad_token_id).sum().item()
        )  # only count non-padding tokens

        return [
            OutputDict(
                output_raw=output,
                metadata={
                    "latency": generation_time / batch_size,
                    "batch_size": batch_size,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
            for output in decoded_outputs
        ]

    def _judge_vllm(self, inputs: list[str]):
        assert self.backend == "vllm", f'Backend should be "vllm" at this point, but got "{self.backend}".'
        assert isinstance(self._model, self._vllm.LLM)

        sampling_params = self._vllm.SamplingParams(
            **self.generation_kwargs
        )  # TODO: somehow make it obvious that this takes vllm arguments
        start = time()  # note that by doing this, we are generalizing latency per prompt as batch_latency/n_prompts
        outputs = self._model.generate(inputs, sampling_params)
        generation_time = time() - start

        batch_size = len(inputs)

        return [
            OutputDict(
                output_raw=output.outputs[0].text,
                metadata={
                    "latency": generation_time / batch_size,
                    "batch_size": batch_size,
                    "input_tokens": len(cast(list[int], output.prompt_token_ids)),
                    "output_tokens": len(output.outputs[0].token_ids),
                },
            )
            for output in outputs
        ]
