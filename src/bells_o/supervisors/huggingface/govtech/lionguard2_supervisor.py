"""LionGuard2Supervisor implementation."""

from os import getenv
from time import time
from typing import Any, Literal

import requests

from bells_o.common import OutputDict, Usage
from bells_o.preprocessors import PreProcessing, TemplateWrapper
from bells_o.result_mappers import lionguard as lionguard_result_mapper

from ..hf_supervisor import HuggingFaceSupervisor


class LionGuard2Supervisor(HuggingFaceSupervisor):
    """Implement the pre-configured govtech/lionguard-2{variant} supervisors from HuggingFace."""

    def __init__(
        self,
        model_id: str,
        pre_processing: list[PreProcessing] = [],
        model_kwargs: dict[str, Any] = {},
        tokenizer_kwargs: dict[str, Any] = {},
        generation_kwargs: dict[str, Any] = {},
        backend: Literal["transformers"] = "transformers",
        api_key: str | None = None,
        api_variable: str | None = "OPENAI_API_KEY",  # has to be variable for different versions
    ):
        """Initialize the supervisor.

        Args:
            model_id (str, optional): The id of the exact model to use. This class supports different LionGuard models.
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            model_kwargs: Keyword arguments to configure the model. Defaults to {}.
            tokenizer_kwargs: Keyword arguments to configure the tokenizer. Defaults to {}.
            generation_kwargs: Keyword arguments to configure generation. Defaults to {}.
            backend (Literal["transformers", "vllm"]): The inference backend to use. Defaults to "transformers".

        """
        self._api_key = api_key
        self._api_variable = api_variable

        self._supported_backends = ["transformers"]
        if model_id == "govtech/lionguard-2-lite":
            pre_processing.append(TemplateWrapper("task: classification | query: {prompt}"))

        super().__init__(
            name=model_id,
            usage=Usage("content_moderation"),
            res_map_fn=lionguard_result_mapper,
            pre_processing=pre_processing,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            generation_kwargs=generation_kwargs,
            provider_name="Govtech",
            backend=backend,
        )

    @property
    def api_key(self) -> str:  # noqa: D102
        if not self._needs_api:
            return ""
        return self._api_key or getenv(self.api_variable, "")

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value
        self._needs_api = True

    @property
    def api_variable(self) -> str:  # noqa: D102
        return self._api_variable if self._api_variable is not None else ""

    @api_variable.setter
    def api_variable(self, value: str | None):
        self._api_variable = value

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
            self._tokenizer = self._get_tokenizer()

    def _apply_chat_template(self, inputs: list[str]) -> list[str]:
        if not self.name == "govtech/lionguard-2-lite":  # this is the only model with a local tokenizer
            return inputs
        else:
            return super()._apply_chat_template(inputs)

    def _get_tokenizer(self):
        if self.name == "govtech/lionguard-2":

            def openai_embedder(inputs: list[str]):
                requests.post("someurl")
                return inputs

            return openai_embedder
            ...  # write some tokenizer function with requests
        if self.name == "govtech/lionguard-2.1":

            def gemini_embedder(inputs: list[str]):
                requests.post("someurl")
                return inputs

            return gemini_embedder
            ...  # write some tokenizer function with requests
        elif self.name == "govtech/lionguard-2-lite":
            from sentence_transformers import SentenceTransformer

            global TSentenceTransformer
            TSentenceTransformer = SentenceTransformer
            self._tokenizer = SentenceTransformer("google/embeddinggemma-300m")

    def _judge_transformers(self, inputs: list[str]) -> list[OutputDict]:
        if not self.name == "govtech/lionguard-2-lite":
            embeddings, input_tokens = self._tokenizer(inputs)
        elif self.name == "govtech/lionguard-2-lite":
            global TSentenceTransformer
            assert isinstance(self._tokenizer, TSentenceTransformer)
            embeddings, input_tokens = self._tokenizer.encode(inputs)

        start = time()
        outputs = self._model.predict(embeddings)  # type: ignore
        generation_time = time() - start

        batch_size = len(inputs)
        output_tokens = 1

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
            for output in outputs
        ]
