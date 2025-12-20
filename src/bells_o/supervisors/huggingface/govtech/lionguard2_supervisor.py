"""LionGuard2Supervisor implementation."""

from os import getenv
from time import time
from typing import Any, Callable, Literal, cast

from torch import Tensor

from bells_o.common import OutputDict, Usage
from bells_o.preprocessors import PreProcessing, TemplateWrapper
from bells_o.result_mappers import lionguard as lionguard_result_mapper

from ..hf_supervisor import HuggingFaceSupervisor


TTokenizer = Callable[[list[str]], tuple[Tensor, list[int]]]


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
        api_variable: str | None = None,
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

        if api_variable is None:
            if model_id == "govtech/lionguard-2":
                api_variable = "OPENAI_API_KEY"
            elif model_id == "govtech/lionguard-2.1":
                api_variable = "GEMINI_API_KEY"

        self.api_variable = api_variable if api_variable else ""

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
    def api_variable(self, value: str):
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
            #   -d '{
            #     "input": "The food was delicious and the waiter...",
            #     "model": "text-embedding-ada-002",
            #     "encoding_format": "float"
            #   }'
            from requests import post

            from bells_o.supervisors import RestSupervisor
            from bells_o.supervisors.rest.auth_mappers import auth_bearer

            # TODO: handle max token length of 8192, or other api errors
            def openai_embedder(inputs: list[str]) -> tuple[Tensor, list[int]]:
                responses = [  # can't use batched input because this limits dimensions to <= 2048
                    post(
                        "https://api.openai.com/v1/embeddings",
                        json={
                            "input": inp,
                            "model": "text-embedding-3-large",
                            "dimensions": 3072,
                            "encoding_format": "float",
                        },
                        headers=auth_bearer(cast(RestSupervisor, self)) | {"Content-Type": "application/json"},
                    ).json()
                    for inp in inputs
                ]

                embeddings = Tensor([response["data"][0]["embedding"] for response in responses])
                input_tokens = [response["usage"]["prompt_tokens"] for response in responses]
                return embeddings, input_tokens

            return openai_embedder
        if self.name == "govtech/lionguard-2.1":
            from requests import post

            def gemini_embedder(inputs: list[str]) -> tuple[Tensor, list[int]]:
                response = post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents",
                    json={
                        "requests": [
                            {
                                "model": "models/gemini-embedding-001",
                                "content": {"parts": [{"text": inp}]},
                            }
                            for inp in inputs
                        ]
                    },
                    headers={"x-goog-api-key": f"{getenv('GEMINI_API_KEY')}"} | {"Content-Type": "application/json"},
                ).json()

                embeddings = Tensor([emb["values"] for emb in response["embeddings"]])
                input_tokens = [1] * len(embeddings)  # there is no information about input tokens
                return embeddings, input_tokens

            return gemini_embedder
        elif self.name == "govtech/lionguard-2-lite":
            from sentence_transformers import SentenceTransformer

            global TSentenceTransformer
            TSentenceTransformer = SentenceTransformer

            global TTensor
            TTensor = Tensor

            return SentenceTransformer("google/embeddinggemma-300m")

    def _judge_transformers(self, inputs: list[str]) -> list[OutputDict]:
        if not self.name == "govtech/lionguard-2-lite":
            embeddings, input_tokens = cast(TTokenizer, self._tokenizer)(inputs)
        else:
            global TSentenceTransformer
            global TTensor

            assert isinstance(self._tokenizer, TSentenceTransformer)

            attention_mask = self._tokenizer.tokenize(inputs)["attention_mask"]
            assert isinstance(attention_mask, TTensor)

            input_tokens = attention_mask.sum(dim=1).tolist()
            embeddings = self._tokenizer.encode(inputs)

        batch_size = len(inputs)
        start = time()
        outputs = self._model.predict(embeddings)  # type: ignore
        generation_time = time() - start

        return [
            OutputDict(
                output_raw=output,
                metadata={
                    "latency": generation_time / batch_size,
                    "batch_size": batch_size,
                    "input_tokens": input_t,
                    "output_tokens": 1,  # only one forward pass
                },
            )
            for output, input_t in zip(outputs, input_tokens)
        ]
