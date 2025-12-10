"""Implement the base class for HuggingFace Inference API supervisors."""

from time import time
from typing import Any

from requests import post

from bells_o.common import AuthMapper, OutputDict, RequestMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing

from ..auth_mappers import auth_bearer
from ..custom_endpoint import RestSupervisor
from ..request_mappers import huggingface as hf_request_map


class HuggingFaceApiSupervisor(RestSupervisor):
    """A concrete class that enables access to HuggingFace models via Inference API."""

    def __init__(
        self,
        model_id: str,
        usage: Usage,
        result_mapper: ResultMapper,
        pre_processing: list[PreProcessing] = None,
        generation_kwargs: dict[str, Any] = None,
        api_key: str | None = None,
        api_variable: str = "HUGGINGFACE_API_KEY",
        provider_name: str = "HuggingFace",
    ):
        """Initialize the HuggingFace API supervisor.

        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-Guard-4-12B").
            usage: The usage type of the supervisor.
            result_mapper: ResultMapper to use for this Supervisor.
            pre_processing: List of PreProcessing steps to apply to prompts.
            generation_kwargs: Keyword arguments to configure generation.
            api_key: API key to use, takes priority over `api_variable`.
            api_variable: Environment variable name that stores the API key.
            provider_name: Name of the provider.
        """
        self.name = model_id
        self.model_id = model_id  # Store for request mapper
        self.provider_name = provider_name
        # Use router API chat completions endpoint
        self.base_url = "https://router.huggingface.co/v1/chat/completions"
        self.usage = usage
        self.res_map_fn = result_mapper
        self.req_map_fn: RequestMapper = hf_request_map
        self.auth_map_fn: AuthMapper = auth_bearer
        self.pre_processing = pre_processing or []
        self.generation_kwargs = generation_kwargs or {}
        self.api_key = api_key
        self.api_variable = api_variable
        self.needs_api = True
        self.custom_header = {}

        super().__post_init__()

    def _judge_sample(self, prompt: str | list[dict[str, str]]) -> OutputDict:
        """Run an individual POST request for inference.

        Overrides base class to handle non-JSON responses gracefully.

        Args:
            prompt: The prompt string or message list to check.

        Returns:
            OutputDict: The output of the Supervisor and corresponding metadata.
        """
        tried_once = False
        no_valid_response = True

        while no_valid_response:
            if tried_once:
                print("INFO: Retrying generation in 5s. Hit rate limit.")
            else:
                print("INFO: Generating judgement.")

            start_time = time()
            headers = self.auth_map_fn(self) | self.custom_header
            response = post(
                self.base_url,
                json=self.req_map_fn(self, prompt),
                headers=headers,
            )
            generation_time = time() - start_time

            tried_once = True
            # Retry on rate limit (429) or model loading (503)
            no_valid_response = response.status_code in (self.rate_limit_code, 503)

        # Handle response - check status code first
        if response.status_code != 200:
            error_text = response.text[:1000] if response.text else 'No response body'
            response_data = {
                "error": f"HTTP {response.status_code}: {error_text}",
                "status_code": response.status_code,
                "url": self.base_url,
            }
        else:
            # Try JSON first, fallback to text
            try:
                response_data = response.json()
            except Exception:
                # If JSON parsing fails, use text response
                response_data = {
                    "error": f"Non-JSON response: {response.text[:500] if response.text else 'Empty response'}"
                }

        return OutputDict(output_raw=response_data, metadata={"latency": generation_time})

    def pre_process(self, inputs: str | list[str]) -> list[str | list[dict[str, str]]]:
        """Apply all preprocessing steps.

        Args:
            inputs: Input string(s) to preprocess.

        Returns:
            List of preprocessed strings or message lists (for chat completions API).
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        if self.pre_processing:
            for pre_processor in self.pre_processing:
                inputs = [pre_processor(input) for input in inputs]

        # Keep message lists as-is for chat completions API, convert others to strings
        processed = []
        for input_val in inputs:
            if isinstance(input_val, list):
                # Keep message list format for chat completions API
                processed.append(input_val)
            else:
                processed.append(str(input_val))

        return processed

    def judge(self, prompts: list[str | list[dict[str, str]]] | str | list[dict[str, str]]) -> list[OutputDict]:
        """Evaluate a (batch of) prompt(s).

        Overrides RestSupervisor to handle HuggingFace API response format.

        Args:
            prompts: List of prompts (strings or message lists) or single prompt.

        Returns:
            List of OutputDict with parsed responses.
        """
        if not prompts:
            return []

        if not isinstance(prompts, list):
            prompts = [prompts]

        # Convert message lists to strings for the base judge method
        # (it will be converted back to messages in the request mapper)
        processed_prompts = []
        for prompt in prompts:
            if isinstance(prompt, list):
                # Keep as list for request mapper
                processed_prompts.append(prompt)
            else:
                processed_prompts.append(str(prompt))

        outputs = super().judge(processed_prompts)
        
        # Parse HuggingFace Router API chat completions response format
        parsed_outputs = []
        for output in outputs:
            raw = output["output_raw"]
            parsed_text = ""
            
            # Handle error responses
            if isinstance(raw, dict) and "error" in raw:
                parsed_text = f"Error: {raw.get('error', 'Unknown error')}"
            elif isinstance(raw, dict):
                # Chat completions format: {"choices": [{"message": {"content": "..."}}]}
                if "choices" in raw and isinstance(raw["choices"], list) and len(raw["choices"]) > 0:
                    choice = raw["choices"][0]
                    if isinstance(choice, dict) and "message" in choice:
                        message = choice["message"]
                        if isinstance(message, dict) and "content" in message:
                            parsed_text = message["content"]
                        else:
                            parsed_text = str(message)
                    else:
                        parsed_text = str(choice)
                # Fallback: try old format with generated_text
                elif "generated_text" in raw:
                    parsed_text = raw["generated_text"]
                else:
                    parsed_text = str(raw)
            elif isinstance(raw, list) and len(raw) > 0:
                # Old format: [{"generated_text": "..."}]
                if isinstance(raw[0], dict):
                    parsed_text = raw[0].get("generated_text", str(raw[0]))
                else:
                    parsed_text = str(raw[0])
            else:
                parsed_text = str(raw) if raw else ""
            
            parsed_outputs.append({
                "output_raw": parsed_text,
                "metadata": output["metadata"],
            })
        
        return parsed_outputs

