from enum import Enum
from time import time
from typing import Any

from bells_o.common import OutputDict, Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import protectai

from ..custom_supervisor import CustomSupervisor


class ProtectAiLlmGuard(CustomSupervisor):
    def __init__(self, pre_processing: list[PreProcessing] = [], **supervisor_kwargs):
        try:
            from llm_guard.input_scanners.prompt_injection import MatchType
            from llm_guard.util import configure_logger

            configure_logger("ERROR")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "This setup requires additional dependencies. The following required module is missing: ['llm-guard']. Please install it with `pip install bells_o[llm-guard]`."
            )
        custom_kwargs = {"threshold": 0.85, "match_type": MatchType.FULL, "use_onnx": True}

        custom_kwargs |= supervisor_kwargs

        super().__init__(
            name="ProtectAI LLM Guard",
            usage=Usage("jailbreak"),
            res_map_fn=protectai,
            pre_processing=pre_processing,
            provider_name="Protect AI",
            **custom_kwargs,
        )

    def _load_supervisor(self):
        from llm_guard.input_scanners import PromptInjection

        self.scanner = PromptInjection(**self.supervisor_kwargs)

    def _judge_sample(self, prompt: str) -> OutputDict:
        start = time()
        output = self.scanner.scan(prompt)
        generation_time = time() - start

        return OutputDict(
            output_raw=output,
            metadata={
                "latency": generation_time,
                "batch_size": 1,
                "input_tokens": len(prompt) / 4,
                "output_tokens": 1,
            },
        )

    def judge(self, inputs: list[str] | str) -> list[OutputDict]:
        if not isinstance(inputs, list):
            inputs = [inputs]

        return [self._judge_sample(inp) for inp in inputs]

    def metadata(self) -> dict[str, Any]:
        """Return metadata dictionary for this Supervisor.

        Returns:
            dict: Dictionary with metadata.

        """
        metadata = super().metadata()
        if self.supervisor_kwargs is not None:
            metadata["supervisor_kwargs"] = self.supervisor_kwargs
            if not isinstance(metadata["supervisor_kwargs"]["match_type"], str):
                if isinstance(metadata["supervisor_kwargs"]["match_type"], Enum):
                    metadata["supervisor_kwargs"]["match_type"] = self.supervisor_kwargs["match_type"].value
                else:
                    raise ValueError(
                        f"Expected Enum or String, but got {type(metadata['supervisor_kwargs']['match_type'])} instead"
                    )
        return metadata
