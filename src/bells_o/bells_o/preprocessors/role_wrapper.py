from typing import Literal

from .preprocessing import PreProcessing


OPPOSITES = {"user": "assistant", "assistant": "user"}


class RoleWrapper(PreProcessing):
    def __init__(
        self,
        role: Literal["user", "assistant"] = "user",
        system_prompt: str | None = None,
        opposite_prompt: str | None = None,
        starts_with: Literal["user", "assistant"] = "user",
    ):
        self.role = role
        self.system_prompt = system_prompt
        self.opposite_prompt = opposite_prompt
        self.starts_with = starts_with

    def process(self, string) -> list[dict[str, str]]:
        """Wrap a string to conform to the dictionary list template necessary for IT models on HF.

        Args:
            string (str): Text string to preprocess.

        Returns:
            list: Wrapped string.

        """
        print(f"DEBUG: {string}, {type(string)}")
        message_list = []
        if self.system_prompt is not None:
            message_list.append({"role": "system", "content": self.system_prompt})
        if self.starts_with != self.role:
            assert self.opposite_prompt, (
                "If the first turn is supposed to be the other role, `opposite_prompt` has to be specified."
            )
            message_list.append({"role": self.starts_with, "content": self.opposite_prompt})
            message_list.append({"role": self.role, "content": string})
        else:
            message_list.append({"role": self.role, "content": string})
            if self.opposite_prompt:
                message_list.append({"role": OPPOSITES[self.role], "content": self.opposite_prompt})

        return message_list
