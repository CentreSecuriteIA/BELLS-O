"""Implement the wrapper to acommodate message dictionaries."""

from typing import Literal

from bells_o.preprocessors import PreProcessing


OPPOSITES = {"user": "assistant", "assistant": "user"}


class RoleWrapper(PreProcessing):
    """Implement the wrapper to acommodate message dictionaries.

    Maps from string to message dictionary.

    """

    def __init__(
        self,
        role: Literal["user", "assistant"] = "user",
        system_prompt: str = "",
        opposite_prompt: str | None = None,
        starts_with: Literal["user", "assistant"] = "user",
    ):
        """Initialize the RoleWrapper.

        This pre processor maps a string to the dictionary list template that specified messages in HF.
        You can choose for what role the prompt string should be used (`assistant` or `user`), if/what the other opposite role should start with,
        and a system prompt string.

        Args:
            role (Literal["user", "assistant"], optional): The role of the prompt string. Defaults to "user".
            system_prompt (str, optional): The system prompt string for the conversation. Defaults to an empty string.
            opposite_prompt (str | None, optional): The string for the opposite role than specified in `role`. Defaults to None.
            starts_with (Literal["user", "assistant"], optional): With which role's message the conversation should start with. Defaults to "user".

        """
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
