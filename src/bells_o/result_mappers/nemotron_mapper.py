"""Implement the result mapping function for nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 API models."""

import re

from bells_o.common import Result, Usage


def mapper(output: str, usage: Usage = Usage()) -> Result:
    """Map the output format of nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 to `Result`.

    Maps a json string for "User Safety" and "Response Safety" with "unsafe" and "safe as values to True and False.

    E.g.:
    {
       "User Safety": "unsafe",
       "Response Safety": "safe",
       "Safety Categories": "guns and illegal weapons"
    }

    Args:
            output: A string of a JSON representation.
            usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
            Result: A Result object with content_moderation flag set based on parsed output.

    """
    text = output.lower()
    user_responses = re.findall(r'"user safety": "(\w*)"', text)
    assistant_responses = re.findall(r'"response safety": "(\w*)"', text)

    user_response_flag = assistant_response_flag = None
    if user_responses:
        user_response_flag = user_responses[0] == "unsafe"  # there should only be one match
    assert isinstance(user_response_flag, bool)

    if assistant_responses:
        assistant_response_flag = assistant_responses[0] == "unsafe"

    if isinstance(assistant_response_flag, bool):
        return Result(content_moderation=assistant_response_flag or user_response_flag)

    return Result(content_moderation=user_response_flag)
