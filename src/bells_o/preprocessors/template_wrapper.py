"""Implement the wrapper to acommodate message dictionaries."""

from bells_o.preprocessors import PreProcessing


OPPOSITES = {"user": "assistant", "assistant": "user"}


class TemplateWrapper(PreProcessing):
    """Implement a preprocessor that wraps a prompt in a template.

    It is essentially a `PreProcessing`-wrapper class for formatting of this nature: `"foo {prompt} foo".format(prompt=sample_prompt)`.

    Make sure to have the `{prompt}` label in your template string.
    ```


    """

    def __init__(
        self,
        template: str,
    ):
        """Initialize the TemplateWrapper.

        This pre processor fills out `{prompt}` label in a template.

        Args:
            template (str): The template string that includes the `{prompt}` label.

        """
        self.template = template

    def process(self, string) -> str:
        """Wrap fill in the `{prompt}` label for a template in .

        Args:
            string (str): Text string to fill into template.

        Returns:
            list: Filled in template string.

        """
        return self.template.format(prompt=string)
