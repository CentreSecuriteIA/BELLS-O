"""Implements Abstract Preprocessing class."""

from abc import ABC, abstractmethod


class PreProcessing(ABC):
    """Abstract PreProcessing class that needs to be concretised."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, string: str, *args, **kwargs) -> str:
        """Call `process` function of class.

        Args:
            string (str): Text string to preprocess.
            *args (Any): Any other arguments that concrete classes need.
            **kwargs (Any): Any other keyword arguments that concrete classes need.

        Returns:
            str: Preprocessed string.

        """
        return self.process(string, *args, **kwargs)

    @abstractmethod
    def process(self, string, *args, **kwargs) -> str:
        """Preprocess a given string.

        Args:
            string (str): Text string to preprocess.
            *args (Any): Any other arguments that concrete classes need.
            **kwargs (Any): Any other keyword arguments that concrete classes need.

        Returns:
            str: Preprocessed string.

        """
        pass
