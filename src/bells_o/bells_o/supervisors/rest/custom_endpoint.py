"""Implement the base class for REST-accessible supersivors."""

from ..supervisor import Supervisor


class RestSupervisor(Supervisor):
    def __post_init__(self):
        """Set up the rest of the supervisor. E.g. load the model from HuggingFace."""
        raise NotImplementedError
        pass

    def judge():
        """Run one evaluation on the supervisor.

        Similar to `forward` in PyTorch, it expects prepped inputs s.t.
        they can be used directly with the supervisor.
        """
        pass
