"""Type hinting for auto_endpoint.py."""

from typing import Literal, overload

from bells_o import PreProcessing, Usage

from .custom_endpoint import RestSupervisor
from .lakeraguard import LakeraGuardDefaultSupervisor, LakeraGuardSupervisor

# TODO: think about using NewType to have the supervisors be different types
class AutoRestSupervisor:
    @overload
    @classmethod
    def load(
        cls,
        endpoint_name: Literal["lakeraguard-default"],
        project_id: str,
        pre_processing: list[PreProcessing] = ...,
        api_key: str | None = ...,
        api_variable: str | None = ...,
    ) -> LakeraGuardDefaultSupervisor: ...
    @overload
    @classmethod
    def load(
        cls,
        endpoint_name: Literal["lakeraguard"],
        project_id: str,
        usage: Usage,
        pre_processing: list[PreProcessing] = ...,
        api_key: str | None = ...,
        api_variable: str | None = ...,
    ) -> LakeraGuardSupervisor: ...
    @overload
    @classmethod
    def load(cls, endpoint_name: str, *args, **kwargs) -> RestSupervisor: ...
