"""Module structure."""

from .auth_bearer_mapper import mapper as auth_bearer
from .ocp_apim_subscription_mapper import mapper as ocp_apim_subscription


__all__ = ["auth_bearer", "ocp_apim_subscription"]
