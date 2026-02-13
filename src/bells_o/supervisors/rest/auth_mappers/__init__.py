"""Module structure."""

from .anthropic_mapper import mapper as anthropic
from .auth_bearer_mapper import mapper as auth_bearer
from .aws_mapper import mapper as aws
from .google_api_key_mapper import mapper as google_api_key
from .ocp_apim_subscription_mapper import mapper as ocp_apim_subscription
from .x_tg_api_key_mapper import mapper as x_tg_api


__all__ = ["auth_bearer", "google_api_key", "ocp_apim_subscription", "anthropic", "aws", "x_tg_api"]
