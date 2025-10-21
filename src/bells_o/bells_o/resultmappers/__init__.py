"""Module structure for mappers."""

from .lakeraguard_mapper import mapper as lakeraguard
from .xguard_mapper import mapper as xguard


__all__ = ["xguard", "lakeraguard"]
