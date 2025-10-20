"""Module structure for mappers."""

from .lakeraguard_mapper import mapper as lakeraguard_mapper
from .xguard_mapper import mapper as xguard_mapper


__all__ = ["xguard_mapper", "lakeraguard_mapper"]
