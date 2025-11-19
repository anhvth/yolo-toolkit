"""Public helpers for label studio pipeline wrappers."""

from .base import PIPELINE_REGISTRY, register_pipeline
from . import moto_plate  # noqa: F401

__all__ = [
    'PIPELINE_REGISTRY',
    'register_pipeline',
]
