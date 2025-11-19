"""Pipeline base utilities and shared defaults."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, Type

DEFAULT_IMG_SIZE = 640
DEFAULT_ROI_CONF_THRES = 0.25
DEFAULT_VEH_CONF_THRES = 0.01
DEFAULT_PLATE_CONF_THRES = 0.01

DEFAULT_ROI_CLASS_NAME = 'ROI'
DEFAULT_VEHICLE_CLASS_NAMES = ('motobike', 'xedap')
DEFAULT_PLATE_CLASS_NAME = 'bienso'

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]

PIPELINE_REGISTRY: dict[str, type['BasePipeline']] = {}

PipelineRegistrar = Callable[[Type['BasePipeline']], Type['BasePipeline']]


def register_pipeline(name: str) -> PipelineRegistrar:
    def decorator(cls: Type['BasePipeline']) -> Type['BasePipeline']:
        if name in PIPELINE_REGISTRY:
            raise KeyError(f'pipeline {name!r} already registered')
        PIPELINE_REGISTRY[name] = cls
        return cls

    return decorator


class BasePipeline(ABC):
    def __init__(self, model_path: str, img_size: int = DEFAULT_IMG_SIZE) -> None:
        self.model_path = model_path
        self.img_size = img_size

    @abstractmethod
    def __call__(self, imgs: Sequence[Any], expect_plates: Sequence[bool]) -> Any:
        raise NotImplementedError()