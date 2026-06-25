"""Model registry: name -> nn.Module factory. Spikformer is registered in C1."""
from __future__ import annotations

from typing import Callable, Dict, List

import torch.nn as nn

from . import dsnn
from .spikformer import spikformer
from .svgg import svgg9, svgg_strided

_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "sew_resnet18": dsnn.sew_resnet18,
    "sew_resnet34": dsnn.sew_resnet34,
    "spiking_vgg16": dsnn.spiking_vgg16,
    "spikformer": spikformer,
    "svgg9": svgg9,
    "svgg_strided": svgg_strided,
}


def register(name: str, builder: Callable[..., nn.Module]) -> None:
    _BUILDERS[name] = builder


def available_models() -> List[str]:
    return sorted(_BUILDERS)


def build_model(name: str, **cfg) -> nn.Module:
    if name not in _BUILDERS:
        raise KeyError(f"unknown model '{name}'; available: {available_models()}")
    model = _BUILDERS[name](**cfg)
    model.eval()
    return model
