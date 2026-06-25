"""D-SNN backbones from SpikingJelly (SEW-ResNet, SpikingVGG).

Thin wrappers that pin the spiking neuron to LIF (the paper's D-SNN neuron) and expose a small,
uniform builder signature. The heavy lifting (layer definitions) is reused from SpikingJelly.
"""
from __future__ import annotations

import torch.nn as nn
from spikingjelly.activation_based import layer, neuron
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg


def sew_resnet18(num_classes: int = 10, cnf: str = "ADD") -> nn.Module:
    return sew_resnet.sew_resnet18(spiking_neuron=neuron.LIFNode, cnf=cnf, num_classes=num_classes)


def sew_resnet34(num_classes: int = 10, cnf: str = "ADD") -> nn.Module:
    return sew_resnet.sew_resnet34(spiking_neuron=neuron.LIFNode, cnf=cnf, num_classes=num_classes)


def spiking_vgg16(num_classes: int = 10) -> nn.Module:
    """SpikingJelly's VGG16 with a global-average-pooling head instead of the ImageNet
    AdaptiveAvgPool2d((7,7)) + Linear(512*7*7, 4096). On small inputs (e.g. 32x32 the conv stack
    already collapses to 512x1x1) the 7x7 pool UPSAMPLES, inflating the classifier fan-in 49x
    (512 -> 25088) — pure padding that blows fc0 up to ~2048 PEs. A 1x1 pool + Linear(512, 4096)
    reflects the real 512-d feature, letting VGG16 fit one 576-PE chip."""
    m = spiking_vgg.spiking_vgg16(spiking_neuron=neuron.LIFNode, num_classes=num_classes)
    m.avgpool = layer.AdaptiveAvgPool2d((1, 1))
    old = m.classifier[0]                       # Linear(512*7*7, 4096)
    head = layer.Linear(512, old.out_features, bias=old.bias is not None)
    head.step_mode = old.step_mode
    m.classifier[0] = head
    return m
