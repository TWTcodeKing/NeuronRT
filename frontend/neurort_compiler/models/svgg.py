"""SVGG9 — a 9-weight-layer spiking VGG for CIFAR-scale inputs (custom; SpikingJelly ships only
VGG11/13/16/19).

Built from SpikingJelly `layer`/`neuron` so a plain T-step forward IS the SpikingJelly reference.
Structure mirrors SpikingJelly's spiking_vgg attribute layout (`features` / `avgpool` / `classifier`)
so the existing `graph/builders.py:build_vgg` DAG builder consumes it unchanged. **No BatchNorm** —
the compiler captures only conv/linear weights, so a BN-free net lets the compiled weights fully
reproduce the layer math (BN folding is a separate follow-up). Global-avg-pool head keeps it small.

6 conv (64,128,256,256,512,512) + 3 FC (512->512->256->10) = 9 weight layers; 4 maxpools.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron

_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]   # 6 conv, 4 maxpool


class SVGG9(nn.Module):
    def __init__(self, num_classes: int = 10, in_ch: int = 3):
        super().__init__()
        feats: list[nn.Module] = []
        c = in_ch
        for v in _CFG:
            if v == "M":
                feats.append(layer.MaxPool2d(kernel_size=2, stride=2))
            else:
                feats.append(layer.Conv2d(c, v, kernel_size=3, padding=1, bias=False))
                feats.append(neuron.LIFNode())
                c = v
        self.features = nn.Sequential(*feats)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            layer.Linear(512, 512, bias=False), neuron.LIFNode(),
            layer.Linear(512, 256, bias=False), neuron.LIFNode(),
            layer.Linear(256, num_classes, bias=False),    # final layer: analog logits (no LIF)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,H,W] -> [B,num_classes] (one step)
        x = self.features(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.classifier(x)


def svgg9(num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    return SVGG9(num_classes=num_classes, in_ch=in_ch)


# (channels, stride) — downsample via stride-2 convs, NO pooling. The NoC end-to-end sim routes a
# pool as additive connectivity (the dendrite SUMS the window), which equals neither MaxPool (max)
# nor AvgPool (sum/N); a pool-free, stride-downsampled net is exactly representable by additive
# dendrites. Sized so the last conv is 1x1 at 8x8 input, making the final AdaptiveAvgPool a no-op.
_CFG_STRIDED = [(64, 1), (128, 2), (256, 2), (512, 2)]


class SVGGStrided(nn.Module):
    def __init__(self, num_classes: int = 10, in_ch: int = 3):
        super().__init__()
        feats: list[nn.Module] = []
        c = in_ch
        for ch, st in _CFG_STRIDED:
            feats.append(layer.Conv2d(c, ch, kernel_size=3, stride=st, padding=1, bias=False))
            feats.append(neuron.LIFNode())
            c = ch
        self.features = nn.Sequential(*feats)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))   # no-op when the last conv is already 1x1
        self.classifier = nn.Sequential(
            layer.Linear(512, 256, bias=False), neuron.LIFNode(),
            layer.Linear(256, num_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.classifier(x)


def svgg_strided(num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    return SVGGStrided(num_classes=num_classes, in_ch=in_ch)
