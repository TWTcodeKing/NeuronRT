"""Dendrite + Spike descriptors — the compressed connectivity primitives (paper Fig. 4).

A `Dendrite` encodes the fan-out pattern of an incoming spike to local post-synaptic neurons via
two base-address lists (nlist/wlist) + local strides. The per-spike `Spike` carries the global
offsets (go1/go2) and the repeat count that shift/extend the pattern. Algorithm 1
(`decompress.decompress`) turns (Dendrite, Spike) into concrete (neuron_addr, weight_value) pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Dendrite:
    id: int
    nlist: List[int]      # base neuron addresses, len == count (signed relative offsets)
    wlist: List[int]      # base weight addresses, len == count
    count: int
    local_off1: int = 0   # lnOffset: neuron-address stride walked by the repeat loop
    local_off2: int = 0   # lwOffset: weight-address stride walked by the repeat loop
    repeat: int = 1       # repeat-loop bound: # feature maps (conv) / # post-neurons (dense) on this PE

    def __post_init__(self):
        assert len(self.nlist) == self.count, "nlist length must equal count"
        assert len(self.wlist) == self.count, "wlist length must equal count"


@dataclass
class Spike:
    """What a firing pre-synaptic neuron emits into a target dendrite (global offsets only)."""
    pre: int            # source (pre-synaptic) neuron id (for bookkeeping / golden checks)
    dendrite_id: int
    go1: int            # GO1: weight-address global offset (selects input channel / weight column)
    go2: int            # GO2: neuron-address global offset (output position)
