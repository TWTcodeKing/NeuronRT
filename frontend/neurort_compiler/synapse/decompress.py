"""Algorithm 1 (Synapse Decompression Flow) — the golden reference.

This is what the C++ Dendrite module will implement in M2. The compiler (`compress.py`) is its
inverse: it must emit dendrites/spikes such that this loop reconstructs the original connectivity.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .dendrite import Dendrite


def decompress(dend: Dendrite, go1: int, go2: int,
               weight_array: Sequence[float]) -> List[Tuple[int, float]]:
    """Reconstruct (neuron_addr, weight_value) pairs for one spike into `dend`.

    The repeat bound lives on the dendrite (constant per PE: # feature maps / post-neurons);
    the spike carries only the global offsets go1/go2.
    """
    out: List[Tuple[int, float]] = []
    for c in range(dend.count):
        w0 = dend.wlist[c] + go1
        n0 = dend.nlist[c] + go2
        for r in range(dend.repeat):
            n = n0 + dend.local_off1 * r
            w = w0 + dend.local_off2 * r
            out.append((n, float(weight_array[w])))
    return out
