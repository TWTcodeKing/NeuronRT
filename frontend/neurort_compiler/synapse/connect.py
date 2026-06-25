"""Ground-truth connectivity enumeration (the reference the compressor must reproduce).

A synapse is a triple (post_neuron_id, pre_neuron_id, weight_value). Neuron ids are flattened:
conv layout (f, h, w) -> (f*H + h)*W + w; dense post = row index, pre = column index.
Weights are rounded to 5 decimals so float comparison is exact for the golden tests.
"""
from __future__ import annotations

from typing import Set, Tuple

import numpy as np

Synapse = Tuple[int, int, float]


def _r(x: float) -> float:
    return round(float(x), 5)


def enumerate_dense(weight: np.ndarray) -> Set[Synapse]:
    """Dense layer weight (N, L): post i receives weight[i, j] from pre j."""
    n, l = weight.shape
    return {(i, j, _r(weight[i, j])) for i in range(n) for j in range(l)}


def conv_out_hw(in_hw: Tuple[int, int], k: int, stride: int, pad: int) -> Tuple[int, int]:
    hi, wi = in_hw
    ho = (hi + 2 * pad - k) // stride + 1
    wo = (wi + 2 * pad - k) // stride + 1
    return ho, wo


def enumerate_conv(weight: np.ndarray, in_hw: Tuple[int, int], stride: int = 1,
                   pad: int = 1) -> Tuple[Set[Synapse], Tuple[int, int]]:
    """Conv weight (F, C, K, K) over input (C, in_hw): standard cross-correlation connectivity."""
    f_out, c_in, k, k2 = weight.shape
    assert k == k2, "only square kernels"
    hi, wi = in_hw
    ho, wo = conv_out_hw(in_hw, k, stride, pad)
    syn: Set[Synapse] = set()
    for f in range(f_out):
        for oh in range(ho):
            for ow in range(wo):
                post = (f * ho + oh) * wo + ow
                for c in range(c_in):
                    for kh in range(k):
                        for kw in range(k):
                            ih = oh * stride - pad + kh
                            iw = ow * stride - pad + kw
                            if 0 <= ih < hi and 0 <= iw < wi:
                                pre = (c * hi + ih) * wi + iw
                                syn.add((post, pre, _r(weight[f, c, kh, kw])))
    return syn, (ho, wo)
