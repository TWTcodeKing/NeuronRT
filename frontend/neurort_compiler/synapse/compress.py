"""Synapse compression: weight tensor -> (dendrites, spikes, dedup'd weight array).

The compression that matters (paper bytes/synapse = 0.05): conv kernels are stored ONCE
(F*C*K*K) and shared across all H*W output positions via per-position global offsets; dendrites
are shared across pre-neurons that have the same valid-entry pattern (interior vs border, and the
stride phase). The result is the inverse of Algorithm 1 (`decompress`): `reconstruct` must equal
the ground-truth `connect.enumerate_*`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from .connect import Synapse, conv_out_hw
from .decompress import decompress
from .dendrite import Dendrite, Spike


@dataclass
class CompiledLayer:
    kind: str                       # 'conv' | 'dense'
    dendrites: List[Dendrite]       # id-assigned, contiguous from 0
    spikes: List[Spike]             # one per pre-synaptic neuron (-> dendrite + global offsets)
    weight_array: np.ndarray        # dedup'd weights (this layer's slice of the weight blob)
    num_post: int                   # post-synaptic neuron count
    num_synapses: int               # logical (uncompressed) synapse count
    out_hw: Optional[Tuple[int, int]] = None


def compress_dense(weight: np.ndarray, n_tokens: int = 1) -> CompiledLayer:
    """Dense weight slice (W_out, L) applied independently to `n_tokens` tokens.

    ONE dendrite of count 1; the W_out outputs of a token are walked by the repeat loop (lnOffset=1
    over neurons, lwOffset=L over weight rows). The SAME weight slice is shared across all tokens,
    so a token-wise Linear is partitioned along its OUTPUT dim (this slice = W[o0:o1, :]) rather
    than replicating the matrix per token. Input neuron (t, i) fires with go1=i (weight column) and
    go2 = t*W_out (the PE-local output base of token t). Reduces to the plain dense layer at
    n_tokens=1. Local post-neuron id = t*W_out + r, i.e. token-major within this PE's slice."""
    w_out, l = weight.shape
    warr = np.ascontiguousarray(weight, dtype=np.float32).reshape(-1)  # row-major: warr[r*L+i]=W[r,i]
    dend = Dendrite(id=0, nlist=[0], wlist=[0], count=1, local_off1=1, local_off2=l, repeat=w_out)
    spikes = [Spike(pre=t * l + i, dendrite_id=0, go1=i, go2=t * w_out)
              for t in range(n_tokens) for i in range(l)]
    return CompiledLayer("dense", [dend], spikes, warr, num_post=n_tokens * w_out,
                         num_synapses=n_tokens * w_out * l)


def compress_conv(weight: np.ndarray, in_hw: Tuple[int, int], stride: int = 1,
                  pad: int = 1) -> CompiledLayer:
    """Conv (F, C, K, K): kernel stored once; dendrites shared by (channel, stride-phase, pattern).

    For a pre-neuron (c, hi, wi), output (f, ho, wo) with ho=(hi+pad-kh)/stride (must be integral
    and in range). Grouping by phase ((hi+pad)%stride) makes ho affine in the entry index, so the
    fan-out splits into a position-independent dendrite (relative nlist/wlist) + a per-position
    global neuron offset go2.
    """
    f_out, c_in, k, _ = weight.shape
    hi_n, wi_n = in_hw
    ho_n, wo_n = conv_out_hw(in_hw, k, stride, pad)
    warr = np.ascontiguousarray(weight, dtype=np.float32).reshape(-1)

    # A dendrite stores only ONE feature map's valid-tap list (<= K^2 entries) plus local offsets;
    # the F feature maps are walked by the repeat loop (lnOffset = Ho*Wo neurons, lwOffset = C*K*K
    # weights), and the C input channels are folded into GO1 = c*K*K on the spike. So the dendrite
    # table is O(K^2) entries regardless of F or C ("constant dendrite count"), and Algorithm 1's
    # repeat loop reconstructs all F*C synapses. Only stride phase + valid-tap pattern distinguish
    # dendrites (interior vs border).
    dmap = {}  # (ph_h, ph_w, valid-pattern) -> Dendrite
    spikes: List[Spike] = []
    for c in range(c_in):
        for hi in range(hi_n):
            for wi in range(wi_n):
                ph_h, ph_w = (hi + pad) % stride, (wi + pad) % stride
                valid = [(kh, kw)
                         for kh in range(k) for kw in range(k)
                         if kh % stride == ph_h and kw % stride == ph_w
                         and 0 <= (hi + pad - kh) // stride < ho_n
                         and 0 <= (wi + pad - kw) // stride < wo_n]
                if not valid:
                    continue
                key = (ph_h, ph_w, tuple(valid))
                dend = dmap.get(key)
                if dend is None:
                    nlist, wlist = [], []
                    for (kh, kw) in valid:   # f=0 only; the repeat loop walks the F feature maps
                        nlist.append(-((kh - ph_h) // stride) * wo_n - ((kw - ph_w) // stride))
                        wlist.append(kh * k + kw)   # widx(f=0, c=0, kh, kw)
                    dend = Dendrite(id=len(dmap), nlist=nlist, wlist=wlist, count=len(nlist),
                                    local_off1=ho_n * wo_n, local_off2=c_in * k * k, repeat=f_out)
                    dmap[key] = dend
                go1 = c * k * k  # weight global offset: select input channel c's kernel slice
                go2 = ((hi + pad) // stride) * wo_n + ((wi + pad) // stride)
                spikes.append(Spike(pre=(c * hi_n + hi) * wi_n + wi,
                                    dendrite_id=dend.id, go1=go1, go2=go2))

    dends = list(dmap.values())
    by_id = {d.id: d for d in dends}
    nsyn = sum(by_id[s.dendrite_id].count * by_id[s.dendrite_id].repeat for s in spikes)
    return CompiledLayer("conv", dends, spikes, warr, num_post=f_out * ho_n * wo_n,
                         num_synapses=nsyn, out_hw=(ho_n, wo_n))


def reconstruct(cl: CompiledLayer) -> Set[Synapse]:
    """Apply Algorithm 1 to every spike -> the (post, pre, weight) set (golden equality check)."""
    by_id = {d.id: d for d in cl.dendrites}
    recon: Set[Synapse] = set()
    for s in cl.spikes:
        for neuron, wval in decompress(by_id[s.dendrite_id], s.go1, s.go2, cl.weight_array):
            recon.add((int(neuron), s.pre, round(float(wval), 5)))
    return recon


def metrics(cl: CompiledLayer, weight_bits: int = 8) -> dict:
    """Storage metrics vs the paper's targets (bytes/neuron ~11, bytes/synapse ~0.05)."""
    weight_bytes = cl.weight_array.size * (weight_bits / 8.0)
    # dendrite header ~ 10 B (nlist/wlist base 16b each, count 8b, 2 local offsets 12b, repeat 16b);
    # each entry (nlist+wlist) = 4 B. With the repeat encoding count is O(K^2), not F*K^2.
    dend_bytes = sum(10 + d.count * 4 for d in cl.dendrites)
    # axon/spike entry ~ (dendrite_id 8 + dst_pe 12 + delay 4 + go1 12 + go2 12) bits = 6 B.
    axon_bytes = len(cl.spikes) * 6
    meta_bytes = dend_bytes + axon_bytes
    return {
        "weight_bytes": weight_bytes,
        "meta_bytes": meta_bytes,
        "total_bytes": weight_bytes + meta_bytes,
        "num_dendrites": len(cl.dendrites),
        "num_spikes": len(cl.spikes),
        "bytes_per_synapse": weight_bytes / max(cl.num_synapses, 1),
        "bytes_per_neuron": meta_bytes / max(cl.num_post, 1),
    }
