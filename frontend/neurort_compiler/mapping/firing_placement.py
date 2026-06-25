"""Firing-rate-aware placement (paper Section C: the compiler "assigns feature maps with highly
dissimilar firing rates to the same PE" / "segregates high-firing neurons into different clusters").

The simulator's DNP gives each PE n_phys = ratio*count physical slots and allocates them in local-id
order; with ~100% allocation it must drop (1-ratio) of every PE's neurons. If hot channels sit
together (contiguous channel partition), some PEs are oversubscribed and reject ACTIVE neurons ->
accuracy collapses. We fix this purely at compile time by PERMUTING each weight layer's output
channels so that (a) each PE's contiguous block holds a firing-balanced mix (round-robin over
rate-sorted channels) and (b) within a block the channels are active-first (so active neurons get
the low local ids that allocate first). The permutation is propagated into the next layer's input
dim, so the network function is unchanged (a relabeling of intermediate channels).
"""
from __future__ import annotations

import numpy as np


def channel_perm(firing: np.ndarray, block_sizes) -> np.ndarray:
    """Return `perm` with perm[new_position] = old_channel. Sort each PE's CONTIGUOUS channel block
    active-first (descending firing), WITHOUT moving channels across PEs. `block_sizes` is the ACTUAL
    per-PE channel count from the partition (sum == C) — reordering within those exact blocks keeps
    each PE's channel SET (so int8 per-PE scale + the function are unchanged) while putting the PE's
    most-active channels at the low local ids the sim's id-order DNP allocation keeps."""
    firing = np.asarray(firing, dtype=np.float64)
    perm = np.arange(len(firing), dtype=np.int64)
    off = 0
    for sz in block_sizes:
        lo, hi = off, off + int(sz)
        block = np.arange(lo, hi)
        perm[lo:hi] = block[np.argsort(firing[lo:hi])[::-1]]   # active-first within this PE's block
        off = hi
    return perm


def apply_firing_placement(weight_layers, chan_firing, block_sizes) -> list[np.ndarray]:
    """In-place, function-preserving. `weight_layers` = the ordered conv/linear modules; the last one
    (the analog logit layer) is NOT permuted. For each earlier layer i, permute its OUTPUT channels
    (weight dim 0) active-first within each PE block and the NEXT layer's INPUT channels (weight dim 1)
    by the same permutation. `chan_firing[i]` = layer i's per-output-channel firing; `block_sizes[i]` =
    layer i's per-PE channel counts (from the partition).

    Works for conv->conv (both weight [out,in,kh,kw]), conv->linear via a channel-preserving
    GAP+flatten (linear weight [out, in==channels]), and linear->linear. Assumes no bias coupling
    that needs reindexing on the INPUT side (svgg9 is bias-free); a permuted layer's own bias (output
    side) would also be gathered by `perm` if present.
    """
    import torch

    perms = []
    for i in range(len(weight_layers) - 1):
        perm_np = channel_perm(chan_firing[i], block_sizes[i])
        perm = torch.as_tensor(perm_np, dtype=torch.long)
        perms.append(perm_np)
        cur, nxt = weight_layers[i], weight_layers[i + 1]
        with torch.no_grad():
            cur.weight.data = cur.weight.data[perm].contiguous()          # output channels
            if getattr(cur, "bias", None) is not None:
                cur.bias.data = cur.bias.data[perm].contiguous()
            nxt.weight.data = nxt.weight.data[:, perm].contiguous()       # next layer's input channels
    return perms
