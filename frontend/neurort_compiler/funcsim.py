"""Functional simulator that runs a COMPILED NeuroRT network forward and checks it against
SpikingJelly.

The NeuroRT forward reassembles every layer's weight from the compiled per-PE artifact (the
partition's per-PE weight slices, optionally re-quantized to int8 exactly as `export/writer.py`
does) and runs a T-step LIF inference with an independent LIF that reproduces SpikingJelly's
`LIFNode` (decay-input, hard reset-to-0): `v += (x - v)/tau; spike = (v >= v_th); v *= (1-spike)`.

Two checks:
  * float weights  -> NeuroRT must match SpikingJelly bit-exact (correctness of the compile +
    topology + LIF reimplementation; the compression is lossless).
  * int8 weights   -> report the firing divergence (the deployment quantization error).

Algorithm 1 (decompress) is exercised + checked separately (`decompress_matches_conv`): for sampled
output neurons it reconstructs the exact incoming (pre, weight) set the compiled dendrites encode
and verifies it equals the dense conv/linear connectivity.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .export.writer import quantize_int8
from .graph.dag import Dag, topo
from .mapping.partition import MappedNetwork


# --------------------------------------------------------------------------------------------
class Lif:
    """SpikingJelly LIFNode (tau, v_threshold, v_reset=0, decay_input, hard reset), stateful."""

    def __init__(self, tau: float, v_th: float):
        self.tau = tau
        self.v_th = v_th
        self.v: torch.Tensor | None = None

    def reset(self) -> None:
        self.v = None

    def step(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        spike = (self.v >= self.v_th).to(x.dtype)
        self.v = self.v * (1.0 - spike)            # hard reset to v_reset=0
        return spike


# --------------------------------------------------------------------------------------------
def reassemble_weight(mapped: MappedNetwork, dag: Dag, node: str, quantize: bool) -> torch.Tensor:
    """Rebuild a conv/dense node's weight tensor from its compiled per-PE slices (in PE order =
    output-channel / output-neuron order). `quantize=True` re-quantizes each PE slice to int8 with
    its own scale (exactly as the exporter), surfacing the per-PE quantization the chip deploys."""
    target = tuple(dag.weights[dag.nodes[node].weight_name].shape)  # (F,C,K,K) or (out,in)
    chunks: List[np.ndarray] = []
    for pe in mapped.pes_of_node(node):
        w = np.asarray(pe.weight, dtype=np.float32)
        if quantize:
            q, scale = quantize_int8(w)
            w = q.astype(np.float32) * scale
        chunks.append(w)
    flat = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return torch.from_numpy(flat.reshape(target).copy())


def _pool(node, x: torch.Tensor) -> torch.Tensor:
    a = node.attrs
    if a.get("pool_type") == "adaptive":
        return F.adaptive_avg_pool2d(x, a["out_hw"])
    kh, kw = a["kernel"]
    sh, sw = a["stride"]
    ph, pw = a.get("padding", (0, 0))
    if a["pool_type"] == "MaxPool2d":
        return F.max_pool2d(x, (kh, kw), (sh, sw), (ph, pw))
    return F.avg_pool2d(x, (kh, kw), (sh, sw), (ph, pw))


def nrt_forward(mapped: MappedNetwork, dag: Dag, image: torch.Tensor, timesteps: int,
                quantize: bool) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Run the compiled network T steps (direct/constant input encoding). Returns
    (per-LIF-node firing-rate map, time-averaged output logits)."""
    tau = float(dag.neuron.get("tau", 2.0))
    v_th = float(dag.neuron.get("v_threshold", 1.0))
    weights = {n.name: reassemble_weight(mapped, dag, n.name, quantize)
               for n in dag.nodes.values() if n.op in ("conv", "dense")}
    # Folded BN bias (per output channel), if any. Applied from the DAG (the binary manifest does
    # not yet carry bias — a deployment follow-up); the conv WEIGHT still comes from the compiled PEs.
    biases = {name: torch.as_tensor(b, dtype=torch.float32) for name, b in dag.biases.items()}
    order = topo(dag)
    out_node = next(n for n in order if n.op == "output")
    analog = set(out_node.inputs)                      # final layer feeds logits -> no LIF
    lifs = {n.name: Lif(tau, v_th) for n in order
            if n.op in ("conv", "dense") and n.name not in analog}

    rate = {name: None for name in lifs}
    logits = torch.zeros(1, dag.nodes[out_node.inputs[0]].num_neurons())
    for _ in range(timesteps):
        act: Dict[str, torch.Tensor] = {}
        for n in order:
            if n.op == "input":
                act[n.name] = image
            elif n.op == "conv":
                x = act[n.inputs[0]]
                st, pd = n.attrs["stride"], n.attrs["padding"]
                cur = F.conv2d(x, weights[n.name], stride=st, padding=pd)
                if n.name in biases:
                    cur = cur + biases[n.name].reshape(1, -1, 1, 1)
                act[n.name] = lifs[n.name].step(cur)
                rate[n.name] = act[n.name] if rate[n.name] is None else rate[n.name] + act[n.name]
            elif n.op == "add":
                s = act[n.inputs[0]]
                for p in n.inputs[1:]:
                    s = s + act[p]                          # SEW ADD: elementwise sum of spike maps
                act[n.name] = s
            elif n.op == "dense":
                x = act[n.inputs[0]]
                if x.dim() == 4:                            # conv/pool map -> first FC: flatten
                    x = torch.flatten(x, 1)                 # token-wise dense ([B,N,E]) keeps its dims
                cur = F.linear(x, weights[n.name])
                if n.name in biases:
                    cur = cur + biases[n.name]
                if n.name in analog:
                    act[n.name] = cur
                    logits = logits + cur
                else:
                    act[n.name] = lifs[n.name].step(cur)
                    rate[n.name] = act[n.name] if rate[n.name] is None else rate[n.name] + act[n.name]
            elif n.op == "reshape":                         # to_tokens: [B,C,H,W] -> [B, H*W, C]
                x = act[n.inputs[0]]
                b, c, h, w = x.shape
                act[n.name] = x.flatten(2).transpose(1, 2).contiguous()
            elif n.op == "matmul_qk":                       # spiking attention scores (per head)
                q, k = act[n.inputs[0]], act[n.inputs[1]]
                b, ntok, e = q.shape
                heads = n.attrs["heads"]; dh = e // heads
                qh = q.reshape(b, ntok, heads, dh).transpose(1, 2)
                kh = k.reshape(b, ntok, heads, dh).transpose(1, 2)
                act[n.name] = (qh @ kh.transpose(-2, -1)) * n.attrs["scale"]   # [B,h,N,N]
            elif n.op == "matmul_av":                       # attn @ V -> [B,N,embed]
                attn, v = act[n.inputs[0]], act[n.inputs[1]]
                b, ntok, e = v.shape
                heads = attn.shape[1]; dh = e // heads
                vh = v.reshape(b, ntok, heads, dh).transpose(1, 2)
                act[n.name] = (attn @ vh).transpose(1, 2).reshape(b, ntok, e)
            elif n.op == "pool":
                if n.attrs.get("pool_type") == "token_mean":
                    act[n.name] = act[n.inputs[0]].mean(dim=1)     # mean over tokens -> [B,embed]
                else:
                    act[n.name] = _pool(n, act[n.inputs[0]])
            elif n.op == "output":
                act[n.name] = act[n.inputs[0]]
            else:
                raise NotImplementedError(f"funcsim: op '{n.op}' (node {n.name}) not handled")
    return {k: (v / timesteps) for k, v in rate.items()}, logits / timesteps


# --------------------------------------------------------------------------------------------
def sj_reference(model, image: torch.Tensor, dag: Dag,
                 timesteps: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """SpikingJelly reference: T-step forward capturing each LIFNode's firing rate + the
    time-averaged logits. Each LIFNode is mapped to the DAG weight node that feeds it by EXECUTION
    order (a Conv2d/Linear forward records itself as `last`; the next LIFNode forward claims it) —
    robust to parallel branches (q/k/v) where topo order would not match registration order."""
    from spikingjelly.activation_based import functional, neuron

    out_node = next(n for n in dag.nodes.values() if n.op == "output")
    analog = set(out_node.inputs)
    mid2name = {n.attrs["mid"]: n.name for n in dag.nodes.values()
                if n.op in ("conv", "dense") and n.name not in analog and "mid" in n.attrs}

    acc: Dict[str, torch.Tensor] = {}
    last = {"name": None}

    def weight_hook(m, _inp, _out):
        last["name"] = mid2name.get(id(m))      # None for the analog head (no LIF follows it)

    def lif_hook(m, _inp, out):
        nm = last["name"]
        if nm is not None:
            acc[nm] = out.detach().clone() if nm not in acc else acc[nm] + out.detach()

    handles = [m.register_forward_hook(weight_hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    handles += [m.register_forward_hook(lif_hook) for m in model.modules()
                if isinstance(m, neuron.LIFNode)]
    functional.reset_net(model)
    logits = torch.zeros(1, dag.nodes[out_node.inputs[0]].num_neurons())
    with torch.no_grad():
        for _ in range(timesteps):
            logits = logits + model(image)
    for h in handles:
        h.remove()
    return {k: v / timesteps for k, v in acc.items()}, logits / timesteps


# --------------------------------------------------------------------------------------------
def _conv_synapses_for_pre(weight: np.ndarray, in_hw, stride: int, pad: int, pre: int):
    """Ground-truth conv synapses {(post_global, weight)} feeding from a single input neuron `pre`
    (computed directly, no full-layer enumeration)."""
    f_out, c_in, k, _ = weight.shape
    hi, wi = in_hw
    from .synapse.connect import conv_out_hw
    ho, wo = conv_out_hw(in_hw, k, stride, pad)
    c = pre // (hi * wi)
    rem = pre % (hi * wi)
    ph, pw = rem // wi, rem % wi
    out = set()
    for f in range(f_out):
        for kh in range(k):
            for kw in range(k):
                oh_n, oh_r = divmod(ph + pad - kh, stride)
                ow_n, ow_r = divmod(pw + pad - kw, stride)
                if oh_r or ow_r or not (0 <= oh_n < ho and 0 <= ow_n < wo):
                    continue
                post = (f * ho + oh_n) * wo + ow_n
                out.add((post, round(float(weight[f, c, kh, kw]), 5)))
    return out


def decompress_matches_conv(mapped: MappedNetwork, dag: Dag, node: str, samples: int = 64) -> bool:
    """Algorithm-1 spot check on the real compiled layer: sample input spikes across the conv's PEs,
    decompress each via Algorithm 1, and verify the produced (global_post, weight) synapses equal
    the dense conv's synapses for that `pre` (restricted to the PE's output range). Cheap — only the
    sampled spikes are decompressed, no full-layer enumeration."""
    from .synapse.decompress import decompress

    n = dag.nodes[node]
    weight = np.asarray(dag.weights[n.weight_name].detach().cpu(), dtype=np.float32)
    _, hi, wi = n.attrs["in_chw"]
    stride, pad = n.attrs["stride"][0], n.attrs["padding"][0]
    pl = mapped.placements[node]
    rng = np.random.default_rng(0)
    pes = mapped.pes_of_node(node)
    # The decompress logic is identical across a node's PEs; spot-check the first few PEs only.
    for block, pe in list(enumerate(pes))[:3]:
        if not pe.spikes:
            continue
        idx = rng.choice(len(pe.spikes), size=min(samples, len(pe.spikes)), replace=False)
        # range of GLOBAL post-neuron ids this PE owns (conv: a contiguous feature-map-channel block)
        lo = pl.local_to_global(block, 0)
        hi_g = pl.local_to_global(block, pe.neuron_count - 1)
        dmap = {d.id: d for d in pe.dendrites}
        for i in idx:
            s = pe.spikes[i]
            got = {(pl.local_to_global(block, int(nloc)), round(float(wv), 5))
                   for nloc, wv in decompress(dmap[s.dendrite_id], s.go1, s.go2, pe.weight)}
            want = {(p, w) for (p, w) in _conv_synapses_for_pre(weight, (hi, wi), stride, pad, s.pre)
                    if lo <= p <= hi_g}
            if got != want:
                return False
    return True
