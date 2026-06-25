"""Per-architecture hand-written DAG builders.

We know each family's topology, so we assemble the DAG explicitly (the flat hook list could not
express residual adds, parallel q/k/v, attention matmuls, or token reshapes). Module output
shapes are captured by one dummy forward (batch stripped, so token-wise Linear keeps its N dim);
weights are read straight from the modules.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

from .dag import Dag, Node


def _capture(model: nn.Module, input_hw: Tuple[int, int], in_ch: int) -> Dict[nn.Module, tuple]:
    """module -> (in_shape, out_shape) with the batch dim stripped (keeps the token dim N)."""
    rec: Dict[nn.Module, tuple] = {}

    def hook(m, inp, out):
        rec[m] = (tuple(int(s) for s in inp[0].shape[1:]), tuple(int(s) for s in out.shape[1:]))

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))]
    functional.reset_net(model)
    with torch.no_grad():
        model(torch.zeros(1, in_ch, *input_hw))
    for h in handles:
        h.remove()
    return rec


def _conv_attrs(m: nn.Conv2d) -> dict:
    return {"stride": tuple(int(s) for s in m.stride),
            "padding": tuple(int(p) for p in m.padding),
            "dilation": tuple(int(d) for d in m.dilation),
            "groups": int(m.groups),
            "kh": int(m.kernel_size[0]), "kw": int(m.kernel_size[1])}


def _pool_attrs(m) -> dict:
    if isinstance(m, nn.AdaptiveAvgPool2d):
        return {"pool_type": "adaptive"}
    def _pair(v):
        return tuple(v) if isinstance(v, (tuple, list)) else (int(v), int(v))
    return {"pool_type": type(m).__name__, "kernel": _pair(m.kernel_size),
            "stride": _pair(m.stride if m.stride is not None else m.kernel_size),
            "padding": _pair(m.padding)}


def _out_shape(cap, m, name):
    """Captured output shape, with a clear error if the module never ran in the dummy forward
    (defined-but-unused, or a conditional branch not taken) instead of a bare KeyError."""
    if m not in cap:
        raise KeyError(f"{name}: module {type(m).__name__} produced no shape — it was not "
                       f"executed in the dummy forward (unused / conditional branch?)")
    return cap[m][1]


def _fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Absorb a BatchNorm2d into the preceding conv (inference fold): returns (W', b') with
    W'[f] = (gamma[f]/sqrt(var[f]+eps)) * W[f] and b'[f] = beta[f] - gamma[f]*mean[f]/sqrt(var[f]+eps)
    (+ the conv's own bias if any). The simulator then sees a single conv-with-bias; BN disappears."""
    w = conv.weight.detach().cpu()
    std = torch.sqrt(bn.running_var.detach().cpu() + bn.eps)
    scale = bn.weight.detach().cpu() / std
    w_f = w * scale.reshape(-1, 1, 1, 1)
    b_conv = conv.bias.detach().cpu() if conv.bias is not None else torch.zeros_like(std)
    b_f = (b_conv - bn.running_mean.detach().cpu()) * scale + bn.bias.detach().cpu()
    return w_f, b_f


def _conv(d, m, cap, prev, name):
    d.weights[name] = m.weight.detach().cpu()
    attrs = {**_conv_attrs(m), "mid": id(m)}    # mid: source module id (LIFNode<->node mapping)
    d.add(Node(name, "conv", _out_shape(cap, m, name), [prev], attrs, weight_name=name))
    return name


def _conv_bn(d, conv, bn, cap, prev, name):
    """A conv with its following BatchNorm folded in (weight + per-channel bias)."""
    w_f, b_f = _fold_bn(conv, bn)
    d.weights[name] = w_f
    d.biases[name] = b_f
    attrs = {**_conv_attrs(conv), "mid": id(conv)}
    d.add(Node(name, "conv", _out_shape(cap, conv, name), [prev], attrs, weight_name=name))
    return name


def _dense(d, m, cap, prev, name):
    d.weights[name] = m.weight.detach().cpu()
    if getattr(m, "bias", None) is not None:
        d.biases[name] = m.bias.detach().cpu()
    d.add(Node(name, "dense", _out_shape(cap, m, name), [prev], {"mid": id(m)}, weight_name=name))
    return name


def _pool(d, m, cap, prev, name):
    attrs = _pool_attrs(m)
    attrs["out_hw"] = _out_shape(cap, m, name)[1:]
    d.add(Node(name, "pool", _out_shape(cap, m, name), [prev], attrs))
    return name


def _neuron_params(model):
    lif = next((m for m in model.modules() if isinstance(m, neuron.LIFNode)), None)
    return {"v_threshold": float(getattr(lif, "v_threshold", 1.0)),
            "tau": float(getattr(lif, "tau", 2.0))} if lif else {}


# ---------------------------------------------------------------------------------------------
def build_vgg(model, name, input_hw, in_ch=3, timesteps=4) -> Dag:
    cap = _capture(model, input_hw, in_ch)
    d = Dag(name, input_hw, timesteps, neuron=_neuron_params(model))
    d.add(Node("input", "input", (in_ch, *input_hw)))
    prev, ci, pi = "input", 0, 0
    for m in model.features:               # Sequential: Conv2d / LIFNode / MaxPool2d
        if isinstance(m, nn.Conv2d):
            prev = _conv(d, m, cap, prev, f"conv{ci}"); ci += 1
        elif isinstance(m, nn.MaxPool2d):
            prev = _pool(d, m, cap, prev, f"pool{pi}"); pi += 1
    prev = _pool(d, model.avgpool, cap, prev, "avgpool")
    fi = 0
    for m in model.classifier:             # Sequential: Linear / LIFNode / Dropout
        if isinstance(m, nn.Linear):
            prev = _dense(d, m, cap, prev, f"fc{fi}"); fi += 1
    d.add(Node("output", "output", d.nodes[prev].shape, [prev]))
    d.validate()
    return d


def build_resnet(model, name, input_hw, in_ch=3, timesteps=4) -> Dag:
    cap = _capture(model, input_hw, in_ch)
    d = Dag(name, input_hw, timesteps, neuron=_neuron_params(model))
    d.add(Node("input", "input", (in_ch, *input_hw)))
    prev = _conv_bn(d, model.conv1, model.bn1, cap, "input", "stem_conv")  # stem: conv->bn->sn->pool
    prev = _pool(d, model.maxpool, cap, prev, "stem_pool")
    bi = 0
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for blk in stage:                  # BasicBlock: conv1,conv2 + SEW-ADD residual
            # This builder models the 2-conv BasicBlock with an additive (SEW 'ADD') residual.
            # A Bottleneck (conv3) would be silently dropped, and a non-ADD cnf (AND/IAND) would be
            # miscompiled as an add — so reject anything we don't actually handle.
            if hasattr(blk, "conv3"):
                raise NotImplementedError(f"{name}: Bottleneck block (conv3) unsupported; "
                                          "only 2-conv BasicBlock is modeled")
            cnf = getattr(blk, "cnf", "ADD")
            if cnf not in (None, "ADD"):
                raise NotImplementedError(f"{name}: SEW cnf='{cnf}' unsupported (only additive ADD)")
            x = prev
            c1 = _conv_bn(d, blk.conv1, blk.bn1, cap, x, f"b{bi}_c1")
            c2 = _conv_bn(d, blk.conv2, blk.bn2, cap, c1, f"b{bi}_c2")
            if blk.downsample is not None:  # downsample = Sequential[Conv2d, BatchNorm2d] (+ downsample_sn LIF)
                ds_conv = next(m for m in blk.downsample if isinstance(m, nn.Conv2d))
                ds_bn = next(m for m in blk.downsample if isinstance(m, nn.BatchNorm2d))
                skip = _conv_bn(d, ds_conv, ds_bn, cap, x, f"b{bi}_ds")
            else:
                skip = x
            add = f"b{bi}_add"
            d.add(Node(add, "add", d.nodes[c2].shape, [c2, skip]))  # SEW 'ADD' connect
            prev = add
            bi += 1
    prev = _pool(d, model.avgpool, cap, prev, "avgpool")
    prev = _dense(d, model.fc, cap, prev, "fc")
    d.add(Node("output", "output", d.nodes[prev].shape, [prev]))
    d.validate()
    return d


def build_spikformer(model, name, input_hw, in_ch=3, timesteps=4) -> Dag:
    cap = _capture(model, input_hw, in_ch)
    d = Dag(name, input_hw, timesteps, neuron=_neuron_params(model))
    d.add(Node("input", "input", (in_ch, *input_hw)))
    prev = _conv_bn(d, model.sps.c1, model.sps.bn1, cap, "input", "sps_c1")  # SPS: conv->bn->lif
    prev = _conv_bn(d, model.sps.c2, model.sps.bn2, cap, prev, "sps_c2")
    embed, hh, ww = d.nodes[prev].shape       # (embed, H', W') -> N tokens of dim=embed
    n_tok = hh * ww
    d.add(Node("tokens", "reshape", (n_tok, embed), [prev],
               {"kind": "to_tokens", "from": (embed, hh, ww)}))
    prev = "tokens"
    for bi, blk in enumerate(model.blocks):
        ssa = blk.attn
        q = _dense(d, ssa.q, cap, prev, f"b{bi}_q")     # q,k,v all consume the block input (parallel)
        k = _dense(d, ssa.k, cap, prev, f"b{bi}_k")
        v = _dense(d, ssa.v, cap, prev, f"b{bi}_v")
        heads = int(ssa.heads)
        qk = f"b{bi}_qk"
        d.add(Node(qk, "matmul_qk", (heads, n_tok, n_tok), [q, k],
                   {"heads": heads, "scale": float(ssa.scale)}))
        av = f"b{bi}_av"
        d.add(Node(av, "matmul_av", (n_tok, embed), [qk, v], {"heads": heads}))
        proj = _dense(d, ssa.proj, cap, av, f"b{bi}_proj")
        add1 = f"b{bi}_add1"
        d.add(Node(add1, "add", (n_tok, embed), [prev, proj]))
        fc1 = _dense(d, blk.mlp.fc1, cap, add1, f"b{bi}_fc1")
        fc2 = _dense(d, blk.mlp.fc2, cap, fc1, f"b{bi}_fc2")
        add2 = f"b{bi}_add2"
        d.add(Node(add2, "add", (n_tok, embed), [add1, fc2]))
        prev = add2
    d.add(Node("pool_tokens", "pool", (embed,), [prev], {"pool_type": "token_mean"}))
    prev = _dense(d, model.head, cap, "pool_tokens", "head")
    d.add(Node("output", "output", d.nodes[prev].shape, [prev]))
    d.validate()
    return d


_BUILDERS = {
    "spiking_vgg16": build_vgg,
    "svgg9": build_vgg,
    "svgg_strided": build_vgg,
    "sew_resnet18": build_resnet,
    "sew_resnet34": build_resnet,
    "spikformer": build_spikformer,
}


def build_dag(model, name, input_hw, in_ch=3, timesteps=4) -> Dag:
    if name not in _BUILDERS:
        raise KeyError(f"no DAG builder for '{name}'")
    return _BUILDERS[name](model, name, input_hw, in_ch, timesteps)
