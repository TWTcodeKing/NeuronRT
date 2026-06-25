"""DAG partition + map to PEs (paper Sec. IV-C).

Every neuron-producing node of the DAG (conv / dense / attention matmul) owns a contiguous block
of neuron ids [0, num_neurons) and is split across PEs:

  * conv  — keep each (H,W) feature map whole, split along F (output channels). Kernels (C,K,K)
            stored once; feature maps per PE bounded by 64KB (shared kernels + 2x ping-pong states).
  * dense — token-wise Linear of shape (N, out): split along the OUTPUT dim (this PE holds all N
            tokens for an output slice). The shared (out,in) weight is split, not replicated.
  * matmul_qk — attention scores (heads,N,N): split along whole HEADS. Unit-weight fan dendrites
            encode connectivity only (the spike x spike product is the M2 Dendrite/Soma's job).
  * matmul_av — attention.V (N,embed): one PE (the qk-fan and v-fan want different contiguity, so
            both are satisfiable only when the whole block is on a single PE).

`Placement.locate(global_neuron)` maps a node's global neuron id -> (PE index, PE-local id); the
router uses it to attach axon entries on the producer PE. `route_dag` wires the whole graph:
conv/dense input spikes are back-traced with `input_sources` (handles residual adds / reshapes /
token pooling), and the attention matmuls get a forward fan pass (q/k -> qk, qk/v -> av).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..graph.dag import Dag, Node, topo
from ..synapse.compress import CompiledLayer, compress_conv, compress_dense, reconstruct
from ..synapse.connect import Synapse, conv_out_hw
from ..synapse.dendrite import Dendrite, Spike
from .dag_route import input_sources

# Hardware limits (paper Sec. IV / flit.hpp).
SRAM_BYTES = 64 * 1024
NUM_PE = 576
MESH_W = 24
COUNT_MAX = 255      # dendrite entry-count field is 8 bits
DENDRITE_MAX = 256   # dendrite_id is 8 bits

SOURCE_OPS = ("conv", "dense", "matmul_qk", "matmul_av")   # nodes that own PEs


_COMBINE_SUM, _COMBINE_MAX, _COMBINE_AVG = 0, 1, 2


def pack_meta(term_id: int, combine: int, avg_n: int = 1) -> int:
    """Per-edge combine descriptor carried in the spike flit: how the CONSUMER folds arrivals from
    this source term. combine 0=SUM / 1=MAX(OR) / 2=AVG(/avg_n). term_id separates an `add`'s
    branches so a MaxPool skip ORs independently of the main SUM term."""
    return ((min(avg_n, 255) & 0xFF) << 8) | ((term_id & 0x3F) << 2) | (combine & 0x3)


@dataclass
class AxonEntry:
    dst_pe: int
    dendrite_id: int
    go1: int
    go2: int
    delay: int = 1
    meta: int = 0               # pack_meta(term_id, combine, avg_n)


MAX_AXON_LEVELS = 3        # nested-loop depth of a compressed axon group (channel x row x col)


@dataclass
class AxonGroup:
    """Compressed axon-table header: one fan-out pattern shared across a NESTED lattice of source
    neurons (the inverse of the dendrite repeat-loop, generalized to multiple dimensions).

    `levels` is innermost-first, each a (count, src_stride, go1_stride, go2_stride). A source neuron
    fires for every combination (r_0..r_{L-1}); it lives at local id `src_base + Σ r_k*src_stride_k`
    and emits a spike to (dst_pe, dendrite_id) with offsets (go1_base + Σ r_k*go1_stride_k,
    go2_base + Σ r_k*go2_stride_k). 1-D (one level) folds a conv feature-map row / token run; 2-D
    adds the row dimension; 3-D adds the channel dimension (go1 advancing per channel) — so a whole
    conv feature map collapses into one header, keeping the axon table O(neurons), not O(synapses)."""
    src_base: int
    go1_base: int
    go2_base: int
    levels: List[Tuple[int, int, int, int]]   # (count, src_stride, go1_stride, go2_stride), inner-first
    dst_pe: int
    dendrite_id: int
    delay: int
    meta: int = 0               # per-edge combine descriptor (pack_meta), shared by the group

    def total(self) -> int:
        t = 1
        for c, *_ in self.levels:
            t *= c
        return t


@dataclass
class PeImage:
    pe_id: int
    coord: Tuple[int, int]
    node: str                   # producing DAG node name
    kind: str
    neuron_base: int            # global (within-node) id of this PE's first post-neuron
    neuron_count: int
    dendrites: List[Dendrite]
    spikes: List[Spike]         # incoming spikes (pre -> dendrite); pre is the node-input id
    weight: np.ndarray          # this PE's weight slice
    budget_bytes: int
    layer: int = -1             # topo ordinal (manifest bookkeeping)
    axons: Dict[int, List[AxonEntry]] = field(default_factory=dict)  # local out-neuron -> entries (flat)
    axon_groups: List[AxonGroup] = field(default_factory=list)       # compressed axon table (export)


@dataclass
class Placement:
    """How a node's global neuron ids are distributed over a contiguous run of PEs."""
    node: str
    kind: str
    pe_start: int               # index into MappedNetwork.pes of this node's first PE
    num_pe: int
    params: Dict                # kind-specific geometry used by locate()

    def locate(self, gid: int) -> Tuple[int, int]:
        """global neuron id -> (PE index in MappedNetwork.pes, PE-local neuron id)."""
        p = self.params
        if self.kind == "conv":
            ho_wo, m = p["ho_wo"], p["m"]
            f = gid // ho_wo
            blk = f // m
            return self.pe_start + blk, gid - (blk * m) * ho_wo
        if self.kind == "dense":
            out, l = p["out"], p["l"]
            t, o = divmod(gid, out)
            blk = o // l
            o0 = blk * l
            width = min(l, out - o0)
            return self.pe_start + blk, t * width + (o - o0)
        if self.kind == "matmul_qk":
            nn, hpp = p["nn"], p["hpp"]      # nn = N*N (neurons per head)
            h = gid // nn
            blk = h // hpp
            return self.pe_start + blk, gid - (blk * hpp) * nn
        if self.kind == "matmul_av":
            return self.pe_start, gid          # single PE
        raise ValueError(f"locate: unknown kind {self.kind}")

    def local_to_global(self, pe_block: int, local: int) -> int:
        """Inverse of locate for a PE at block offset `pe_block` within this node (golden check)."""
        p = self.params
        if self.kind == "conv":
            return pe_block * p["m"] * p["ho_wo"] + local
        if self.kind == "dense":
            out, l = p["out"], p["l"]
            o0 = pe_block * l
            width = min(l, out - o0)
            t, o_rel = divmod(local, width)
            return t * out + o0 + o_rel
        raise ValueError(f"local_to_global unsupported for {self.kind}")


@dataclass
class MappedNetwork:
    model_name: str
    pes: List[PeImage]
    placements: Dict[str, Placement]              # node name -> Placement
    num_pe_used: int
    skipped_nodes: List[Tuple[str, str]]          # (node, op) not partitioned (none, in practice)
    neuron: Dict = field(default_factory=dict)    # LIF params (tau, v_threshold) for the Soma

    def pes_of_node(self, node: str) -> List[PeImage]:
        pl = self.placements[node]
        return self.pes[pl.pe_start:pl.pe_start + pl.num_pe]


def _to_np(weight) -> np.ndarray:
    if hasattr(weight, "detach"):
        return weight.detach().cpu().numpy()
    return np.asarray(weight)


def _coord(pe_id: int) -> Tuple[int, int]:
    return (pe_id % MESH_W, pe_id // MESH_W)


# --------------------------------------------------------------------------------------------
# Per-node budget formulas (Eq. 5-7, extended to the token dim).
def conv_maps_per_pe(c_in: int, k: int, ho: int, wo: int,
                     weight_bytes: int = 1, state_bytes: int = 1) -> int:
    """Feature maps per PE: shared kernels (C*K*K) + 2x ping-pong states per map. The repeat-loop
    keeps the dendrite count O(K^2) regardless of m, so m is bounded only by SRAM."""
    per_map_cost = c_in * k * k * weight_bytes + 2 * ho * wo * state_bytes
    return max(1, SRAM_BYTES // per_map_cost)


def dense_neurons_per_pe(fan_in: int, n_tokens: int = 1,
                         weight_bytes: int = 1, state_bytes: int = 1) -> int:
    """Output neurons per PE for a token-wise dense: the shared (slice) weight row costs `fan_in`
    bytes and is reused by all tokens, while each output carries 2*n_tokens ping-pong states.
    l = floor(64KB / (fan_in + 2*n_tokens)); reduces to Eq.7 at n_tokens=1."""
    return max(1, SRAM_BYTES // (fan_in * weight_bytes + 2 * n_tokens * state_bytes))


# --------------------------------------------------------------------------------------------
def _conv_geom(node: Node, weight: np.ndarray):
    f_out, c_in, k, _ = weight.shape
    _, h_in, w_in = _input_chw(node)
    st = node.attrs.get("stride", (1, 1))
    pd = node.attrs.get("padding", (0, 0))
    if st[0] != st[1] or pd[0] != pd[1]:
        raise NotImplementedError(f"{node.name}: asymmetric stride/padding {st}/{pd} unsupported")
    if node.attrs.get("groups", 1) != 1:
        raise NotImplementedError(f"{node.name}: grouped conv (groups={node.attrs['groups']}) unsupported")
    if tuple(node.attrs.get("dilation", (1, 1))) != (1, 1):
        raise NotImplementedError(f"{node.name}: dilated conv {node.attrs['dilation']} unsupported")
    return f_out, c_in, k, h_in, w_in, st[0], pd[0]


def _input_chw(node: Node) -> Tuple[int, int, int]:
    """The conv's input (C,H,W) is recoverable from its output shape + attrs are not enough, so the
    builder stores it: we read it back from attrs set in partition (see _partition_conv caller)."""
    return node.attrs["in_chw"]


def _partition_conv(node: Node, weight: np.ndarray, pe_id0: int) -> Tuple[List[PeImage], Placement]:
    f_out, c_in, k, h_in, w_in, stride, pad = _conv_geom(node, weight)
    ho, wo = conv_out_hw((h_in, w_in), k, stride, pad)
    if (f_out, ho, wo) != tuple(node.shape):
        raise ValueError(f"{node.name}: computed conv output {(f_out, ho, wo)} != node shape "
                         f"{tuple(node.shape)} (ceil_mode / output_padding not modeled)")
    m = conv_maps_per_pe(c_in, k, ho, wo)
    images, pe_id = [], pe_id0
    for f0 in range(0, f_out, m):
        f1 = min(f0 + m, f_out)
        cl = compress_conv(weight[f0:f1], (h_in, w_in), stride, pad)
        nbytes = cl.weight_array.size + 2 * cl.num_post
        images.append(PeImage(pe_id, _coord(pe_id), node.name, "conv",
                              neuron_base=f0 * ho * wo, neuron_count=cl.num_post,
                              dendrites=cl.dendrites, spikes=cl.spikes,
                              weight=cl.weight_array, budget_bytes=nbytes))
        pe_id += 1
    pl = Placement(node.name, "conv", pe_id0, len(images), {"ho_wo": ho * wo, "m": m})
    return images, pl


def _partition_dense(node: Node, weight: np.ndarray, pe_id0: int) -> Tuple[List[PeImage], Placement]:
    out_features, in_features = weight.shape
    n_tokens = node.shape[0] if len(node.shape) == 2 else 1
    l = dense_neurons_per_pe(in_features, n_tokens)
    images, pe_id = [], pe_id0
    for o0 in range(0, out_features, l):
        o1 = min(o0 + l, out_features)
        cl = compress_dense(weight[o0:o1], n_tokens)
        nbytes = cl.weight_array.size + 2 * cl.num_post
        images.append(PeImage(pe_id, _coord(pe_id), node.name, "dense",
                              neuron_base=o0 * max(n_tokens, 1), neuron_count=cl.num_post,
                              dendrites=cl.dendrites, spikes=cl.spikes,
                              weight=cl.weight_array, budget_bytes=nbytes))
        pe_id += 1
    pl = Placement(node.name, "dense", pe_id0, len(images), {"out": out_features, "l": l})
    return images, pl


_UNIT_W = np.ones(1, dtype=np.float32)   # placeholder weight for attention fan dendrites


def _partition_matmul_qk(node: Node, dag: Dag, pe_id0: int) -> Tuple[List[PeImage], Placement]:
    heads, n_tok, _ = node.shape
    nn = n_tok * n_tok
    hpp = max(1, (SRAM_BYTES - _UNIT_W.size) // (2 * nn))   # whole heads per PE (states only)
    images, pe_id = [], pe_id0
    for h0 in range(0, heads, hpp):
        h1 = min(h0 + hpp, heads)
        ncount = (h1 - h0) * nn
        dends = [Dendrite(0, [0], [0], 1, local_off1=1, local_off2=0, repeat=n_tok),    # q-fan: scores(h,i,*)
                 Dendrite(1, [0], [0], 1, local_off1=n_tok, local_off2=0, repeat=n_tok)]  # k-fan: scores(h,*,j)
        images.append(PeImage(pe_id, _coord(pe_id), node.name, "matmul_qk",
                              neuron_base=h0 * nn, neuron_count=ncount, dendrites=dends,
                              spikes=[], weight=_UNIT_W, budget_bytes=_UNIT_W.size + 2 * ncount))
        pe_id += 1
    pl = Placement(node.name, "matmul_qk", pe_id0, len(images), {"nn": nn, "hpp": hpp})
    return images, pl


def _partition_matmul_av(node: Node, dag: Dag, pe_id0: int) -> Tuple[List[PeImage], Placement]:
    n_tok, embed = node.shape
    heads = node.attrs["heads"]
    dh = embed // heads
    ncount = n_tok * embed
    if _UNIT_W.size + 2 * ncount > SRAM_BYTES:
        raise NotImplementedError(f"{node.name}: attention.V block ({ncount} neurons) exceeds one "
                                  "PE; per-token-block splitting of matmul_av is not implemented")
    dends = [Dendrite(0, [0], [0], 1, local_off1=1, local_off2=0, repeat=dh),       # qk-fan: out(i, head slice)
             Dendrite(1, [0], [0], 1, local_off1=embed, local_off2=0, repeat=n_tok)]  # v-fan: out(*, e)
    img = PeImage(pe_id0, _coord(pe_id0), node.name, "matmul_av",
                  neuron_base=0, neuron_count=ncount, dendrites=dends, spikes=[],
                  weight=_UNIT_W, budget_bytes=_UNIT_W.size + 2 * ncount)
    pl = Placement(node.name, "matmul_av", pe_id0, 1, {"embed": embed, "heads": heads})
    return [img], pl


def partition_dag(dag: Dag) -> MappedNetwork:
    """Partition every neuron-producing node of the DAG onto PEs (placement only; see route_dag)."""
    pes: List[PeImage] = []
    placements: Dict[str, Placement] = {}
    skipped: List[Tuple[str, str]] = []
    pe_id = 0
    for ordinal, node in enumerate(topo(dag)):
        if node.op == "conv":
            w = _to_np(dag.weights[node.weight_name])
            node.attrs.setdefault("in_chw", _producer_chw(dag, node))
            imgs, pl = _partition_conv(node, w, pe_id)
        elif node.op == "dense":
            imgs, pl = _partition_dense(node, _to_np(dag.weights[node.weight_name]), pe_id)
        elif node.op == "matmul_qk":
            imgs, pl = _partition_matmul_qk(node, dag, pe_id)
        elif node.op == "matmul_av":
            imgs, pl = _partition_matmul_av(node, dag, pe_id)
        else:
            continue                       # input / pool / add / reshape / output produce no PEs
        for img in imgs:
            img.layer = ordinal
        placements[node.name] = pl
        pes.extend(imgs)
        pe_id += len(imgs)
    return MappedNetwork(dag.model_name, pes, placements, pe_id, skipped, dict(dag.neuron))


def _producer_chw(dag: Dag, conv_node: Node) -> Tuple[int, int, int]:
    """(C,H,W) feeding a conv = its single producer's output shape (input node or a pool/conv)."""
    return dag.nodes[conv_node.inputs[0]].shape


def _edge_terms(dag: Dag, node: Node) -> Dict[str, Tuple[int, int, int]]:
    """For each SOURCE_OPS node feeding `node` (through dissolved pool/reshape/add relays), the
    (term_id, combine, avg_n) describing how the consumer folds that source's arrivals:
      * a path through a MaxPool -> MAX (OR the window); through AvgPool(s) -> AVG (sum / window);
      * plain / residual-add branch -> SUM.
    Each `add` branch is a distinct term_id, so a MaxPool skip ORs independently of the main term."""
    found: Dict[str, Tuple[int, int]] = {}   # source -> (combine, avg_n)

    def walk(cur: Node, has_max: bool, avg_n: int) -> None:
        for inp in cur.inputs:
            p = dag.nodes[inp]
            if p.op in SOURCE_OPS:
                combine = _COMBINE_MAX if has_max else (_COMBINE_AVG if avg_n > 1 else _COMBINE_SUM)
                found[inp] = (combine, avg_n)
            elif p.op == "pool":
                hm = has_max or p.attrs.get("pool_type") == "MaxPool2d"
                an = avg_n
                if p.attrs.get("pool_type") != "MaxPool2d":
                    s = dag.nodes[p.inputs[0]].shape
                    n_in = s[1] * s[2] if len(s) == 3 else 1
                    n_out = p.shape[1] * p.shape[2] if len(p.shape) == 3 else 1
                    an *= max(1, n_in // max(1, n_out))
                walk(p, hm, an)
            elif p.op in ("reshape", "add"):
                walk(p, has_max, avg_n)

    walk(node, False, 1)
    return {src: (tid, found[src][0], found[src][1]) for tid, src in enumerate(sorted(found))}


# --------------------------------------------------------------------------------------------
def _source_nodes(dag: Dag, node: Node) -> set:
    """SOURCE_OPS nodes feeding `node` through dissolved relays (pool/reshape/add); input -> none."""
    out: set = set()
    for inp in node.inputs:
        p = dag.nodes[inp]
        if p.op in SOURCE_OPS:
            out.add(inp)
        elif p.op in ("pool", "reshape", "add"):
            out |= _source_nodes(dag, p)
    return out


def node_depths(dag: Dag) -> Dict[str, int]:
    """Pipeline stage of each weight node = #weight-layers on the longest path from the input. A
    delay-1 pipeline advances one stage per weight layer, so a residual's main and skip paths reach
    an `add` at different stages; per-edge axon delay = depth(consumer) - depth(source) re-aligns
    them (the paper's configurable synaptic delay)."""
    depths: Dict[str, int] = {}
    for n in topo(dag):
        if n.op in SOURCE_OPS:
            depths[n.name] = 1 + max((depths[s] for s in _source_nodes(dag, n)), default=0)
    return depths


def route_dag(dag: Dag, mapped: MappedNetwork) -> List[Tuple[str, int]]:
    """Fill PeImage.axons in place. Returns [(node, num_axon_entries)] per producing node.

    conv/dense: each downstream PE's input spike `pre` is back-traced to its upstream source
    neurons (input_sources handles residual adds / reshape / pooling); an AxonEntry is appended on
    each source PE pointing at the downstream PE + dendrite + offsets. Source = the network input
    means external stimulus (no on-chip producer) and is skipped.
    matmul: a forward fan pass wires q/k -> qk and qk/v -> av through the unit-weight fan dendrites.
    """
    memo: dict = {}
    depths = node_depths(dag)
    routed: List[Tuple[str, int]] = []

    def add_entry(src_node: str, src_neuron: int, dst_pe_id: int, dend: int, go1: int, go2: int,
                  delay: int, meta: int):
        if dag.nodes[src_node].op == "input":
            return 0
        pe_idx, local = mapped.placements[src_node].locate(src_neuron)
        mapped.pes[pe_idx].axons.setdefault(local, []).append(
            AxonEntry(dst_pe_id, dend, go1, go2, delay=delay, meta=meta))
        return 1

    for node in topo(dag):
        if node.op in ("conv", "dense"):
            terms = _edge_terms(dag, node)             # source -> (term_id, combine, avg_n)
            cnt = 0
            for pe in mapped.pes_of_node(node.name):
                for s in pe.spikes:
                    for src_node, src_neuron in input_sources(dag, node.name, s.pre, memo):
                        # delay-1 per pipeline stage; longer for a shorter (skip) source path.
                        delay = depths[node.name] - depths.get(src_node, depths[node.name] - 1)
                        tid, comb, an = terms.get(src_node, (0, 0, 1))
                        cnt += add_entry(src_node, src_neuron, pe.pe_id, s.dendrite_id,
                                         s.go1, s.go2, delay, pack_meta(tid, comb, an))
            routed.append((node.name, cnt))
        elif node.op == "matmul_qk":
            routed.append((node.name, _route_qk(dag, mapped, node, depths)))
        elif node.op == "matmul_av":
            routed.append((node.name, _route_av(dag, mapped, node, depths)))
    return routed


def _route_qk(dag: Dag, mapped: MappedNetwork, node: Node, depths: Dict[str, int]) -> int:
    q, k = node.inputs
    heads, n_tok, _ = node.shape
    embed = dag.nodes[q].shape[1]
    dh = embed // heads
    qk_pl = mapped.placements[node.name]
    dq = depths[node.name] - depths[q]
    dk = depths[node.name] - depths[k]
    cnt = 0
    for g in range(n_tok * embed):                 # q neuron (i, e) -> scores (h, i, *)  [dendrite 0]
        i, e = divmod(g, embed)
        h = e // dh
        pe_idx, go2 = qk_pl.locate(h * n_tok * n_tok + i * n_tok)
        src_idx, src_local = mapped.placements[q].locate(g)
        mapped.pes[src_idx].axons.setdefault(src_local, []).append(
            AxonEntry(mapped.pes[pe_idx].pe_id, 0, 0, go2, delay=dq)); cnt += 1
    for g in range(n_tok * embed):                 # k neuron (j, e) -> scores (h, *, j)  [dendrite 1]
        j, e = divmod(g, embed)
        h = e // dh
        pe_idx, go2 = qk_pl.locate(h * n_tok * n_tok + j)
        src_idx, src_local = mapped.placements[k].locate(g)
        mapped.pes[src_idx].axons.setdefault(src_local, []).append(
            AxonEntry(mapped.pes[pe_idx].pe_id, 1, 0, go2, delay=dk)); cnt += 1
    return cnt


def _route_av(dag: Dag, mapped: MappedNetwork, node: Node, depths: Dict[str, int]) -> int:
    qk, v = node.inputs
    n_tok, embed = node.shape
    heads = node.attrs["heads"]
    dh = embed // heads
    av_pl = mapped.placements[node.name]
    d_qk = depths[node.name] - depths[qk]
    d_v = depths[node.name] - depths[v]
    cnt = 0
    nn = n_tok * n_tok
    for g in range(heads * nn):                    # score (h, i, j) -> out(i, head slice)  [dendrite 0]
        h, rem = divmod(g, nn)
        i, _j = divmod(rem, n_tok)
        pe_idx, go2 = av_pl.locate(i * embed + h * dh)
        src_idx, src_local = mapped.placements[qk].locate(g)
        mapped.pes[src_idx].axons.setdefault(src_local, []).append(
            AxonEntry(mapped.pes[pe_idx].pe_id, 0, 0, go2, delay=d_qk)); cnt += 1
    for g in range(n_tok * embed):                 # v neuron (j, e) -> out(*, e)  [dendrite 1]
        _j, e = divmod(g, embed)
        pe_idx, go2 = av_pl.locate(e)
        src_idx, src_local = mapped.placements[v].locate(g)
        mapped.pes[src_idx].axons.setdefault(src_local, []).append(
            AxonEntry(mapped.pes[pe_idx].pe_id, 1, 0, go2, delay=d_v)); cnt += 1
    return cnt


def axon_entry_count(mapped: MappedNetwork) -> int:
    return sum(len(v) for pe in mapped.pes for v in pe.axons.values())


# --------------------------------------------------------------------------------------------
# Axon-table compression: fold per-neuron entries into NESTED arithmetic-lattice headers (the
# inverse of the dendrite repeat-loop). Source neurons sharing a (dst_pe, dendrite_id, delay)
# target whose (src, go1, go2) form a multi-dimensional arithmetic lattice collapse into one
# AxonGroup. Greedy: detect the longest inner run, then how many times that block repeats with a
# fixed outer stride (next level), up to MAX_AXON_LEVELS. verify_axons() is the golden safety net.
def _sub(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _run_len(pts, i: int) -> Tuple[int, Tuple[int, int, int]]:
    """Longest arithmetic run of points starting at i (src strictly increasing)."""
    n = len(pts)
    if i + 1 >= n:
        return 1, (1, 0, 0)
    s = _sub(pts[i + 1], pts[i])
    if s[0] <= 0:
        return 1, (1, 0, 0)
    j = i + 1
    while j + 1 < n and _sub(pts[j + 1], pts[j]) == s:
        j += 1
    return j - i + 1, s


def _block_repeats(pts, i: int, block: int, outer: Tuple[int, int, int]) -> int:
    """How many consecutive size-`block` blocks from i are block0 shifted by m*outer."""
    n, k = len(pts), 1
    while i + (k + 1) * block <= n:
        shift = (outer[0] * k, outer[1] * k, outer[2] * k)
        if any(_sub(pts[i + k * block + t], pts[i + t]) != shift for t in range(block)):
            break
        k += 1
    return k


def _build_levels(pts, i: int) -> List[Tuple[int, int, int, int]]:
    c0, s0 = _run_len(pts, i)
    levels = [(c0, s0[0], s0[1], s0[2])]
    block = c0
    while len(levels) < MAX_AXON_LEVELS and i + 2 * block <= len(pts):
        outer = _sub(pts[i + block], pts[i])
        if outer[0] <= 0:
            break
        k = _block_repeats(pts, i, block, outer)
        if k <= 1:
            break
        levels.append((k, outer[0], outer[1], outer[2]))
        block *= k
    return levels


def _compress_pe_axons(pe: PeImage) -> List[AxonGroup]:
    channels: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int]]] = {}
    for src, entries in pe.axons.items():
        for e in entries:
            channels.setdefault((e.dst_pe, e.dendrite_id, e.delay, e.meta), []).append(
                (src, e.go1, e.go2))

    groups: List[AxonGroup] = []
    for (dst_pe, dend, delay, meta), triples in channels.items():
        pts = sorted(set(triples))                       # distinct, ordered by (src, go1, go2)
        i, n = 0, len(pts)
        while i < n:
            levels = _build_levels(pts, i)
            total = 1
            for c, *_ in levels:
                total *= c
            s, g1, g2 = pts[i]
            groups.append(AxonGroup(s, g1, g2, levels, dst_pe, dend, delay, meta))
            i += total
    return groups


def compress_axons(mapped: MappedNetwork) -> None:
    """Fill PeImage.axon_groups for every PE (call after route_dag)."""
    for pe in mapped.pes:
        pe.axon_groups = _compress_pe_axons(pe)


def decompress_axon_group(g: AxonGroup) -> List[Tuple[int, int, int, int, int, int, int]]:
    """(src, dst_pe, dendrite_id, go1, go2, delay, meta) tuples — golden reconstruction (nested)."""
    out: List[Tuple[int, int, int, int, int, int, int]] = []
    for rs in product(*(range(c) for c, *_ in g.levels)):
        src, go1, go2 = g.src_base, g.go1_base, g.go2_base
        for r, (_, ss, gs1, gs2) in zip(rs, g.levels):
            src += r * ss
            go1 += r * gs1
            go2 += r * gs2
        out.append((src, g.dst_pe, g.dendrite_id, go1, go2, g.delay, g.meta))
    return out


def verify_axons(mapped: MappedNetwork) -> bool:
    """Golden: the compressed groups decompress to EXACTLY the flat per-neuron axon entries."""
    for pe in mapped.pes:
        flat = {(src, e.dst_pe, e.dendrite_id, e.go1, e.go2, e.delay, e.meta)
                for src, entries in pe.axons.items() for e in entries}
        recon = {t for g in pe.axon_groups for t in decompress_axon_group(g)}
        if flat != recon:
            return False
    return True


def axon_group_count(mapped: MappedNetwork) -> int:
    return sum(len(pe.axon_groups) for pe in mapped.pes)


# Binary axon-record field widths (mirrored by the C++ loader + schema).
_U32 = 1 << 32
_U16 = 1 << 16
MAX_AXON_DELAY = 15        # 4-bit flit delay field


def validate_axon_groups(mapped: MappedNetwork) -> List[str]:
    """Semantic checks on compressed axon groups before serialization (empty == all good):
    target dendrite exists on the destination PE, source run stays within the PE, fields fit."""
    errs: List[str] = []
    pe_dends = {pe.pe_id: {d.id for d in pe.dendrites} for pe in mapped.pes}
    for pe in mapped.pes:
        for g in pe.axon_groups:
            if not (0 <= g.dst_pe < mapped.num_pe_used and g.dst_pe < _U16):
                errs.append(f"PE {pe.pe_id}: axon dst_pe {g.dst_pe} out of range"); continue
            if g.dendrite_id not in pe_dends[g.dst_pe]:
                errs.append(f"PE {pe.pe_id}: axon targets dendrite {g.dendrite_id} absent on PE {g.dst_pe}")
            if not (0 <= g.delay <= MAX_AXON_DELAY):
                errs.append(f"PE {pe.pe_id}: axon delay {g.delay} > {MAX_AXON_DELAY}")
            if not (1 <= len(g.levels) <= MAX_AXON_LEVELS):
                errs.append(f"PE {pe.pe_id}: axon group has {len(g.levels)} levels")
            # src strides are non-negative (sorted lattice); the max-corner id must stay in the PE.
            last = g.src_base + sum((c - 1) * ss for c, ss, _, _ in g.levels)
            if not (0 <= g.src_base and 0 <= last < pe.neuron_count):
                errs.append(f"PE {pe.pe_id}: axon src lattice [{g.src_base}..{last}] outside "
                            f"[0,{pe.neuron_count})")
            if not (0 <= g.src_base < _U32 and all(0 < c < _U32 for c, *_ in g.levels)):
                errs.append(f"PE {pe.pe_id}: axon src_base/count exceed u32")
    return errs


# --------------------------------------------------------------------------------------------
def reconstruct_node(mapped: MappedNetwork, node: str) -> set:
    """Union of per-PE decompressed connectivity in GLOBAL (within-node) post ids — golden check
    for conv / dense losslessness (matmul carries no static weights)."""
    pl = mapped.placements[node]
    out: set = set()
    for block, pe in enumerate(mapped.pes_of_node(node)):
        cl = CompiledLayer(pe.kind, pe.dendrites, pe.spikes, pe.weight, pe.neuron_count, 0)
        for (post, pre, wv) in reconstruct(cl):
            out.add((pl.local_to_global(block, post), pre, wv))
    return out


def validate(mapped: MappedNetwork) -> List[str]:
    """Return a list of constraint violations (empty == all good)."""
    errs: List[str] = []
    if mapped.num_pe_used > NUM_PE:
        errs.append(f"uses {mapped.num_pe_used} PEs > {NUM_PE}")
    for pe in mapped.pes:
        if pe.budget_bytes > SRAM_BYTES:
            errs.append(f"PE {pe.pe_id}: {pe.budget_bytes} B > {SRAM_BYTES} B SRAM")
        if len(pe.dendrites) > DENDRITE_MAX:
            errs.append(f"PE {pe.pe_id}: {len(pe.dendrites)} dendrites > {DENDRITE_MAX}")
        for d in pe.dendrites:
            if d.count > COUNT_MAX:
                errs.append(f"PE {pe.pe_id} dendrite {d.id}: count {d.count} > {COUNT_MAX}")
                break
    return errs
