"""Edge-typed routing over the DAG.

`edge_preimage` answers, for one connectivity-only node (pool / reshape / add), which producer
neurons feed a given output neuron. `sources` composes those preimages — walking back through the
connectivity-only nodes until it reaches weight/compute nodes (conv / dense / matmul) or the
input — to find the upstream weight-node neurons that feed a downstream weight-node's input. This
replaces the flat-list adjacency: it routes residual adds (two producers per output), reshapes
(token transpose), and pooling windows correctly.
"""
from __future__ import annotations

from typing import List, Tuple

from ..graph.dag import Dag, Node

# Nodes that PRODUCE neurons (routing stops here): a weight layer, attention cluster, or the input.
SOURCE_OPS = {"conv", "dense", "matmul_qk", "matmul_av", "input"}
# Connectivity-only nodes that merely re-map neurons (routing recurses through them).
RELAY_OPS = {"pool", "reshape", "add"}


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _pool_preimage(prod_shape: Tuple[int, ...], node: Node, oid: int) -> List[int]:
    _, ho, wo = prod_shape
    _, h1, w1 = node.shape
    c = oid // (h1 * w1)
    rem = oid % (h1 * w1)
    oh, ow = rem // w1, rem % w1
    if node.attrs.get("pool_type") == "adaptive":
        hs = range((oh * ho) // h1, _ceil_div((oh + 1) * ho, h1))
        ws = range((ow * wo) // w1, _ceil_div((ow + 1) * wo, w1))
    else:
        kh, kw = node.attrs["kernel"]
        sh, sw = node.attrs["stride"]
        ph, pw = node.attrs.get("padding", (0, 0))
        hs = [oh * sh - ph + dh for dh in range(kh) if 0 <= oh * sh - ph + dh < ho]
        ws = [ow * sw - pw + dw for dw in range(kw) if 0 <= ow * sw - pw + dw < wo]
    return [(c * ho + h) * wo + w for h in hs for w in ws]


def _reshape_preimage(node: Node, oid: int) -> int:
    """to_tokens: producer (embed,H,W) -> output (N=H*W, embed); output (t,d) <- producer (d, t//W, t%W)."""
    assert node.attrs.get("kind") == "to_tokens", f"unsupported reshape {node.attrs}"
    embed, h, w = node.attrs["from"]
    t, d = oid // embed, oid % embed   # output neuron (token t, dim d)
    return d * (h * w) + t             # producer (d, t//W, t%W) flat = d*H*W + t


def edge_preimage(dag: Dag, node: Node, oid: int) -> List[Tuple[str, int]]:
    """Producer (node_name, neuron_id) pairs feeding output neuron `oid` of a relay node."""
    if node.op == "add":
        return [(p, oid) for p in node.inputs]          # element-wise: same index from each producer
    if node.op == "reshape":
        return [(node.inputs[0], _reshape_preimage(node, oid))]
    if node.op == "pool":
        if node.attrs.get("pool_type") == "token_mean":  # (N,embed) -> (embed,): all tokens of dim oid
            prod = dag.nodes[node.inputs[0]]
            n_tok, embed = prod.shape
            return [(node.inputs[0], t * embed + oid) for t in range(n_tok)]
        prod = node.inputs[0]
        return [(prod, pid) for pid in _pool_preimage(dag.nodes[prod].shape, node, oid)]
    raise ValueError(f"edge_preimage: '{node.op}' is not a relay node")


def sources(dag: Dag, node_name: str, neuron_id: int,
            memo: dict = None) -> List[Tuple[str, int]]:
    """Source-node neurons feeding (node_name, neuron_id), composing relay preimages.

    Recursion terminates because the DAG is acyclic (Dag.validate raises on cycles). `memo` caches
    (node_name, neuron_id) -> sources so a shared upstream sub-path (residual adds re-fan the same
    neuron into both producers; token reshape feeds parallel q/k/v) is traced once, not per consumer.
    """
    if memo is None:
        memo = {}
    key = (node_name, neuron_id)
    cached = memo.get(key)
    if cached is not None:
        return cached
    node = dag.nodes[node_name]
    if node.op in SOURCE_OPS:
        result = [(node_name, neuron_id)]
    else:
        result = []
        for pname, pid in edge_preimage(dag, node, neuron_id):
            result.extend(sources(dag, pname, pid, memo))
    memo[key] = result
    return result


def input_sources(dag: Dag, weight_node_name: str, input_id: int,
                  memo: dict = None) -> List[Tuple[str, int]]:
    """For a single-input weight node (conv/dense), the upstream source neurons feeding its input
    neuron `input_id`. Attention matmuls have two operands and are routed by the mapping layer's
    matmul pass — not here — so guard against silently dropping the second operand."""
    node = dag.nodes[weight_node_name]
    assert node.inputs, f"{weight_node_name} has no input"
    assert len(node.inputs) == 1, (f"{weight_node_name} ({node.op}) has {len(node.inputs)} inputs; "
                                   "input_sources is for single-input nodes (matmul routed separately)")
    return sources(dag, node.inputs[0], input_id, memo)


def matmul_preimage(dag: Dag, node: Node, oid: int) -> List[Tuple[str, int]]:
    """Operand source neurons feeding attention-output neuron `oid` (graph-level; the mapping layer
    routes the inverse fan).  dh = embed/heads is the per-head width.

      matmul_qk score (h,i,j) <- q[i, head h slice] and k[j, head h slice]
      matmul_av out   (i, e)  <- scores (h=e//dh, i, all j) and v[all j, e]
    """
    q_or_qk, k_or_v = node.inputs
    if node.op == "matmul_qk":
        heads, n_tok, _ = node.shape
        embed = dag.nodes[q_or_qk].shape[1]
        dh = embed // heads
        h, rem = divmod(oid, n_tok * n_tok)
        i, j = divmod(rem, n_tok)
        qs = [(q_or_qk, i * embed + h * dh + d) for d in range(dh)]
        ks = [(k_or_v, j * embed + h * dh + d) for d in range(dh)]
        return qs + ks
    if node.op == "matmul_av":
        n_tok, embed = node.shape
        heads = node.attrs["heads"]
        dh = embed // heads
        i, e = divmod(oid, embed)
        h = e // dh
        scores = [(q_or_qk, h * n_tok * n_tok + i * n_tok + j) for j in range(n_tok)]
        vs = [(k_or_v, j * embed + e) for j in range(n_tok)]
        return scores + vs
    raise ValueError(f"matmul_preimage: {node.op} is not an attention matmul")
