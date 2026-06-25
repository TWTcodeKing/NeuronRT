"""DAG IR — the topology-aware network representation that replaces the flat layer list.

A Node is one operator; its `shape` is the OUTPUT activation shape with the batch dim stripped
(so neuron count = prod(shape), token-aware: a token-wise Linear on (N, dim) has N*dim neurons).
Edges are `inputs` (names of producer nodes). This captures residual adds (op='add', two inputs),
parallel branches (several consumers of one node), attention matmuls (no static weight), and
reshape/transpose (index permutation) — none of which a sequential list can express.

Op set:
  input    - network input (image / tokens)
  conv     - Conv2d; shape (F,H,W); attrs stride/padding/kh/kw; weight (F,C,K,K)
  dense    - Linear, possibly token-wise; shape (N,out) or (out,); weight (out,in)
  pool     - Max/Avg/AdaptiveAvg pool; shape (C,H,W); attrs kernel/stride/padding or 'adaptive'
  add      - element-wise residual add of >=2 same-shape producers
  reshape  - permutation/flatten/transpose; shape is the new shape; attrs describe the index map
  matmul_qk- attention scores Q.K^T; shape (heads,N,N); inputs [q, k]
  matmul_av- attention.V; shape (N,dim); inputs [attn, v]
  output   - logits sink
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

VALID_OPS = {"input", "conv", "dense", "pool", "add", "reshape", "matmul_qk", "matmul_av", "output"}


@dataclass
class Node:
    name: str
    op: str
    shape: Tuple[int, ...]                 # output activation shape, batch stripped
    inputs: List[str] = field(default_factory=list)  # producer node names (edges)
    attrs: Dict = field(default_factory=dict)
    weight_name: Optional[str] = None      # key into Dag.weights (conv/dense)

    def num_neurons(self) -> int:
        return int(math.prod(self.shape)) if self.shape else 0


@dataclass
class Dag:
    model_name: str
    input_hw: Tuple[int, int]
    timesteps: int = 4
    nodes: Dict[str, Node] = field(default_factory=dict)
    weights: Dict[str, "object"] = field(default_factory=dict)  # name -> torch/np weight tensor
    biases: Dict[str, "object"] = field(default_factory=dict)   # name -> per-out-channel bias (BN-fold)
    neuron: Dict = field(default_factory=dict)                  # LIF params

    def add(self, node: Node) -> Node:
        if node.name in self.nodes:
            raise ValueError(f"duplicate node name {node.name}")
        self.nodes[node.name] = node
        return node

    def consumers(self, name: str) -> List[Node]:
        return [n for n in self.nodes.values() if name in n.inputs]

    def of_op(self, op: str) -> List[Node]:
        return [n for n in topo(self) if n.op == op]

    def total_neurons(self) -> int:
        return sum(n.num_neurons() for n in self.nodes.values())

    def validate(self) -> None:
        for n in self.nodes.values():
            if n.op not in VALID_OPS:
                raise ValueError(f"{n.name}: unknown op {n.op}")
            for p in n.inputs:
                if p not in self.nodes:
                    raise ValueError(f"{n.name}: input '{p}' is not a node")
            if n.op == "input" and n.inputs:
                raise ValueError(f"input node {n.name} must have no inputs")
            if n.op in ("conv", "dense", "pool", "reshape") and len(n.inputs) != 1:
                raise ValueError(f"{n.name} ({n.op}) needs exactly 1 input, got {len(n.inputs)}")
            if n.op == "add" and len(n.inputs) < 2:
                raise ValueError(f"add node {n.name} needs >= 2 inputs")
            if n.op == "add":
                shp = {self.nodes[p].shape for p in n.inputs}
                if len(shp) != 1:
                    raise ValueError(f"add node {n.name}: producers differ in shape {shp}")
            if n.op == "matmul_qk" and len(n.inputs) != 2:
                raise ValueError(f"matmul_qk {n.name} needs 2 inputs (q, k)")
            if n.op == "matmul_av" and len(n.inputs) != 2:
                raise ValueError(f"matmul_av {n.name} needs 2 inputs (attn, v)")
            if n.op == "reshape" and n.attrs.get("kind") == "to_tokens":
                src = n.attrs.get("from")
                if src is None or math.prod(src) != n.num_neurons():
                    raise ValueError(f"reshape {n.name}: 'from' {src} neuron count "
                                     f"!= output {n.shape} ({n.num_neurons()})")
        topo(self)  # raises on cycle

        # Structural sink + reachability: exactly one 'output', it is a true sink, and every node
        # lies on a path from an input (an orphan branch would silently inflate the neuron budget).
        outs = [n.name for n in self.nodes.values() if n.op == "output"]
        if len(outs) != 1:
            raise ValueError(f"expected exactly one output node, got {outs}")
        if self.consumers(outs[0]):
            raise ValueError(f"output node {outs[0]} must be a sink (has consumers)")
        children, _ = _adjacency(self)
        reachable: set = set()
        stack = [n.name for n in self.nodes.values() if n.op == "input"]
        while stack:
            x = stack.pop()
            if x in reachable:
                continue
            reachable.add(x)
            stack.extend(children[x])
        orphans = [name for name in self.nodes if name not in reachable]
        if orphans:
            raise ValueError(f"nodes unreachable from any input (orphans): {orphans}")

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for n in self.nodes.values():
            counts[n.op] = counts.get(n.op, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return f"{self.model_name}: {len(self.nodes)} nodes ({parts}), {self.total_neurons():,} neurons"


def _adjacency(dag: Dag) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """children[producer] -> consumers, and in-degree counted ONCE per distinct producer edge.

    Counting distinct producers (not occurrences) is what makes a duplicate input (e.g. add
    [x, x]) sound: x is decremented once and the consumer still reaches in-degree 0.
    """
    children: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = {name: 0 for name in dag.nodes}
    for n in dag.nodes.values():
        for p in dict.fromkeys(n.inputs):      # distinct producers, order-preserving
            children[p].append(n.name)
            indeg[n.name] += 1
    return children, indeg


def topo(dag: Dag) -> List[Node]:
    """Kahn topological order; raises ValueError on a cycle. O(V + E) via a children map +
    a heap (lexicographic tie-break keeps the order deterministic across runs/thread counts)."""
    children, indeg = _adjacency(dag)
    ready = [name for name, d in indeg.items() if d == 0]
    heapq.heapify(ready)
    out: List[Node] = []
    while ready:
        name = heapq.heappop(ready)
        out.append(dag.nodes[name])
        for c in children[name]:
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(ready, c)
    if len(out) != len(dag.nodes):
        raise ValueError("DAG has a cycle")
    return out
