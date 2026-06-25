"""End-to-end real-network simulation on the C++ NoC, validated against the functional reference.

Compile a small SVGG9, compute the first-layer conv(image) input current, hand the compiled network
+ input to the C++ NetworkRunner (delay-1 pipeline over the real NoC), then compare its steady-state
per-layer firing rates to the int8 funcsim reference (== SpikingJelly int8). Pipelined rates converge
to the single-step rates (phase-shifted by depth), so we expect a close match, not bit-exact.
"""
import os
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurort_compiler.funcsim import nrt_forward, reassemble_weight
from neurort_compiler.graph.builders import build_dag
from neurort_compiler.graph.dag import topo
from neurort_compiler.export.writer import write_network
from neurort_compiler.mapping.partition import (compress_axons, node_depths, partition_dag,
                                                route_dag, validate)
from neurort_compiler.models.registry import build_model

MODEL = "spikformer"
HW = 32
GAIN = 8.0
MEASURE = 32
SIM = "/home/twt/NeuronRT/build/sim/neurort_sim"


def write_attention_bin(path, dag, mapped):
    """Per SSA block: q/k/v/proj PE ids + n_tok/embed/heads/scale + coproc_delay (= proj−q depth gap,
    the attention's pipeline stages). The C++ runner runs the data-dependent matmul as a coprocessor."""
    dep = node_depths(dag)
    blocks = [n for n in topo(dag) if n.op == "matmul_av"]
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", len(blocks)))
        for av in blocks:
            qk = dag.nodes[av.inputs[0]]
            v = av.inputs[1]
            q, k = qk.inputs
            proj = next(n.name for n in dag.consumers(av.name) if n.op == "dense")
            n_tok, embed = dag.nodes[q].shape
            fh.write(struct.pack("<IIIdI", n_tok, embed, int(qk.attrs["heads"]),
                                 float(qk.attrs["scale"]), dep[proj] - dep[q]))
            for node in (q, k, v, proj):
                pes = [pe.pe_id for pe in mapped.pes_of_node(node)]
                fh.write(struct.pack("<I", len(pes)))
                fh.write(struct.pack(f"<{len(pes)}I", *pes))


def main():
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    model = build_model(MODEL, num_classes=10)
    for m in model.modules():                                   # randomize BN + scale (exercise firing)
        if isinstance(m, nn.BatchNorm2d):
            f = m.running_mean.numel()
            m.running_mean.copy_(torch.randn(f, generator=g) * 0.5)
            m.running_var.copy_(torch.rand(f, generator=g) * 0.5 + 0.5)
            with torch.no_grad():
                m.weight.copy_(torch.randn(f, generator=g) * 0.3 + 1.0)
                m.bias.copy_(torch.randn(f, generator=g) * 0.2)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.mul_(GAIN)

    dag = build_dag(model, MODEL, (HW, HW))
    dag.validate()
    mapped = partition_dag(dag)
    route_dag(dag, mapped)
    compress_axons(mapped)
    assert validate(mapped) == [], validate(mapped)[:3]
    out_node = next(n for n in dag.nodes.values() if n.op == "output")
    analog = set(out_node.inputs)
    names = [n.name for n in topo(dag) if n.op in ("conv", "dense") and n.name not in analog]
    depth = len(names)
    print(f"[compile] svgg9 @ {HW}x{HW}: {mapped.num_pe_used} PEs, {depth} weight layers")

    out_dir = "/tmp/e2e_svgg9"
    os.makedirs(out_dir, exist_ok=True)
    write_network(mapped, out_dir, timesteps=MEASURE)

    image = torch.randn(1, 3, HW, HW, generator=g)

    # int8 funcsim reference (single-step, == SpikingJelly int8) per-layer firing rate.
    ref_rate, _ = nrt_forward(mapped, dag, image, MEASURE, quantize=True)

    # First-layer input current = conv(image)+bias with the int8 conv0 weight (what the chip sees).
    in_node = next(n for n in dag.nodes.values()
                   if n.op == "conv" and dag.nodes[n.inputs[0]].op == "input")
    w = reassemble_weight(mapped, dag, in_node.name, quantize=True)
    cur = F.conv2d(image, w, stride=in_node.attrs["stride"], padding=in_node.attrs["padding"])
    if in_node.name in dag.biases:
        cur = cur + torch.as_tensor(dag.biases[in_node.name], dtype=torch.float32).reshape(1, -1, 1, 1)
    cur_flat = cur.flatten().detach().numpy().astype(np.float64)

    # Write input.bin: warmup, measure, then each input PE's local current slice. (tau/v_th now
    # live in the manifest's neuron params, read by the C++ runner.)
    warmup = depth + 40
    in_pes = mapped.pes_of_node(in_node.name)
    with open(os.path.join(out_dir, "input.bin"), "wb") as fh:
        fh.write(struct.pack("<II", warmup, MEASURE))
        fh.write(struct.pack("<I", len(in_pes)))
        for pe in in_pes:
            seg = cur_flat[pe.neuron_base:pe.neuron_base + pe.neuron_count]
            fh.write(struct.pack("<II", pe.pe_id, pe.neuron_count))
            fh.write(seg.tobytes())

    # Attention co-processors (Spikformer SSA blocks), if any.
    attn_path = os.path.join(out_dir, "attention.bin")
    if os.path.exists(attn_path):
        os.remove(attn_path)                       # avoid a stale file polluting a non-attention run
    if any(n.op == "matmul_av" for n in dag.nodes.values()):
        write_attention_bin(attn_path, dag, mapped)

    # Run the C++ end-to-end NoC simulation.
    print(f"[run] {SIM} --network {out_dir}  (warmup={warmup}, measure={MEASURE})")
    r = subprocess.run([SIM, "--network", out_dir], capture_output=True, text=True)
    print("  " + r.stdout.strip().replace("\n", "\n  "))
    if r.returncode != 0:
        print("  STDERR:", r.stderr.strip()); return 1

    # Read firing.bin -> per-PE counts -> per-node global steady-state rate.
    with open(os.path.join(out_dir, "firing.bin"), "rb") as fh:
        npe = struct.unpack("<I", fh.read(4))[0]
        fc = {}
        for _ in range(npe):
            pe_id, cnt = struct.unpack("<II", fh.read(8))
            fc[pe_id] = np.frombuffer(fh.read(4 * cnt), dtype=np.uint32)

    print(f"\n  {'layer':10} {'neurons':>8} {'refRate':>9} {'cppRate':>9} {'rateMAE':>9} {'maxΔ':>8}")
    worst = 0.0
    for nm in names:
        node = dag.nodes[nm]
        ncount = node.num_neurons()
        cpp = np.zeros(ncount, dtype=np.float64)
        ref = np.zeros(ncount, dtype=np.float64)
        # A token-wise dense ([N,out]) is laid out output-slice-major, token-major within the slice
        # (neuron_base=o0*N, local=t*w_out+r) — NOT the funcsim ref's [N,out] flatten. Reorder ref to
        # the per-PE layout so the per-neuron comparison lines up. conv / plain dense stay canonical.
        token_dense = node.op == "dense" and len(node.shape) == 2
        if token_dense:
            N = node.shape[0]
            ref_t = ref_rate[nm].detach().numpy().astype(np.float64).reshape(N, node.shape[1])
        else:
            ref_flat = ref_rate[nm].flatten().detach().numpy().astype(np.float64)
        for pe in mapped.pes_of_node(nm):
            b, c = pe.neuron_base, pe.neuron_count
            cpp[b:b + c] = fc[pe.pe_id]
            if token_dense:
                w_out, o0 = c // N, b // N
                ref[b:b + c] = ref_t[:, o0:o0 + w_out].reshape(-1)
            else:
                ref[b:b + c] = ref_flat[b:b + c]
        cpp /= MEASURE
        mae = float(np.abs(ref - cpp).mean())
        mx = float(np.abs(ref - cpp).max())
        worst = max(worst, mx)
        print(f"  {nm:10} {node.num_neurons():8d} {ref.mean():9.4f} {cpp.mean():9.4f} {mae:9.5f} {mx:8.4f}")
    print(f"\n[RESULT] worst per-neuron firing-rate Δ (C++ pipeline vs int8 funcsim) = {worst:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
