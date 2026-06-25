"""End-to-end accuracy check: compile SVGG9 and verify the NeuroRT functional simulation reproduces
SpikingJelly's firing behaviour (float = bit-exact gate; int8 = deployment tolerance)."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from neurort_compiler.funcsim import (Lif, decompress_matches_conv, nrt_forward, sj_reference)
from neurort_compiler.graph.builders import build_dag
from neurort_compiler.graph.dag import topo
from neurort_compiler.mapping.partition import compress_axons, partition_dag, route_dag, validate
from neurort_compiler.models.registry import build_model

T = 8


def lif_unit_check():
    """My Lif must equal SpikingJelly LIFNode on an arbitrary current sequence."""
    from spikingjelly.activation_based import neuron
    rng = torch.Generator().manual_seed(1)
    x = torch.randn(20, 50, generator=rng) * 1.5
    sj = neuron.LIFNode(); sj.reset()
    mine = Lif(tau=2.0, v_th=1.0)
    bad = 0
    for t in range(x.shape[0]):
        s_sj = sj(x[t])
        s_me = mine.step(x[t])
        bad += int((s_sj != s_me).sum())
    print(f"[LIF unit] mismatching spikes vs SpikingJelly LIFNode: {bad}  -> {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


def get_input():
    """A real CIFAR10 image if one is already extracted on disk, else a deterministic synthetic one.
    (We do NOT download — firing-EQUIVALENCE to SpikingJelly is input-agnostic; the same input is
    fed to both sides. Point this at a real CIFAR10 dir to swap in real data.)"""
    import os
    root = "/tmp/cifar10"
    if os.path.isdir(os.path.join(root, "cifar-10-batches-py")):
        import torchvision
        import torchvision.transforms as TT
        ds = torchvision.datasets.CIFAR10(root, train=False, download=False, transform=TT.ToTensor())
        img, label = ds[0]
        return ((img - 0.5) / 0.5).unsqueeze(0), f"CIFAR10 test[0] (label={label})"
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 3, 32, 32, generator=g), "synthetic 3x32x32 (real CIFAR10 not extracted on disk)"


def compare(nrt_rate, sj_rate, names):
    print(f"\n  {'layer':8} {'neurons':>8} {'SJ rate':>9} {'NRT rate':>9} {'rateMAE':>9} {'maxΔ':>8} {'disagree%':>9}")
    worst = 0.0
    for nm in names:
        a, b = sj_rate[nm].flatten(), nrt_rate[nm].flatten()
        mae = float((a - b).abs().mean())
        mx = float((a - b).abs().max())
        disagree = float(((a > 0) != (b > 0)).float().mean()) * 100.0
        worst = max(worst, mx)
        print(f"  {nm:8} {a.numel():8d} {a.mean():9.4f} {b.mean():9.4f} {mae:9.5f} {mx:8.4f} {disagree:9.2f}")
    return worst


def main():
    torch.manual_seed(0)
    # No trained VGG9-CIFAR10 weights on disk, so scale the (shared) random weights to drive healthy
    # firing through all layers — the equivalence check is independent of the weight values.
    gain = float(os.environ.get("SVGG9_GAIN", "8.0"))
    ok_lif = lif_unit_check()

    model = build_model("svgg9", num_classes=10)
    if gain != 1.0:
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    m.weight.mul_(gain)
        print(f"[setup] scaled conv/linear weights x{gain} (shared by both sides) to exercise firing")
    dag = build_dag(model, "svgg9", (32, 32))
    dag.validate()
    mapped = partition_dag(dag)
    route_dag(dag, mapped)
    compress_axons(mapped)
    errs = validate(mapped)
    print(f"\n[compile] {dag.summary()}")
    print(f"[compile] {mapped.num_pe_used} PEs, validate={'OK' if not errs else errs[:2]}")

    # Algorithm-1 (decompress) golden on every conv layer.
    conv_nodes = [n.name for n in dag.nodes.values() if n.op == "conv"]
    alg1 = all(decompress_matches_conv(mapped, dag, c) for c in conv_nodes)
    print(f"[Algorithm 1] decompress == dense conv connectivity on {len(conv_nodes)} conv layers: "
          f"{'OK' if alg1 else 'FAIL'}")

    image, src = get_input()
    print(f"\n[input] {src}; T={T} timesteps")
    out_node = next(n for n in dag.nodes.values() if n.op == "output")
    analog = set(out_node.inputs)
    names = [n.name for n in topo(dag) if n.op in ("conv", "dense") and n.name not in analog]

    sj_rate, sj_logits = sj_reference(model, image, dag, T)

    print("\n===== FLOAT weights: NeuroRT vs SpikingJelly (must be bit-exact) =====")
    nrt_rate_f, nrt_logits_f = nrt_forward(mapped, dag, image, T, quantize=False)
    worst_f = compare(nrt_rate_f, sj_rate, names)
    logit_dl = float((sj_logits - nrt_logits_f).abs().max())
    exact = worst_f < 1e-5 and logit_dl < 1e-4
    print(f"  worst firing-rate Δ = {worst_f:.2e}, logits maxΔ = {logit_dl:.2e}  -> "
          f"{'EXACT MATCH ✓' if exact else 'MISMATCH ✗'}")

    print("\n===== INT8 weights: NeuroRT vs SpikingJelly (deployment, tolerance) =====")
    nrt_rate_q, nrt_logits_q = nrt_forward(mapped, dag, image, T, quantize=True)
    worst_q = compare(nrt_rate_q, sj_rate, names)
    sj_arg = int(sj_logits.argmax()); q_arg = int(nrt_logits_q.argmax())
    print(f"  worst firing-rate Δ = {worst_q:.4f}; output argmax  SJ={sj_arg}  NRT-int8={q_arg}  "
          f"({'agree' if sj_arg == q_arg else 'DIFFER'})")
    # mean spike agreement across all LIF neurons
    tot = sum(sj_rate[n].numel() for n in names)
    agree = sum(int((((sj_rate[n].flatten() > 0) == (nrt_rate_q[n].flatten() > 0)).sum())) for n in names)
    print(f"  int8 spike-pattern agreement: {agree}/{tot} neurons = {100.0*agree/tot:.2f}%")

    ok = ok_lif and alg1 and exact
    print(f"\n[RESULT] LIF={ok_lif} Alg1={alg1} float-exact={exact}  => "
          f"{'PIPELINE ACCURATE (lossless vs SpikingJelly)' if ok else 'NEEDS INVESTIGATION'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
