"""Cross-topology functional check: compile a model and verify the NeuroRT functional simulation
reproduces SpikingJelly's firing (float = bit-exact gate; int8 = deployment tolerance).

BatchNorm running stats + affine params are randomized so the compile-time BN FOLD is non-trivially
exercised (an unfolded or wrongly-folded conv would then diverge from SpikingJelly).

Usage: python validate_arch.py [model_name]   (default sew_resnet18)
"""
import os
import sys

import torch
import torch.nn as nn

from neurort_compiler.funcsim import decompress_matches_conv, nrt_forward, sj_reference
from neurort_compiler.graph.builders import build_dag
from neurort_compiler.graph.dag import topo
from neurort_compiler.mapping.partition import compress_axons, partition_dag, route_dag, validate
from neurort_compiler.models.registry import build_model

T = 8


def randomize_bn(model, g):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            f = m.running_mean.numel()
            m.running_mean.copy_(torch.randn(f, generator=g) * 0.5)
            m.running_var.copy_(torch.rand(f, generator=g) * 0.5 + 0.5)
            with torch.no_grad():
                m.weight.copy_(torch.randn(f, generator=g) * 0.3 + 1.0)
                m.bias.copy_(torch.randn(f, generator=g) * 0.2)


def compare(nrt_rate, sj_rate, names):
    print(f"  {'layer':10} {'neurons':>8} {'SJ rate':>9} {'NRT rate':>9} {'rateMAE':>9} {'maxΔ':>7} {'disagree%':>9}")
    worst = 0.0
    for nm in names:
        a, b = sj_rate[nm].flatten(), nrt_rate[nm].flatten()
        mae = float((a - b).abs().mean()); mx = float((a - b).abs().max())
        dis = float(((a > 0) != (b > 0)).float().mean()) * 100.0
        worst = max(worst, mx)
        print(f"  {nm:10} {a.numel():8d} {a.mean():9.4f} {b.mean():9.4f} {mae:9.5f} {mx:7.4f} {dis:9.2f}")
    return worst


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "sew_resnet18"
    gain = float(os.environ.get("ARCH_GAIN", "3.0"))
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    model = build_model(name, num_classes=10)
    randomize_bn(model, g)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.mul_(gain)

    dag = build_dag(model, name, (32, 32))
    dag.validate()
    mapped = partition_dag(dag); route_dag(dag, mapped); compress_axons(mapped)
    print(f"[compile] {dag.summary()}")
    print(f"[compile] {mapped.num_pe_used} PEs, validate={'OK' if not validate(mapped) else validate(mapped)[:2]}, "
          f"BN folded into {len(dag.biases)} convs")

    convs = [n.name for n in dag.nodes.values() if n.op == "conv"]
    alg1 = all(decompress_matches_conv(mapped, dag, c) for c in convs)
    print(f"[Algorithm 1] decompress == dense conv on {len(convs)} convs (incl. BN-folded): {'OK' if alg1 else 'FAIL'}")

    image = torch.randn(1, 3, 32, 32, generator=g)
    out_node = next(n for n in dag.nodes.values() if n.op == "output")
    analog = set(out_node.inputs)
    names = [n.name for n in topo(dag) if n.op in ("conv", "dense") and n.name not in analog]
    sj_rate, sj_logits = sj_reference(model, image, dag, T)

    print(f"\n===== {name}: FLOAT weights vs SpikingJelly (must be bit-exact) =====")
    nrt_rate_f, nrt_logits_f = nrt_forward(mapped, dag, image, T, quantize=False)
    worst_f = compare(nrt_rate_f, sj_rate, names)
    ldl = float((sj_logits - nrt_logits_f).abs().max())
    exact = worst_f < 1e-5 and ldl < 1e-4
    print(f"  worst firing-rate Δ = {worst_f:.2e}, logits maxΔ = {ldl:.2e}  -> {'EXACT MATCH ✓' if exact else 'MISMATCH ✗'}")

    print(f"\n===== {name}: INT8 weights vs SpikingJelly (deployment, tolerance) =====")
    nrt_rate_q, nrt_logits_q = nrt_forward(mapped, dag, image, T, quantize=True)
    worst_q = compare(nrt_rate_q, sj_rate, names)
    tot = sum(sj_rate[n].numel() for n in names)
    agree = sum(int(((sj_rate[n].flatten() > 0) == (nrt_rate_q[n].flatten() > 0)).sum()) for n in names)
    print(f"  worst Δ={worst_q:.4f}; argmax SJ={int(sj_logits.argmax())} NRT-int8={int(nrt_logits_q.argmax())}; "
          f"spike agreement {100.0*agree/tot:.2f}%")

    print(f"\n[RESULT {name}] Alg1={alg1} float-exact={exact} => "
          f"{'TOPOLOGY ACCURATE (lossless vs SpikingJelly)' if (alg1 and exact) else 'NEEDS INVESTIGATION'}")
    return 0 if (alg1 and exact) else 1


if __name__ == "__main__":
    sys.exit(main())
