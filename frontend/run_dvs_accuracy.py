"""Classification accuracy of DNP on the DVS workload (trained SEW-ResNet18, DVS128Gesture, T=16).

The chip sim gave firing-fidelity (1.2% MAE at 15-39x storage) but a transient DVS clip suffers a
pipeline-vs-single-step mismatch, so for TOP-1 we use the trained multi-step model (correct residual +
GAP + fc forward) and emulate DNP by keeping, per PE, only the ratio*count most-active neurons (by
profiled union firing) and zeroing the rest's spikes — exactly DNP's effect (a neuron with no slot
never fires). DVS firing is so sparse (union ~6.6%) that ratio >= ~0.07 keeps all active neurons ->
lossless; this measures where it stays lossless and where it degrades.

Usage:  PYTHONPATH=. python run_dvs_accuracy.py --ckpt checkpoints/sew_resnet18_dvsgesture.pth
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional, layer, neuron
from spikingjelly.activation_based.model import sew_resnet
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

from neurort_compiler.graph.builders import build_dag
from neurort_compiler.graph.dag import topo
from neurort_compiler.mapping.partition import partition_dag

T = 16


def make_model(dev, step):
    m = sew_resnet.sew_resnet18(spiking_neuron=neuron.LIFNode, cnf="ADD", num_classes=11)
    m.conv1 = layer.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    functional.set_step_mode(m, step)
    return m.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sew_resnet18_dvsgesture.pth")
    ap.add_argument("--data", default="/data/twt/datasets/dvs128gesture")
    ap.add_argument("--profile-batches", type=int, default=8)
    ap.add_argument("--ratios", default="0.10,0.07,0.05,0.03")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.ckpt, map_location=dev)["model"]

    net = make_model(dev, "m"); net.load_state_dict(sd)
    lifs = [m for m in net.modules() if isinstance(m, neuron.LIFNode)]

    # compile (single-step) for the PE layout; map LIF layers <-> compiled nodes by execution order.
    cnet = make_model("cpu", "s"); cnet.load_state_dict(sd)
    dag = build_dag(cnet, "sew_resnet18", (128, 128), in_ch=2)
    mapped = partition_dag(dag)
    out_node = next(n for n in dag.nodes.values() if n.op == "output")
    analog = set(out_node.inputs)
    names = [n.name for n in topo(dag) if n.op in ("conv", "dense") and n.name not in analog]
    assert len(names) == len(lifs), f"{len(names)} names vs {len(lifs)} LIF"
    nname = {nm: dag.nodes[nm].num_neurons() for nm in names}

    test = DVS128Gesture(args.data, train=False, data_type="frame", frames_number=T, split_by="number")
    pes_of = {nm: list(mapped.pes_of_node(nm)) for nm in names}

    # DNP is INPUT-ADAPTIVE: per clip it allocates the neurons active FOR THAT clip (the others never
    # claim a slot). So per sample we (1) forward to get this clip's per-neuron firing, (2) per PE keep
    # the ratio*count most-active THIS-clip neurons, zero the rest's spikes, (3) re-forward -> logits.
    # If a PE's per-clip active set fits ratio*count, all its active neurons fire -> lossless.
    ratios = [float(r) for r in args.ratios.split(",")]
    correct = {r: 0 for r in [None] + ratios}
    agree = {r: 0 for r in ratios}
    # per-clip max per-PE active fraction (the ratio needed for that clip to be lossless)
    need_ratio = []
    n = 0
    for s in range(len(test)):
        x = torch.as_tensor(test[s][0]).float().unsqueeze(1).to(dev)   # [T,1,2,128,128]
        y = int(test[s][1])
        # (1) dense forward, capture per-clip firing
        cap = {}
        def hk(nm):
            def h(m, i, o): cap[nm] = (o.detach()[:, 0] > 0).any(0).reshape(-1)   # [n] fired this clip
            return h
        hs = [lif.register_forward_hook(hk(nm)) for nm, lif in zip(names, lifs)]
        functional.reset_net(net); dense_pred = int(net(x).mean(0).argmax(1).item())
        for h in hs:
            h.remove()
        correct[None] += int(dense_pred == y)
        # per-clip per-PE active fraction
        mx = 0.0
        for nm in names:
            a = cap[nm].float().cpu().numpy()
            for pe in pes_of[nm]:
                mx = max(mx, a[pe.neuron_base:pe.neuron_base + pe.neuron_count].mean())
        need_ratio.append(mx)
        # (2)+(3) per ratio: keep top-(r*count) by THIS clip's firing per PE
        for r in ratios:
            masks = {}
            for nm in names:
                keep = torch.zeros(nname[nm])
                a = cap[nm].float().cpu().numpy()
                for pe in pes_of[nm]:
                    lo, hi = pe.neuron_base, pe.neuron_base + pe.neuron_count
                    k = max(1, int(r * pe.neuron_count))
                    keep[lo + np.argsort(a[lo:hi])[::-1][:k]] = 1.0
                masks[nm] = keep.to(dev)
            hk2 = [lif.register_forward_hook(
                lambda m, i, o, mm=masks[nm]: o * mm.view((1, 1) + o.shape[2:]))
                for nm, lif in zip(names, lifs)]
            functional.reset_net(net); p = int(net(x).mean(0).argmax(1).item())
            for h in hk2:
                h.remove()
            correct[r] += int(p == y); agree[r] += int(p == dense_pred)
        n += 1
        if n % 64 == 0:
            print(f"  ...{n}/{len(test)}", flush=True)

    print(f"\n  [dvs] sew_resnet18 DVS128Gesture, {n} test clips, T={T}, {mapped.num_pe_used} PEs "
          f"(input-adaptive DNP keep)")
    print(f"  per-clip max per-PE active fraction: mean {np.mean(need_ratio)*100:.1f}%  "
          f"p90 {np.percentile(need_ratio,90)*100:.1f}%  max {np.max(need_ratio)*100:.1f}%")
    tot = sum(nname[nm] for nm in names)
    print(f"\n  {'config':14} {'storage':>8} {'top1':>7} {'=dense':>8}")
    print(f"  {'dense':14} {'1.00x':>8} {correct[None]/n*100:6.1f}% {'100.0%':>8}")
    for r in ratios:
        kept = sum(max(1, int(r * pe.neuron_count)) for nm in names for pe in pes_of[nm])
        print(f"  ratio {r:<8.3f} {tot/kept:7.2f}x {correct[r]/n*100:6.1f}% {agree[r]/n*100:7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
