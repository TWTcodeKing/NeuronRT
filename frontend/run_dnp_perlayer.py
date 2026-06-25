"""Explore 4x+ DNP storage via PER-LAYER physical ratios (Python upper bound, activity-channel keep).

Uniform-ratio DNP is bottlenecked at 4x by the dense fc layers (firing 37-47% > 25%). But the conv
layers are ~99.5% of all neurons (147466 / 148234) and fire only 8-20%, while fc is 768 neurons. So a
NON-UNIFORM ratio — tight on conv, generous on fc — should reach ~4x AGGREGATE storage (conv dominates
the count) while both layer types keep their active sets (conv firing < conv-ratio, fc firing <
fc-ratio) => near-lossless. This motivates per-PE physical slots in the sim/compiler.

Reports top-1 (real CIFAR test set) + aggregate storage (sum logical / sum kept slots) for several
(conv_ratio, fc_ratio) points, using the implementable channel-level activity keep.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision as tv
import torchvision.transforms as TT
from spikingjelly.activation_based import functional, neuron

import run_dnp_sweep as S
from neurort_compiler.models.registry import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/svgg9_cifar10.pth")
    ap.add_argument("--data", default="/data/twt/datasets/cifar10")
    ap.add_argument("--profile-images", type=int, default=256)
    ap.add_argument("--test-images", type=int, default=2000)
    ap.add_argument("--T", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model("svgg9", num_classes=10).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    dag, mapped, names, *_ = S.compile_trained(args.ckpt)
    lifs = [m for m in model.modules() if isinstance(m, neuron.LIFNode)]
    nname = {nm: dag.nodes[nm].num_neurons() for nm in names}
    shp = {nm: dag.nodes[nm].shape for nm in names}
    hw = {nm: (shp[nm][1] * shp[nm][2] if len(shp[nm]) == 3 else 1) for nm in names}
    is_conv = {nm: len(shp[nm]) == 3 for nm in names}

    tf = TT.Compose([TT.ToTensor(), TT.Normalize(S.CIFAR_MEAN, S.CIFAR_STD)])
    test = tv.datasets.CIFAR10(args.data, train=False, transform=tf, download=False)

    # profile per-neuron firing (offline ranking set)
    fr = {nm: torch.zeros(nname[nm], device=dev) for nm in names}

    def acc_hook(nm):
        def h(mod, i, o):
            fr[nm].add_(o.detach().reshape(o.shape[0], -1).sum(0))
        return h
    hs = [lif.register_forward_hook(acc_hook(nm)) for nm, lif in zip(names, lifs)]
    with torch.no_grad():
        for x, _ in DataLoader(Subset(test, range(args.profile_images)), batch_size=128):
            x = x.to(dev); functional.reset_net(model)
            for _ in range(args.T):
                model(x)
    for h in hs:
        h.remove()
    fr = {nm: v.cpu().numpy() for nm, v in fr.items()}

    def masks(ratio_of):
        out = {}
        for nm in names:
            keep = np.zeros(nname[nm], dtype=np.float32)
            r, h = ratio_of(nm), hw[nm]
            for pe in mapped.pes_of_node(nm):
                lo, hi = pe.neuron_base, pe.neuron_base + pe.neuron_count
                nch = pe.neuron_count // h
                crate = fr[nm][lo:hi].reshape(nch, h).mean(1)
                for c in np.argsort(crate)[::-1][:int(r * nch)]:
                    keep[lo + c * h: lo + (c + 1) * h] = 1.0
            out[nm] = torch.from_numpy(keep).to(dev)
        return out

    def storage(ratio_of):
        tot = kept = 0
        for nm in names:
            for pe in mapped.pes_of_node(nm):
                nch = pe.neuron_count // hw[nm]
                kept += int(ratio_of(nm) * nch) * hw[nm]
                tot += pe.neuron_count
        return tot / max(1, kept)

    loader = DataLoader(Subset(test, range(args.test_images)), batch_size=200)

    def accuracy(mask):
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                functional.reset_net(model)
                hooks = ([lif.register_forward_hook(
                    lambda m, i, o, mm=mask[nm]: o * mm.view((1,) + o.shape[1:]))
                    for nm, lif in zip(names, lifs)] if mask else [])
                acc = None
                for _ in range(args.T):
                    out = model(x); acc = out if acc is None else acc + out
                for h in hooks:
                    h.remove()
                correct += (acc.argmax(1) == y).sum().item(); total += y.size(0)
        return correct / total

    print(f"\n  {'config':22} {'conv_r':>7} {'fc_r':>5} {'storage':>8} {'top1':>7}")
    base = accuracy(None)
    print(f"  {'baseline (no DNP)':22} {'-':>7} {'-':>5} {'1.00x':>8} {base*100:6.1f}%")
    configs = [("uniform 0.25", 0.25, 0.25), ("uniform 0.50", 0.50, 0.50),
               ("conv0.25 fc1.0", 0.25, 1.0), ("conv0.25 fc0.5", 0.25, 0.5),
               ("conv0.20 fc0.6", 0.20, 0.6), ("conv0.18 fc0.7", 0.18, 0.7),
               ("conv0.15 fc0.7", 0.15, 0.7)]
    for label, cr, fcr in configs:
        rof = lambda nm, cr=cr, fcr=fcr: cr if is_conv[nm] else fcr
        print(f"  {label:22} {cr:7.2f} {fcr:5.2f} {storage(rof):7.2f}x {accuracy(masks(rof))*100:6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
