"""Quick upper-bound test of firing-rate-aware placement for DNP (does prioritizing active neurons
recover accuracy under a tight physical/logical ratio?).

DNP gives each PE n_phys = ratio*count slots; with ~100% allocation it must drop (1-ratio) of every
PE's neurons. WHICH neurons survive is the whole game. We emulate the survivors per PE on the trained
svgg9 (real CIFAR test set) under three policies and measure top-1 accuracy:
  * none         — full network (baseline)
  * id-first     — keep local ids [0, ratio*count) per PE  == the current DNP (id-order allocation)
  * activity-top — keep the ratio*count HIGHEST firing-rate neurons per PE == firing-rate-aware placement

A dropped neuron's firing is zeroed (no slot -> never fires -> doesn't propagate), exactly DNP's effect.
This abstracts the sim's dynamics to isolate "which neurons get slots", giving the best case for each
policy. activity-top >> id-first confirms the compiler optimization is worth implementing.
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
    ap.add_argument("--profile-images", type=int, default=256)   # firing-rate ranking (offline, like the compiler)
    ap.add_argument("--test-images", type=int, default=2000)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--ratios", default="0.5,0.25")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model("svgg9", num_classes=10).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    dag, mapped, names, *_ = S.compile_trained(args.ckpt)
    lifs = [m for m in model.modules() if isinstance(m, neuron.LIFNode)]
    assert len(lifs) == len(names)
    nname = {nm: dag.nodes[nm].num_neurons() for nm in names}

    tf = TT.Compose([TT.ToTensor(), TT.Normalize(S.CIFAR_MEAN, S.CIFAR_STD)])
    test = tv.datasets.CIFAR10(args.data, train=False, transform=tf, download=False)

    def forward_T(x, masks=None):
        """T-step mean logits; if masks given, zero each LIF layer's dropped neurons every step."""
        functional.reset_net(model)
        hs = []
        if masks is not None:
            for nm, lif in zip(names, lifs):
                m = masks[nm]
                hs.append(lif.register_forward_hook(
                    lambda mod, i, o, m=m: o * m.view((1,) + o.shape[1:])))
        acc = None
        for _ in range(args.T):
            out = model(x)
            acc = out if acc is None else acc + out
        for h in hs:
            h.remove()
        return acc / args.T

    # 1) profile per-neuron firing rate (offline ranking set)
    fr = {nm: torch.zeros(nname[nm], device=dev) for nm in names}

    def acc_hook(nm):
        def h(mod, i, o):                       # MUST return None (non-None replaces the layer output)
            fr[nm].add_(o.detach().reshape(o.shape[0], -1).sum(0))
        return h
    hs = [lif.register_forward_hook(acc_hook(nm)) for nm, lif in zip(names, lifs)]
    prof = DataLoader(Subset(test, range(args.profile_images)), batch_size=128)
    with torch.no_grad():
        for x, _ in prof:
            x = x.to(dev); functional.reset_net(model)
            for _ in range(args.T):
                model(x)
    for h in hs:
        h.remove()
    fr = {nm: v.cpu().numpy() for nm, v in fr.items()}

    # 2) build per-PE keep masks for each policy/ratio
    def build_masks(ratio, policy):
        masks = {}
        for nm in names:
            keep = np.zeros(nname[nm], dtype=np.float32)
            for pe in mapped.pes_of_node(nm):
                lo, hi = pe.neuron_base, pe.neuron_base + pe.neuron_count
                k = int(ratio * pe.neuron_count)
                if policy == "id":
                    keep[lo:lo + k] = 1.0
                elif policy == "act":  # activity-top: highest firing rate NEURONS within this PE
                    order = np.argsort(fr[nm][lo:hi])[::-1]
                    keep[lo + order[:k]] = 1.0
                else:  # "chan": keep highest firing-rate CHANNELS whole (== compiler channel placement)
                    shp = dag.nodes[nm].shape
                    hw = shp[1] * shp[2] if len(shp) == 3 else 1
                    c0, nch = lo // hw, pe.neuron_count // hw
                    crate = fr[nm][lo:hi].reshape(nch, hw).mean(1)
                    kc = int(ratio * nch)
                    for c in np.argsort(crate)[::-1][:kc]:
                        keep[lo + c * hw: lo + (c + 1) * hw] = 1.0
            masks[nm] = torch.from_numpy(keep).to(dev)
        return masks

    # 3) measure top-1 on the test set
    loader = DataLoader(Subset(test, range(args.test_images)), batch_size=200)
    def accuracy(masks):
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                pred = forward_T(x, masks).argmax(1)
                correct += (pred == y).sum().item(); total += y.size(0)
        return correct / total

    print(f"\n  policy            " + "  ".join(f"r={r}" for r in args.ratios.split(",")))
    base = accuracy(None)
    print(f"  none (baseline)    {base*100:5.1f}%")
    for policy, label in [("id", "id-first (=DNP)  "), ("chan", "activity-channel "), ("act", "activity-neuron  ")]:
        accs = [accuracy(build_masks(float(r), policy)) for r in args.ratios.split(",")]
        print(f"  {label} " + "  ".join(f"{a*100:5.1f}%" for a in accs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
