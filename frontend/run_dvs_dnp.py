"""DNP behavior on the DVS workload (trained SEW-ResNet18, DVS128Gesture, T=16).

The static-CIFAR study found DNP's lossless ceiling is ~2x because input is constant -> every neuron
receives input every timestep (~100% per-step working set) -> age-pruning reclaims nothing. Event
streams are TEMPORALLY SPARSE: events (and thus each neuron's input) occur only in some timesteps, so
a neuron is idle in stretches -> age-pruning reclaims its slot -> the peak simultaneously-mapped slots
is the PER-TIMESTEP active set, far below the union over T. That is the regime for the paper's 5x.

We profile per-neuron, per-timestep ALLOCATION (LIF receives nonzero input) over T=16 and report, per
layer + aggregate: the per-step working set, the union-over-T working set, and the DNP peak slots under
age-pruning AGE_THRESH=k (a neuron holds a slot while its last input was within k steps). storage =
n_log / peak. k=inf (no aging) = lazy-alloc only (peak=union); k=1 = aggressive (peak=max per-step).

(The chip sim can't run sew_resnet18 @128x128 — >576 PEs, no temporal SRAM folding yet — so this is the
direct measurement of the temporal sparsity DNP exploits, at the model level.)
"""
import argparse

import numpy as np
import torch
from spikingjelly.activation_based import functional, neuron

from train_dvs_sewresnet import build_model
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def age_peak(active, k):
    """active[T, n] binary (neuron active = fired). Peak simultaneously-mapped slots under
    AGE_THRESH=k: a neuron holds a slot iff it was active within the last k steps (reclaim after k
    idle steps). k>=T => union (no reclaim); k=1 => only currently-active. (Firing is the clean
    temporal-activity signal; the sim's spike-driven allocation tracks it, unlike the BN-biased input
    which is nonzero ~everywhere.)"""
    T = active.shape[0]
    mapped = np.zeros_like(active)
    last = np.full(active.shape[1], -10_000)
    for t in range(T):
        last = np.where(active[t] > 0, t, last)
        mapped[t] = (t - last) < k
    return int(mapped.sum(1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sew_resnet18_dvsgesture.pth")
    ap.add_argument("--data", default="/data/twt/datasets/dvs128gesture")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--T", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = build_model(11).to(dev).eval()
    net.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    test = DVS128Gesture(args.data, train=False, data_type="frame", frames_number=args.T, split_by="number")

    lifs = [m for m in net.modules() if isinstance(m, neuron.LIFNode)]
    cap = {}

    def post(idx):
        def h(mod, inp, out):
            cap[idx] = (out[:, 0] > 0).reshape(out.shape[0], -1).cpu().numpy()   # [T, n] fired per step
        return h
    hs = [lif.register_forward_hook(post(i)) for i, lif in enumerate(lifs)]

    nlif = len(lifs)
    # accumulators per LIF layer
    n_neu = [0] * nlif
    sum_perstep = np.zeros(nlif)      # mean per-step active count
    sum_union = np.zeros(nlif)        # union active count
    sum_peak = {k: np.zeros(nlif) for k in (1, 2, 4, 8)}
    correct = 0

    for s in range(args.n_samples):
        x, y = test[s]
        xb = torch.as_tensor(x).float().unsqueeze(1).to(dev)   # [T,1,2,128,128]
        functional.reset_net(net)
        with torch.no_grad():
            out = net(xb)                                       # [T,1,11]
        correct += int(out.mean(0).argmax(1).item() == int(y))
        for i in range(nlif):
            a = cap[i]                                          # [T, n]
            n_neu[i] = a.shape[1]
            sum_perstep[i] += a.sum(1).mean()
            sum_union[i] += (a.sum(0) > 0).sum()
            for k in sum_peak:
                sum_peak[k][i] += age_peak(a, k)
    for h in hs:
        h.remove()

    ns = args.n_samples
    tot = sum(n_neu)
    print(f"\n[dvs] sew_resnet18 DVS128Gesture, {ns} samples, T={args.T}, "
          f"profiling-acc {correct/ns*100:.1f}%  ({nlif} LIF layers, {tot} neurons)")
    perstep = sum_perstep.sum() / ns
    union = sum_union.sum() / ns
    print(f"\n  aggregate working set (fraction of {tot} neurons):")
    print(f"    per-timestep active : {perstep/tot*100:5.1f}%  -> {tot/max(1,perstep):5.2f}x  (age-pruning floor)")
    print(f"    union over T={args.T:<2d}    : {union/tot*100:5.1f}%  -> {tot/max(1,union):5.2f}x  (lazy-alloc only)")
    print(f"\n  DNP peak slots & storage by AGE_THRESH:")
    for k in (1, 2, 4, 8):
        pk = sum_peak[k].sum() / ns
        print(f"    age={k:<2d}: peak {pk/tot*100:5.1f}% -> {tot/max(1,pk):5.2f}x storage")
    print(f"    age>=T (no prune)  -> {tot/max(1,union):5.2f}x  (== lazy-alloc, union)")
    print(f"\n  temporal-sparsity factor (union / per-step) = {union/max(1,perstep):.2f}x  "
          f"(static CIFAR ~= 1.0; >1 means age-pruning reclaims idle neurons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
