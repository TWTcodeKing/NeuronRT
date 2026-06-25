"""Validate DNP age-pruning on a SYNTHETIC temporally-sparse workload (the regime the paper targets).

Static CIFAR gives no temporal idleness, so age-pruning never reclaims (see run_dnp_accuracy.py). Here
we drive the trained svgg9's input with a ROTATING active band: the 32-row input is split into K bands;
in each L-step window only one band carries current (conv0 of a real image, masked), the rest are 0.
A neuron in band b is then idle for (K-1)*L steps per period -> its age_cnt climbs -> age-pruning
reclaims its slot -> it is re-allocated when band b is active again (the membrane lost while idle had
already decayed, so this is ~lossless).

We compare, at the SAME low physical/logical ratio: age-pruning OFF (lazy-alloc + cap -> every band
eventually allocates -> working set fills -> rejects) vs age-pruning ON (idle bands reclaimed -> only
the active set stays mapped -> few rejects, firing preserved). age-ON having far fewer rejects + lower
firing divergence at equal/greater storage reduction demonstrates the reclamation mechanism.

Usage:  PYTHONPATH=. python run_dnp_temporal.py --ckpt checkpoints/svgg9_cifar10.pth
"""
import argparse
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as TT

import run_dnp_sweep as S

K = 4          # spatial bands (over the 32 input rows)
L = 8          # timesteps each band stays active  -> period P = K*L = 32
PERIOD = K * L


def base_current(dag, mapped, in_node, w_in, image):
    cur = F.conv2d(image, w_in, stride=in_node.attrs["stride"], padding=in_node.attrs["padding"])
    if in_node.name in dag.biases:
        cur = cur + torch.as_tensor(dag.biases[in_node.name], dtype=torch.float32).reshape(1, -1, 1, 1)
    _, c, h, w = cur.shape
    return cur.flatten().detach().numpy().astype(np.float64), h, w


def write_input_seq(dag, mapped, in_node, w_in, image, warmup):
    base, H, W = base_current(dag, mapped, in_node, w_in, image)   # conv0 output (C,H,W), id=c*H*W+row*W+col
    band_h = max(1, H // K)
    in_pes = mapped.pes_of_node(in_node.name)
    # input.bin (warmup/measure + a constant fallback = the full base current; overridden per-step below)
    with open(os.path.join(S.OUT, "input.bin"), "wb") as fh:
        fh.write(struct.pack("<III", warmup, S.MEASURE, len(in_pes)))
        for pe in in_pes:
            seg = base[pe.neuron_base:pe.neuron_base + pe.neuron_count]
            fh.write(struct.pack("<II", pe.pe_id, pe.neuron_count))
            fh.write(seg.tobytes())
    # input_seq.bin: PERIOD frames per input PE; frame t keeps only band (t//L)%K (row-masked).
    rows = (np.arange(base.size) % (H * W)) // W           # row of each conv0 neuron
    active_band = rows // band_h                            # which band each neuron belongs to
    with open(os.path.join(S.OUT, "input_seq.bin"), "wb") as fh:
        fh.write(struct.pack("<II", PERIOD, len(in_pes)))
        for pe in in_pes:
            lo, hi = pe.neuron_base, pe.neuron_base + pe.neuron_count
            seg, ab = base[lo:hi], active_band[lo:hi]
            fh.write(struct.pack("<II", pe.pe_id, pe.neuron_count))
            for t in range(PERIOD):
                win = (t // L) % K
                fh.write((seg * (ab == win)).tobytes())    # zero out neurons outside the active band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/svgg9_cifar10.pth")
    ap.add_argument("--data", default="/data/twt/datasets/cifar10")
    ap.add_argument("--n-images", type=int, default=3)
    args = ap.parse_args()

    S.MEASURE = PERIOD                                      # measure exactly one full rotation
    dag, mapped, names, in_node, w_in, _ = S.compile_trained(args.ckpt)
    warmup = 2 * PERIOD + len(names)                        # a couple rotations to reach steady pruning
    tf = TT.Compose([TT.ToTensor(), TT.Normalize(S.CIFAR_MEAN, S.CIFAR_STD)])
    test = tv.datasets.CIFAR10(args.data, train=False, transform=tf, download=False)
    print(f"[temporal] rotating band: K={K} bands x L={L} steps, period={PERIOD}, warmup={warmup}")

    # tight ratio (< the full-period working set) so age-OFF MUST reject accumulated bands; compare
    # against aggressive age-pruning that reclaims idle bands fast. fires-preserved = the real proxy.
    configs = [(["--dnp-off"], "dense"),
               (["--dnp-ratio", "0.20"], "ratio0.20 age-OFF"),
               (["--dnp-ratio", "0.20", "--dnp-age", "2"], "ratio0.20 age=2"),
               (["--dnp-ratio", "0.20", "--dnp-age", "4"], "ratio0.20 age=4")]
    agg = {l: dict(storage=[], fires=[], prune=[], reject=[]) for _, l in configs}

    for k in range(args.n_images):
        write_input_seq(dag, mapped, in_node, w_in, test[k][0].unsqueeze(0), warmup)
        base_fc, _ = S.run(["--dnp-off"])
        base_fires = sum(int(c.sum()) for c in base_fc.values())
        for flags, l in configs:
            fc, dnp = S.run(flags)
            agg[l]["fires"].append(sum(int(c.sum()) for c in fc.values()) / max(1, base_fires))
            if dnp:
                tl = sum(r["n_log"] for r in dnp); tp = sum(r["peak"] for r in dnp)
                agg[l]["storage"].append(tl / max(1, tp))
                agg[l]["prune"].append(sum(r["prune"] for r in dnp))
                agg[l]["reject"].append(sum(r["reject"] for r in dnp))

    print(f"\n  {'config':20} {'storage':>8} {'fires_kept':>11} {'prunes':>10} {'rejects':>10}  (avg/{args.n_images})")
    for _, l in configs:
        a = agg[l]
        st = np.mean(a["storage"]) if a["storage"] else 1.0
        print(f"  {l:20} {st:7.2f}x {np.mean(a['fires'])*100:10.1f}% "
              f"{(np.mean(a['prune']) if a['prune'] else 0):10.0f} "
              f"{(np.mean(a['reject']) if a['reject'] else 0):10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
