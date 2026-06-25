"""Train SEW-ResNet18 on DVS128Gesture (T=16) — a temporally-sparse, event-stream workload, the regime
where NeuroRT's DNP delivers its real win (neurons idle between events -> age-pruning reclaims slots).

DVS128Gesture frames are pre-integrated to T=16 (frames_number_16_split_by_number, already cached).
SEW-ResNet18 is adapted to 2 polarity input channels and 11 gesture classes; multi-step BPTT with the
LIF surrogate. The trained checkpoint feeds the later DNP evaluation, where the T=16 event frames drive
the sim's time-varying input (input_seq.bin) so age-pruning sees genuine temporal idleness.

Usage:  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python train_dvs_sewresnet.py --epochs 100 --out checkpoints/sew_resnet18_dvsgesture.pth
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional, layer, neuron
from spikingjelly.activation_based.model import sew_resnet
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def build_model(num_classes):
    m = sew_resnet.sew_resnet18(spiking_neuron=neuron.LIFNode, cnf="ADD", num_classes=num_classes)
    # DVS has 2 polarity channels; use spikingjelly's layer.Conv2d so set_step_mode wraps it for [T,B,..]
    m.conv1 = layer.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    functional.set_step_mode(m, "m")                                             # multi-step [T,B,C,H,W]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/twt/datasets/dvs128gesture")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/sew_resnet18_dvsgesture.pth")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    train = DVS128Gesture(args.data, train=True, data_type="frame", frames_number=args.T, split_by="number")
    test = DVS128Gesture(args.data, train=False, data_type="frame", frames_number=args.T, split_by="number")
    tl = DataLoader(train, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(test, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"[cfg] dvsgesture T={args.T} train={len(train)} test={len(test)} batch={args.batch} "
          f"epochs={args.epochs} lr={args.lr} dev={dev}", flush=True)

    net = build_model(11).to(dev)
    print(f"[model] sew_resnet18 (2ch, 11 classes): {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    def step_batch(x, y, train_mode):
        x = x.float().transpose(0, 1).to(dev, non_blocking=True)   # [B,T,2,H,W] -> [T,B,2,H,W]
        y = y.to(dev, non_blocking=True)
        functional.reset_net(net)
        logits = net(x).mean(0)                                    # [T,B,11] -> [B,11]
        loss = crit(logits, y)
        if train_mode:
            opt.zero_grad(); loss.backward(); opt.step()
        return loss.item() * y.size(0), (logits.argmax(1) == y).sum().item(), y.size(0)

    best = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ep in range(args.epochs):
        net.train(); t0 = time.time(); ls = corr = seen = 0
        for bi, (x, y) in enumerate(tl):
            if args.smoke and bi >= args.smoke:
                break
            l, c, n = step_batch(x, y, True); ls += l; corr += c; seen += n
        sched.step()
        net.eval(); vc = vs = 0
        with torch.no_grad():
            for bi, (x, y) in enumerate(vl):
                if args.smoke and bi >= args.smoke:
                    break
                _, c, n = step_batch(x, y, False); vc += c; vs += n
        tr, va = corr / max(1, seen), vc / max(1, vs)
        print(f"epoch {ep:3d}  lr {sched.get_last_lr()[0]:.5f}  loss {ls/max(1,seen):.4f}  "
              f"train {tr:.4f}  test {va:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if va > best:
            best = va
            torch.save({"model": net.state_dict(), "acc": best, "epoch": ep, "T": args.T,
                        "arch": "sew_resnet18", "dataset": "dvs128gesture", "num_classes": 11}, args.out)
            print(f"  * saved best {best:.4f} -> {args.out}", flush=True)
    print(f"[done] best test acc {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
