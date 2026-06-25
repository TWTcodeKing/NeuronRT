"""Train SVGG9 (9-weight-layer spiking VGG) on CIFAR-10 with SpikingJelly surrogate-gradient BPTT.

The trained weights are the input to the NeuroRT compiler: a TRAINED classifier produces realistic
output sparsity (learned feature selectivity -> many persistently-silent "zombie" neurons), which is
exactly what Dynamic Neuron Pruning exploits — unlike random weights, where ~every neuron is active.

Runs the single-step model in a manual T-step loop (functional.reset_net per sample, mean of the
per-step analog logits as the readout), identical to how funcsim/the NoC sim run the compiled net.
Saves the best test-accuracy checkpoint (state_dict) for later compilation + DNP evaluation.

Usage (pin a GPU + run in tmux):
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python train_svgg9_cifar10.py --epochs 120 --out checkpoints/svgg9_cifar10.pth
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as TT
from spikingjelly.activation_based import functional

from neurort_compiler.models.registry import build_model

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_loaders(root, batch, workers):
    tf_train = TT.Compose([TT.RandomCrop(32, padding=4), TT.RandomHorizontalFlip(),
                           TT.ToTensor(), TT.Normalize(CIFAR_MEAN, CIFAR_STD)])
    tf_test = TT.Compose([TT.ToTensor(), TT.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train = tv.datasets.CIFAR10(root, train=True, transform=tf_train, download=False)
    test = tv.datasets.CIFAR10(root, train=False, transform=tf_test, download=False)
    train_loader = DataLoader(train, batch, shuffle=True, num_workers=workers, pin_memory=True,
                              drop_last=True, persistent_workers=workers > 0)
    test_loader = DataLoader(test, 256, shuffle=False, num_workers=workers, pin_memory=True,
                             persistent_workers=workers > 0)
    return train_loader, test_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/twt/datasets/cifar10")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/svgg9_cifar10.pth")
    ap.add_argument("--init-gain", type=float, default=2.5,
                    help="conv/linear init gain; BN-free spiking VGG dies at PyTorch's default init "
                         "(a=sqrt(5)) -> use kaiming-relu * gain so spikes propagate through all layers")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, run only this many train batches/epoch (test)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"[cfg] device={dev} epochs={args.epochs} batch={args.batch} T={args.T} lr={args.lr} "
          f"wd={args.wd} out={args.out}", flush=True)

    train_loader, test_loader = build_loaders(args.data, args.batch, args.workers)
    net = build_model("svgg9", num_classes=10).to(dev)
    for m in net.modules():   # spiking-friendly init (see --init-gain): keep all layers firing
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            with torch.no_grad():
                m.weight.mul_(args.init_gain)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[model] svgg9: {n_params/1e6:.2f}M params  init_gain={args.init_gain}", flush=True)

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    def forward_T(x):
        """Single-step net in a T-step loop; mean of the per-step analog logits (rate readout)."""
        functional.reset_net(net)
        acc = None
        for _ in range(args.T):
            out = net(x)
            acc = out if acc is None else acc + out
        return acc / args.T

    best = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        seen = correct = 0
        loss_sum = 0.0
        for bi, (x, y) in enumerate(train_loader):
            if args.smoke and bi >= args.smoke:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            logits = forward_T(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            seen += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        sched.step()
        train_acc = correct / max(1, seen)

        net.eval()
        tseen = tcorrect = 0
        with torch.no_grad():
            for bi, (x, y) in enumerate(test_loader):
                if args.smoke and bi >= args.smoke:
                    break
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                logits = forward_T(x)
                tcorrect += (logits.argmax(1) == y).sum().item()
                tseen += y.size(0)
        test_acc = tcorrect / max(1, tseen)

        print(f"epoch {ep:3d}  lr {sched.get_last_lr()[0]:.4f}  loss {loss_sum/max(1,seen):.4f}  "
              f"train {train_acc:.4f}  test {test_acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if test_acc > best:
            best = test_acc
            torch.save({"model": net.state_dict(), "acc": best, "epoch": ep, "T": args.T,
                        "arch": "svgg9"}, args.out)
            print(f"  * saved best {best:.4f} -> {args.out}", flush=True)
    print(f"[done] best test acc {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
