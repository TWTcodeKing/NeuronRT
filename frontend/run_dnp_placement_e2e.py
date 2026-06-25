"""End-to-end validation of firing-rate-aware placement on the real C++ DNP sim.

Compile the trained svgg9 two ways and run the actual NoC + DNP sim at a sweep of physical ratios,
measuring top-1 (last-LIF spikes x int8 output weights), for:
  * BASE   — current contiguous channel partition
  * PLACED — firing-rate-aware channel permutation (mapping/firing_placement.py), function-preserving

Expectation (from the Python upper-bound): base collapses under capping (~random at ratio 0.5),
placed recovers to ~baseline at ratio 0.5 (2x storage), confirming the compiler optimization works on
the actual DNP hardware model.

Usage:  PYTHONPATH=. python run_dnp_placement_e2e.py --ckpt checkpoints/svgg9_cifar10.pth --n-images 20
"""
import argparse

import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as TT
from spikingjelly.activation_based import functional, neuron

import run_dnp_sweep as S
from neurort_compiler.funcsim import reassemble_weight
from neurort_compiler.graph.builders import build_dag
from neurort_compiler.graph.dag import topo
from neurort_compiler.mapping.partition import partition_dag
from neurort_compiler.mapping.firing_placement import apply_firing_placement
from neurort_compiler.models.registry import build_model


def weight_modules(model):
    mods = []
    for seq in (model.features, model.classifier):
        for m in seq:
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                mods.append(m)
    return mods                                   # [conv0..conv5, fc0, fc1, fc2]


def profile_channel_firing(model, lifs, names, dag, imgs, T, dev):
    """Per-OUTPUT-CHANNEL firing rate of each permuted layer (the LIF after it), in model order."""
    acc = {nm: None for nm in names}

    def hook(nm):
        def h(mod, i, o):
            f = o.detach()
            ch = f.shape[1] if f.dim() >= 2 else f.shape[-1]
            acc[nm] = (f.reshape(f.shape[0], ch, -1).mean((0, 2)) if acc[nm] is None
                       else acc[nm] + f.reshape(f.shape[0], ch, -1).mean((0, 2)))
        return h
    hs = [lif.register_forward_hook(hook(nm)) for nm, lif in zip(names, lifs)]
    with torch.no_grad():
        for x in imgs:
            functional.reset_net(model)
            for _ in range(T):
                model(x.to(dev))
    for h in hs:
        h.remove()
    return [acc[nm].cpu().numpy() for nm in names]


def accuracy(dag, mapped, names, in_node, w_in, out_dir, test, n_images, warmup, flags):
    S.OUT = out_dir
    fc2 = next(n for n in dag.nodes.values() if n.op == "output").inputs[0]
    fc1 = dag.nodes[fc2].inputs[0]
    W = reassemble_weight(mapped, dag, fc2, quantize=True).detach().numpy()   # [10, n_fc1]
    nf1 = dag.nodes[fc1].num_neurons()
    correct = 0
    for k in range(n_images):
        S.write_input_bin(dag, mapped, in_node, w_in, test[k][0].unsqueeze(0), warmup)
        fc, _ = S.run(flags)
        rate = np.zeros(nf1)
        for pe in mapped.pes_of_node(fc1):
            rate[pe.neuron_base:pe.neuron_base + pe.neuron_count] = fc[pe.pe_id]
        correct += int(np.argmax(W @ (rate / S.MEASURE)) == int(test[k][1]))
    return correct / n_images


def compile_variant(model, out_dir):
    S.OUT = out_dir
    dag, mapped, names, in_node, w_in, warmup = S.compile_trained(None, model=model)
    return dict(dag=dag, mapped=mapped, names=names, in_node=in_node, w_in=w_in, warmup=warmup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/svgg9_cifar10.pth")
    ap.add_argument("--data", default="/data/twt/datasets/cifar10")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--profile-images", type=int, default=64)
    ap.add_argument("--T", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tf = TT.Compose([TT.ToTensor(), TT.Normalize(S.CIFAR_MEAN, S.CIFAR_STD)])
    test = tv.datasets.CIFAR10(args.data, train=False, transform=tf, download=False)

    # BASE model + dry partition (for per-layer num_pe), then profile firing, then PLACED model.
    base_model = build_model("svgg9", num_classes=10).to(dev).eval()
    base_model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    dag0 = build_dag(base_model.cpu(), "svgg9", (S.HW, S.HW)); base_model.to(dev)
    mapped0 = partition_dag(dag0)
    out_node = next(n for n in dag0.nodes.values() if n.op == "output")
    analog = set(out_node.inputs)
    names = [n.name for n in topo(dag0) if n.op in ("conv", "dense") and n.name not in analog]
    # ACTUAL per-PE channel counts from the partition (channels = neurons / spatial-size); reorder
    # active-first within these exact blocks so each PE's channel set (-> int8 scale, function) is kept.
    block_sizes = []
    for nm in names:
        shp = dag0.nodes[nm].shape
        hw = shp[1] * shp[2] if len(shp) == 3 else 1
        block_sizes.append([pe.neuron_count // hw for pe in mapped0.pes_of_node(nm)])

    lifs = [m for m in base_model.modules() if isinstance(m, neuron.LIFNode)]
    imgs = [test[k][0].unsqueeze(0) for k in range(args.profile_images)]
    chan_fire = profile_channel_firing(base_model, lifs, names, dag0, imgs, args.T, dev)
    print("[profile] per-layer mean channel firing:",
          ", ".join(f"{nm}={f.mean():.3f}" for nm, f in zip(names, chan_fire)))

    placed_model = build_model("svgg9", num_classes=10).eval()
    placed_model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    perms = apply_firing_placement(weight_modules(placed_model), chan_fire, block_sizes)
    print(f"[placement] permuted {len(perms)} layers (firing-balanced, active-first per PE)")

    base = compile_variant(base_model.cpu(), "/tmp/dnp_base")
    placed = compile_variant(placed_model, "/tmp/dnp_placed")

    print(f"\n  variant   {'dense':>7} {'ratio0.5':>9} {'ratio0.25':>10}  (top-1 on {args.n_images} real CIFAR, real DNP sim)")
    for label, v in [("BASE  ", base), ("PLACED", placed)]:
        accs = []
        for flags in (["--dnp-off"], ["--dnp-ratio", "0.5"], ["--dnp-ratio", "0.25"]):
            accs.append(accuracy(v["dag"], v["mapped"], v["names"], v["in_node"], v["w_in"],
                                 ("/tmp/dnp_base" if label.strip() == "BASE" else "/tmp/dnp_placed"),
                                 test, args.n_images, v["warmup"], flags))
        print(f"  {label}   {accs[0]*100:6.1f}% {accs[1]*100:8.1f}% {accs[2]*100:9.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
