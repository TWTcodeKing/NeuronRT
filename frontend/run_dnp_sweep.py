"""Soma-DNP (Algorithm 2) end-to-end evaluation on the C++ NoC with a TRAINED svgg9 + REAL CIFAR-10.

A trained classifier has learned feature selectivity -> real implicit output sparsity (many neurons
persistently silent on a given input). That is the regime DNP is designed for, unlike random weights
(where ~every neuron is active). We compile the trained net once, then for each of several real test
images: run the dense Soma (baseline) and DNP at a sweep of physical/logical ratios + prune
thresholds, and report (averaged over images):
  * GOLDEN #1 — DNP with n_phys >= n_log, pruning off (--dnp-ratio 1.0) reproduces the dense baseline
    firing BYTE-IDENTICALLY (lossless virtual-memory plumbing), and the lazy-alloc working set
    (peak physical slots / logical) now drops below 1 because silent neurons never claim a slot.
  * sweep — storage reduction (logical / peak slots), firing divergence vs the dense baseline (the
    accuracy proxy: low rateMAE on the last LIF layer => preserved logits), and prune / reject counts.

Usage:  PYTHONPATH=. python run_dnp_sweep.py --ckpt checkpoints/svgg9_cifar10.pth --n-images 4
"""
import argparse
import os
import struct
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as TT

from neurort_compiler.funcsim import reassemble_weight
from neurort_compiler.graph.builders import build_dag
from neurort_compiler.graph.dag import topo
from neurort_compiler.export.writer import write_network
from neurort_compiler.mapping.partition import compress_axons, partition_dag, route_dag, validate
from neurort_compiler.models.registry import build_model

HW = 32
MEASURE = 32
SIM = "/home/twt/NeuronRT/build/sim/neurort_sim"
OUT = "/tmp/dnp_svgg9"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def compile_trained(ckpt_path, model=None):
    if model is None:                              # else: caller passes a (possibly permuted) model
        model = build_model("svgg9", num_classes=10)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"[ckpt] {ckpt_path}  test_acc={ckpt.get('acc'):.4f}  epoch={ckpt.get('epoch')}")

    dag = build_dag(model, "svgg9", (HW, HW))
    dag.validate()
    mapped = partition_dag(dag)
    route_dag(dag, mapped)
    compress_axons(mapped)
    assert validate(mapped) == [], validate(mapped)[:3]
    out_node = next(n for n in dag.nodes.values() if n.op == "output")
    analog = set(out_node.inputs)
    names = [n.name for n in topo(dag) if n.op in ("conv", "dense") and n.name not in analog]
    os.makedirs(OUT, exist_ok=True)
    write_network(mapped, OUT, timesteps=MEASURE)
    ap = os.path.join(OUT, "attention.bin")          # feed-forward net: no stale attention sidecar
    if os.path.exists(ap):
        os.remove(ap)

    in_node = next(n for n in dag.nodes.values()
                   if n.op == "conv" and dag.nodes[n.inputs[0]].op == "input")
    w_in = reassemble_weight(mapped, dag, in_node.name, quantize=True)   # int8 conv0 (what the chip uses)
    warmup = len(names) + 40
    print(f"[compile] svgg9: {mapped.num_pe_used} PEs, {len(names)} weight layers, warmup={warmup}")
    return dag, mapped, names, in_node, w_in, warmup


def write_input_bin(dag, mapped, in_node, w_in, image, warmup):
    cur = F.conv2d(image, w_in, stride=in_node.attrs["stride"], padding=in_node.attrs["padding"])
    if in_node.name in dag.biases:
        cur = cur + torch.as_tensor(dag.biases[in_node.name], dtype=torch.float32).reshape(1, -1, 1, 1)
    cur_flat = cur.flatten().detach().numpy().astype(np.float64)
    in_pes = mapped.pes_of_node(in_node.name)
    with open(os.path.join(OUT, "input.bin"), "wb") as fh:
        fh.write(struct.pack("<III", warmup, MEASURE, len(in_pes)))
        for pe in in_pes:
            seg = cur_flat[pe.neuron_base:pe.neuron_base + pe.neuron_count]
            fh.write(struct.pack("<II", pe.pe_id, pe.neuron_count))
            fh.write(seg.tobytes())


def run(flags):
    r = subprocess.run([SIM, "--network", OUT] + flags, capture_output=True, text=True)
    if r.returncode != 0:
        print("  STDERR:", r.stderr.strip())
        raise RuntimeError(f"sim failed ({r.returncode})")
    return read_firing(), read_dnp()


def read_firing():
    fc = {}
    with open(os.path.join(OUT, "firing.bin"), "rb") as fh:
        npe = struct.unpack("<I", fh.read(4))[0]
        for _ in range(npe):
            pe_id, cnt = struct.unpack("<II", fh.read(8))
            fc[pe_id] = np.frombuffer(fh.read(4 * cnt), dtype=np.uint32).copy()
    return fc


def read_dnp():
    path = os.path.join(OUT, "dnp.bin")
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, "rb") as fh:
        n = struct.unpack("<I", fh.read(4))[0]
        for _ in range(n):
            pe, n_log, n_phys, peak, reject = struct.unpack("<IIIII", fh.read(20))
            (prune,) = struct.unpack("<Q", fh.read(8))
            recs.append(dict(pe=pe, n_log=n_log, n_phys=n_phys, peak=peak, reject=reject, prune=prune))
    return recs


def firing_mae(a, b):
    num = den = 0.0
    for pe_id, ca in a.items():
        num += np.abs(ca.astype(np.float64) - b[pe_id].astype(np.float64)).sum()
        den += ca.size
    return num / den / MEASURE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/svgg9_cifar10.pth")
    ap.add_argument("--data", default="/data/twt/datasets/cifar10")
    ap.add_argument("--n-images", type=int, default=4)
    args = ap.parse_args()

    dag, mapped, names, in_node, w_in, warmup = compile_trained(args.ckpt)
    tf = TT.Compose([TT.ToTensor(), TT.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test = tv.datasets.CIFAR10(args.data, train=False, transform=tf, download=False)

    configs = [(["--dnp-ratio", "0.50"], "ratio 0.50"),
               (["--dnp-ratio", "0.30"], "ratio 0.30"),
               (["--dnp-ratio", "0.25"], "ratio 0.25"),
               (["--dnp-ratio", "0.125"], "ratio 0.125"),
               (["--dnp-ratio", "0.50", "--dnp-pot", "0.02"], "ratio 0.50 pot.02"),
               (["--dnp-ratio", "0.25", "--dnp-pot", "0.02"], "ratio 0.25 pot.02")]
    agg = {label: dict(storage=[], mae=[], prune=[], reject=[]) for _, label in configs}
    base_rates, ws_ratios, identical_all = [], [], True

    for k in range(args.n_images):
        image = test[k][0].unsqueeze(0)
        write_input_bin(dag, mapped, in_node, w_in, image, warmup)

        base_fc, _ = run(["--dnp-off"])
        n_neu = sum(c.size for c in base_fc.values())
        base_rates.append(sum(int(c.sum()) for c in base_fc.values()) / (n_neu * MEASURE))

        g1_fc, g1_dnp = run(["--dnp-ratio", "1.0"])
        identical_all &= all(np.array_equal(base_fc[p], g1_fc[p]) for p in base_fc)
        tl = sum(r["n_log"] for r in g1_dnp); tp = sum(r["peak"] for r in g1_dnp)
        ws_ratios.append(tl / max(1, tp))

        for flags, label in configs:
            fc, dnp = run(flags)
            tl = sum(r["n_log"] for r in dnp); tp = sum(r["peak"] for r in dnp)
            agg[label]["storage"].append(tl / max(1, tp))
            agg[label]["mae"].append(firing_mae(fc, base_fc))
            agg[label]["prune"].append(sum(r["prune"] for r in dnp))
            agg[label]["reject"].append(sum(r["reject"] for r in dnp))

    print(f"\n[baseline] trained-net firing rate on real CIFAR images: {np.mean(base_rates)*100:.1f}% "
          f"(over {args.n_images} images)")
    print(f"[golden#1] --dnp-ratio 1.0 vs dense: {'IDENTICAL' if identical_all else 'MISMATCH'};  "
          f"lazy-alloc working set => {np.mean(ws_ratios):.2f}x storage (lossless)")
    print(f"\n  {'config':22} {'storage':>8} {'rateMAE':>9} {'prunes':>10} {'rejects':>10}  (avg/{args.n_images} imgs)")
    for _, label in configs:
        a = agg[label]
        print(f"  {label:22} {np.mean(a['storage']):7.2f}x {np.mean(a['mae']):9.5f} "
              f"{np.mean(a['prune']):10.0f} {np.mean(a['reject']):10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
