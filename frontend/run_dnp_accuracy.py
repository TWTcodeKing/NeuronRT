"""Classification accuracy of the trained svgg9 under Soma-DNP, on the real NoC + real CIFAR-10.

The C++ sim outputs per-LIF-layer firing; the final layer is analog logits. We reconstruct the logits
the chip would produce as W_fc2(int8) @ (last-LIF-layer firing rate), take argmax, and compare to the
true label (top-1) and to the dense-Soma prediction (agreement). This is the metric the paper cares
about: does DNP preserve accuracy while cutting neuron-state storage?

Usage:  PYTHONPATH=. python run_dnp_accuracy.py --ckpt checkpoints/svgg9_cifar10.pth --n-images 40
"""
import argparse

import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as TT

from neurort_compiler.funcsim import reassemble_weight
import run_dnp_sweep as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/svgg9_cifar10.pth")
    ap.add_argument("--data", default="/data/twt/datasets/cifar10")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--ratios", default="0.50,0.25,0.125", help="comma-separated phys/logical ratios")
    args = ap.parse_args()

    dag, mapped, names, in_node, w_in, warmup = S.compile_trained(args.ckpt)
    out_node = next(n for n in dag.nodes.values() if n.op == "output")
    fc2_name = out_node.inputs[0]                       # analog output Linear (logits)
    fc1_name = dag.nodes[fc2_name].inputs[0]            # last LIF layer feeding the logits
    W_fc2 = reassemble_weight(mapped, dag, fc2_name, quantize=True).detach().numpy()  # [10, n_fc1]
    n_fc1 = dag.nodes[fc1_name].num_neurons()
    print(f"[head] logits = W_fc2{list(W_fc2.shape)} @ rate({fc1_name}, {n_fc1} neurons)")

    def predict(fc):
        rate = np.zeros(n_fc1)
        for pe in mapped.pes_of_node(fc1_name):         # fc1 is plain dense (n_tokens=1) -> canonical
            rate[pe.neuron_base:pe.neuron_base + pe.neuron_count] = fc[pe.pe_id]
        rate /= S.MEASURE
        return int(np.argmax(W_fc2 @ rate))

    tf = TT.Compose([TT.ToTensor(), TT.Normalize(S.CIFAR_MEAN, S.CIFAR_STD)])
    test = tv.datasets.CIFAR10(args.data, train=False, transform=tf, download=False)

    configs = [(["--dnp-off"], "dense")]
    for r in args.ratios.split(","):
        configs.append((["--dnp-ratio", r.strip()], f"ratio {r.strip()}"))
    correct = {l: 0 for _, l in configs}
    agree = {l: 0 for _, l in configs}
    storage = {l: [] for _, l in configs}

    for k in range(args.n_images):
        img = test[k][0].unsqueeze(0)
        label = int(test[k][1])
        S.write_input_bin(dag, mapped, in_node, w_in, img, warmup)
        preds = {}
        for flags, l in configs:
            fc, dnp = S.run(flags)
            p = predict(fc)
            preds[l] = p
            correct[l] += int(p == label)
            if dnp:
                tl = sum(r["n_log"] for r in dnp)
                tp = sum(r["peak"] for r in dnp)
                storage[l].append(tl / max(1, tp))
        for _, l in configs:
            agree[l] += int(preds[l] == preds["dense"])
        if (k + 1) % 10 == 0:
            print(f"  ...{k+1}/{args.n_images} images")

    n = args.n_images
    print(f"\n  {'config':12} {'top1':>7} {'=dense':>8} {'storage':>8}  (n={n} real CIFAR test images)")
    for _, l in configs:
        s = np.mean(storage[l]) if storage[l] else 1.0
        print(f"  {l:12} {correct[l]/n*100:6.1f}% {agree[l]/n*100:7.1f}% {s:7.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
