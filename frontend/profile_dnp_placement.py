"""Does firing-rate-aware placement help DNP on trained svgg9? (premise check before implementing.)

DNP rejects ACTIVE neurons when a PE's working set (neurons that receive input -> claim a slot)
exceeds its physical slots. The compiler currently maps CONTIGUOUS output channels to each PE, so if
hot channels cluster, some PEs are oversubscribed (reject -> accuracy loss) while others sit nearly
empty. The paper spreads dissimilar-rate neurons across PEs to even this out.

We profile, per weight layer, each neuron's "receives input" rate (the allocation set) on real CIFAR,
then compare the per-PE working-set fraction under the current CONTIGUOUS channel assignment vs a
firing-BALANCED one (deal rate-sorted channels round-robin). If contiguous max >> balanced max, the
optimization has headroom: at physical ratio R, a PE with working-set fraction > R rejects.
"""
import argparse

import numpy as np
import torch
from spikingjelly.activation_based import functional, neuron

import run_dnp_sweep as S
import torchvision as tv
import torchvision.transforms as TT
from neurort_compiler.models.registry import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/svgg9_cifar10.pth")
    ap.add_argument("--data", default="/data/twt/datasets/cifar10")
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.5, help="physical/logical ratio to score against")
    args = ap.parse_args()

    model = build_model("svgg9", num_classes=10)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    dag, mapped, names, in_node, w_in, _ = S.compile_trained(args.ckpt)

    lifs = [m for m in model.modules() if isinstance(m, neuron.LIFNode)]
    assert len(lifs) == len(names), f"{len(lifs)} LIF vs {len(names)} names"
    recv = {nm: None for nm in names}      # per-neuron: received nonzero input on ANY step (this image)
    fires = {nm: None for nm in names}     # per-neuron: # spikes over the T steps (this image)

    hooks = []
    for nm, lif in zip(names, lifs):
        def pre(m, inp, nm=nm):
            x = inp[0].detach()
            flat = (x.abs() > 1e-9).reshape(x.shape[0], -1).float()   # [B, neurons] received this step
            recv[nm] = flat.clone() if recv[nm] is None else torch.maximum(recv[nm], flat)
        def post(m, inp, out, nm=nm):
            f = out.detach().reshape(out.shape[0], -1)                # [B, neurons] spikes this step
            fires[nm] = f.clone() if fires[nm] is None else fires[nm] + f
        hooks.append(lif.register_forward_pre_hook(pre))
        hooks.append(lif.register_forward_hook(post))

    tf = TT.Compose([TT.ToTensor(), TT.Normalize(S.CIFAR_MEAN, S.CIFAR_STD)])
    test = tv.datasets.CIFAR10(args.data, train=False, transform=tf, download=False)

    # accumulate per-neuron allocation rate + firing rate over images
    alloc = {nm: None for nm in names}
    fire = {nm: None for nm in names}
    for k in range(args.n_images):
        for nm in names:
            recv[nm] = fires[nm] = None
        x = test[k][0].unsqueeze(0)
        functional.reset_net(model)
        with torch.no_grad():
            for _ in range(args.T):
                model(x)
        for nm in names:
            a = recv[nm][0].numpy()                       # [neurons] 0/1 allocated this image
            fr = (fires[nm][0].numpy() > 0).astype(float)  # [neurons] fired at least once this image
            alloc[nm] = a.copy() if alloc[nm] is None else alloc[nm] + a
            fire[nm] = fr.copy() if fire[nm] is None else fire[nm] + fr
    for nm in names:
        alloc[nm] /= args.n_images
        fire[nm] /= args.n_images                          # fraction of images where the neuron fires
    for h in hooks:
        h.remove()

    # The lever is FIRING (active neurons), not allocation (~100%). At ratio R, if each PE's firing
    # fraction <= R, prioritizing active neurons' allocation lets them ALL fit -> lossless. Report the
    # per-PE firing fraction: contiguous max vs a firing-balanced placement, and how many PEs have
    # firing > R (where even firing-first can't fit, so cross-PE rebalance is needed).
    print(f"\n  profiling on {args.n_images} real CIFAR images, T={args.T}, scored at ratio={args.ratio}")
    print(f"  {'layer':8} {'neurons':>8} {'PEs':>4} {'wset':>6} {'fire_avg':>8} {'fire_maxPE':>11} "
          f"{'bal_maxPE':>10} {'fire>R contig/bal':>18}")
    tot_c = tot_b = tot_pe = 0
    for nm in names:
        node = dag.nodes[nm]
        a, fr = alloc[nm], fire[nm]
        pes = mapped.pes_of_node(nm)
        npe = len(pes)
        contig = np.array([fr[pe.neuron_base:pe.neuron_base + pe.neuron_count].mean() for pe in pes])
        order = np.argsort(fr)
        bal = np.zeros(npe); cnt = np.zeros(npe)
        for i, idx in enumerate(order):
            bal[i % npe] += fr[idx]; cnt[i % npe] += 1
        bal = bal / np.maximum(1, cnt)
        c_over = int((contig > args.ratio).sum()); b_over = int((bal > args.ratio).sum())
        tot_c += c_over; tot_b += b_over; tot_pe += npe
        print(f"  {nm:8} {node.num_neurons():8d} {npe:4d} {a.mean():6.2f} {fr.mean():8.3f} "
              f"{contig.max():11.3f} {bal.max():10.3f} {str(c_over)+'/'+str(b_over):>18}")
    print(f"\n  alloc(working set) ~ {np.mean([alloc[n].mean() for n in names]):.2f} (near 1 -> capping "
          f"hits every PE); firing avg ~ {np.mean([fire[n].mean() for n in names]):.3f}")
    print(f"  PEs with firing-fraction > ratio {args.ratio} (active set can't fit even if prioritized): "
          f"contiguous={tot_c}/{tot_pe}, firing-balanced={tot_b}/{tot_pe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
