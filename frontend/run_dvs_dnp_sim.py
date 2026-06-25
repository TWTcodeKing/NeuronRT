"""Real C++ NoC + DNP simulation on the DVS workload (trained SEW-ResNet18, DVS128Gesture, T=16).

The model fits one chip (210 PEs < 576). The T=16 event frames are time-varying input: each timestep
the stem PEs are fed conv1(frame_t) via input_seq.bin (cycling, period 16), so the sim sees genuine
temporal idleness between events -> age-pruning reclaims slots. We run dense Soma (baseline) and DNP
(lazy-alloc + age-pruning) and report per-PE peak slots / prunes / rejects (storage) and firing
divergence vs the dense baseline (the accuracy proxy), averaged over a few real test clips.

Usage:  PYTHONPATH=. python run_dvs_dnp_sim.py --ckpt checkpoints/sew_resnet18_dvsgesture.pth --n-samples 3
"""
import argparse
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron
from spikingjelly.activation_based.model import sew_resnet
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

from neurort_compiler.funcsim import reassemble_weight
from neurort_compiler.graph.builders import build_dag
from neurort_compiler.graph.dag import topo
from neurort_compiler.export.writer import write_network
from neurort_compiler.mapping.partition import compress_axons, partition_dag, route_dag, validate
import run_dnp_sweep as S

OUT = "/tmp/dvs_net"
T = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sew_resnet18_dvsgesture.pth")
    ap.add_argument("--data", default="/data/twt/datasets/dvs128gesture")
    ap.add_argument("--n-samples", type=int, default=3)
    args = ap.parse_args()
    S.OUT = OUT
    S.MEASURE = T

    m = sew_resnet.sew_resnet18(spiking_neuron=neuron.LIFNode, cnf="ADD", num_classes=11)
    m.conv1 = layer.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    functional.set_step_mode(m, "s")
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"]); m.eval()

    dag = build_dag(m, "sew_resnet18", (128, 128), in_ch=2)
    dag.validate()
    mapped = partition_dag(dag)
    route_dag(dag, mapped)
    compress_axons(mapped)
    errs = validate(mapped)
    out_node = next(n for n in dag.nodes.values() if n.op == "output")
    analog = set(out_node.inputs)
    names = [n.name for n in topo(dag) if n.op in ("conv", "dense") and n.name not in analog]
    os.makedirs(OUT, exist_ok=True)
    write_network(mapped, OUT, timesteps=T)
    if os.path.exists(os.path.join(OUT, "attention.bin")):
        os.remove(os.path.join(OUT, "attention.bin"))
    in_node = next(n for n in dag.nodes.values()
                   if n.op == "conv" and dag.nodes[n.inputs[0]].op == "input")
    w_in = reassemble_weight(mapped, dag, in_node.name, quantize=True)
    bias = (torch.as_tensor(dag.biases[in_node.name], dtype=torch.float32).reshape(1, -1, 1, 1)
            if in_node.name in dag.biases else None)
    stem = mapped.pes_of_node(in_node.name)
    warmup = len(names) + 2 * T
    print(f"[compile] DVS sew_resnet18 @128: {mapped.num_pe_used} PEs (validate {errs[:1] or 'OK'}), "
          f"{len(names)} layers, stem {len(stem)} PEs, warmup={warmup}", flush=True)

    test = DVS128Gesture(args.data, train=False, data_type="frame", frames_number=T, split_by="number")

    def frame_currents(frames):                       # [T,2,128,128] -> [T, stem_neurons] int8-conv current
        cur = []
        for t in range(T):
            c = F.conv2d(frames[t:t + 1], w_in, stride=in_node.attrs["stride"], padding=in_node.attrs["padding"])
            if bias is not None:
                c = c + bias
            cur.append(c.flatten().detach().numpy().astype(np.float64))
        return np.stack(cur)                          # [T, stem_neurons]

    def write_inputs(cur_t):
        with open(os.path.join(OUT, "input.bin"), "wb") as fh:    # constant fallback = frame 0
            fh.write(struct.pack("<III", warmup, T, len(stem)))
            for pe in stem:
                fh.write(struct.pack("<II", pe.pe_id, pe.neuron_count))
                fh.write(cur_t[0, pe.neuron_base:pe.neuron_base + pe.neuron_count].tobytes())
        with open(os.path.join(OUT, "input_seq.bin"), "wb") as fh:  # T time-varying frames
            fh.write(struct.pack("<II", T, len(stem)))
            for pe in stem:
                fh.write(struct.pack("<II", pe.pe_id, pe.neuron_count))
                for t in range(T):
                    fh.write(cur_t[t, pe.neuron_base:pe.neuron_base + pe.neuron_count].tobytes())

    configs = [(["--dnp-off"], "dense"),
               (["--dnp-ratio", "0.10"], "ratio0.10 lazy"),
               (["--dnp-ratio", "0.10", "--dnp-age", "2"], "ratio0.10 age2"),
               (["--dnp-ratio", "0.05", "--dnp-age", "2"], "ratio0.05 age2")]
    agg = {l: dict(storage=[], mae=[], prune=[], reject=[]) for _, l in configs}

    for s in range(args.n_samples):
        frames = torch.as_tensor(test[s][0]).float()
        write_inputs(frame_currents(frames))
        base_fc, _ = S.run(["--dnp-off"])
        for flags, l in configs:
            fc, dnp = S.run(flags)
            agg[l]["mae"].append(S.firing_mae(fc, base_fc))
            if dnp:
                tl = sum(r["n_log"] for r in dnp); tp = sum(r["peak"] for r in dnp)
                agg[l]["storage"].append(tl / max(1, tp))
                agg[l]["prune"].append(sum(r["prune"] for r in dnp))
                agg[l]["reject"].append(sum(r["reject"] for r in dnp))

    print(f"\n  {'config':18} {'storage':>8} {'rateMAE':>9} {'prunes':>10} {'rejects':>10}  (avg/{args.n_samples} clips)")
    for _, l in configs:
        a = agg[l]
        st = np.mean(a["storage"]) if a["storage"] else 1.0
        print(f"  {l:18} {st:7.2f}x {np.mean(a['mae']):9.5f} "
              f"{(np.mean(a['prune']) if a['prune'] else 0):10.0f} {(np.mean(a['reject']) if a['reject'] else 0):10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
