"""NeuroRT compiler CLI.

Example:
  python -m neurort_compiler.cli --model spiking_vgg16 --input 32x32 --timesteps 4 --out build/net

The full pipeline (graph -> synapse -> compress -> map -> export) is wired in across sub-tasks
C2-C6. This C0 skeleton builds the model and prints a summary.
"""
from __future__ import annotations

import argparse
from typing import Tuple

from .export.writer import write_network
from .graph.builders import build_dag
from .mapping.partition import (axon_entry_count, axon_group_count, compress_axons,
                                partition_dag, route_dag, validate)
from .models.registry import available_models, build_model


def parse_hw(s: str) -> Tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--input must be HxW, got '{s}'")
    return int(parts[0]), int(parts[1])


def main(argv=None) -> int:
    p = argparse.ArgumentParser("neurort-compiler")
    p.add_argument("--model", required=True, choices=available_models())
    p.add_argument("--input", type=parse_hw, default=(32, 32), help="input size HxW")
    p.add_argument("--timesteps", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--out", default="build/net")
    args = p.parse_args(argv)

    h, w = args.input
    model = build_model(args.model, num_classes=args.num_classes)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"built {args.model}: {n_params / 1e6:.2f}M params, input {h}x{w}, T={args.timesteps}")

    dag = build_dag(model, args.model, (h, w), timesteps=args.timesteps)
    print(f"  graph: {dag.summary()}")
    mapped = partition_dag(dag)
    routed = route_dag(dag, mapped)
    compress_axons(mapped)
    flat, grp = axon_entry_count(mapped), axon_group_count(mapped)
    print(f"  mapped: {mapped.num_pe_used} PEs, {flat:,} axon entries -> {grp:,} compressed "
          f"groups ({flat / max(grp, 1):.1f}x) over {len(routed)} producing nodes")
    errs = validate(mapped)
    over_576 = [e for e in errs if "PEs >" in e]
    if over_576:
        print(f"  WARNING: {over_576[0]} (needs folding to fit a single chip)")
    other = [e for e in errs if "PEs >" not in e]
    if other:
        print(f"  WARNING: {len(other)} per-PE constraint violation(s), e.g. {other[0]}")

    manifest = write_network(mapped, args.out, timesteps=args.timesteps)
    print(f"  exported {args.out}/manifest.json + {manifest['weight_blob']} "
          f"({manifest['weight_blob_bytes']:,} weight bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
