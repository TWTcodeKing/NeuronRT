"""Export a MappedNetwork to the compiled format: manifest.json + <model>.weights.bin +
<model>.axons.bin.

Two binary sidecars keep the manifest small: the weight blob (dedup'd int8 weights) and the AXON
blob (compressed axon-table groups, 32 bytes each — see AxonGroup). The manifest carries per-PE
spans into both blobs plus the dendrite table; the axon table is NOT inlined (it would be O(synapses)
of verbose JSON — e.g. ~381 MB for Spikformer before this change)."""
from __future__ import annotations

import json
import os
import struct
from typing import Tuple

import numpy as np

from ..mapping.partition import MAX_AXON_LEVELS, MappedNetwork, validate_axon_groups
from . import schema

# Little-endian axon group record (mirrored by the C++ NetworkImage loader), 66 bytes at 3 levels:
#   src_base u32 | go1_base i32 | go2_base i32 |
#   MAX_AXON_LEVELS x (count u32 | src_stride i32 | go1_stride i32 | go2_stride i32) |
#   dst_pe u16 | meta u16 | dendrite_id u8 | delay u8
# `meta` = per-edge combine descriptor (term_id/combine/avg_n). Levels are inner-first; unused
# levels are padded with count=1, strides=0 (a no-op factor).
AXON_REC = struct.Struct("<Iii" + "Iiii" * MAX_AXON_LEVELS + "HHBB")


def quantize_int8(weight: np.ndarray) -> Tuple[np.ndarray, float]:
    """Symmetric per-PE int8 quantization. Returns (int8 array, scale) with w ~= int8 * scale."""
    if weight.size and not np.all(np.isfinite(weight)):
        raise ValueError("non-finite weight (NaN/Inf) cannot be quantized — clamp the model first")
    amax = float(np.max(np.abs(weight))) if weight.size else 0.0
    scale = amax / 127.0 if amax > 0 else 1.0
    q = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    return q, scale


def build_manifest(mapped: MappedNetwork, weight_bits: int = 8, timesteps: int = 4, dnp: dict = None):
    errs = validate_axon_groups(mapped)
    if errs:
        raise ValueError(f"axon group validation failed ({len(errs)}): {errs[:3]}")
    wblob = bytearray()
    ablob = bytearray()
    pes_json = []
    for pe in mapped.pes:
        q, scale = quantize_int8(pe.weight)
        woff = len(wblob)
        wblob += q.tobytes()
        aoff = len(ablob)
        for g in pe.axon_groups:
            lv = list(g.levels) + [(1, 0, 0, 0)] * (MAX_AXON_LEVELS - len(g.levels))
            flat = [v for level in lv for v in level]    # (count, src_stride, go1_stride, go2_stride)*L
            ablob += AXON_REC.pack(g.src_base, g.go1_base, g.go2_base, *flat,
                                   g.dst_pe, g.meta, g.dendrite_id, g.delay)
        dends = [{"id": d.id, "count": d.count, "repeat": d.repeat,
                  "local_off1": d.local_off1, "local_off2": d.local_off2,
                  "nlist": [int(x) for x in d.nlist], "wlist": [int(x) for x in d.wlist]}
                 for d in pe.dendrites]
        pes_json.append({
            "pe": pe.pe_id, "coord": list(pe.coord), "layer": pe.layer, "kind": pe.kind,
            "neuron_base": pe.neuron_base, "neuron_count": pe.neuron_count,
            "dendrites": dends,
            "axon_span": {"offset": aoff, "groups": len(pe.axon_groups)},
            "weight_span": {"offset": woff, "bytes": len(q.tobytes()), "scale": scale},
        })

    manifest = {
        "format_version": schema.FORMAT_VERSION,
        "model": mapped.model_name,
        "chip": {"mesh_w": schema.MESH_W, "mesh_h": schema.MESH_H, "num_pe": schema.NUM_PE,
                 "sram_bytes_per_pe": schema.SRAM_BYTES, "weight_bits": weight_bits},
        "timesteps": timesteps,
        "neuron": {"tau": float(mapped.neuron.get("tau", 2.0)),
                   "v_threshold": float(mapped.neuron.get("v_threshold", 1.0))},
        "weight_blob": f"{mapped.model_name}.weights.bin",
        "weight_blob_bytes": len(wblob),
        "axon_blob": f"{mapped.model_name}.axons.bin",
        "axon_blob_bytes": len(ablob),
        "axon_record_bytes": AXON_REC.size,
        "pes": pes_json,
    }
    if dnp is not None:   # optional Soma-DNP config (virtual-memory neuron state). Absent => disabled.
        manifest["dnp"] = dict(dnp)
    return manifest, bytes(wblob), bytes(ablob)


def write_network(mapped: MappedNetwork, out_dir: str,
                  weight_bits: int = 8, timesteps: int = 4, dnp: dict = None) -> dict:
    manifest, wblob, ablob = build_manifest(mapped, weight_bits, timesteps, dnp=dnp)
    schema.validate_manifest(manifest)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(out_dir, manifest["weight_blob"]), "wb") as f:
        f.write(wblob)
    with open(os.path.join(out_dir, manifest["axon_blob"]), "wb") as f:
        f.write(ablob)
    return manifest
