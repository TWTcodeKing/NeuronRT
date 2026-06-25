"""Compiled-format constants + manifest validation (mirrored by the C++ NetworkImage loader)."""
from __future__ import annotations

import math

FORMAT_VERSION = 1
MESH_W = 24
MESH_H = 24
NUM_PE = 576
SRAM_BYTES = 64 * 1024

MAX_DENDRITE_ID = 255      # 8 bits
MAX_COUNT = 255            # 8 bits (dendrite entry count)
MAX_DELAY = 15             # 4 bits
MAX_TARGET_COUNT = 4095    # 12 bits


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"manifest invalid: {msg}")


def validate_manifest(m: dict) -> None:
    _check(m.get("format_version") == FORMAT_VERSION, "format_version")
    chip = m["chip"]
    num_pe = chip["num_pe"]
    mesh_w, mesh_h = chip["mesh_w"], chip["mesh_h"]
    sram = chip["sram_bytes_per_pe"]
    _check(mesh_w > 0 and mesh_h > 0 and num_pe > 0, "mesh dims / num_pe must be > 0")
    _check(num_pe <= NUM_PE, f"num_pe {num_pe} > {NUM_PE}")

    rec_bytes = m["axon_record_bytes"]
    total_weight_bytes = 0
    axon_off = 0                  # axon groups are stored contiguously in the axon blob, PE by PE
    seen_pe = set()
    for pe in m["pes"]:
        pid = pe["pe"]
        _check(0 <= pid < num_pe, f"pe id {pid} out of range")
        _check(pid not in seen_pe, f"duplicate pe id {pid}")
        seen_pe.add(pid)
        x, y = pe["coord"]
        _check(x == pid % mesh_w and y == pid // mesh_w, f"pe {pid} coord {pe['coord']}")
        _check(0 <= x < mesh_w and 0 <= y < mesh_h, f"pe {pid} coord out of mesh")

        ndend = len(pe["dendrites"])
        _check(ndend <= MAX_DENDRITE_ID + 1, f"pe {pid}: {ndend} dendrites > 256")
        ids = set()
        for d in pe["dendrites"]:
            _check(0 <= d["id"] <= MAX_DENDRITE_ID, f"pe {pid} dendrite id {d['id']}")
            _check(d["id"] not in ids, f"pe {pid} duplicate dendrite id {d['id']}")
            ids.add(d["id"])
            _check(0 <= d["count"] <= MAX_COUNT, f"pe {pid} dendrite {d['id']} count {d['count']}")
            _check(d["repeat"] >= 1, f"pe {pid} dendrite {d['id']} repeat {d['repeat']}")
            _check(len(d["nlist"]) == d["count"] and len(d["wlist"]) == d["count"],
                   f"pe {pid} dendrite {d['id']} list length != count")

        # Axon table is a binary sidecar; the manifest only carries the per-PE span. Group records
        # are laid out contiguously (semantics validated by the writer + the C++ loader).
        asp = pe["axon_span"]
        _check(asp["offset"] == axon_off, f"pe {pid} axon span not contiguous")
        _check(asp["groups"] >= 0, f"pe {pid} negative axon groups")
        axon_off += asp["groups"] * rec_bytes

        sp = pe["weight_span"]
        _check(math.isfinite(sp["scale"]), f"pe {pid} weight scale not finite")
        budget = sp["bytes"] + 2 * pe["neuron_count"]   # weights + ping-pong neuron states
        _check(budget <= sram, f"pe {pid}: {budget} B > {sram} B SRAM")
        total_weight_bytes += sp["bytes"]

    _check(total_weight_bytes == m["weight_blob_bytes"],
           f"weight_blob_bytes {m['weight_blob_bytes']} != sum of spans {total_weight_bytes}")
    _check(axon_off == m["axon_blob_bytes"],
           f"axon_blob_bytes {m['axon_blob_bytes']} != sum of spans {axon_off}")

    if "dnp" in m:   # optional Soma-DNP config block
        d = m["dnp"]
        _check(isinstance(d.get("enabled", False), bool), "dnp.enabled must be bool")
        _check(d.get("n_phys", 0) > 0 or 0.0 < d.get("phys_ratio", 0.25) <= 1.0,
               "dnp: need n_phys > 0 or 0 < phys_ratio <= 1")
