# NeuroRT compiled-network format (v1)

The compiler emits a **JSON manifest** + two **binary blobs**: the weight blob (`<model>.weights.bin`)
and the **axon blob** (`<model>.axons.bin`, the compressed axon table). This is the co-design
contract between the Python compiler and the C++ sim's `NetworkImage` loader. Field bit-widths
mirror the sim's `flit.hpp` and the paper's Fig. 4 memory primitives; the C++ loader validates
every field against them. The axon table is a binary sidecar, not inline JSON — inlining it would
make the manifest O(synapses) (≈381 MB for Spikformer); compressed + binary it is ≈1.6 MB.

## Bit-width limits (must hold; asserted on both sides)

| field | bits | max | source |
|---|---|---|---|
| `dendrite_id` | 8 | 255 | flit `kSpikeDendriteBits` |
| `count` (dendrite entries) | 8 | 255 | dendrite header |
| `delay` (axon group) | 4 | 15 | flit `kSpikeDelayBits` (D-SNN = 1) |
| `dst_pe` (axon group) | — | <num_pe | PE id (coord = id%mesh_w, id/mesh_w) |
| `pe` id | — | <num_pe (≤576) | mesh |

`nlist` / `wlist` / `local_off1` / `local_off2` / `go1` / `go2` are the compiler's **signed
relative offsets** (e.g. conv `nlist[c] = f·Ho·Wo − kh·Wo − kw`, decoded as `post = nlist[c] + go2`,
`weight = wlist[c] + go1`). They are stored as JSON ints (loaded as int32). The hardware
address-width check (final decoded address fits the PE's neuron/weight space) is a sim-runtime
(M2) concern, not a load-time one — the loader validates structure, the bit-constrained fields
above, list-length == count, per-PE bytes ≤ SRAM, and blob size.

## manifest.json

```jsonc
{
  "format_version": 1,
  "model": "spiking_vgg16",
  "chip": { "mesh_w": 24, "mesh_h": 24, "num_pe": 576, "sram_bytes_per_pe": 65536, "weight_bits": 8 },
  "timesteps": 4,
  "weight_blob": "spiking_vgg16.weights.bin",
  "weight_blob_bytes": 1234567,
  "axon_blob": "spiking_vgg16.axons.bin",
  "axon_blob_bytes": 234567,
  "axon_record_bytes": 32,
  "pes": [
    { "pe": 0, "coord": [0, 0], "layer": 0, "kind": "conv",
      "neuron_base": 0, "neuron_count": 512,
      "dendrites": [
        { "id": 0, "count": 9, "repeat": 28, "local_off1": 64, "local_off2": 72,
          "nlist": [ /* signed offsets, length == count (one feature map's <=K^2 taps) */ ],
          "wlist": [ /* signed offsets, length == count */ ] }
      ],
      "axon_span": { "offset": 0, "groups": 36 },
      "weight_span": { "offset": 0, "bytes": 972, "scale": 0.0123 } }
  ]
}
```
`weight_span.scale` dequantizes the int8 blob slice: `w_float ≈ int8 * scale`. `nlist`/`wlist`
index within this PE's slice (`wlist[c] + go1` in `[0, bytes)`). `axon_span` = this PE's slice of
the axon blob (`offset` bytes in, `groups` records of `axon_record_bytes` each); spans are
contiguous and `Σ groups * axon_record_bytes == axon_blob_bytes`.

## Weight blob

Raw bytes; `pes[i].weight_span` = `{offset, bytes}` slice. `weight_bits` = element precision
(8 = int8). Conv kernels are stored **once** and shared across feature maps via global/local
offsets, so `Σ weight_span.bytes` ≪ the dense parameter count. `weight_blob_bytes == Σ bytes`.

## Axon blob (compressed axon table)

The axon table is the **inverse of the dendrite repeat-loop**, generalized to a nested lattice:
source neurons that share a fan-out pattern collapse into one **axon group** of up to 3 nested
levels (inner-first), each `(count, src_stride, go1_stride, go2_stride)`. 1-D folds a conv
feature-map row / token run; 2-D adds the row dimension; 3-D adds the channel dimension (`go1`
advancing per channel) — so a whole conv feature map collapses into one header. This keeps the
axon table O(neurons), not O(synapses) (Spikformer: 2.24M flat entries → 2.5K groups, **890×**;
ResNet18 10.8×).

Each group is a fixed **64-byte little-endian record** (`axon_record_bytes`):

| offset | field | type | meaning |
|---|---|---|---|
| 0 | `src_base` | u32 | source neuron (PE-local id) at all-zero indices |
| 4 | `go1_base` | i32 | weight global offset at all-zero indices |
| 8 | `go2_base` | i32 | neuron global offset at all-zero indices |
| 12 + 16·L | `count` | u32 | level L run length (≥1; padding levels = 1) |
| 16 + 16·L | `src_stride` | i32 | level L source-neuron step |
| 20 + 16·L | `go1_stride` | i32 | level L go1 step |
| 24 + 16·L | `go2_stride` | i32 | level L go2 step |
| 60 | `dst_pe` | u16 | destination PE id |
| 62 | `dendrite_id` | u8 | target dendrite on `dst_pe` |
| 63 | `delay` | u8 | axon delay (≤15) |

(L ∈ [0, 3); the three level records occupy bytes 12..59.) Decompression: for every index tuple
`(r_0, r_1, r_2)` with `r_k ∈ [0, count_k)`, a source neuron at `src_base + Σ r_k·src_stride_k`
emits a spike to (`dst_pe`, `dendrite_id`) carrying `go1 = go1_base + Σ r_k·go1_stride_k`,
`go2 = go2_base + Σ r_k·go2_stride_k`. The compiler verifies the groups decompress to exactly the
flat per-neuron entries (`mapping/partition.py:verify_axons`); the C++ loader re-validates each
group's destination dendrite exists and its source lattice stays within the PE's neurons.

## Decompression contract (paper Algorithm 1)

The compression that beats O(n²): a dendrite stores only ONE feature map's `<=K^2` taps; the
**local offsets** (`local_off1`/`local_off2`) + the **repeat** bound walk the remaining feature
maps (conv) or post-neurons (dense), and the per-spike **global offsets** (`go1`/`go2`) select
the input channel / output position. So the dendrite table is O(K²) entries regardless of F, C
(conv) or N (dense). A `SPIKE(dendrite_id, go1, go2)` reconstructs `(neuron, weight)` pairs:

```
e = dendrites[dendrite_id]
for c in range(e.count):              # e.count <= K^2 (conv) or 1 (dense)
    w = wlist[c] + go1
    n = nlist[c] + go2
    for r in range(e.repeat):         # e.repeat = #feature-maps (conv) / #post-neurons (dense)
        emit (neuron = n + e.local_off1 * r, weight = blob[w + e.local_off2 * r])
```
e.g. conv: `local_off1 = Ho*Wo`, `local_off2 = C*K*K`, `repeat = F`, `go1 = c*K*K`; the loop
emits all F*K² synapses of an input spike. dense: `count=1`, `local_off1=1`, `local_off2=L`,
`repeat=N`, `go1=j`.

The compiler is the **inverse**: it produces `nlist/wlist/offsets/repeat/go*` such that this loop
reconstructs exactly the original layer connectivity and weight values. `synapse/decompress.py`
implements this loop as the golden reference for the lossless-compression test.
