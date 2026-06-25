# NeuroRT

A simulator + compiler for **NeuroRT** — a Process-Near-Memory (PNM) neuromorphic
processor that deploys Spiking Neural Networks (SNNs) on a single-chip 24×24 NoC mesh
(576 PEs, 333 MHz, 28nm). The two hardware contributions modeled here are **Synapse
Compression** (Algorithm 1, dendrite decompression) and **Dynamic Neuron Pruning (DNP)**
(Algorithm 2, virtual-memory neuron store with age/potential pruning).

This repo contains:
- `sim/` — multi-threaded C++20 cycle-stepped NoC + PE simulator (the `BspEngine`),
  with a per-action energy model.
- `frontend/` — Python compiler that turns a trained SNN into a per-PE, compressed,
  sim-loadable network image (`manifest.json` + binary weight/axon blobs), plus the
  experiment harnesses.

See `CLAUDE.md` for the detailed architecture and current validation state. See
`paper.pdf` for the spec.

---

## 1. Build the C++ simulator

Requirements: CMake ≥ 3.20, g++ ≥ 11 (C++20), OpenMP. The first configure pulls
GoogleTest + nlohmann/json via FetchContent (**network required once**).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Offline build (no network): pass `-DNEURORT_BUILD_TESTS=OFF` and point CMake at a local
copy of the json single-header with `-DNEURORT_JSON_DIR=<dir containing nlohmann/json.hpp>`.

Treat compiler warnings as errors (CI mode): `-DNEURORT_WERROR=ON`.

## 2. Run the simulator

Two modes, both from the **repo root** (config paths are repo-root-relative):

### (a) Synthetic NoC traffic (no model)

Drives the mesh with a traffic pattern and reports cycles / spike latency / energy.

```bash
./build/sim/neurort_sim \
  --config configs/chip_default.json \
  --traffic configs/traffic/uniform_random.json \
  [--threads N] [--dump out.json]
```

Traffic patterns in `configs/traffic/`: `tree_sync_only`, `neighbor_xy`,
`uniform_random`, `hotspot_center`.

### (b) Compiled network end-to-end on the NoC

Runs a real compiled network image on the full PE pipeline (Axon-in → Dendrite →
Soma → Axon-out → real XY-routed NoC). Reads `<dir>/manifest.json` + `input.bin`,
writes `firing.bin`, and reports **energy + latency over the steady-state measure
window** plus a full per-action breakdown to `energy.json`.

```bash
./build/sim/neurort_sim --network <dir> [--energy-table configs/energy_table_28nm.json]
```

### Common flags (run `sim/app/main.cpp` for the source of truth)

| Flag | Meaning |
|------|---------|
| `--config <path>` | Chip/NoC/sim config (default `configs/chip_default.json`). |
| `--traffic <path>` | Traffic pattern file (synthetic mode). |
| `--network <dir>` | Compiled-network directory (end-to-end mode). |
| `--energy-table <path>` | Swappable pJ/action table (default `configs/energy_table_28nm.json`, **placeholder values**). |
| `--threads N` | `>0` honored verbatim; `0`/unset = auto, capped at 8 (results are thread-count-independent). |
| `--dump <path>` | Write a JSON report (synthetic mode). |
| `--dnp-ratio R` | Enable DNP with physical/logical ratio `R` (e.g. `0.5` = 2× storage). |
| `--dnp-age T` | Enable DNP age-pruning with `AGE_THRESH = T`. |
| `--dnp-pot P` | Enable DNP potential-pruning with `POT_THRESH = P`. |
| `--dnp-off` | Force dense Soma (disable DNP), overriding the manifest. |
| `--dnp-skip-pruned` | Enable the sticky-skip of pruned/unmapped neurons in the Dendrite. |

> The DNP flags **override** the manifest's optional `"dnp"` block; without them the
> manifest's config (or dense, if absent) is used.

## 3. Run the tests

```bash
ctest --test-dir build --output-on-failure          # all
ctest --test-dir build -R Engine --output-on-failure # one (regex)
./build/sim/test_engine                              # or run a binary directly
```

**Determinism gate** (headline invariant — output identical for any thread count):

```bash
for n in 1 2 4 8; do
  OMP_NUM_THREADS=$n ./build/sim/neurort_sim \
    --config configs/chip_default.json \
    --traffic configs/traffic/uniform_random.json --dump s_$n.json
done   # s_*.json must be byte-identical
```

---

## 4. Compile a model (Python frontend)

Env: the `spik-yolo` conda env (PyTorch + SpikingJelly + einops). pytest is **not**
installed; tests use the dependency-free runner.

```bash
PY=/home/twt/.conda/envs/spik-yolo/bin/python
cd frontend
$PY run_tests.py            # all frontend tests (optional substring filter)
```

Compile a model to a network image (`<out>/manifest.json` + `<model>.weights.bin` +
`<model>.axons.bin`):

```bash
$PY -m neurort_compiler.cli \
  --model svgg9 \
  --input 32x32 \
  --timesteps 4 \
  --num-classes 10 \
  --out build/net
```

Available models: `sew_resnet18`, `sew_resnet34`, `spiking_vgg16`, `spikformer`,
`svgg9` (a BN-free 9-weight-layer spiking VGG).

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | (required) | One of the models above. |
| `--input` | `32x32` | Input spatial size `HxW`. |
| `--timesteps` | `4` | SNN timesteps `T`. |
| `--num-classes` | `10` | Output classes. |
| `--out` | `build/net` | Output directory. |

The compiler pipeline: `models/` → `graph/` (DAG IR, BN folded into conv at compile
time) → `synapse/` (connectivity + Algorithm-1 compression) → `mapping/` (partition to
576 PEs, route, compress axons) → `export/` (manifest + binary blobs). The format
contract is `frontend/neurort_compiler/format.md`; the C++ loader is
`sim/.../network/network_image.{hpp,cpp}`.

---

## 5. Run experiments

The experiment harnesses live in `frontend/` and drive the C++ simulator
end-to-end. All run from `frontend/` with `$PY` = the `spik-yolo` env python.

### 5.1 End-to-end correctness vs the int8 funcsim reference

Compiles a model, computes the layer-1 input current, runs the C++ NoC sim, and compares
steady-state per-layer firing rates to the int8 functional-sim / SpikingJelly reference.

```bash
cd frontend
$PY run_e2e.py
```

(Defaults to `svgg9` at 32×32 with random weights exercising firing; edit the
`MODEL`/`HW`/`MEASURE` constants at the top of the file to retarget.)

### 5.2 DNP on static CIFAR-10 (svgg9)

Requires the trained checkpoint `frontend/checkpoints/svgg9_cifar10.pth` and CIFAR-10 at
`/data/twt/datasets/cifar10`.

```bash
# Accuracy vs DNP ratio, firing-rate-aware placement ON vs OFF (the key comparison)
$PY run_dnp_placement_e2e.py --ckpt checkpoints/svgg9_cifar10.pth \
    --data /data/twt/datasets/cifar10 --n-images 20 --T 4

# Per-layer DNP ratios (exploring beyond the uniform 2× ceiling)
$PY run_dnp_perlayer.py --ckpt checkpoints/svgg9_cifar10.pth \
    --data /data/twt/datasets/cifar10 --T 4

# Python upper-bound for keep-ratio (no NoC sim, fast)
$PY run_dnp_placement_test.py
```

Key flags: `--ckpt`, `--data`, `--n-images` / `--test-images`, `--profile-images`,
`--T`. The C++ DNP flags (`--dnp-ratio`, `--dnp-age`, `--dnp-pot`, `--dnp-off`,
`--dnp-skip-pruned`) are passed through to the simulator by these harnesses.

### 5.3 DNP on DVS (DVS128Gesture, SEW-ResNet18) — the temporal-sparse regime

Requires `frontend/checkpoints/sew_resnet18_dvsgesture.pth` and DVS128Gesture at
`/data/twt/datasets/dvs128gesture`.

```bash
# End-to-end on the real C++ NoC + DNP chip sim, reports storage× vs firing-MAE
$PY run_dvs_dnp_sim.py --ckpt checkpoints/sew_resnet18_dvsgesture.pth \
    --data /data/twt/datasets/dvs128gesture --n-samples 3

# True classification accuracy (top-1) vs storage — the real metric
$PY run_dvs_accuracy.py --ckpt checkpoints/sew_resnet18_dvsgesture.pth \
    --data /data/twt/datasets/dvs128gesture --ratios 0.10,0.07,0.05,0.03
```

### 5.4 Training / retraining the models

```bash
$PY train_svgg9_cifar10.py        # svgg9 on CIFAR-10 -> checkpoints/svgg9_cifar10.pth
$PY train_dvs_sewresnet.py        # SEW-ResNet18 on DVS128Gesture -> checkpoints/sew_resnet18_dvsgesture.pth
```

### 5.5 Functional-equivalence validation vs SpikingJelly

```bash
$PY validate_arch.py <model>      # svgg9 | sew_resnet18 | spikformer
```

Runs the compiled network through an independent LIF forward and checks firing/logits
match SpikingJelly bit-exactly (float) — locks the compile→connectivity→LIF chain.

---

## 6. Configuring the chip / experiment parameters

- **`configs/chip_default.json`** — chip (`freq_hz`, `tech_nm`, `sram_kb_per_pe`),
  NoC (`width`/`height`, `link_latency`, `num_vc`, `credit_init`), and sim
  (`num_timesteps`, `traffic_file`, `energy_table_file`, `seed`, `num_threads`).
  Override with `--config`. Defaults: 24×24 mesh, 64 KB/PE, 333 MHz, 28nm.
- **`configs/energy_table_28nm.json`** — pJ per `ActionKind`. **Placeholder values** —
  re-characterize via CACTI for absolute energy claims; the relative structure is
  faithful. Swap with `--energy-table`.
- **`configs/traffic/*.json`** — synthetic traffic patterns for NoC-only runs.
- **Per-PE DNP** — set in the manifest's optional `"dnp"` block at compile time, or
  overridden at run time with the `--dnp-*` flags above.

### Workload→DNP cheat sheet (validated, see `CLAUDE.md` for the full reasoning)

| Workload | Lossless DNP storage | Notes |
|----------|----------------------|-------|
| Static CIFAR-10 (svgg9) | ~2× | Set by conv firing *distribution*; needs firing-rate-aware placement to be usable at all. |
| DVS (DVS128Gesture, T=16) | ~3.3× lossless, ~5× @ ~4% acc cost | Temporal sparsity lets a per-clip adaptive keep retain discriminative neurons. |
| Synthetic temporal input | toward the 5× headline | Age-pruning activates only when neurons go idle in stretches (validated on a rotating-band workload). |

---

## Project layout

```
sim/
  app/main.cpp              # CLI entry point (synthetic + --network modes)
  include/neurort/          # headers
  src/{common,engine,functional,network,stats,timing,energy}/   # 3-layer decoupled core
  tests/                    # gtest unit + golden tests
frontend/
  neurort_compiler/         # the compiler (graph/synapse/mapping/export)
  run_*.py                  # experiment harnesses (drive the C++ sim end-to-end)
  train_*.py                # model training
  checkpoints/              # trained .pth
  tests/                    # frontend tests (run_tests.py runner)
configs/                    # chip/NoC/energy/traffic configs
```

The `sim/` core enforces a strict 3-layer decoupling (functional / engine / energy),
locked by the `decoupling_audit` ctest. See `CLAUDE.md` for the full architecture.
