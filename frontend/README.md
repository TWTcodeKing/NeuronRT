# NeuroRT compiler frontend (`neurort_compiler`)

Turns trained SNN models into a per-PE, compressed, sim-loadable network image
(JSON manifest + binary weight blob). Scope: **SEW-ResNet18/34, SpikingVGG16, Spikformer**.

## Environment

Runs in the `spik-yolo` conda env (PyTorch 2.8 + SpikingJelly 0.0.0.0.14 + einops). pytest is
NOT installed, so tests use the dependency-free `run_tests.py`.

```bash
PY=/home/twt/.conda/envs/spik-yolo/bin/python
cd frontend
$PY run_tests.py            # run all tests (optionally pass a substring filter)
$PY -m neurort_compiler.cli --model spiking_vgg16 --input 32x32 --timesteps 4 --out build/net
```

## Pipeline

`models/` (model defs) → `graph/` (NeuralGraph IR via shape-inference hooks) → `synapse/`
(tensor→connectivity, compression, Algorithm-1 decompress golden) → `mapping/` (layer-wise
partition to 576 PEs) → `export/` (manifest + weight blob). The format contract is `format.md`;
the C++ side is `sim/.../network/network_image.{hpp,cpp}`.
