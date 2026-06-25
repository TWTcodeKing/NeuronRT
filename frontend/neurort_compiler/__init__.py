"""NeuroRT compiler frontend.

Turns trained SNN models (SEW-ResNet, SpikingVGG, Spikformer) into a per-PE, compressed,
sim-loadable network image (JSON manifest + binary weight blob). Runs in the `spik-yolo` conda
env (PyTorch + SpikingJelly). See format.md for the compiled-file contract.
"""

__version__ = "0.1.0"
