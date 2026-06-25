#pragma once

namespace neurort {

// The single source of truth for the leaky-integrate-and-fire membrane step, shared by Soma (dense
// neuron state) and Dnp (virtual-memory neuron state) so the two never diverge. Reproduces
// SpikingJelly's decay-input, hard-reset-to-0 LIFNode: given this timestep's input current I,
//   v += (I - v) / tau ;  spike = (v >= v_threshold) ;  on spike v <- 0 (hard reset).
// Returns true iff the neuron fired this step. `v` persists across timesteps in the caller's store.
inline bool lif_step(double& v, double input, double tau, double v_threshold) {
  v += (input - v) / tau;
  if (v >= v_threshold) {
    v = 0.0;   // hard reset to v_reset = 0
    return true;
  }
  return false;
}

}  // namespace neurort
