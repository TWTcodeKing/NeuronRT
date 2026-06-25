#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "neurort/common/types.hpp"
#include "neurort/functional/lif.hpp"
#include "neurort/stats/thread_stats.hpp"

namespace neurort {

// M2 Soma — normal membrane-potential update + firing (LIFNode; no Dynamic Neuron Pruning yet).
// Reproduces SpikingJelly's decay-input, hard-reset-to-0 LIF: for each neuron, given this
// timestep's input current I (the Dendrite's accumulated synaptic weight),
//   v += (I - v) / tau ;  spike = (v >= v_threshold) ;  v *= (1 - spike)    (hard reset to 0)
// The membrane potential v persists across timesteps; `fire` returns the local ids that spiked.
class Soma {
 public:
  Soma(std::size_t num_neurons, double tau, double v_threshold)
      : tau_(tau), v_th_(v_threshold), v_(num_neurons, 0.0) {}

  // Integrate the per-neuron input current, fire, reset. Appends fired local-neuron ids to `out`.
  // Energy: per neuron per step the LIF reads + writes its membrane potential (neuron-state SRAM)
  // and does the decay-integrate + threshold compare.
  void fire(const std::vector<double>& current, std::vector<std::uint32_t>& out, ThreadStats& ts) {
    for (std::size_t n = 0; n < v_.size(); ++n) {
      ts.bump(ActionKind::SramAccess, 2);   // membrane-potential read + write-back
      ts.bump(ActionKind::Mul);             // (I - v) * (1/tau)
      ts.bump(ActionKind::Add);             // v += ...
      ts.bump(ActionKind::Comp);            // v >= v_th
      if (lif_step(v_[n], current[n], tau_, v_th_)) out.push_back(static_cast<std::uint32_t>(n));
    }
  }

  double potential(std::size_t n) const { return v_[n]; }
  std::size_t num_neurons() const { return v_.size(); }
  void reset() { std::fill(v_.begin(), v_.end(), 0.0); }

 private:
  double tau_;
  double v_th_;
  std::vector<double> v_;   // membrane potential per local neuron (persists across timesteps)
};

}  // namespace neurort
