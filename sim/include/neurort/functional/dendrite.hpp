#pragma once
#include <cstdint>
#include <vector>

#include "neurort/common/types.hpp"
#include "neurort/functional/flit.hpp"
#include "neurort/network/network_image.hpp"

namespace neurort {

class ThreadStats;

// Algorithm 1 (Synapse Decompression Flow), storage-agnostic core. For a SPIKE carrying global
// offsets (go1, go2) into dendrite `d`, emit each synapse's (neuron_addr, weight_addr) — both
// PE-local addresses formed from the dendrite's signed-relative base lists plus the per-repeat
// local strides. This is the exact inverse of the compiler's compression (the frontend's
// `synapse/decompress.py` golden); `emit` receives ints so callers can bounds-check before use.
template <class Emit>
inline void decompress(const DendriteImage& d, int go1, int go2, Emit&& emit) {
  for (int c = 0; c < static_cast<int>(d.count); ++c) {
    const int w0 = d.wlist[static_cast<std::size_t>(c)] + go1;
    const int n0 = d.nlist[static_cast<std::size_t>(c)] + go2;
    for (int r = 0; r < d.repeat; ++r) {
      emit(n0 + d.local_off1 * r, w0 + d.local_off2 * r);
    }
  }
}

// M2 Dendrite module (paper Fig.4, Algorithm 1). Decompresses incoming SPIKEs and accumulates the
// dequantized synaptic weights into per-neuron membrane potentials. Models `num_alus` parallel
// synaptic ALUs (paper: 64/PE, configurable): a spike expanding to K = count*repeat synapses
// drains in ceil(K / num_alus) cycles. The membrane potentials are the Soma's input — Dynamic
// Neuron Pruning (Algorithm 2) consumes and resets them in the next M2 step.
//
// Bind-by-reference: the NetworkImage must outlive the Dendrite (it owns the dendrite table + the
// weight blob this reads).
class Dendrite {
 public:
  Dendrite(const NetworkImage& net, PeId pe, int num_alus = kDendriteAlus);

  // Decompress + accumulate one SPIKE (Algorithm 1). Returns the ALU-cycles it consumed.
  // Throws std::out_of_range if a decoded neuron/weight address escapes this PE (a corrupt image;
  // valid compiled networks never trip it).
  std::uint64_t process_spike(std::uint8_t dendrite_id, int go1, int go2, ThreadStats& ts);

  // Convenience: a delivered SPIKE flit -> process_spike; non-spike flits are ignored (return 0).
  std::uint64_t process(const Flit& f, ThreadStats& ts);

  // Synapses a spike into `dendrite_id` expands to (count * repeat), and the matching ALU-cycle cost.
  std::uint64_t synaptic_ops(std::uint8_t dendrite_id) const;
  static std::uint64_t alu_cycles(std::uint64_t ops, int num_alus) {
    const auto a = static_cast<std::uint64_t>(num_alus > 0 ? num_alus : 1);
    return (ops + a - 1) / a;
  }

  double potential(std::size_t neuron) const { return potential_.at(neuron); }
  const std::vector<double>& potentials() const { return potential_; }
  std::size_t neuron_count() const { return potential_.size(); }
  int num_alus() const { return num_alus_; }
  void reset();  // zero all membrane potentials (SNN timestep boundary)

 private:
  const DendriteImage& find(std::uint8_t dendrite_id) const;

  const NetworkImage* net_;
  const PeNetImage* pe_;
  int num_alus_;
  std::vector<double> potential_;       // membrane potential per PE-local neuron
};

}  // namespace neurort
