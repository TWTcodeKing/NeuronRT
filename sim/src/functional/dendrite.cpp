#include "neurort/functional/dendrite.hpp"

#include <algorithm>
#include <stdexcept>

#include "neurort/stats/thread_stats.hpp"

namespace neurort {

Dendrite::Dendrite(const NetworkImage& net, PeId pe, int num_alus)
    : net_(&net), pe_(&net.pes().at(pe)), num_alus_(num_alus),
      potential_(pe_->neuron_count, 0.0) {
  if (num_alus_ < 1) throw std::invalid_argument("Dendrite: num_alus must be >= 1");
}

const DendriteImage& Dendrite::find(std::uint8_t dendrite_id) const {
  for (const auto& d : pe_->dendrites) {
    if (d.id == dendrite_id) return d;
  }
  throw std::out_of_range("Dendrite: SPIKE targets an unknown dendrite id on this PE");
}

std::uint64_t Dendrite::synaptic_ops(std::uint8_t dendrite_id) const {
  const DendriteImage& d = find(dendrite_id);
  return static_cast<std::uint64_t>(d.count) * static_cast<std::uint64_t>(d.repeat);
}

std::uint64_t Dendrite::process_spike(std::uint8_t dendrite_id, int go1, int go2, ThreadStats& ts) {
  const DendriteImage& d = find(dendrite_id);
  ts.bump(ActionKind::SramAccess);  // read the dendrite header (connectivity table)

  std::uint64_t ops = 0;
  const std::size_t n_neurons = potential_.size();
  decompress(d, go1, go2, [&](int n, int w) {
    if (n < 0 || static_cast<std::size_t>(n) >= n_neurons) {
      throw std::out_of_range("Dendrite: decoded neuron address escapes this PE");
    }
    const double weight = net_->weight(*pe_, static_cast<std::size_t>(w));  // bounds-checks w
    potential_[static_cast<std::size_t>(n)] += weight;
    ts.bump(ActionKind::SramAccess);  // 6T weight-array read
    ts.bump(ActionKind::Acc);         // membrane-potential accumulate
    ++ops;
  });
  return alu_cycles(ops, num_alus_);
}

std::uint64_t Dendrite::process(const Flit& f, ThreadStats& ts) {
  if (f.type != FlitType::Spike) return 0;
  return process_spike(f.dendrite_id, f.global_off1, f.global_off2, ts);
}

void Dendrite::reset() { std::fill(potential_.begin(), potential_.end(), 0.0); }

}  // namespace neurort
