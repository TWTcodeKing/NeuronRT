#pragma once
#include <cstdint>

#include "neurort/common/types.hpp"
#include "neurort/stats/action.hpp"

namespace neurort {

// Per-thread latency/throughput accumulator (merged across threads at report time).
struct LatencyAccum {
  std::uint64_t delivered = 0;       // flits ejected at destination
  std::uint64_t total_latency = 0;   // sum of (eject_cycle - inject_cycle)
  std::uint64_t max_latency = 0;

  void record(Cycle inject_cycle, Cycle eject_cycle) {
    const std::uint64_t lat = eject_cycle - inject_cycle;
    ++delivered;
    total_latency += lat;
    if (lat > max_latency) max_latency = lat;
  }
  void merge(const LatencyAccum& o) {
    delivered += o.delivered;
    total_latency += o.total_latency;
    if (o.max_latency > max_latency) max_latency = o.max_latency;
  }
  double mean_latency() const {
    return delivered ? static_cast<double>(total_latency) / static_cast<double>(delivered) : 0.0;
  }
};

// Lightweight view handed to Tickable::compute(). Binds the entity's tile action-counters and
// a per-thread latency accumulator. Hot path: bump() is a single array increment, no locks.
class ThreadStats {
 public:
  ThreadStats(ActionCounters& counters, LatencyAccum& lat)
      : counters_(&counters), lat_(&lat) {}

  void bump(ActionKind k, std::uint64_t n = 1) { counters_->bump(k, n); }
  void record_latency(Cycle inject_cycle, Cycle eject_cycle) {
    lat_->record(inject_cycle, eject_cycle);
  }

 private:
  ActionCounters* counters_;
  LatencyAccum* lat_;
};

}  // namespace neurort
