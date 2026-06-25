#pragma once
#include <cstddef>
#include <vector>

#include "neurort/stats/action.hpp"
#include "neurort/stats/thread_stats.hpp"

namespace neurort {

// Owns per-thread, per-tile action counters and per-thread latency accumulators. Each thread
// writes ONLY its own row (ctr_[tid]), so the hot path is lock-free. At report time the rows are
// merged in a fixed order; since the values are integers, the merged result is bit-identical
// regardless of thread count (the determinism guarantee).
class Stats {
 public:
  Stats(std::size_t num_tiles, int num_threads)
      : num_tiles_(num_tiles),
        num_threads_(num_threads),
        ctr_(static_cast<std::size_t>(num_threads), std::vector<ActionCounters>(num_tiles)),
        lat_(static_cast<std::size_t>(num_threads)) {}

  // View handed to a tile's compute() on thread `tid`.
  ThreadStats view(int tid, std::uint32_t tile) {
    return ThreadStats(ctr_[static_cast<std::size_t>(tid)][tile], lat_[static_cast<std::size_t>(tid)]);
  }

  ActionCounters merge_counters() const {
    ActionCounters g;
    for (std::size_t tile = 0; tile < num_tiles_; ++tile) {
      for (int tid = 0; tid < num_threads_; ++tid) {
        g.merge(ctr_[static_cast<std::size_t>(tid)][tile]);
      }
    }
    return g;
  }
  LatencyAccum merge_latency() const {
    LatencyAccum g;
    for (int tid = 0; tid < num_threads_; ++tid) g.merge(lat_[static_cast<std::size_t>(tid)]);
    return g;
  }

  void reset() {   // zero all action + latency counters (e.g. at the warmup->measure boundary)
    for (auto& row : ctr_) for (auto& c : row) c = ActionCounters{};
    for (auto& l : lat_) l = LatencyAccum{};
  }

  std::size_t num_tiles() const { return num_tiles_; }
  int num_threads() const { return num_threads_; }

 private:
  std::size_t num_tiles_;
  int num_threads_;
  std::vector<std::vector<ActionCounters>> ctr_;  // [thread][tile]
  std::vector<LatencyAccum> lat_;                 // [thread]
};

}  // namespace neurort
