#pragma once
#include <cstdint>

#include "neurort/common/types.hpp"

namespace neurort {

class ThreadStats;  // defined in stats/ (M1.6); a per-thread view for bumping action counters

// Every simulated entity (Router, PE, ...) implements Tickable. The BSP engine drives all
// entities through two phases per NoC cycle, separated by a barrier:
//
//   PHASE A  compute(): read CUR buffers of self + neighbours; write ONLY own NEXT buffers.
//                       Must not mutate another entity's CUR/visible state.
//   PHASE B  commit():  finalize own NEXT (carry-forward unsent, merge staging) and advance
//                       own double buffers. Touches only self.
//
// Because compute() reads CUR and writes NEXT, and commit() is the only place buffers advance
// (single-threaded per entity), the simulation is bit-identical regardless of thread count.
class Tickable {
 public:
  virtual ~Tickable() = default;

  virtual void compute(Cycle now, ThreadStats& ts) = 0;
  virtual void commit(Cycle now) = 0;

  // Stable identity for fixed-order iteration and per-tile stats indexing.
  virtual std::uint32_t tile_index() const = 0;
};

}  // namespace neurort
