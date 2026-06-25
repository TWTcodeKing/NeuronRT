#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace neurort {

// The one coupling between the simulator and the energy layer: integer action counters.
// functional/timing code only ever calls ActionCounters::bump(); energy/ reads them later.
// (No floats, no behaviour here — keeps the dependency arrow clean: functional -> stats,
//  energy -> stats, and energy is NEVER included by functional/timing.)
enum class ActionKind : std::uint16_t {
  // NoC
  RouterRouteCompute,
  RouterArbitrate,
  RouterBufferWrite,
  RouterBufferRead,
  LinkTraversal,
  CreditReturn,
  // PE
  SpikeInject,
  SpikeEject,
  SyncFlitEmit,
  // compute primitives (defined now so the energy table is complete; used from M2)
  Mul,
  Acc,
  Add,
  And,
  Comp,
  Mux,
  Reg,
  Sft,
  // memory
  SramAccess,
  DramAccess,
  ScratchpadAccess,
  // Dynamic Neuron Pruning (Soma DNP, Algorithm 2) — virtual-memory neuron-state management
  MapTableRead,    // logical -> physical slot lookup
  MapTableWrite,   // logical->phys mapping create / invalidate
  FreeListPop,     // allocate a physical slot from the free list
  FreeListPush,    // return a reclaimed physical slot to the free list
  AgeTick,         // age_cnt increment for one valid slot
  PruneScan,       // one slot examined against the prune thresholds
  ReclaimOp,       // a slot reclaimed (zombie neuron pruned)
  kNumKinds
};
inline constexpr std::size_t kNumActionKinds = static_cast<std::size_t>(ActionKind::kNumKinds);

const char* to_cstr(ActionKind k);

struct ActionCounters {
  std::array<std::uint64_t, kNumActionKinds> c{};

  void bump(ActionKind k, std::uint64_t n = 1) { c[static_cast<std::size_t>(k)] += n; }
  std::uint64_t get(ActionKind k) const { return c[static_cast<std::size_t>(k)]; }
  void merge(const ActionCounters& o) {
    for (std::size_t i = 0; i < kNumActionKinds; ++i) c[i] += o.c[i];
  }
};

}  // namespace neurort
