#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "neurort/common/types.hpp"
#include "neurort/functional/lif.hpp"

namespace neurort {

class ThreadStats;

// ----------------------------------------------------------------------------------------------
// Dynamic Neuron Pruning (Soma, paper Algorithm 2).
//
// A per-PE VIRTUAL-MEMORY neuron store: n_phys physical slots (paper: 4096) logically manage up to
// n_log logical neurons (paper: 16384 at the 25% sweet spot). It exploits implicit output sparsity
// ("zombie" neurons that stay inactive) — only neurons that are mapped to a physical slot hold
// membrane state and fire; idle / sub-threshold neurons are reclaimed so their slots serve others.
//
// SEAM (see plan): the NetworkPe delay ring + accumulate stay LOGICAL-indexed and unchanged. Dnp
// owns only the persistent membrane state. `step` consumes a dense logical current vector and emits
// fired LOGICAL ids (via Rev_Map), so Axon-out is untouched. With n_phys >= n_log and pruning off
// (age_thresh = max, pot_thresh = -inf) it reproduces dense Soma's firing exactly (golden #1).
// ----------------------------------------------------------------------------------------------

// Reclaim a slot when age_cnt >= age_thresh (idle too long) OR membrane <= pot_thresh (too low).
struct DnpThresholds {
  std::uint32_t age_thresh = std::numeric_limits<std::uint32_t>::max();   // idle-timesteps; max => off
  double        pot_thresh = -std::numeric_limits<double>::infinity();    // -inf => off
};

// Pluggable threshold policy. Phase 1 = constant (FixedThresholdPolicy). Phase 2's adaptive Eqs 1-4
// (AGE_THRESH = w*(f/r)+(1-w)*f_t, POT_THRESH adaptation) drop in here without touching Dnp/NetworkPe.
class ThresholdPolicy {
 public:
  virtual ~ThresholdPolicy() = default;
  // Called once per timestep BEFORE the prune scan. Observes this step's firing + occupancy.
  virtual DnpThresholds update(std::uint64_t fires_this_step, std::uint32_t valid_slots,
                               Timestep t) = 0;
};

class FixedThresholdPolicy : public ThresholdPolicy {
 public:
  explicit FixedThresholdPolicy(DnpThresholds th) : th_(th) {}
  DnpThresholds update(std::uint64_t, std::uint32_t, Timestep) override { return th_; }

 private:
  DnpThresholds th_;
};

// Per-PE DNP configuration (carried in the manifest's optional "dnp" block / a runtime override).
struct DnpConfig {
  bool          enabled    = false;
  std::uint32_t n_phys     = 0;       // 0 => derive per-PE from phys_ratio
  double        phys_ratio = 0.25;    // n_phys = ceil(phys_ratio * n_log) when n_phys == 0
  std::uint32_t age_thresh = std::numeric_limits<std::uint32_t>::max();
  double        pot_thresh = -std::numeric_limits<double>::infinity();
  // Dendrite-side skip (paper "ignore update for [pruned]"): once a logical neuron has been mapped
  // then reclaimed (a pruned zombie), the Dendrite skips its synaptic weight-read + accumulate. This
  // is STICKY after prune — a pruned neuron no longer receives input, so UpdateFSM's allocate-on-input
  // never re-maps it (consistent: no input => no allocation). A NEVER-mapped neuron is NOT skipped
  // (ever_mapped_=0), so its first input still triggers allocation (the allocate-on-input bootstrap).
  // Without this the Dendrite reads weights for every decoded neuron regardless of DNP => DNP saves
  // storage but ~0 energy; with it, pruned zombies skip the dominant SRAM weight reads => real win.
  bool          skip_pruned = false;

  // n_phys for a PE of `n_log` logical neurons, clamped to [1, n_log] (more physical than logical is
  // pointless, and the clamp keeps the float->uint cast well-defined for any phys_ratio).
  std::uint32_t resolve_phys(std::uint32_t n_log) const {
    if (n_phys > 0) return n_phys;
    double p = std::ceil(phys_ratio * static_cast<double>(n_log));
    if (p < 1.0) p = 1.0;
    if (p > static_cast<double>(n_log)) p = static_cast<double>(n_log);
    return static_cast<std::uint32_t>(p);
  }
};

class Dnp {
 public:
  Dnp(std::uint32_t n_log, std::uint32_t n_phys, double tau, double v_threshold,
      std::unique_ptr<ThresholdPolicy> policy, bool skip_pruned = false);

  // One timestep (Fig.5 ping-pong: finalize t-1, fire into t). `current` is LOGICAL-indexed
  // (size n_log) — the consumed ring slot. Appends fired LOGICAL ids (ascending) to `out`. `t` is
  // the absolute timestep (for the threshold policy; unused by FixedThresholdPolicy).
  void step(const std::vector<double>& current, std::vector<std::uint32_t>& out, Timestep t,
            ThreadStats& ts);

  void reset();          // clear all slots/maps, rebuild the free list (fresh-run boundary)
  void reset_metrics();  // zero peak/prune/reject/alloc but KEEP neuron state (warmup->measure split)

  // ---- metrics (the energy/memory headline numbers) ----
  std::uint32_t n_log()        const { return n_log_; }
  std::uint32_t n_phys()       const { return n_phys_; }
  std::uint32_t live_slots()   const { return live_; }     // currently mapped
  std::uint32_t peak_slots()   const { return peak_; }     // high-water mark -> storage proxy
  std::uint64_t prune_count()  const { return prunes_; }
  std::uint64_t reject_count() const { return rejects_; }  // UpdateFSM no-free-slot drops
  std::uint64_t alloc_count()  const { return allocs_; }

  // ---- inspection (tests) ----
  bool   mapped(std::uint32_t logical) const { return logical < n_log_ && map_valid_[logical]; }
  // Should the Dendrite SKIP this neuron's synaptic weight-read? True only for pruned zombies: a
  // neuron that was mapped (ever_mapped_) but is currently unmapped (map_valid_=0, i.e. reclaimed).
  // Never-mapped neurons return false so their first input allocates a slot (bootstrap).
  bool   should_skip(std::uint32_t logical) const {
    return skip_pruned_ && logical < n_log_ && ever_mapped_[logical] && !map_valid_[logical];
  }
  double potential(std::uint32_t logical) const {
    return mapped(logical) ? state_[map_table_[logical]] : 0.0;
  }
  // Physical slot backing `logical` (only valid when mapped(logical)); for tests/inspection.
  std::uint32_t phys_slot(std::uint32_t logical) const { return map_table_[logical]; }

 private:
  bool          free_empty() const { return free_head_ == free_tail_; }
  std::uint32_t free_pop(ThreadStats& ts);
  void          free_push(std::uint32_t phys, ThreadStats& ts);
  void          reclaim(std::uint32_t phys, ThreadStats& ts);

  // Five per-PHYSICAL-slot arrays (Algorithm 2), sized n_phys:
  std::vector<double>        state_;       // State    : membrane potential V (persists)
  std::vector<std::uint8_t>  valid_;       // Valid    : slot occupied
  std::vector<std::uint32_t> age_;         // Age_Cnt  : idle timesteps since last update
  std::vector<std::uint8_t>  pending_;     // pending  : reclaimed, not yet re-initialised
  std::vector<std::uint32_t> rev_map_;     // Rev_Map  : phys -> logical
  std::vector<double>        slot_input_;  // scratch  : this step's input current per slot
  std::vector<std::uint8_t>  fired_mask_;  // scratch  : slot fired this step (protect its reset-to-0)

  // logical -> phys (sized n_log):
  std::vector<std::uint32_t> map_table_;   // map_table : logical -> phys slot
  std::vector<std::uint8_t>  map_valid_;   // map_valid
  std::vector<std::uint8_t>  ever_mapped_; // logical ever allocated (sticky-skip gate; sized n_log)

  // FIFO free_list of physical slot ids (paper free_head/free_tail), ring of size n_phys+1.
  std::vector<std::uint32_t> free_list_;
  std::uint32_t free_head_ = 0, free_tail_ = 0;

  std::uint32_t n_log_, n_phys_;
  double tau_, v_th_;
  std::unique_ptr<ThresholdPolicy> policy_;
  bool skip_pruned_ = false;

  std::uint32_t live_ = 0, peak_ = 0;
  std::uint64_t prunes_ = 0, rejects_ = 0, allocs_ = 0;
};

}  // namespace neurort
