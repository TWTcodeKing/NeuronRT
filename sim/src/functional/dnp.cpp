#include "neurort/functional/dnp.hpp"

#include <algorithm>
#include <utility>

#include "neurort/stats/action.hpp"
#include "neurort/stats/thread_stats.hpp"

namespace neurort {

Dnp::Dnp(std::uint32_t n_log, std::uint32_t n_phys, double tau, double v_threshold,
         std::unique_ptr<ThresholdPolicy> policy, bool skip_pruned)
    : n_log_(n_log), n_phys_(n_phys), tau_(tau), v_th_(v_threshold), policy_(std::move(policy)),
      skip_pruned_(skip_pruned) {
  state_.assign(n_phys_, 0.0);
  valid_.assign(n_phys_, 0);
  age_.assign(n_phys_, 0);
  pending_.assign(n_phys_, 0);
  rev_map_.assign(n_phys_, 0);
  slot_input_.assign(n_phys_, 0.0);
  fired_mask_.assign(n_phys_, 0);
  map_table_.assign(n_log_, 0);
  map_valid_.assign(n_log_, 0);
  ever_mapped_.assign(n_log_, 0);
  free_list_.assign(static_cast<std::size_t>(n_phys_) + 1, 0);   // ring: empty <=> head == tail
  for (std::uint32_t i = 0; i < n_phys_; ++i) free_list_[i] = i;  // all slots free initially
  free_head_ = 0;
  free_tail_ = n_phys_;
}

void Dnp::reset() {
  std::fill(state_.begin(), state_.end(), 0.0);
  std::fill(valid_.begin(), valid_.end(), 0);
  std::fill(age_.begin(), age_.end(), 0);
  std::fill(pending_.begin(), pending_.end(), 0);
  std::fill(slot_input_.begin(), slot_input_.end(), 0.0);
  std::fill(fired_mask_.begin(), fired_mask_.end(), 0);
  std::fill(map_valid_.begin(), map_valid_.end(), 0);
  std::fill(ever_mapped_.begin(), ever_mapped_.end(), 0);
  for (std::uint32_t i = 0; i < n_phys_; ++i) free_list_[i] = i;
  free_head_ = 0;
  free_tail_ = n_phys_;
  live_ = peak_ = 0;
  prunes_ = rejects_ = allocs_ = 0;
}

// Reset only the measurement counters (NOT the neuron state) — used at the warmup->measure boundary
// so the reported storage/pruning numbers cover the measure window, like fire_counts_.
void Dnp::reset_metrics() {
  peak_ = live_;                  // high-water restarts from the current occupancy
  prunes_ = rejects_ = allocs_ = 0;
}

std::uint32_t Dnp::free_pop(ThreadStats& ts) {
  const std::uint32_t p = free_list_[free_head_];                 // caller guarantees !free_empty()
  free_head_ = (free_head_ + 1) % static_cast<std::uint32_t>(free_list_.size());
  ts.bump(ActionKind::FreeListPop);
  return p;
}

void Dnp::free_push(std::uint32_t phys, ThreadStats& ts) {
  free_list_[free_tail_] = phys;
  free_tail_ = (free_tail_ + 1) % static_cast<std::uint32_t>(free_list_.size());
  ts.bump(ActionKind::FreeListPush);
}

void Dnp::reclaim(std::uint32_t phys, ThreadStats& ts) {
  valid_[phys] = 0;
  pending_[phys] = 1;                       // reclaimed; cleared when the slot is re-allocated
  map_valid_[rev_map_[phys]] = 0;           // logical neuron unmapped -> future input re-allocates
  ts.bump(ActionKind::MapTableWrite);
  free_push(phys, ts);
  ts.bump(ActionKind::ReclaimOp);
  --live_;
  ++prunes_;
}

void Dnp::step(const std::vector<double>& current, std::vector<std::uint32_t>& out, Timestep t,
               ThreadStats& ts) {
  const std::size_t out0 = out.size();      // append-only contract (matches Soma::fire)

  // 1. AGEINCREMENT — every valid slot ages one timestep.
  for (std::uint32_t i = 0; i < n_phys_; ++i) {
    if (valid_[i]) {
      ++age_[i];
      ts.bump(ActionKind::AgeTick);
    }
  }

  // 2. UpdateFSM — route each logical neuron's input to its physical slot (allocate on demand).
  for (std::uint32_t L = 0; L < n_log_; ++L) {
    const double in = current[L];
    if (in == 0.0) continue;                // no synaptic input -> no update / no allocation
    if (map_valid_[L]) {                    // mapped
      const std::uint32_t p = map_table_[L];
      ts.bump(ActionKind::MapTableRead);
      if (!pending_[p]) {                   // mapped & !pending -> stage input, reset age
        slot_input_[p] = in;
        age_[p] = 0;
        ts.bump(ActionKind::Acc);
      }                                     // mapped & pending -> ignore (single-thread no-op)
    } else if (!free_empty()) {             // unmapped & free slot -> allocate
      const std::uint32_t p = free_pop(ts);
      map_table_[L] = p;
      map_valid_[L] = 1;
      ever_mapped_[L] = 1;                  // first allocation marks it; sticky-skip applies if later pruned
      rev_map_[p] = L;
      valid_[p] = 1;
      pending_[p] = 0;
      state_[p] = 0.0;                      // fresh membrane (a re-allocated neuron loses its past)
      age_[p] = 0;
      slot_input_[p] = in;
      ts.bump(ActionKind::MapTableWrite);
      ts.bump(ActionKind::Acc);
      ++allocs_;
      ++live_;
    } else {                               // unmapped & no free slot -> reject (input dropped)
      ++rejects_;
    }
  }
  peak_ = std::max(peak_, live_);           // intra-step high-water (before any reclaim this step)

  // 3. LIF + FIRE — finalize every valid !pending slot (leak applies even with zero input, exactly
  //    like dense Soma which integrates every neuron every step). slot_input defaults to 0.
  for (std::uint32_t i = 0; i < n_phys_; ++i) {
    if (!valid_[i] || pending_[i]) continue;
    const bool fired = lif_step(state_[i], slot_input_[i], tau_, v_th_);
    slot_input_[i] = 0.0;                   // consume; clean for next step
    ts.bump(ActionKind::SramAccess, 2);     // membrane-potential read + write-back
    ts.bump(ActionKind::Mul);               // (I - v) * (1/tau)
    ts.bump(ActionKind::Add);               // v += ...
    ts.bump(ActionKind::Comp);              // v >= v_th
    if (fired) {
      out.push_back(rev_map_[i]);
      age_[i] = 0;                          // firing counts as activity (resets the idle counter)
      fired_mask_[i] = 1;                   // protect this slot's transient reset-to-0 from POT-prune
    }
  }

  // 4. threshold policy (Phase 1 fixed -> constant; Phase 2 adaptive Eqs 1-4 drop in here).
  const DnpThresholds th =
      policy_->update(static_cast<std::uint64_t>(out.size() - out0), live_, t);

  // 5. PRUNE + RECLAIM — zombie neurons (paper Algo 2 + text): idle too long (age_cnt >= AGE_THRESH)
  //    OR membrane too low (V_n <= POT_THRESH). POT-pruning deliberately targets neurons that DO
  //    receive input every step but stay silent (never fire) — so it is NOT gated on idleness; only a
  //    slot that FIRED this step is skipped, since its membrane was transiently reset to 0 (an active
  //    neuron, not a zombie). A just-fired slot has age==0, so age-pruning never hits it either.
  for (std::uint32_t i = 0; i < n_phys_; ++i) {
    if (!valid_[i] || pending_[i]) continue;
    ts.bump(ActionKind::PruneScan);
    if (fired_mask_[i]) { fired_mask_[i] = 0; continue; }   // fired this step -> active; clear & skip
    if (age_[i] >= th.age_thresh || state_[i] <= th.pot_thresh) reclaim(i, ts);
  }

  // 6. emit fired ids in ascending logical order (== dense Soma's emission order).
  std::sort(out.begin() + static_cast<std::ptrdiff_t>(out0), out.end());
}

}  // namespace neurort
