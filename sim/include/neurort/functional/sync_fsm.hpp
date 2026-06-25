#pragma once
#include <cstdint>
#include <vector>

#include "neurort/common/types.hpp"
#include "neurort/functional/flit.hpp"

namespace neurort {

class ThreadStats;

// Per-PE state in the decentralized two-stage barrier (paper Sec. IV-D):
//   Computing      - executing this timestep's work (injecting/forwarding SPIKEs).
//   WaitingChildren- local work done; awaiting REPORT from every child subtree.
//   WaitingNext    - (non-root) REPORT sent to parent; awaiting NEXTTIMESTEP.
//   Advancing      - NEXTTIMESTEP forwarded to children; ready for the engine to begin t+1.
enum class PeSyncState : std::uint8_t { Computing, WaitingChildren, WaitingNext, Advancing };

// Drives one PE through the REPORT-up / NEXTTIMESTEP-down handshake. Communicates purely via
// sync flits routed over the NoC, so the barrier is fully decentralized. Deadlock is avoided by
// the rule: a PE may not inject timestep-t SPIKEs until it has received AND forwarded the
// NEXTTIMESTEP entering t; with XY in-order delivery, NEXT cannot be overtaken by later SPIKEs.
class SyncController {
 public:
  SyncController(PeId id, int num_pe, int mesh_width = kMeshW);

  // Accumulate a received sync flit (REPORT from a child / NEXTTIMESTEP from parent / BROADCAST).
  void on_received(const Flit& f);

  // Advance the FSM for the current timestep. `local_done` = this PE's local work for the current
  // timestep is finished. Appends any sync flits to emit (REPORT to parent, NEXTTIMESTEP to
  // children). Idempotent until the next external event.
  void step(bool local_done, std::vector<Flit>& to_inject, ThreadStats& ts);

  // Engine advances the whole chip once every PE is Advancing (the per-timestep barrier).
  bool ready_to_advance() const { return state_ == PeSyncState::Advancing; }
  void begin_next_timestep();

  // Deadlock-avoidance gate consulted by the PE before injecting SPIKEs.
  bool may_inject_spike(Timestep t) const {
    return state_ == PeSyncState::Computing && current_t_ == t;
  }

  Timestep current_timestep() const { return current_t_; }
  PeSyncState state() const { return state_; }
  bool is_root() const { return is_root_; }
  int child_count() const { return child_count_; }

  // BROADCAST gate (layernorm/self-attention state exchange; wired now, exercised from M2).
  void expect_broadcasts(int n) { broadcasts_expected_ = n; }
  bool broadcasts_satisfied() const { return broadcasts_got_ >= broadcasts_expected_; }
  int broadcasts_received() const { return broadcasts_got_; }

 private:
  Flit make_report() const;
  Flit make_next(PeId child) const;
  void emit_next_to_children(std::vector<Flit>& out, ThreadStats& ts);

  PeId id_;
  int num_pe_;
  int mesh_width_;
  bool is_root_;
  int child_count_;

  Timestep current_t_ = 0;
  PeSyncState state_ = PeSyncState::Computing;
  int children_reported_ = 0;
  bool next_received_ = false;

  int broadcasts_expected_ = 0;
  int broadcasts_got_ = 0;
};

}  // namespace neurort
