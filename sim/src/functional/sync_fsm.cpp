#include "neurort/functional/sync_fsm.hpp"

#include "neurort/functional/tree_topology.hpp"
#include "neurort/stats/thread_stats.hpp"

namespace neurort {

SyncController::SyncController(PeId id, int num_pe, int mesh_width)
    : id_(id),
      num_pe_(num_pe),
      mesh_width_(mesh_width),
      is_root_(tree::is_root(id)),
      child_count_(tree::child_count(id, num_pe)) {}

Flit SyncController::make_report() const {
  Flit f;
  f.type = FlitType::Report;
  f.src = to_coord(id_, mesh_width_);
  f.dst = to_coord(tree::parent(id_), mesh_width_);
  f.timestep = current_t_;
  f.sync_payload = id_;  // which child is reporting
  return f;
}

Flit SyncController::make_next(PeId child) const {
  Flit f;
  f.type = FlitType::NextTimestep;
  f.src = to_coord(id_, mesh_width_);
  f.dst = to_coord(child, mesh_width_);
  f.timestep = current_t_;  // the completed timestep; recipient advances to current_t_ + 1
  return f;
}

void SyncController::emit_next_to_children(std::vector<Flit>& out, ThreadStats& ts) {
  if (tree::has_left(id_, num_pe_)) {
    out.push_back(make_next(tree::left_child(id_)));
    ts.bump(ActionKind::SyncFlitEmit);
  }
  if (tree::has_right(id_, num_pe_)) {
    out.push_back(make_next(tree::right_child(id_)));
    ts.bump(ActionKind::SyncFlitEmit);
  }
}

void SyncController::on_received(const Flit& f) {
  switch (f.type) {
    case FlitType::Report:
      if (f.timestep == current_t_) ++children_reported_;
      break;
    case FlitType::NextTimestep:
      if (f.timestep == current_t_) next_received_ = true;
      break;
    case FlitType::Broadcast:
      if (f.timestep == current_t_) ++broadcasts_got_;  // ignore stale broadcasts from a past timestep
      break;
    case FlitType::Spike:
      break;  // not a sync event
  }
}

void SyncController::step(bool local_done, std::vector<Flit>& to_inject, ThreadStats& ts) {
  bool progress = true;
  while (progress) {
    progress = false;
    switch (state_) {
      case PeSyncState::Computing:
        if (local_done) {
          state_ = PeSyncState::WaitingChildren;
          progress = true;
        }
        break;

      case PeSyncState::WaitingChildren:
        if (children_reported_ >= child_count_) {
          if (is_root_) {
            // Stage 2 origin: all subtrees reported -> broadcast NEXTTIMESTEP down.
            emit_next_to_children(to_inject, ts);
            state_ = PeSyncState::Advancing;
          } else {
            // Stage 1: report this subtree's completion up to the parent.
            to_inject.push_back(make_report());
            ts.bump(ActionKind::SyncFlitEmit);
            state_ = PeSyncState::WaitingNext;
          }
          progress = true;
        }
        break;

      case PeSyncState::WaitingNext:
        if (next_received_) {
          // Forward NEXTTIMESTEP to children BEFORE we may inject any t+1 SPIKE (deadlock rule).
          emit_next_to_children(to_inject, ts);
          state_ = PeSyncState::Advancing;
          progress = true;
        }
        break;

      case PeSyncState::Advancing:
        break;  // wait for the engine to begin the next timestep
    }
  }
}

void SyncController::begin_next_timestep() {
  ++current_t_;
  state_ = PeSyncState::Computing;
  children_reported_ = 0;
  next_received_ = false;
  broadcasts_got_ = 0;
}

}  // namespace neurort
