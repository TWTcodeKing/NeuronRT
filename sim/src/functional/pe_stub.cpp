#include "neurort/functional/pe_stub.hpp"

#include "neurort/functional/router.hpp"
#include "neurort/functional/traffic.hpp"
#include "neurort/stats/thread_stats.hpp"

namespace neurort {

ProcessingElementStub::ProcessingElementStub(PeId id, int num_pe, int mesh_width,
                                             std::uint32_t tile_index)
    : id_(id), tile_index_(tile_index), sync_(id, num_pe, mesh_width) {}

void ProcessingElementStub::begin_timestep(Timestep /*t*/) { spikes_generated_ = false; }

void ProcessingElementStub::compute(Cycle now, ThreadStats& ts) {
  // (a) Drain flits ejected to us last cycle.
  while (!eject_in_.empty_cur()) {
    const Flit& f = eject_in_.front_cur();
    if (is_sync(f.type)) sync_.on_received(f);  // spikes are discarded by the M1 stub
    eject_in_.pop_cur();
  }

  // (b) Generate this timestep's spikes once we are allowed to inject for it.
  const Timestep t = sync_.current_timestep();
  if (!spikes_generated_ && sync_.may_inject_spike(t)) {
    if (traffic_ != nullptr) {
      traffic_->spikes_for(id_, t, scratch_);
      for (const auto& f : scratch_) pending_spikes_.push_back(f);
    }
    spikes_generated_ = true;
  }

  // (c) Advance the sync FSM; queue any REPORT / NEXTTIMESTEP it emits.
  sync_emit_.clear();
  sync_.step(local_done(), sync_emit_, ts);
  for (const auto& f : sync_emit_) pending_sync_.push_back(f);

  // (d) Inject — sync flits first (avoid sync starvation), then spikes — until backpressure.
  while (!pending_sync_.empty()) {
    Flit f = pending_sync_.front();
    f.inject_cycle = now;
    if (!router_->try_inject(f)) break;
    pending_sync_.pop_front();
  }
  while (!pending_spikes_.empty()) {
    Flit f = pending_spikes_.front();
    f.id = next_flit_id_++;
    f.inject_cycle = now;
    if (!router_->try_inject(f)) break;
    ts.bump(ActionKind::SpikeInject);
    pending_spikes_.pop_front();
  }
}

void ProcessingElementStub::commit(Cycle /*now*/) { eject_in_.commit(); }

}  // namespace neurort
