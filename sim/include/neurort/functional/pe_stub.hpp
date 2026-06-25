#pragma once
#include <cstdint>
#include <deque>
#include <vector>

#include "neurort/common/ports.hpp"
#include "neurort/common/types.hpp"
#include "neurort/functional/flit.hpp"
#include "neurort/functional/pe_receiver.hpp"
#include "neurort/functional/sync_fsm.hpp"
#include "neurort/functional/tickable.hpp"

namespace neurort {

class Router;
class TrafficSource;

// M1 Processing Element: a traffic injector/sink + sync-protocol client. No Dendrite/Soma compute
// yet (that is M2+). Still exercises the ping-pong / single-writer discipline so the engine path
// is identical to what M2 will need.
//
// Dataflow per cycle (PHASE A compute):
//   (a) drain flits the router ejected to us last cycle (sync -> SyncController; spikes discarded);
//   (b) generate this timestep's spikes once (gated by the deadlock rule may_inject_spike);
//   (c) advance the sync FSM, queueing any REPORT/NEXTTIMESTEP to inject;
//   (d) inject queued flits into the router Local port (sync first, then spikes) until backpressure.
class ProcessingElementStub final : public Tickable, public PeReceiver {
 public:
  ProcessingElementStub(PeId id, int num_pe, int mesh_width, std::uint32_t tile_index);

  void attach_router(Router* r) { router_ = r; }
  void attach_traffic(TrafficSource* t) { traffic_ = t; }

  // Router delivers an ejected flit into our ingress staging (single writer = the router).
  void receive_eject(const Flit& f) override { eject_in_.push_next(f); }

  void compute(Cycle now, ThreadStats& ts) override;
  void commit(Cycle now) override;
  std::uint32_t tile_index() const override { return tile_index_; }

  SyncController& sync() { return sync_; }
  const SyncController& sync() const { return sync_; }

  void begin_timestep(Timestep t);  // single-threaded, at the timestep boundary

  // This PE's local work for the current timestep is done once all its spikes are injected.
  bool local_done() const { return spikes_generated_ && pending_spikes_.empty(); }

  std::size_t eject_occupancy() const { return eject_in_.size_cur(); }

 private:
  PeId id_;
  std::uint32_t tile_index_;
  Router* router_ = nullptr;
  TrafficSource* traffic_ = nullptr;
  SyncController sync_;

  DoubleBufferedFifo<Flit, kInBufCap> eject_in_{};  // flits ejected to us by our router
  std::deque<Flit> pending_spikes_;                 // spikes awaiting injection (backpressure retry)
  std::deque<Flit> pending_sync_;                   // REPORT/NEXTTIMESTEP awaiting injection
  std::vector<Flit> scratch_;                       // reused traffic buffer
  std::vector<Flit> sync_emit_;                     // reused sync-emit buffer

  bool spikes_generated_ = false;
  FlitId next_flit_id_ = 0;
};

}  // namespace neurort
