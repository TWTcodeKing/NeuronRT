#pragma once
#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "neurort/common/ports.hpp"
#include "neurort/common/types.hpp"
#include "neurort/functional/axon_out.hpp"
#include "neurort/functional/dnp.hpp"
#include "neurort/functional/flit.hpp"
#include "neurort/functional/pe_receiver.hpp"
#include "neurort/functional/soma.hpp"
#include "neurort/functional/sync_fsm.hpp"
#include "neurort/functional/tickable.hpp"

namespace neurort {

class Router;
class NetworkImage;
struct PeNetImage;

// M2 network PE: a real compiled-network core on the Fig.5 cross-timestep pipeline.
//   Axon-in/Dendrite : a delivered SPIKE decompresses (Algorithm 1) + accumulates its synaptic
//                      current into the membrane buffer for timestep (t + axon_delay) — a RING of
//                      delay buffers, so a residual skip's longer delay re-aligns it with the main
//                      path at the join (the paper's configurable synaptic delay). MaxPool inputs
//                      (pool_mode==MAX) OR by deduping arrivals per (dendrite,go1,go2) per buffer;
//                      AvgPool scales by 1/pool_n.
//   Soma             : each timestep fires (LIF, no DNP) from the buffer destined for THIS step.
//   Axon-out         : packages fired neurons into NoC flits via the compiled axon table.
// An input-layer PE is seeded with a fixed input current each timestep instead of the NoC.
class NetworkPe final : public Tickable, public PeReceiver {
 public:
  NetworkPe(const NetworkImage& net, PeId id, int num_pe, int mesh_w, std::uint32_t tile_index,
            double tau, double v_threshold, int ring_size, const DnpConfig& dnp_cfg = {});

  void attach_router(Router* r) { router_ = r; }
  void set_input_current(std::vector<double> cur) {  // also used per-timestep by the attention coproc
    is_input_ = true;
    input_current_ = std::move(cur);
  }
  bool is_input() const { return is_input_; }
  void set_inert() { inert_ = true; }   // never fires (qk/av matmul PEs — the coproc does attention)
  bool dnp_enabled() const { return dnp_.has_value(); }
  const Dnp* dnp() const { return dnp_ ? &*dnp_ : nullptr; }   // DNP metrics (null if plain Soma)
  const std::vector<std::uint32_t>& fired() const { return fired_; }  // neurons that fired THIS step
  std::uint32_t neuron_base() const;
  std::uint32_t neuron_count() const { return static_cast<std::uint32_t>(soma_.num_neurons()); }
  double weight_at(std::size_t i) const;   // dequantized weight i of this PE (for proj reassembly)

  void receive_eject(const Flit& f) override { eject_in_.push_next(f); }

  void begin_timestep(Timestep t, ThreadStats& ts);
  void compute(Cycle now, ThreadStats& ts) override;
  void commit(Cycle /*now*/) override { eject_in_.commit(); }
  std::uint32_t tile_index() const override { return tile_index_; }

  SyncController& sync() { return sync_; }
  const SyncController& sync() const { return sync_; }
  bool local_done() const { return spikes_queued_ && pending_spikes_.empty(); }
  std::size_t eject_occupancy() const { return eject_in_.size_cur(); }

  const std::vector<std::uint64_t>& fire_counts() const { return fire_counts_; }
  void reset_counts() {
    std::fill(fire_counts_.begin(), fire_counts_.end(), 0);
    if (dnp_) dnp_->reset_metrics();   // measure-window DNP metrics (keeps neuron state warm)
  }
  std::uint64_t spikes_received() const { return spikes_received_; }
  std::uint64_t total_fires() const {
    std::uint64_t s = 0;
    for (auto c : fire_counts_) s += c;
    return s;
  }

 private:
  void accumulate(std::uint8_t dendrite_id, int go1, int go2, int delay, int meta, ThreadStats& ts);

  PeId id_;
  std::uint32_t tile_index_;
  Router* router_ = nullptr;
  const NetworkImage* net_;
  const PeNetImage* img_;

  Soma soma_;
  std::optional<Dnp> dnp_;   // engaged iff DNP enabled; else the plain-Soma path runs (unchanged)
  AxonOut axon_;
  SyncController sync_;

  int ring_size_;                                    // #delay buffers (>= max axon delay + 1)
  Timestep cur_t_ = 0;
  std::vector<std::vector<double>> ring_;            // ring_[t % ring_size] = current destined for t
  std::vector<std::unordered_set<std::uint64_t>> seen_;  // MAX-term dedup (term,dendrite,go), per ring slot

  bool is_input_ = false;
  bool inert_ = false;
  std::vector<double> input_current_;
  std::vector<double> cur_current_;
  std::vector<std::uint64_t> fire_counts_;

  DoubleBufferedFifo<Flit, kInBufCap> eject_in_{};
  std::deque<Flit> pending_spikes_;
  std::deque<Flit> pending_sync_;
  std::vector<Flit> sync_emit_;
  std::vector<std::uint32_t> fired_;
  bool spikes_queued_ = false;
  std::uint64_t spikes_received_ = 0;
  FlitId next_flit_id_ = 0;
};

}  // namespace neurort
