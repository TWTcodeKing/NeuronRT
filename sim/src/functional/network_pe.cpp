#include "neurort/functional/network_pe.hpp"

#include <memory>
#include <stdexcept>

#include "neurort/functional/dendrite.hpp"   // decompress() (Algorithm 1)
#include "neurort/functional/router.hpp"
#include "neurort/network/network_image.hpp"
#include "neurort/stats/thread_stats.hpp"

namespace neurort {

NetworkPe::NetworkPe(const NetworkImage& net, PeId id, int num_pe, int mesh_w,
                     std::uint32_t tile_index, double tau, double v_threshold, int ring_size,
                     const DnpConfig& dnp_cfg)
    : id_(id),
      tile_index_(tile_index),
      net_(&net),
      img_(&net.pes().at(id)),
      soma_(net.pes().at(id).neuron_count, tau, v_threshold),
      axon_(net.pes().at(id), mesh_w),
      sync_(id, num_pe, mesh_w),
      ring_size_(ring_size < 1 ? 1 : ring_size),
      ring_(static_cast<std::size_t>(ring_size_),
            std::vector<double>(net.pes().at(id).neuron_count, 0.0)),
      seen_(static_cast<std::size_t>(ring_size_)),
      cur_current_(net.pes().at(id).neuron_count, 0.0),
      fire_counts_(net.pes().at(id).neuron_count, 0) {
  if (dnp_cfg.enabled) {  // Soma DNP: virtual-memory neuron state with N_phys < N_log physical slots.
    const std::uint32_t n_log = net.pes().at(id).neuron_count;
    dnp_.emplace(n_log, dnp_cfg.resolve_phys(n_log), tau, v_threshold,
                 std::make_unique<FixedThresholdPolicy>(
                     DnpThresholds{dnp_cfg.age_thresh, dnp_cfg.pot_thresh}),
                 dnp_cfg.skip_pruned);
  }
}

std::uint32_t NetworkPe::neuron_base() const { return img_->neuron_base; }
double NetworkPe::weight_at(std::size_t i) const { return net_->weight(*img_, i); }

void NetworkPe::accumulate(std::uint8_t dendrite_id, int go1, int go2, int delay, int meta,
                           ThreadStats& ts) {
  // The synapse contributes `delay` timesteps after arrival (delay-1 = next step). Route it to the
  // ring buffer destined for that future timestep so residual/skip paths re-align at their join.
  const std::size_t slot =
      static_cast<std::size_t>((cur_t_ + static_cast<Timestep>(delay <= 0 ? 1 : delay)) % ring_size_);

  // Per-edge combine (meta): MAX(1) OR-folds a MaxPool window — dedup per (term,dendrite,go) within
  // the slot, separately per source term so a residual's main SUM term is unaffected; AVG(2) scales
  // by 1/avg_n; SUM(0) accumulates. term and avg_n share the spike flit's spare meta field.
  const int combine = meta & 0x3;
  const int term = (meta >> 2) & 0x3F;
  const double scale = (combine == 2) ? 1.0 / static_cast<double>((meta >> 8) & 0xFF) : 1.0;
  if (combine == 1) {
    const std::uint64_t key = (static_cast<std::uint64_t>(term) << 56) |
                              (static_cast<std::uint64_t>(dendrite_id) << 48) |
                              (static_cast<std::uint64_t>(go1 & 0xFFFFFF) << 24) |
                              static_cast<std::uint64_t>(go2 & 0xFFFFFF);
    if (!seen_[slot].insert(key).second) return;  // window neuron already OR'd into this term
  }

  const DendriteImage* d = nullptr;
  for (const auto& dd : img_->dendrites) {
    if (dd.id == dendrite_id) { d = &dd; break; }
  }
  if (d == nullptr) throw std::out_of_range("NetworkPe: SPIKE targets unknown dendrite id");
  ts.bump(ActionKind::SramAccess);  // dendrite-header read
  std::vector<double>& buf = ring_[slot];
  const std::size_t n_neurons = buf.size();
  decompress(*d, go1, go2, [&](int n, int w) {
    if (n < 0 || static_cast<std::size_t>(n) >= n_neurons) {
      throw std::out_of_range("NetworkPe: decoded neuron address escapes this PE");
    }
    // DNP sticky-skip (paper "ignore update for [pruned]"): a pruned zombie (was mapped, now
    // reclaimed) skips the synaptic weight-read + accumulate — the dominant SRAM cost. A never-mapped
    // neuron is NOT skipped so its first input still allocates a slot (bootstrap). Bump only the
    // map-table check (1 cheap read) for skipped neurons.
    if (dnp_ && dnp_->should_skip(static_cast<std::uint32_t>(n))) {
      ts.bump(ActionKind::MapTableRead);
      return;
    }
    buf[static_cast<std::size_t>(n)] += net_->weight(*img_, static_cast<std::size_t>(w)) * scale;
    ts.bump(ActionKind::SramAccess, 3);  // weight read + membrane-current read-modify-write
    ts.bump(ActionKind::Mul);            // int8 weight * dequant scale
    ts.bump(ActionKind::Acc);            // accumulate into membrane current
  });
}

void NetworkPe::begin_timestep(Timestep t, ThreadStats& ts) {
  cur_t_ = t;
  if (t > 0) sync_.begin_next_timestep();

  const std::size_t slot = static_cast<std::size_t>(t % ring_size_);
  cur_current_ = is_input_ ? input_current_ : ring_[slot];   // current destined for THIS timestep
  std::fill(ring_[slot].begin(), ring_[slot].end(), 0.0);    // free this slot (reused at t+ring_size)
  seen_[slot].clear();

  fired_.clear();
  pending_spikes_.clear();
  spikes_queued_ = true;
  if (inert_) return;   // qk/av matmul PEs never fire/inject (attention runs in the coprocessor)

  if (dnp_) dnp_->step(cur_current_, fired_, cur_t_, ts);   // Soma DNP (virtual-memory neuron state)
  else soma_.fire(cur_current_, fired_, ts);                // plain dense LIF (unchanged path)
  for (std::uint32_t n : fired_) ++fire_counts_[n];

  for (std::uint32_t n : fired_) {
    ts.bump(ActionKind::SramAccess);   // axon-table lookup for this firing neuron's fan-out
    for (const auto& tgt : axon_.targets(n)) {
      Flit f;
      f.type = FlitType::Spike;
      f.dendrite_id = tgt.dendrite_id;
      f.dst = tgt.dst;
      f.src = to_coord(id_);
      f.axon_delay = tgt.delay;
      f.global_off1 = static_cast<std::uint16_t>(tgt.go1);
      f.global_off2 = static_cast<std::uint16_t>(tgt.go2);
      f.sync_payload = tgt.meta;   // spare field on SPIKE flits carries the per-edge combine meta
      f.timestep = t;
      pending_spikes_.push_back(f);
    }
  }
}

void NetworkPe::compute(Cycle now, ThreadStats& ts) {
  while (!eject_in_.empty_cur()) {
    const Flit& f = eject_in_.front_cur();
    if (is_sync(f.type)) {
      sync_.on_received(f);
    } else {
      accumulate(f.dendrite_id, f.global_off1, f.global_off2, f.axon_delay, f.sync_payload, ts);
      ++spikes_received_;
    }
    eject_in_.pop_cur();
  }

  sync_emit_.clear();
  sync_.step(local_done(), sync_emit_, ts);
  for (const auto& f : sync_emit_) pending_sync_.push_back(f);

  while (!pending_sync_.empty()) {
    Flit f = pending_sync_.front();
    f.inject_cycle = now;
    if (!router_->try_inject(f)) break;
    pending_sync_.pop_front();
  }
  while (!pending_spikes_.empty() && sync_.may_inject_spike(sync_.current_timestep())) {
    Flit f = pending_spikes_.front();
    f.id = next_flit_id_++;
    f.inject_cycle = now;
    if (!router_->try_inject(f)) break;
    ts.bump(ActionKind::SpikeInject);
    pending_spikes_.pop_front();
  }
}

}  // namespace neurort
