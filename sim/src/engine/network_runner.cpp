#include "neurort/engine/network_runner.hpp"

#include <algorithm>

#include "neurort/functional/router.hpp"
#include "neurort/stats/thread_stats.hpp"

namespace neurort {
namespace {
constexpr Cycle kWatchdog = 1'000'000;
}  // namespace

NetworkRunner::NetworkRunner(const NetworkImage& net, double tau, double v_threshold,
                             std::optional<DnpConfig> dnp_override)
    : net_(&net),
      mesh_([&] {
        NoCConfig c;
        c.width = net.chip().mesh_w;
        c.height = net.chip().mesh_h;
        return c;
      }()),
      stats_(2u * mesh_.size(), 1) {
  const int n = static_cast<int>(net.pes().size());
  const int mesh_w = mesh_.width();
  const double use_tau = tau > 0.0 ? tau : net.tau();
  const double use_vth = v_threshold > 0.0 ? v_threshold : net.v_threshold();
  const DnpConfig dnp_cfg = dnp_override.value_or(net.dnp());   // CLI override else manifest
  // Ring depth = max per-edge axon delay across the network + 1 (delay-balanced residual skips).
  int max_delay = 1;
  for (const auto& pe : net.pes()) {
    for (const auto& g : pe.axon_groups) max_delay = std::max<int>(max_delay, g.delay);
  }
  const int ring_size = max_delay + 1;
  pes_.reserve(net.pes().size());  // reserve => stable addresses for router<->PE cross-pointers
  for (int id = 0; id < n; ++id) {
    pes_.emplace_back(net, static_cast<PeId>(id), n, mesh_w,
                      static_cast<std::uint32_t>(mesh_.size()) + static_cast<std::uint32_t>(id),
                      use_tau, use_vth, ring_size, dnp_cfg);
  }
  for (int id = 0; id < n; ++id) {
    mesh_.router(static_cast<PeId>(id)).set_local_pe(&pes_[static_cast<std::size_t>(id)]);
    pes_[static_cast<std::size_t>(id)].attach_router(&mesh_.router(static_cast<PeId>(id)));
    if (net.pes()[static_cast<std::size_t>(id)].kind.rfind("matmul", 0) == 0) {
      pes_[static_cast<std::size_t>(id)].set_inert();   // qk/av: attention runs in the coprocessor
    }
  }
}

void NetworkRunner::set_input_current(PeId pe, std::vector<double> current) {
  pes_[pe].set_input_current(std::move(current));
}

void NetworkRunner::set_input_sequence(PeId pe, std::vector<std::vector<double>> frames) {
  if (frames.empty()) return;
  input_period_ = static_cast<Timestep>(frames.size());   // all sequences share one period
  input_seq_.emplace_back(pe, std::move(frames));
}

void NetworkRunner::add_attn_coproc(AttnCoproc c) {
  c.unit = AttentionUnit(c.n_tok, c.embed, c.heads, c.scale);
  for (int p : c.proj_pes) {                       // proj is driven by the co-processor (input layer)
    pes_[static_cast<std::size_t>(p)].set_input_current(
        std::vector<double>(pes_[static_cast<std::size_t>(p)].neuron_count(), 0.0));
  }
  coprocs_.push_back(std::move(c));
}

// Map a token-wise-dense PE's fired LOCAL neuron -> (token, dim): output-split, local = t*width +
// (dim - o0), neuron_base = o0 * N.
namespace {
void set_qkv(AttentionUnit& u, char which, const NetworkPe& pe, int n_tok) {
  const int width = static_cast<int>(pe.neuron_count()) / n_tok;
  const int o0 = (width > 0) ? static_cast<int>(pe.neuron_base()) / n_tok : 0;
  for (std::uint32_t loc : pe.fired()) {
    const int t = static_cast<int>(loc) / width;
    const int dim = o0 + static_cast<int>(loc) % width;
    if (which == 'q') u.set_q(t, dim);
    else if (which == 'k') u.set_k(t, dim);
    else u.set_v(t, dim);
  }
}
}  // namespace

void NetworkRunner::coproc_read(AttnCoproc& c) {
  c.unit.clear();
  for (int p : c.q_pes) set_qkv(c.unit, 'q', pes_[static_cast<std::size_t>(p)], c.n_tok);
  for (int p : c.k_pes) set_qkv(c.unit, 'k', pes_[static_cast<std::size_t>(p)], c.n_tok);
  for (int p : c.v_pes) set_qkv(c.unit, 'v', pes_[static_cast<std::size_t>(p)], c.n_tok);
  c.buf.push_back(c.unit.out());
}

void NetworkRunner::coproc_feed(AttnCoproc& c) {
  if (static_cast<int>(c.buf.size()) < c.coproc_delay) return;  // pipeline still filling -> proj=0
  const std::vector<double> out = std::move(c.buf.front());
  c.buf.pop_front();
  // proj_membrane[t,o] = sum_i out[t,i] * W_proj[o,i]; drive each proj PE's output slice.
  for (int p : c.proj_pes) {
    NetworkPe& pe = pes_[static_cast<std::size_t>(p)];
    const int width = static_cast<int>(pe.neuron_count()) / c.n_tok;
    std::vector<double> ic(pe.neuron_count(), 0.0);
    for (int t = 0; t < c.n_tok; ++t) {
      for (int r = 0; r < width; ++r) {
        double m = 0.0;
        for (int i = 0; i < c.embed; ++i) m += out[t * c.embed + i] * pe.weight_at(r * c.embed + i);
        ic[static_cast<std::size_t>(t) * width + r] = m;
      }
    }
    pe.set_input_current(std::move(ic));
  }
}

std::size_t NetworkRunner::inflight() const {
  std::size_t n = 0;
  for (const auto& r : mesh_.routers()) n += r.input_occupancy();
  for (const auto& pe : pes_) n += pe.eject_occupancy();
  return n;
}

bool NetworkRunner::all_advancing() const {
  for (const auto& pe : pes_) {
    if (!pe.sync().ready_to_advance()) return false;
  }
  return true;
}

void NetworkRunner::step_cycle() {
  for (auto& pe : pes_) {
    ThreadStats ts = stats_.view(0, pe.tile_index());
    pe.compute(cycle_, ts);
  }
  for (auto& r : mesh_.routers()) {
    ThreadStats ts = stats_.view(0, r.tile_index());
    r.compute(cycle_, ts);
  }
  for (auto& pe : pes_) pe.commit(cycle_);
  for (auto& r : mesh_.routers()) r.commit(cycle_);
  ++cycle_;
}

void NetworkRunner::run(Timestep timesteps) {
  // `abs_t_` is the ABSOLUTE timestep and persists across run() calls, so the warmup->measure split
  // (run(warmup); reset_counts(); run(measure)) presents one continuous timeline: begin_timestep
  // re-arms the sync FSM (its `t>0` guard) and indexes the ring delay buffers on abs_t_, instead of
  // a per-call index that restarts at 0 (which left the FSM in Advancing and dropped measure-step-0's
  // spike delivery + mis-slotted warmup-tail delayed currents at the boundary).
  for (Timestep i = 0; i < timesteps && !deadlock_; ++i, ++abs_t_) {
    // Time-varying input (synthetic temporal workload / DVS): feed frame[abs_t % period] each step.
    if (input_period_ > 0) {
      const std::size_t f = static_cast<std::size_t>(abs_t_ % input_period_);
      for (auto& [pe, frames] : input_seq_) pes_[pe].set_input_current(frames[f]);
    }
    for (auto& c : coprocs_) coproc_feed(c);   // set proj input (delayed SSA out) before firing
    for (auto& pe : pes_) {
      ThreadStats ts = stats_.view(0, pe.tile_index());
      pe.begin_timestep(abs_t_, ts);
    }
    for (auto& c : coprocs_) coproc_read(c);    // read this step's q/k/v spikes -> SSA out -> buffer
    const Cycle barrier_start = cycle_;
    while (!all_advancing()) {
      step_cycle();
      if (cycle_ - barrier_start > kWatchdog) { deadlock_ = true; break; }
    }
    // Drain: deliver every spike injected this timestep so it accumulates into the destination
    // Dendrite BEFORE the next timestep snapshots it (preserves the delay-1 pipeline semantics).
    const Cycle drain_start = cycle_;
    while (inflight() > 0 && !deadlock_) {
      step_cycle();
      if (cycle_ - drain_start > kWatchdog) { deadlock_ = true; break; }
    }
  }
}

}  // namespace neurort
