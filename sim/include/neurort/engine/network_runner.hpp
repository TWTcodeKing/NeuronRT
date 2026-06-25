#pragma once
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "neurort/common/config.hpp"
#include "neurort/common/types.hpp"
#include "neurort/functional/attention_unit.hpp"
#include "neurort/functional/mesh.hpp"
#include "neurort/functional/network_pe.hpp"
#include "neurort/network/network_image.hpp"
#include "neurort/stats/stats.hpp"

namespace neurort {

// One Spiking-Self-Attention block run as a side-channel co-processor (the data-dependent matmul
// additive dendrites can't do). Reads the q/k/v PEs' spikes each timestep, computes the fused SSA
// `out`, delays it `coproc_delay` steps (= the attention's pipeline depth, so proj fires at its
// DAG stage), turns it into proj's input current via proj's weights, and drives the proj PEs
// (marked input-layer). The qk/av PEs still exist on the NoC but proj ignores their arrivals.
struct AttnCoproc {
  std::vector<int> q_pes, k_pes, v_pes, proj_pes;  // PE ids
  int n_tok = 0, embed = 0, heads = 0, coproc_delay = 1;
  double scale = 1.0;
  AttentionUnit unit{0, 0, 1, 1.0};
  std::deque<std::vector<double>> buf;             // `out` delayed by coproc_delay
};

// Runs a compiled NetworkImage end-to-end on the real NoC, single-threaded, using the Fig.5
// delay-1 pipeline: each SNN timestep every PE fires from its previous-step current, Axon-out
// packs flits, the mesh routes them (XY + credits + cycle timing), and Axon-in/Dendrite accumulate
// at the destination for the next step. The decentralized sync tree (over the used PEs) closes
// each timestep; an extra drain guarantees every spike is delivered before the next step.
class NetworkRunner {
 public:
  // tau / v_threshold default to the manifest's neuron params (negative => use the image's).
  // dnp_override (if set) overrides the manifest's DNP config (CLI --dnp-ratio / --dnp-off).
  NetworkRunner(const NetworkImage& net, double tau = -1.0, double v_threshold = -1.0,
                std::optional<DnpConfig> dnp_override = std::nullopt);

  // Seed an input-layer PE with its (constant) input current = the analog first-layer pre-activation
  // (e.g. conv(image)); call once before run() for every input PE.
  void set_input_current(PeId pe, std::vector<double> current);

  // Time-varying input: `frames[t % period]` is fed to `pe` each timestep (synthetic temporal
  // workload / DVS event frames). All sequences must share the same period (= frames.size()).
  void set_input_sequence(PeId pe, std::vector<std::vector<double>> frames);

  // Register a SSA block; marks its proj PEs as co-processor-driven input layers.
  void add_attn_coproc(AttnCoproc c);

  void run(Timestep timesteps);
  void reset_counts() {                       // at the warmup->measure boundary: zero firing + the
    for (auto& pe : pes_) pe.reset_counts();   // action/latency counters so energy/latency reflect the
    stats_.reset();                            // steady-state measure window (cycles keep counting)
    measure_start_cycle_ = cycle_;
  }
  Cycle measure_cycles() const { return cycle_ - measure_start_cycle_; }  // cycles in the measure window

  // Per-(global within node) firing count over the whole run, for the PE owning `pe`.
  const std::vector<std::uint64_t>& fire_counts(PeId pe) const { return pes_[pe].fire_counts(); }
  const NetworkPe& pe(PeId p) const { return pes_[p]; }
  std::size_t num_pe() const { return pes_.size(); }
  Cycle total_cycles() const { return cycle_; }
  bool deadlock() const { return deadlock_; }
  const Stats& stats() const { return stats_; }
  std::size_t inflight() const;

 private:
  bool all_advancing() const;
  void step_cycle();  // one NoC cycle: compute all, then commit all (single-threaded)
  void coproc_feed(AttnCoproc& c);   // drive proj input from the delayed `out` (before begin_timestep)
  void coproc_read(AttnCoproc& c);   // read q/k/v spikes -> SSA out -> buffer (after begin_timestep)

  const NetworkImage* net_;
  Mesh mesh_;
  std::vector<NetworkPe> pes_;   // index == PeId (the used PEs, 0..n-1)
  std::vector<AttnCoproc> coprocs_;
  std::vector<std::pair<PeId, std::vector<std::vector<double>>>> input_seq_;  // time-varying input
  Timestep input_period_ = 0;                                                // 0 => constant input
  Stats stats_;
  Cycle cycle_ = 0;
  Cycle measure_start_cycle_ = 0;   // cycle_ at the warmup->measure boundary (reset_counts)
  Timestep abs_t_ = 0;   // absolute SNN timestep, CONTINUES across run() calls (warmup then measure)
  bool deadlock_ = false;
};

}  // namespace neurort
