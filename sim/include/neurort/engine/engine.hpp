#pragma once
#include <cstddef>
#include <memory>
#include <vector>

#include "neurort/common/config.hpp"
#include "neurort/common/types.hpp"
#include "neurort/functional/mesh.hpp"
#include "neurort/functional/pe_stub.hpp"
#include "neurort/functional/tickable.hpp"
#include "neurort/functional/traffic.hpp"
#include "neurort/stats/stats.hpp"

namespace neurort {

// The BSP cycle-stepped engine. Builds the mesh + PEs from config, then runs:
//   for each SNN timestep:
//     advance sync controllers (begin_next_timestep) and reset PE per-timestep flags;
//     repeat NoC cycles { PHASE A compute | barrier | PHASE B commit | barrier | single } until
//     every PE's SyncController is Advancing (the decentralized per-timestep barrier).
//   then drain in-flight spikes so every injected flit is delivered.
//
// Determinism: each cycle, compute() reads CUR buffers and writes own NEXT; commit() merges. Each
// thread writes only its own Stats row. Results (counters, cycles, latency) are bit-identical for
// any thread count.
class BspEngine {
 public:
  BspEngine(const Config& cfg, std::unique_ptr<TrafficSource> traffic);
  BspEngine(const BspEngine&) = delete;
  BspEngine& operator=(const BspEngine&) = delete;

  void run();

  Cycle total_cycles() const { return cycle_; }
  bool deadlock_detected() const { return deadlock_; }
  const Stats& stats() const { return stats_; }
  Mesh& mesh() { return mesh_; }
  int num_threads() const { return num_threads_; }
  std::size_t inflight() const;  // flits resident in routers + undrained PE ejects

 private:
  void run_timestep(Timestep t);
  void drain();
  bool all_advancing() const;

  Config cfg_;
  std::unique_ptr<TrafficSource> traffic_;
  Mesh mesh_;
  std::vector<ProcessingElementStub> pes_;  // index == PeId
  std::vector<Tickable*> schedule_;         // PEs (PeId order) ++ routers (PeId order)
  int num_threads_;
  Stats stats_;
  Cycle cycle_ = 0;
  bool deadlock_ = false;
};

}  // namespace neurort
