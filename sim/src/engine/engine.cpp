#include "neurort/engine/engine.hpp"

#include <omp.h>

#include "neurort/functional/router.hpp"

namespace neurort {
namespace {
// Each BSP NoC cycle is two short parallel-for sweeps over ~2*num_pe tiny tasks separated by
// barriers. The work per task is a handful of flit ops, so beyond a modest thread count the
// per-cycle barrier/sync cost dominates and more threads only slow the run down (and on a
// many-core host, num_threads=0 -> omp_get_max_threads() can mean hundreds of threads, which
// crawls). Cap the AUTO default; an explicit sim.num_threads > 0 is always honoured verbatim.
// Determinism is independent of thread count, so this only affects speed, never results.
constexpr int kAutoThreadCap = 8;
int resolve_threads(const SimConfig& s) {
  if (s.num_threads > 0) return s.num_threads;
  const int avail = omp_get_max_threads();
  return avail < kAutoThreadCap ? avail : kAutoThreadCap;
}
constexpr Cycle kWatchdogCyclesPerPhase = 1'000'000;  // safety net against a non-terminating run
}  // namespace

BspEngine::BspEngine(const Config& cfg, std::unique_ptr<TrafficSource> traffic)
    : cfg_(cfg),
      traffic_(std::move(traffic)),
      mesh_(cfg.noc),
      num_threads_(resolve_threads(cfg.sim)),
      stats_(2u * mesh_.size(), num_threads_) {
  const int num_pe = static_cast<int>(mesh_.size());
  const int width = mesh_.width();

  pes_.reserve(mesh_.size());  // reserve => stable addresses for cross-pointers
  for (int id = 0; id < num_pe; ++id) {
    pes_.emplace_back(static_cast<PeId>(id), num_pe, width,
                      static_cast<std::uint32_t>(num_pe + id));  // PE tiles: [num_pe, 2*num_pe)
  }

  // Wire PE <-> router and traffic.
  for (int id = 0; id < num_pe; ++id) {
    Router& r = mesh_.router(static_cast<PeId>(id));
    ProcessingElementStub& pe = pes_[static_cast<std::size_t>(id)];
    pe.attach_router(&r);
    pe.attach_traffic(traffic_.get());
    r.set_local_pe(&pe);
  }

  // Fixed-order schedule: PEs (PeId order) then routers (PeId order).
  schedule_.reserve(2u * mesh_.size());
  for (auto& pe : pes_) schedule_.push_back(&pe);
  for (auto& r : mesh_.routers()) schedule_.push_back(&r);
}

bool BspEngine::all_advancing() const {
  for (const auto& pe : pes_) {
    if (!pe.sync().ready_to_advance()) return false;
  }
  return true;
}

std::size_t BspEngine::inflight() const {
  std::size_t n = 0;
  for (const auto& r : mesh_.routers()) n += r.input_occupancy();
  for (const auto& pe : pes_) n += pe.eject_occupancy();
  return n;
}

void BspEngine::run_timestep(Timestep /*t*/) {
  bool done = false;
  const Cycle start = cycle_;
  const int n = static_cast<int>(schedule_.size());

#pragma omp parallel num_threads(num_threads_)
  {
    const int tid = omp_get_thread_num();
    for (;;) {
      // PHASE A: compute (read CUR, write own NEXT). Implicit barrier ends the omp-for.
#pragma omp for schedule(static)
      for (int i = 0; i < n; ++i) {
        ThreadStats ts = stats_.view(tid, schedule_[static_cast<std::size_t>(i)]->tile_index());
        schedule_[static_cast<std::size_t>(i)]->compute(cycle_, ts);
      }
      // PHASE B: commit (merge staging, swap). Implicit barrier ends the omp-for.
#pragma omp for schedule(static)
      for (int i = 0; i < n; ++i) {
        schedule_[static_cast<std::size_t>(i)]->commit(cycle_);
      }
#pragma omp single
      {
        ++cycle_;
        done = all_advancing();
        if (!done && (cycle_ - start) > kWatchdogCyclesPerPhase) deadlock_ = true;
      }
      // Implicit barrier after omp-single makes `done`/`deadlock_` visible to all threads.
      if (done || deadlock_) break;
    }
  }
}

void BspEngine::drain() {
  // Deliver any spikes still in flight after the last timestep (single-threaded; deterministic).
  const Cycle cap = cycle_ + kWatchdogCyclesPerPhase;
  const int n = static_cast<int>(schedule_.size());
  while (inflight() > 0 && cycle_ < cap) {
    for (int i = 0; i < n; ++i) {
      ThreadStats ts = stats_.view(0, schedule_[static_cast<std::size_t>(i)]->tile_index());
      schedule_[static_cast<std::size_t>(i)]->compute(cycle_, ts);
    }
    for (int i = 0; i < n; ++i) schedule_[static_cast<std::size_t>(i)]->commit(cycle_);
    ++cycle_;
  }
  if (inflight() > 0) deadlock_ = true;  // failed to drain
}

void BspEngine::run() {
  for (Timestep t = 0; t < cfg_.sim.num_timesteps && !deadlock_; ++t) {
    if (t > 0) {
      for (auto& pe : pes_) pe.sync().begin_next_timestep();
    }
    for (auto& pe : pes_) pe.begin_timestep(t);
    run_timestep(t);
  }
  if (!deadlock_) drain();
}

}  // namespace neurort
