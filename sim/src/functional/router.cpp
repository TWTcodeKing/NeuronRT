#include "neurort/functional/router.hpp"

#include <cassert>

#include "neurort/functional/routing.hpp"
#include "neurort/stats/thread_stats.hpp"

namespace neurort {

Router::Router(PeId id, Coord pos, int credit_init)
    : id_(id), pos_(pos), inject_credit_(credit_init) {}

bool Router::try_inject(const Flit& f) {
  if (inject_credit_ <= 0) return false;            // backpressure: Local input full
  in_[static_cast<std::size_t>(Dir::Local)].push_next(f);
  --inject_credit_;
  return true;
}

int Router::neighbor_count() const {
  int n = 0;
  for (Dir d : {Dir::East, Dir::West, Dir::North, Dir::South}) {
    if (neighbor_[static_cast<std::size_t>(d)] != nullptr) ++n;
  }
  return n;
}

std::size_t Router::input_occupancy() const {
  std::size_t n = 0;
  for (const auto& fifo : in_) n += fifo.size_cur();
  return n;
}

void Router::return_credit(Dir in_port, ThreadStats& ts) {
  ts.bump(ActionKind::CreditReturn);
  if (in_port == Dir::Local) {
    pe_credit_return_.stage_add(1);  // free a Local slot -> credit back to the PE
    return;
  }
  Router* up = neighbor_[static_cast<std::size_t>(in_port)];
  if (up != nullptr) {
    // Our input `in_port` faces neighbour `up`; from up's view we are direction opposite(in_port),
    // so up's output toward us is out_credit_[opposite(in_port)]. Stage a credit there.
    up->stage_credit_return(opposite(in_port));
  }
}

void Router::compute(Cycle now, ThreadStats& ts) {
  // 1. Each non-empty input requests one output (the XY route of its head flit).
  std::array<Dir, kPortCount> req;
  for (std::size_t p = 0; p < kPortCount; ++p) {
    if (in_[p].empty_cur()) {
      req[p] = Dir::kNumPorts;  // sentinel: no request
    } else {
      ts.bump(ActionKind::RouterBufferRead);
      req[p] = route_xy(pos_, in_[p].front_cur().dst);
      ts.bump(ActionKind::RouterRouteCompute);
    }
  }

  // 2. For each output (fixed Dir order), serve at most one requesting input (deterministic RR).
  for (std::size_t oi = 0; oi < kPortCount; ++oi) {
    const Dir o = static_cast<Dir>(oi);
    int chosen = -1;
    for (std::size_t k = 0; k < kPortCount; ++k) {
      const std::size_t p = (rr_cursor_[oi] + k) % kPortCount;
      if (req[p] == o) {
        chosen = static_cast<int>(p);
        break;
      }
    }
    if (chosen < 0) continue;  // no input wants this output
    ts.bump(ActionKind::RouterArbitrate);

    const Dir in_port = static_cast<Dir>(chosen);
    const Flit& head = in_[static_cast<std::size_t>(chosen)].front_cur();

    // A cardinal route must have a wired neighbour; otherwise the flit's dst is off-mesh and it
    // would stall forever (no credit, never forwarded). Fail loudly rather than hang the watchdog.
    assert(o == Dir::Local || neighbor_[oi] != nullptr);

    if (o == Dir::Local) {
      // Reached destination PE: deliver to the local PE (sync flits feed its SyncController;
      // spikes are consumed by the M1 stub). Latency is tracked for spikes only.
      if (local_pe_ != nullptr) local_pe_->receive_eject(head);
      if (head.type == FlitType::Spike) {
        ts.bump(ActionKind::SpikeEject);
        ts.record_latency(head.inject_cycle, now);
      }
      in_[static_cast<std::size_t>(chosen)].pop_cur();
      return_credit(in_port, ts);
      rr_cursor_[oi] = static_cast<std::uint8_t>((chosen + 1) % kPortCount);
    } else if (out_credit_[oi] > 0) {
      // Forward to neighbour: write its input staging, consume credit, free our slot.
      neighbor_[oi]->receive_flit(opposite(o), head);
      ts.bump(ActionKind::RouterBufferWrite);
      ts.bump(ActionKind::LinkTraversal);
      --out_credit_[oi];
      in_[static_cast<std::size_t>(chosen)].pop_cur();
      return_credit(in_port, ts);
      rr_cursor_[oi] = static_cast<std::uint8_t>((chosen + 1) % kPortCount);
    }
    // else: no credit -> backpressure; leave the flit, don't advance the cursor (retry next cycle).
  }
}

void Router::commit(Cycle /*now*/) {
  for (auto& fifo : in_) fifo.commit();  // merge arrivals (staged by upstream) into visible ring
  for (std::size_t d = 0; d < kPortCount; ++d) {
    out_credit_[d] += static_cast<int>(cred_inbox_[d].take());  // reclaim returned credits
  }
  inject_credit_ += static_cast<int>(pe_credit_return_.take());
}

}  // namespace neurort
