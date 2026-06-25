#pragma once
#include <array>
#include <cstdint>

#include "neurort/common/ports.hpp"
#include "neurort/common/types.hpp"
#include "neurort/functional/credit.hpp"
#include "neurort/functional/flit.hpp"
#include "neurort/functional/pe_receiver.hpp"
#include "neurort/functional/tickable.hpp"

namespace neurort {

// One mesh router + its attached PE port. Input-buffered, XY-routed, credit backpressured.
//
// M1 folds the 1-cycle link into the router-to-router handoff: routing a flit out direction o
// writes it directly into the downstream neighbour's input STAGING (single writer = this
// router) and stages a credit return to the upstream of the consumed port. The 1-cycle hop
// latency emerges from the buffer commit/swap. Multi-cycle links (Link as a separate Tickable)
// are deferred to when link_latency > 1 is needed.
//
// Determinism: route_xy is pure; per-output arbitration scans inputs in fixed Dir order from a
// deterministic round-robin cursor. No thread id / time / pointer ever influences a decision.
class Router final : public Tickable {
 public:
  Router(PeId id, Coord pos, int credit_init);

  // ---- wiring (set at build time, before any tick) ----
  void set_neighbor(Dir d, Router* r) { neighbor_[static_cast<std::size_t>(d)] = r; }
  void set_link_credit(Dir d, int c) { out_credit_[static_cast<std::size_t>(d)] = c; }
  void set_local_pe(PeReceiver* pe) { local_pe_ = pe; }

  // ---- single-writer entry points called by neighbours / the PE in PHASE A ----
  void receive_flit(Dir in_port, const Flit& f) {  // downstream neighbour delivers into our input
    in_[static_cast<std::size_t>(in_port)].push_next(f);
  }
  void stage_credit_return(Dir out_port) {  // downstream returns a credit for our output port
    cred_inbox_[static_cast<std::size_t>(out_port)].stage_add(1);
  }
  // PE / test injects into the Local input (credit-gated). Returns false on backpressure.
  bool try_inject(const Flit& f);

  // ---- Tickable ----
  void compute(Cycle now, ThreadStats& ts) override;
  void commit(Cycle now) override;
  std::uint32_t tile_index() const override { return id_; }

  // ---- accessors (tests / wiring / stats) ----
  PeId id() const { return id_; }
  Coord pos() const { return pos_; }
  Router* neighbor(Dir d) const { return neighbor_[static_cast<std::size_t>(d)]; }
  int neighbor_count() const;
  int out_credit(Dir d) const { return out_credit_[static_cast<std::size_t>(d)]; }
  int inject_credit() const { return inject_credit_; }
  std::size_t input_occupancy() const;  // sum of visible (cur) flits over all input ports

 private:
  void return_credit(Dir in_port, ThreadStats& ts);

  PeId id_;
  Coord pos_;

  std::array<DoubleBufferedFifo<Flit, kInBufCap>, kPortCount> in_{};  // one input FIFO per Dir
  std::array<Router*, kPortCount> neighbor_{};                        // null at edges / Local
  std::array<int, kPortCount> out_credit_{};      // free slots in each downstream neighbour input
  std::array<StagedCounter, kPortCount> cred_inbox_{};  // credit returns from downstream (per out port)
  std::array<std::uint8_t, kPortCount> rr_cursor_{};    // deterministic round-robin per output

  PeReceiver* local_pe_ = nullptr;  // receives flits ejected at this PE (Local output)
  int inject_credit_;                 // PE -> Local input credits
  StagedCounter pe_credit_return_{};  // Local slots freed this cycle, returned to the PE
};

}  // namespace neurort
