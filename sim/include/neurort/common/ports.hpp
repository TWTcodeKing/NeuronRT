#pragma once
#include <array>
#include <cassert>
#include <cstddef>

namespace neurort {

// Fixed-capacity, double-buffered FIFO — THE determinism primitive of the BSP engine.
//
// Two disjoint memory regions:
//   * visible ring (cur): read via front_cur()/pop_cur(); mutated only by the OWNER.
//   * staging (next):      written via push_next(); mutated only by the single UPSTREAM writer.
//
// In one NoC cycle (PHASE A, parallel): the upstream entity push_next()s while the owner
// front_cur()/pop_cur()s. These touch disjoint fields (stage_/stage_n_ vs buf_/head_/count_),
// so concurrent calls on the SAME object from two threads are race-free WITHOUT locks.
// commit() (PHASE B, single-threaded for this object) merges staging into the ring.
//
// IMPORTANT: push_next() never reads count_ (the owner mutates it concurrently). Capacity is
// guaranteed upstream by the credit system, not by inspecting this FIFO's live occupancy.
template <class T, std::size_t Cap>
class DoubleBufferedFifo {
 public:
  // ---- visible (cur) side: the FIFO as seen this cycle (owner reads/pops) ----
  bool empty_cur() const { return count_ == 0; }
  std::size_t size_cur() const { return count_; }
  const T& front_cur() const {
    assert(count_ > 0);
    return buf_[head_];
  }
  void pop_cur() {
    assert(count_ > 0);
    head_ = (head_ + 1) % Cap;
    --count_;
  }

  // ---- staging (next) side: invisible until commit() (single upstream writer) ----
  std::size_t staged() const { return stage_n_; }
  bool can_push_next() const { return count_ + stage_n_ < Cap; }
  void push_next(const T& v) {
    // Race-free: touches ONLY the staging counter. count_ is mutated concurrently by the owner's
    // pop_cur() in the same phase, so push_next must never read it. The combined-occupancy bound
    // (count_ + stage_n_ <= Cap) is guaranteed by the credit system and asserted in commit().
    assert(stage_n_ < Cap);
    stage_[stage_n_++] = v;
  }

  // ---- commit (PHASE B, single owner): merge staging into the visible ring; FIFO order preserved ----
  void commit() {
    assert(count_ + stage_n_ <= Cap);  // safe to read count_ here: commit is single-threaded for this object
    for (std::size_t i = 0; i < stage_n_; ++i) {
      buf_[(head_ + count_) % Cap] = stage_[i];
      ++count_;
    }
    stage_n_ = 0;
  }

  static constexpr std::size_t capacity() { return Cap; }
  // total occupancy after the next commit (visible + staged); for tests/asserts only.
  std::size_t occupancy() const { return count_ + stage_n_; }

 private:
  std::array<T, Cap> buf_{};    // visible ring
  std::array<T, Cap> stage_{};  // staging area
  std::size_t head_ = 0;
  std::size_t count_ = 0;
  std::size_t stage_n_ = 0;
};

}  // namespace neurort
