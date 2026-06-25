#pragma once
#include "neurort/common/types.hpp"

// Synchronization tree over the PEs (paper Sec. IV-D), embedded as a binary heap over the
// row-major PeId: root = PE 0 = PE(0,0), children of i are (2i+1, 2i+2), parent is (i-1)/2.
// Pure arithmetic => deterministic, trivially testable, and a one-file seam to later swap for
// an XY-aligned spanning tree if sync latency matters. Logical tree edges are NOT physical mesh
// neighbours; REPORT/NEXTTIMESTEP flits traverse multiple hops via XY routing.
namespace neurort::tree {

inline constexpr PeId kRoot = 0;

inline constexpr bool is_root(PeId i) { return i == kRoot; }
inline constexpr PeId parent(PeId i) { return static_cast<PeId>((i - 1) / 2); }  // precondition: i != root
inline constexpr PeId left_child(PeId i) { return static_cast<PeId>(2 * i + 1); }
inline constexpr PeId right_child(PeId i) { return static_cast<PeId>(2 * i + 2); }

inline constexpr bool has_left(PeId i, int n) { return static_cast<int>(left_child(i)) < n; }
inline constexpr bool has_right(PeId i, int n) { return static_cast<int>(right_child(i)) < n; }
inline constexpr bool is_leaf(PeId i, int n) { return !has_left(i, n); }

inline constexpr int child_count(PeId i, int n) {
  return (has_left(i, n) ? 1 : 0) + (has_right(i, n) ? 1 : 0);
}

}  // namespace neurort::tree
