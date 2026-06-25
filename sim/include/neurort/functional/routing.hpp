#pragma once
#include "neurort/common/types.hpp"

namespace neurort {

// Dimension-order (XY) routing: resolve X first, then Y; Local when at the destination.
// Convention: East = +x, West = -x, South = +y (row increases), North = -y.
// XY routing is deterministic and deadlock-free on a mesh (no cyclic channel dependency),
// and guarantees in-order delivery for a fixed (src,dst) pair — relied on by the sync protocol.
inline Dir route_xy(Coord here, Coord dst) {
  if (dst.x != here.x) return dst.x > here.x ? Dir::East : Dir::West;
  if (dst.y != here.y) return dst.y > here.y ? Dir::South : Dir::North;
  return Dir::Local;
}

inline Dir opposite(Dir d) {
  switch (d) {
    case Dir::East: return Dir::West;
    case Dir::West: return Dir::East;
    case Dir::North: return Dir::South;
    case Dir::South: return Dir::North;
    default: return Dir::Local;
  }
}

// One mesh step in direction d (no bounds checking; see has_neighbor).
inline Coord step(Coord c, Dir d) {
  switch (d) {
    case Dir::East: return Coord{static_cast<std::uint8_t>(c.x + 1), c.y};
    case Dir::West: return Coord{static_cast<std::uint8_t>(c.x - 1), c.y};
    case Dir::South: return Coord{c.x, static_cast<std::uint8_t>(c.y + 1)};
    case Dir::North: return Coord{c.x, static_cast<std::uint8_t>(c.y - 1)};
    default: return c;
  }
}

// Whether moving from c in direction d stays inside a width x height mesh.
inline bool has_neighbor(Coord c, Dir d, int width = kMeshW, int height = kMeshH) {
  switch (d) {
    case Dir::East: return c.x + 1 < width;
    case Dir::West: return c.x > 0;
    case Dir::South: return c.y + 1 < height;
    case Dir::North: return c.y > 0;
    default: return false;
  }
}

}  // namespace neurort
