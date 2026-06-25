#pragma once
#include <cstddef>
#include <cstdint>

namespace neurort {

// ---- Strong-ish typedefs (prevent mixing ids / cycles / timesteps) ----
using Cycle    = std::uint64_t;  // NoC clock cycles within the current sim run
using Timestep = std::uint64_t;  // SNN timesteps
using PeId     = std::uint16_t;  // 0..575, row-major: id = y*W + x
using FlitId   = std::uint64_t;  // monotonic injection id (latency tracking + deterministic tie-break)

// ---- Mesh geometry ----
inline constexpr int kMeshW = 24;
inline constexpr int kMeshH = 24;
inline constexpr int kNumPe = kMeshW * kMeshH;  // 576

// PE coordinate. x = column, y = row. XY routing resolves X first, then Y.
struct Coord {
  std::uint8_t x{};
  std::uint8_t y{};
  bool operator==(const Coord&) const = default;
};

inline constexpr PeId to_id(Coord c, int width = kMeshW) {
  return static_cast<PeId>(c.y * width + c.x);
}
inline constexpr Coord to_coord(PeId id, int width = kMeshW) {
  return Coord{static_cast<std::uint8_t>(id % width),
               static_cast<std::uint8_t>(id / width)};
}

// Router/PE ports. Local = the attached PE; the four cardinals are mesh neighbours.
enum class Dir : std::uint8_t { Local = 0, East, West, North, South, kNumPorts };
inline constexpr std::size_t kPortCount = static_cast<std::size_t>(Dir::kNumPorts);  // 5

// Physical router input-buffer capacity (paper: Axon-in holds <= 32 flits). The effective
// depth is governed at run time by credits (credit_init <= kInBufCap); the array is fixed.
inline constexpr std::size_t kInBufCap = 32;

// Dendrite ALUs per PE (paper Sec. IV / Fig.4): synaptic accumulations processed in parallel.
// A spike's K = count*repeat synapses drain in ceil(K / num_alus) cycles. Configurable via
// ChipConfig::num_dendrite_alus.
inline constexpr int kDendriteAlus = 64;

}  // namespace neurort
