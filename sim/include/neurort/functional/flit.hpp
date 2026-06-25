#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "neurort/common/types.hpp"

namespace neurort {

// The four NoC event types (paper Sec. IV-D). Only Spike carries synaptic payload;
// the other three are control/synchronization events that ride the same NoC.
enum class FlitType : std::uint8_t { Spike = 0, Report, NextTimestep, Broadcast };

inline constexpr bool is_sync(FlitType t) { return t != FlitType::Spike; }
const char* to_cstr(FlitType t);

// One "fat" logical flit. M1 carries all fields in a single POD; the true 32-bit
// multi-flit packetization is accounted for separately via wire_flit_count() so the
// timing/energy layers can bill real wire traffic without the functional layer doing
// wormhole segmentation. POD + standard-layout => memcpy-safe across double buffers and
// directly usable by the later compact binary traffic format.
struct alignas(8) Flit {
  FlitType type{FlitType::Spike};
  std::uint8_t dendrite_id{};      // Fig.4: 8b  — target dendrite within dst PE
  Coord dst{};                     // Fig.4 "Core": target PE (x,y)
  Coord src{};                     // origin PE (latency accounting + deterministic tie-break)
  std::uint8_t axon_delay{1};      // Fig.4: 4b  — D-SNN = 1; BIS-SNN configurable
  std::uint16_t global_off1{};     // Fig.4 GO1: 12b
  std::uint16_t global_off2{};     // Fig.4 GO2: 12b
  Timestep timestep{};             // SNN timestep this flit belongs to
  FlitId id{};                     // monotonic injection id (not hardware; tracking/tie-break)
  Cycle inject_cycle{};            // NoC cycle when injected (not hardware; latency measurement)
  std::uint16_t sync_payload{};    // REPORT: reporting PE id; BROADCAST: value bucket

  bool operator==(const Flit&) const = default;
};
static_assert(std::is_trivially_copyable_v<Flit>);
static_assert(std::is_standard_layout_v<Flit>);

// ---- Paper Fig.4 SPIKE payload bit-widths (for wire-flit billing, not functional state) ----
inline constexpr int kSpikeDendriteBits = 8;
inline constexpr int kSpikeCoreBits = 12;   // encodes (x,y)
inline constexpr int kSpikeDelayBits = 4;
inline constexpr int kSpikeGOBits = 12;      // each of GO1, GO2
inline constexpr int kSpikePayloadBits =
    kSpikeDendriteBits + kSpikeCoreBits + kSpikeDelayBits + 2 * kSpikeGOBits;  // = 48
inline constexpr int kWireFlitBits = 32;

// Number of physical 32-bit wire flits this logical flit maps to.
// SPIKE: ceil(48/32) = 2; sync events: 1.
int wire_flit_count(const Flit& f);

// Fixed-size POD (de)serialization — basis for the later compact binary traffic format.
inline constexpr std::size_t kFlitBytes = sizeof(Flit);
void serialize(const Flit& f, std::byte* out);   // writes kFlitBytes bytes
Flit deserialize(const std::byte* in);            // reads kFlitBytes bytes

std::string to_string(const Flit& f);

}  // namespace neurort
