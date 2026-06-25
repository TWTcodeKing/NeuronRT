#pragma once
#include <cstdint>
#include <vector>

#include "neurort/common/types.hpp"
#include "neurort/functional/flit.hpp"

namespace neurort {

// Source of SPIKE flits for the PE stubs. The seam shared by M1 synthetic patterns and (later)
// real compiler-emitted spikes. MUST be a pure function of (src, t): the same query always
// returns the same flits, with no shared mutable RNG state — this is what keeps the multithreaded
// engine deterministic. The engine assigns flit id / inject_cycle at injection.
class TrafficSource {
 public:
  virtual ~TrafficSource() = default;
  virtual void spikes_for(PeId src, Timestep t, std::vector<Flit>& out) = 0;
};

class NoTraffic final : public TrafficSource {
 public:
  void spikes_for(PeId, Timestep, std::vector<Flit>& out) override { out.clear(); }
};

// Counter-based hashing (SplitMix64-style); pure, no state -> deterministic across threads.
inline std::uint64_t mix3(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
  std::uint64_t x = a * 0x9E3779B97F4A7C15ull + b * 0xBF58476D1CE4E5B9ull +
                    c * 0x94D049BB133111EBull + 0x2545F4914F6CDD1Dull;
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return x;
}

// Each PE emits `spikes_per_pe` spikes per timestep (for t < max_t) to uniformly-random
// destinations (excluding itself), derived purely from (seed, src, t, k).
class UniformRandomTraffic final : public TrafficSource {
 public:
  UniformRandomTraffic(unsigned seed, int width, int height, int spikes_per_pe, Timestep max_t)
      : seed_(seed), w_(width), h_(height), rate_(spikes_per_pe), max_t_(max_t) {}

  void spikes_for(PeId src, Timestep t, std::vector<Flit>& out) override {
    out.clear();
    if (t >= max_t_) return;
    const Coord sc = to_coord(src, w_);
    for (int k = 0; k < rate_; ++k) {
      const std::uint64_t r = mix3(seed_, (static_cast<std::uint64_t>(src) << 20) ^ t,
                                   static_cast<std::uint64_t>(k));
      std::uint8_t dx = static_cast<std::uint8_t>(r % static_cast<std::uint64_t>(w_));
      std::uint8_t dy = static_cast<std::uint8_t>((r / static_cast<std::uint64_t>(w_)) %
                                                  static_cast<std::uint64_t>(h_));
      if (dx == sc.x && dy == sc.y) {  // avoid trivial self-destination
        dx = static_cast<std::uint8_t>((dx + 1) % w_);
      }
      Flit f;
      f.type = FlitType::Spike;
      f.src = sc;
      f.dst = Coord{dx, dy};
      f.dendrite_id = static_cast<std::uint8_t>((r >> 16) & 0xFF);
      f.axon_delay = 1;
      f.timestep = t;
      out.push_back(f);
    }
  }

 private:
  unsigned seed_;
  int w_, h_, rate_;
  Timestep max_t_;
};

// Every PE (except the centre) sends to a single hotspot. Stresses NoC congestion / backpressure.
class HotspotTraffic final : public TrafficSource {
 public:
  HotspotTraffic(Coord center, int width, int spikes_per_pe, Timestep max_t)
      : center_(center), w_(width), rate_(spikes_per_pe), max_t_(max_t) {}

  void spikes_for(PeId src, Timestep t, std::vector<Flit>& out) override {
    out.clear();
    if (t >= max_t_) return;
    const Coord sc = to_coord(src, w_);
    if (sc == center_) return;
    for (int k = 0; k < rate_; ++k) {
      Flit f;
      f.type = FlitType::Spike;
      f.src = sc;
      f.dst = center_;
      f.axon_delay = 1;
      f.timestep = t;
      out.push_back(f);
    }
  }

 private:
  Coord center_;
  int w_, rate_;
  Timestep max_t_;
};

// Each PE sends one spike to its east neighbour (east-edge PEs send nothing). Deterministic,
// hand-verifiable single-hop traffic: a golden anchor for delivery count and minimum latency.
class NeighborXyTraffic final : public TrafficSource {
 public:
  NeighborXyTraffic(int width, int height, Timestep max_t)
      : w_(width), h_(height), max_t_(max_t) {}

  void spikes_for(PeId src, Timestep t, std::vector<Flit>& out) override {
    out.clear();
    if (t >= max_t_) return;
    const Coord sc = to_coord(src, w_);
    if (sc.x + 1 >= w_) return;  // east edge: no east neighbour
    Flit f;
    f.type = FlitType::Spike;
    f.src = sc;
    f.dst = Coord{static_cast<std::uint8_t>(sc.x + 1), sc.y};
    f.axon_delay = 1;
    f.timestep = t;
    out.push_back(f);
  }

 private:
  int w_, h_;
  Timestep max_t_;
};

}  // namespace neurort
