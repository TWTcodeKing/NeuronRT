#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "neurort/common/types.hpp"
#include "neurort/network/network_image.hpp"

namespace neurort {

// M2 Axon-out — when a neuron fires, look up its axon entries and emit one spike flit per target
// (destination core, dendrite id, global offsets). The compiled axon table is the COMPRESSED
// nested-lattice groups (AxonGroupImage); this decodes them ONCE at construction into a per-source
// flit-target list (the hardware decodes on the fly; the sim precomputes for speed).
class AxonOut {
 public:
  struct Target {
    Coord dst;                 // destination PE coord
    std::uint8_t dendrite_id;
    std::uint8_t delay;
    std::uint16_t meta;        // per-edge combine descriptor (term/combine/avg_n)
    int go1;
    int go2;
  };

  AxonOut(const PeNetImage& pe, int mesh_w) {
    // Decompress each nested group: for every index tuple, the source neuron at
    // src_base + Sum r_k*src_stride_k fires to dst_pe/dendrite with offsets go_base + Sum r_k*go_stride_k.
    for (const auto& g : pe.axon_groups) {
      const Coord dst = to_coord(g.dst_pe, mesh_w);
      std::vector<std::uint32_t> idx(kAxonLevels, 0);
      const std::uint64_t total = group_total(g);
      for (std::uint64_t flat = 0; flat < total; ++flat) {
        std::int64_t src = g.src_base;
        int go1 = g.go1_base, go2 = g.go2_base;
        for (std::size_t L = 0; L < kAxonLevels; ++L) {
          src += static_cast<std::int64_t>(idx[L]) * g.levels[L].src_stride;
          go1 += static_cast<int>(idx[L]) * g.levels[L].go1_stride;
          go2 += static_cast<int>(idx[L]) * g.levels[L].go2_stride;
        }
        targets_[static_cast<std::uint32_t>(src)].push_back(
            Target{dst, g.dendrite_id, g.delay, g.meta, go1, go2});
        advance(idx, g);
      }
    }
  }

  // Targets a firing source neuron projects to (empty if it has no outgoing axon — e.g. output layer).
  const std::vector<Target>& targets(std::uint32_t src_neuron) const {
    static const std::vector<Target> kEmpty;
    auto it = targets_.find(src_neuron);
    return it == targets_.end() ? kEmpty : it->second;
  }

  std::size_t num_targets() const {
    std::size_t n = 0;
    for (const auto& kv : targets_) n += kv.second.size();
    return n;
  }

 private:
  static std::uint64_t group_total(const AxonGroupImage& g) {
    std::uint64_t t = 1;
    for (const auto& lv : g.levels) t *= lv.count;
    return t;
  }
  static void advance(std::vector<std::uint32_t>& idx, const AxonGroupImage& g) {
    for (std::size_t L = 0; L < kAxonLevels; ++L) {   // odometer over the nested levels
      if (++idx[L] < g.levels[L].count) return;
      idx[L] = 0;
    }
  }

  std::unordered_map<std::uint32_t, std::vector<Target>> targets_;
};

}  // namespace neurort
