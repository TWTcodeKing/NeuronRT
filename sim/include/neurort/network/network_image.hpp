#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "neurort/common/types.hpp"
#include "neurort/functional/dnp.hpp"

namespace neurort {

// In-memory image of a compiled network (Python compiler output: manifest.json + weights.bin).
// M1/C7 scope: PARSE + VALIDATE only. The real Dendrite/Soma consume these structures in M2.
// nlist/wlist/local_off/go1/go2 are the compiler's SIGNED relative offsets (decoded at sim time
// as `post = nlist[c] + go2`, `weight = wlist[c] + go1`).

struct DendriteImage {
  std::uint8_t id{};
  std::uint8_t count{};
  int repeat{1};           // Algorithm-1 repeat bound: # feature maps (conv) / post-neurons (dense)
  int local_off1{};        // lnOffset (neuron stride walked by the repeat loop)
  int local_off2{};        // lwOffset (weight stride)
  std::vector<int> nlist;  // length == count
  std::vector<int> wlist;  // length == count
};

// Compressed axon-table header (inverse of the dendrite repeat-loop, generalized to a nested
// lattice). `levels` is inner-first; a source neuron fires for every index combination
// (r_0..r_{L-1}), living at `src_base + Σ r_k*levels[k].src_stride` and emitting a spike to PE
// `dst_pe`, dendrite `dendrite_id`, with offsets (go1_base + Σ r_k*go1_stride,
// go2_base + Σ r_k*go2_stride). Parsed from the 64-byte little-endian records in the axon blob;
// unused levels have count == 1 (a no-op factor).
inline constexpr std::size_t kAxonLevels = 3;

struct AxonGroupLevel {
  std::uint32_t count{1};
  std::int32_t src_stride{};
  std::int32_t go1_stride{};
  std::int32_t go2_stride{};
};

struct AxonGroupImage {
  std::uint32_t src_base{};
  std::int32_t go1_base{};
  std::int32_t go2_base{};
  std::array<AxonGroupLevel, kAxonLevels> levels{};
  std::uint16_t dst_pe{};      // destination PE id (coord = {id % mesh_w, id / mesh_w})
  std::uint16_t meta{};        // per-edge combine: bits[0:2]=combine(0 SUM/1 MAX/2 AVG), [2:8]=term, [8:16]=avg_n
  std::uint8_t dendrite_id{};
  std::uint8_t delay{};
};

struct WeightSpan {
  std::uint64_t offset{};
  std::uint64_t bytes{};
  double scale{1.0};  // dequantize: w_float = int8 * scale
};

struct PeNetImage {
  std::uint16_t pe{};
  Coord coord{};
  int layer{};
  std::string kind;
  std::uint32_t neuron_base{};
  std::uint32_t neuron_count{};
  std::vector<DendriteImage> dendrites;
  std::vector<AxonGroupImage> axon_groups;
  WeightSpan weight_span;
};

struct ChipMeta {
  int mesh_w{kMeshW};
  int mesh_h{kMeshH};
  int num_pe{kNumPe};
  int sram_bytes_per_pe{64 * 1024};
  int weight_bits{8};
};

class NetworkImage {
 public:
  // Parse the manifest + its sibling weight blob and validate the contract. Throws
  // std::runtime_error on any violation. No Dendrite/Soma computation is performed.
  static NetworkImage load(const std::string& manifest_path);

  const ChipMeta& chip() const { return chip_; }
  const std::string& model() const { return model_; }
  std::uint64_t timesteps() const { return timesteps_; }
  double tau() const { return tau_; }                  // LIF params for the Soma (M2)
  double v_threshold() const { return v_threshold_; }
  const DnpConfig& dnp() const { return dnp_; }         // optional Soma-DNP config (disabled if absent)
  const std::vector<PeNetImage>& pes() const { return pes_; }
  const std::vector<std::int8_t>& weight_blob() const { return blob_; }

  // Dequantized weight for PE `p` at local weight index `i` (bounds-checked).
  double weight(const PeNetImage& p, std::size_t i) const {
    const std::size_t idx = p.weight_span.offset + i;
    if (i >= p.weight_span.bytes || idx >= blob_.size()) {
      throw std::out_of_range("NetworkImage::weight: index past this PE's weight span");
    }
    return static_cast<double>(blob_[idx]) * p.weight_span.scale;
  }

 private:
  ChipMeta chip_{};
  std::string model_;
  std::uint64_t timesteps_{0};
  double tau_{2.0};
  double v_threshold_{1.0};
  DnpConfig dnp_{};
  std::vector<PeNetImage> pes_;
  std::vector<std::int8_t> blob_;
};

}  // namespace neurort
