#include "neurort/network/network_image.hpp"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace neurort {
namespace {
using nlohmann::json;

constexpr std::uint64_t kMaxDendriteId = 255;   // 8 bits
constexpr std::uint64_t kMaxCount = 255;        // 8 bits
constexpr std::uint64_t kMaxDelay = 15;         // 4 bits
// Little-endian axon group record: src_base u32 | go1_base i32 | go2_base i32 |
// kAxonLevels x (count u32 | src_stride i32 | go1_stride i32 | go2_stride i32) | dst_pe u16 |
// meta u16 | dendrite_id u8 | delay u8.
constexpr std::size_t kAxonRecBytes = 12 + 16 * kAxonLevels + 6;

void check(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error("NetworkImage: " + msg);
}

std::uint32_t rd_u32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::int32_t rd_i32(const std::uint8_t* p) { return static_cast<std::int32_t>(rd_u32(p)); }
std::uint16_t rd_u16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

Coord parse_coord(const json& j, int mesh_w, int mesh_h) {
  check(j.is_array() && j.size() == 2, "coord must be [x, y]");
  const int x = j[0].get<int>();  // range-check the RAW value before the uint8 truncation
  const int y = j[1].get<int>();
  check(0 <= x && x < mesh_w && 0 <= y && y < mesh_h, "coord out of mesh");
  return Coord{static_cast<std::uint8_t>(x), static_cast<std::uint8_t>(y)};
}

std::vector<std::int8_t> read_blob(const std::string& path, std::uint64_t expect_bytes) {
  std::ifstream f(path, std::ios::binary);
  check(static_cast<bool>(f), "cannot open weight blob " + path);
  f.seekg(0, std::ios::end);
  const std::streamoff sz = f.tellg();
  f.seekg(0, std::ios::beg);
  check(sz >= 0 && static_cast<std::uint64_t>(sz) == expect_bytes,
        "weight blob size " + std::to_string(sz) + " != weight_blob_bytes " +
            std::to_string(expect_bytes));
  std::vector<std::int8_t> blob(static_cast<std::size_t>(sz));
  if (sz > 0) {
    f.read(reinterpret_cast<char*>(blob.data()), sz);
    check(f.gcount() == sz, "short read of weight blob " + path);
  }
  return blob;
}

std::vector<std::uint8_t> read_u8_blob(const std::string& path, std::uint64_t expect_bytes) {
  std::ifstream f(path, std::ios::binary);
  check(static_cast<bool>(f), "cannot open axon blob " + path);
  f.seekg(0, std::ios::end);
  const std::streamoff sz = f.tellg();
  f.seekg(0, std::ios::beg);
  check(sz >= 0 && static_cast<std::uint64_t>(sz) == expect_bytes,
        "axon blob size " + std::to_string(sz) + " != axon_blob_bytes " +
            std::to_string(expect_bytes));
  std::vector<std::uint8_t> blob(static_cast<std::size_t>(sz));
  if (sz > 0) {
    f.read(reinterpret_cast<char*>(blob.data()), sz);
    check(f.gcount() == sz, "short read of axon blob " + path);
  }
  return blob;
}

}  // namespace

NetworkImage NetworkImage::load(const std::string& manifest_path) {
  std::ifstream mf(manifest_path);
  check(static_cast<bool>(mf), "cannot open manifest " + manifest_path);
  std::stringstream ss;
  ss << mf.rdbuf();
  const json j = json::parse(ss.str());

  NetworkImage img;
  check(j.at("format_version").get<int>() == 1, "unsupported format_version");
  img.model_ = j.at("model").get<std::string>();
  img.timesteps_ = j.at("timesteps").get<std::uint64_t>();
  if (j.contains("neuron")) {
    img.tau_ = j.at("neuron").value("tau", 2.0);
    img.v_threshold_ = j.at("neuron").value("v_threshold", 1.0);
  }
  check(img.tau_ > 0.0, "neuron.tau must be > 0");

  if (j.contains("dnp")) {  // optional Soma-DNP config; absent => disabled (plain dense Soma)
    const json& d = j.at("dnp");
    img.dnp_.enabled = d.value("enabled", false);
    img.dnp_.n_phys = d.value("n_phys", 0u);
    img.dnp_.phys_ratio = d.value("phys_ratio", 0.25);
    img.dnp_.age_thresh = d.value("age_thresh", std::numeric_limits<std::uint32_t>::max());
    img.dnp_.pot_thresh = d.value("pot_thresh", -std::numeric_limits<double>::infinity());
    check(img.dnp_.n_phys > 0 || (img.dnp_.phys_ratio > 0.0 && img.dnp_.phys_ratio <= 1.0),
          "dnp: need n_phys > 0 or 0 < phys_ratio <= 1");
  }

  const json& chip = j.at("chip");
  img.chip_ = ChipMeta{chip.at("mesh_w").get<int>(), chip.at("mesh_h").get<int>(),
                       chip.at("num_pe").get<int>(), chip.at("sram_bytes_per_pe").get<int>(),
                       chip.at("weight_bits").get<int>()};
  const ChipMeta& c = img.chip_;
  check(c.mesh_w > 0 && c.mesh_h > 0 && c.num_pe > 0, "mesh dims / num_pe must be > 0");
  check(c.num_pe <= kNumPe, "num_pe " + std::to_string(c.num_pe) + " > " + std::to_string(kNumPe));

  const std::uint64_t blob_bytes = j.at("weight_blob_bytes").get<std::uint64_t>();
  const std::uint64_t axon_blob_bytes = j.at("axon_blob_bytes").get<std::uint64_t>();
  check(j.at("axon_record_bytes").get<std::size_t>() == kAxonRecBytes, "axon_record_bytes != 32");
  std::uint64_t span_total = 0;
  std::uint64_t axon_span_total = 0;                       // running byte offset into the axon blob
  std::vector<std::pair<std::uint64_t, std::uint64_t>> axon_spans;  // (offset, groups) per PE
  std::set<int> seen_pe;

  for (const json& pj : j.at("pes")) {
    PeNetImage pe;
    pe.pe = static_cast<std::uint16_t>(pj.at("pe").get<int>());
    check(pe.pe < c.num_pe, "pe id out of range");
    check(seen_pe.insert(pe.pe).second, "duplicate pe id");
    pe.coord = parse_coord(pj.at("coord"), c.mesh_w, c.mesh_h);
    check(pe.coord.x == pe.pe % c.mesh_w && pe.coord.y == pe.pe / c.mesh_w, "coord != pe id");
    pe.layer = pj.at("layer").get<int>();
    pe.kind = pj.at("kind").get<std::string>();
    pe.neuron_base = pj.at("neuron_base").get<std::uint32_t>();
    pe.neuron_count = pj.at("neuron_count").get<std::uint32_t>();

    std::set<int> dids;
    for (const json& dj : pj.at("dendrites")) {
      DendriteImage d;
      d.id = static_cast<std::uint8_t>(dj.at("id").get<int>());
      check(dj.at("id").get<std::uint64_t>() <= kMaxDendriteId, "dendrite id > 255");
      check(dids.insert(d.id).second, "duplicate dendrite id");
      d.count = static_cast<std::uint8_t>(dj.at("count").get<int>());
      check(dj.at("count").get<std::uint64_t>() <= kMaxCount, "dendrite count > 255");
      d.repeat = dj.at("repeat").get<int>();
      check(d.repeat >= 1, "dendrite repeat < 1");
      d.local_off1 = dj.at("local_off1").get<int>();
      d.local_off2 = dj.at("local_off2").get<int>();
      d.nlist = dj.at("nlist").get<std::vector<int>>();
      d.wlist = dj.at("wlist").get<std::vector<int>>();
      check(d.nlist.size() == d.count && d.wlist.size() == d.count, "nlist/wlist length != count");
      pe.dendrites.push_back(std::move(d));
    }
    check(pe.dendrites.size() <= kMaxDendriteId + 1, "more than 256 dendrites on a PE");

    const json& asp = pj.at("axon_span");
    const std::uint64_t aoff = asp.at("offset").get<std::uint64_t>();
    const std::uint64_t agroups = asp.at("groups").get<std::uint64_t>();
    check(aoff == axon_span_total, "axon spans not contiguous");
    axon_spans.emplace_back(aoff, agroups);
    axon_span_total += agroups * kAxonRecBytes;

    const json& sp = pj.at("weight_span");
    pe.weight_span = WeightSpan{sp.at("offset").get<std::uint64_t>(),
                                sp.at("bytes").get<std::uint64_t>(), sp.at("scale").get<double>()};
    check(pe.weight_span.offset == span_total, "weight spans not contiguous");
    span_total += pe.weight_span.bytes;
    const std::uint64_t budget = pe.weight_span.bytes + 2ull * pe.neuron_count;
    check(budget <= static_cast<std::uint64_t>(c.sram_bytes_per_pe),
          "PE " + std::to_string(pe.pe) + " budget " + std::to_string(budget) + " > SRAM");

    img.pes_.push_back(std::move(pe));
  }

  check(span_total == blob_bytes, "sum of weight spans != weight_blob_bytes");
  check(axon_span_total == axon_blob_bytes, "sum of axon spans != axon_blob_bytes");

  // PE vector index must equal pe id — downstream code (and axon dst resolution below) index by id.
  for (std::size_t i = 0; i < img.pes_.size(); ++i) {
    check(img.pes_[i].pe == i, "PE ids must be contiguous 0..n-1 in manifest order");
  }

  const std::filesystem::path dir = std::filesystem::path(manifest_path).parent_path();
  img.blob_ = read_blob((dir / j.at("weight_blob").get<std::string>()).string(), blob_bytes);

  // Parse the compressed axon table from its binary sidecar and validate every group: target
  // dendrite exists on the destination PE, and the source-neuron run stays within the PE.
  const std::vector<std::uint8_t> ab =
      read_u8_blob((dir / j.at("axon_blob").get<std::string>()).string(), axon_blob_bytes);
  for (std::size_t i = 0; i < img.pes_.size(); ++i) {
    PeNetImage& pe = img.pes_[i];
    const std::uint64_t off = axon_spans[i].first, ng = axon_spans[i].second;
    pe.axon_groups.reserve(ng);
    for (std::uint64_t g = 0; g < ng; ++g) {
      const std::uint8_t* p = ab.data() + off + g * kAxonRecBytes;
      AxonGroupImage a;
      a.src_base = rd_u32(p);
      a.go1_base = rd_i32(p + 4);
      a.go2_base = rd_i32(p + 8);
      std::int64_t last = static_cast<std::int64_t>(a.src_base);   // max-corner source id
      for (std::size_t L = 0; L < kAxonLevels; ++L) {
        const std::uint8_t* q = p + 12 + 16 * L;
        AxonGroupLevel lv{rd_u32(q), rd_i32(q + 4), rd_i32(q + 8), rd_i32(q + 12)};
        check(lv.count >= 1, "axon group level count < 1");
        last += static_cast<std::int64_t>(lv.count - 1) * lv.src_stride;
        a.levels[L] = lv;
      }
      a.dst_pe = rd_u16(p + 12 + 16 * kAxonLevels);
      a.meta = rd_u16(p + 12 + 16 * kAxonLevels + 2);
      a.dendrite_id = p[12 + 16 * kAxonLevels + 4];
      a.delay = p[12 + 16 * kAxonLevels + 5];
      check(a.delay <= kMaxDelay, "axon delay > 15");
      check(a.dst_pe < img.pes_.size(), "axon dst_pe id out of range");
      check(a.src_base < pe.neuron_count && last >= 0 &&
                last < static_cast<std::int64_t>(pe.neuron_count),
            "axon source lattice outside the PE's neurons");
      bool found = false;
      for (const auto& d : img.pes_[a.dst_pe].dendrites) {
        if (d.id == a.dendrite_id) { found = true; break; }
      }
      check(found, "axon group targets a dendrite id absent on its destination PE");
      pe.axon_groups.push_back(a);
    }
  }
  return img;
}

}  // namespace neurort
