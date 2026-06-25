#pragma once
#include <string>

#include "neurort/common/types.hpp"

namespace neurort {

// Network-on-chip parameters. Defaults match the paper (24x24, 32-flit Axon-in buffer).
struct NoCConfig {
  int width = kMeshW;
  int height = kMeshH;
  int link_latency = 1;   // cycles per hop; also the conservative-PDES lookahead
  int num_vc = 1;         // M1 = 1; designed for 2 (class0 sync > class1 spike)
  // Axon-in input-buffer depth in flits == initial per-link/inject credits. Must be in
  // [1, kInBufCap] (the physical array is fixed at kInBufCap); enforced by Config validation.
  int credit_init = static_cast<int>(kInBufCap);
};

// Chip-level parameters. freq_hz/tech_nm/sram_kb_per_pe are recorded metadata for energy/area
// reporting (not yet functional in M1). The PE count is derived from noc.width*height, so it is
// intentionally NOT a config field (avoids a knob that silently disagrees with the mesh).
struct ChipConfig {
  double freq_hz = 333e6;
  int tech_nm = 28;
  int sram_kb_per_pe = 64;
  int num_dendrite_alus = kDendriteAlus;  // parallel synaptic ALUs per PE (paper: 64)
};

// Per-run simulation parameters.
struct SimConfig {
  Timestep num_timesteps = 16;
  std::string traffic_file;
  std::string energy_table_file;
  unsigned seed = 0;     // deterministic traffic seed
  int num_threads = 0;   // 0 => OpenMP default; results MUST be identical for any value
};

struct Config {
  ChipConfig chip;
  NoCConfig noc;
  SimConfig sim;

  // Parse from a JSON file / string. Missing keys keep the struct defaults above.
  static Config load(const std::string& json_path);
  static Config from_json_string(const std::string& json_text);
};

}  // namespace neurort
