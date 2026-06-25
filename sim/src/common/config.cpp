#include "neurort/common/config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace neurort {
namespace {
using nlohmann::json;

// Assign j[key] into out only if present; otherwise out keeps its default.
template <class T>
void get_to_if(const json& j, const char* key, T& out) {
  if (j.contains(key)) {
    j.at(key).get_to(out);
  }
}

Config parse(const json& j) {
  Config c;
  if (j.contains("chip")) {
    const auto& x = j.at("chip");
    get_to_if(x, "freq_hz", c.chip.freq_hz);
    get_to_if(x, "tech_nm", c.chip.tech_nm);
    get_to_if(x, "sram_kb_per_pe", c.chip.sram_kb_per_pe);
    get_to_if(x, "num_dendrite_alus", c.chip.num_dendrite_alus);
  }
  if (j.contains("noc")) {
    const auto& x = j.at("noc");
    get_to_if(x, "width", c.noc.width);
    get_to_if(x, "height", c.noc.height);
    get_to_if(x, "link_latency", c.noc.link_latency);
    get_to_if(x, "num_vc", c.noc.num_vc);
    get_to_if(x, "credit_init", c.noc.credit_init);
  }
  if (j.contains("sim")) {
    const auto& x = j.at("sim");
    get_to_if(x, "num_timesteps", c.sim.num_timesteps);
    get_to_if(x, "traffic_file", c.sim.traffic_file);
    get_to_if(x, "energy_table_file", c.sim.energy_table_file);
    get_to_if(x, "seed", c.sim.seed);
    get_to_if(x, "num_threads", c.sim.num_threads);
  }

  // ---- validation ----
  if (c.noc.width < 1 || c.noc.height < 1) {
    throw std::runtime_error("Config: noc.width and noc.height must be >= 1");
  }
  if (c.noc.credit_init < 1 || static_cast<std::size_t>(c.noc.credit_init) > kInBufCap) {
    throw std::runtime_error("Config: noc.credit_init must be in [1, " + std::to_string(kInBufCap) +
                             "] (the physical Axon-in buffer capacity)");
  }
  if (c.noc.link_latency < 1) {
    throw std::runtime_error("Config: noc.link_latency must be >= 1");
  }
  if (c.chip.num_dendrite_alus < 1) {
    throw std::runtime_error("Config: chip.num_dendrite_alus must be >= 1");
  }
  return c;
}

}  // namespace

Config Config::from_json_string(const std::string& json_text) {
  return parse(json::parse(json_text));
}

Config Config::load(const std::string& json_path) {
  std::ifstream f(json_path);
  if (!f) {
    throw std::runtime_error("Config::load: cannot open " + json_path);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return from_json_string(ss.str());
}

}  // namespace neurort
