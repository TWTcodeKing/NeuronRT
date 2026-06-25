#include "neurort/energy/energy_table.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace neurort {

EnergyTable EnergyTable::from_json_string(const std::string& text) {
  const nlohmann::json j = nlohmann::json::parse(text);
  EnergyTable t;
  for (std::size_t i = 0; i < kNumActionKinds; ++i) {
    const ActionKind k = static_cast<ActionKind>(i);
    const char* name = to_cstr(k);
    if (j.contains(name)) {
      t.set(k, j.at(name).get<double>());
    }
  }
  return t;
}

EnergyTable EnergyTable::load_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("EnergyTable::load_json: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return from_json_string(ss.str());
}

}  // namespace neurort
