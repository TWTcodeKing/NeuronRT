#pragma once
#include <array>
#include <string>

#include "neurort/stats/action.hpp"

namespace neurort {

// Per-action energy (pJ), keyed by ActionKind. Loaded from JSON ({ "<ActionName>": <pJ>, ... });
// unknown keys are ignored, missing actions default to 0. Swappable: M1 ships placeholder values
// (SATA_Sim 16-bit / ~45 nm); they will be re-characterized for 28 nm via CACTI without touching
// any simulator code (the energy layer is a pure consumer of action counts).
class EnergyTable {
 public:
  EnergyTable() = default;  // all-zero

  double pj(ActionKind k) const { return pj_[static_cast<std::size_t>(k)]; }
  void set(ActionKind k, double v) { pj_[static_cast<std::size_t>(k)] = v; }

  static EnergyTable load_json(const std::string& path);
  static EnergyTable from_json_string(const std::string& text);

 private:
  std::array<double, kNumActionKinds> pj_{};
};

}  // namespace neurort
