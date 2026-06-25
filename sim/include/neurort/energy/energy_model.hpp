#pragma once
#include <utility>

#include "neurort/energy/energy_table.hpp"
#include "neurort/stats/action.hpp"

namespace neurort {

// Pure consumer of action counts: energy = sum over actions of count[k] * pj[k]. Invoked only at
// report time, never on the hot path. The SATA_Sim methodology: a composite action's pJ is itself
// a sum of primitive-gate energies, precomputed into the table.
class EnergyModel {
 public:
  explicit EnergyModel(EnergyTable table) : tbl_(std::move(table)) {}

  double action_pj(const ActionCounters& c, ActionKind k) const {
    return static_cast<double>(c.get(k)) * tbl_.pj(k);
  }
  double total_pj(const ActionCounters& c) const {
    double e = 0.0;
    for (std::size_t i = 0; i < kNumActionKinds; ++i) {
      e += action_pj(c, static_cast<ActionKind>(i));
    }
    return e;
  }
  const EnergyTable& table() const { return tbl_; }

 private:
  EnergyTable tbl_;
};

}  // namespace neurort
