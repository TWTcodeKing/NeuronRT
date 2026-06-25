#pragma once
#include <cstdint>

namespace neurort {

// A credit counter written by a single downstream entity (stage_add, PHASE A) and drained by
// the owner (take, PHASE B). Because writer and reader operate in different BSP phases, no lock
// is needed and the result is thread-count independent.
struct StagedCounter {
  std::uint32_t staged = 0;
  void stage_add(std::uint32_t n = 1) { staged += n; }
  std::uint32_t take() {
    const std::uint32_t s = staged;
    staged = 0;
    return s;
  }
};

}  // namespace neurort
