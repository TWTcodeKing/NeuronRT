#pragma once
#include <cstddef>
#include <vector>

#include "neurort/common/config.hpp"
#include "neurort/common/types.hpp"
#include "neurort/functional/router.hpp"

namespace neurort {

// Owns all routers and wires the width x height mesh (XY neighbours + initial link credits).
// Non-copyable; movable (vector move keeps router addresses stable, so neighbour pointers and
// the tickable list remain valid).
class Mesh {
 public:
  explicit Mesh(const NoCConfig& cfg);
  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;
  Mesh(Mesh&&) = default;
  Mesh& operator=(Mesh&&) = default;

  Router& router(PeId id) { return routers_[id]; }
  const Router& router(PeId id) const { return routers_[id]; }
  Router& router(Coord c) { return routers_[to_id(c, width_)]; }

  int width() const { return width_; }
  int height() const { return height_; }
  std::size_t size() const { return routers_.size(); }

  std::vector<Router>& routers() { return routers_; }
  const std::vector<Router>& routers() const { return routers_; }

 private:
  void build(const NoCConfig& cfg);

  int width_;
  int height_;
  std::vector<Router> routers_;  // index == PeId
};

}  // namespace neurort
