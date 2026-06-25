#include "neurort/functional/mesh.hpp"

#include "neurort/functional/routing.hpp"

namespace neurort {

Mesh::Mesh(const NoCConfig& cfg) : width_(cfg.width), height_(cfg.height) { build(cfg); }

void Mesh::build(const NoCConfig& cfg) {
  const int n = width_ * height_;
  routers_.reserve(static_cast<std::size_t>(n));  // reserve => no realloc => stable addresses
  for (int id = 0; id < n; ++id) {
    const Coord c = to_coord(static_cast<PeId>(id), width_);
    routers_.emplace_back(static_cast<PeId>(id), c, cfg.credit_init);
  }

  // Wire cardinal neighbours and seed each outgoing link with credit_init credits.
  for (int id = 0; id < n; ++id) {
    Router& r = routers_[static_cast<std::size_t>(id)];
    const Coord c = r.pos();
    for (Dir d : {Dir::East, Dir::West, Dir::North, Dir::South}) {
      if (has_neighbor(c, d, width_, height_)) {
        Router* nb = &routers_[to_id(step(c, d), width_)];
        r.set_neighbor(d, nb);
        r.set_link_credit(d, cfg.credit_init);
      }
    }
  }
}

}  // namespace neurort
