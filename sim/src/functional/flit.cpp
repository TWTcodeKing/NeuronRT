#include "neurort/functional/flit.hpp"

#include <cstring>

namespace neurort {

const char* to_cstr(FlitType t) {
  switch (t) {
    case FlitType::Spike: return "SPIKE";
    case FlitType::Report: return "REPORT";
    case FlitType::NextTimestep: return "NEXTTIMESTEP";
    case FlitType::Broadcast: return "BROADCAST";
  }
  return "?";
}

int wire_flit_count(const Flit& f) {
  if (is_sync(f.type)) {
    return 1;  // control/sync events are a single 32-bit flit
  }
  return (kSpikePayloadBits + kWireFlitBits - 1) / kWireFlitBits;  // ceil(48/32) = 2
}

void serialize(const Flit& f, std::byte* out) {
  std::memcpy(out, &f, kFlitBytes);
}

Flit deserialize(const std::byte* in) {
  Flit f;
  std::memcpy(&f, in, kFlitBytes);
  return f;
}

std::string to_string(const Flit& f) {
  std::string s = to_cstr(f.type);
  s += " src(" + std::to_string(f.src.x) + "," + std::to_string(f.src.y) + ")";
  s += "->dst(" + std::to_string(f.dst.x) + "," + std::to_string(f.dst.y) + ")";
  s += " dendr=" + std::to_string(f.dendrite_id);
  s += " delay=" + std::to_string(f.axon_delay);
  s += " t=" + std::to_string(f.timestep);
  s += " id=" + std::to_string(f.id);
  return s;
}

}  // namespace neurort
