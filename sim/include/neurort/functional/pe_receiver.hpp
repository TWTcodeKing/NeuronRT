#pragma once
#include "neurort/functional/flit.hpp"

namespace neurort {

// Anything a Router can eject a delivered flit into (the M1 traffic stub, or the M2 network PE).
// Decouples the Router from a concrete PE type so the same NoC drives synthetic traffic or a real
// compiled network.
class PeReceiver {
 public:
  virtual ~PeReceiver() = default;
  virtual void receive_eject(const Flit& f) = 0;
};

}  // namespace neurort
