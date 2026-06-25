#include "neurort/stats/action.hpp"

namespace neurort {

const char* to_cstr(ActionKind k) {
  switch (k) {
    case ActionKind::RouterRouteCompute: return "RouterRouteCompute";
    case ActionKind::RouterArbitrate: return "RouterArbitrate";
    case ActionKind::RouterBufferWrite: return "RouterBufferWrite";
    case ActionKind::RouterBufferRead: return "RouterBufferRead";
    case ActionKind::LinkTraversal: return "LinkTraversal";
    case ActionKind::CreditReturn: return "CreditReturn";
    case ActionKind::SpikeInject: return "SpikeInject";
    case ActionKind::SpikeEject: return "SpikeEject";
    case ActionKind::SyncFlitEmit: return "SyncFlitEmit";
    case ActionKind::Mul: return "Mul";
    case ActionKind::Acc: return "Acc";
    case ActionKind::Add: return "Add";
    case ActionKind::And: return "And";
    case ActionKind::Comp: return "Comp";
    case ActionKind::Mux: return "Mux";
    case ActionKind::Reg: return "Reg";
    case ActionKind::Sft: return "Sft";
    case ActionKind::SramAccess: return "SramAccess";
    case ActionKind::DramAccess: return "DramAccess";
    case ActionKind::ScratchpadAccess: return "ScratchpadAccess";
    case ActionKind::MapTableRead: return "MapTableRead";
    case ActionKind::MapTableWrite: return "MapTableWrite";
    case ActionKind::FreeListPop: return "FreeListPop";
    case ActionKind::FreeListPush: return "FreeListPush";
    case ActionKind::AgeTick: return "AgeTick";
    case ActionKind::PruneScan: return "PruneScan";
    case ActionKind::ReclaimOp: return "ReclaimOp";
    case ActionKind::kNumKinds: return "kNumKinds";
  }
  return "?";
}

}  // namespace neurort
