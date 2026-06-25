#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "neurort/common/config.hpp"
#include "neurort/energy/energy_model.hpp"
#include "neurort/energy/energy_table.hpp"
#include "neurort/engine/engine.hpp"
#include "neurort/engine/network_runner.hpp"
#include "neurort/functional/traffic.hpp"
#include "neurort/network/network_image.hpp"

using namespace neurort;

namespace {

template <class T>
T read_pod(std::ifstream& f) {
  T v{};
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
  return v;
}

// --network <dir>: run a real compiled network end-to-end on the NoC. Reads dir/manifest.json +
// dir/input.bin (tau, v_th, warmup, measure, per-input-PE input currents), runs the delay-1
// pipeline (warmup to fill the pipeline, then measure steady-state firing), writes dir/firing.bin.
int run_network(const std::string& dir, std::optional<DnpConfig> dnp_override,
                const std::string& energy_table) {
  const NetworkImage img = NetworkImage::load(dir + "/manifest.json");
  std::ifstream in(dir + "/input.bin", std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + dir + "/input.bin");
  const std::uint32_t warmup = read_pod<std::uint32_t>(in);
  const std::uint32_t measure = read_pod<std::uint32_t>(in);
  const std::uint32_t n_in = read_pod<std::uint32_t>(in);

  NetworkRunner runner(img, -1.0, -1.0, dnp_override);  // tau/v_th + DNP from manifest (or override)
  for (std::uint32_t i = 0; i < n_in; ++i) {
    const std::uint32_t pe = read_pod<std::uint32_t>(in);
    const std::uint32_t cnt = read_pod<std::uint32_t>(in);
    std::vector<double> cur(cnt);
    in.read(reinterpret_cast<char*>(cur.data()), static_cast<std::streamsize>(cnt * sizeof(double)));
    runner.set_input_current(static_cast<PeId>(pe), std::move(cur));
  }

  // Optional time-varying input (synthetic temporal workload / DVS): per input PE, `period` frames
  // of current; frame[t % period] is fed each timestep. Overrides the constant input.bin above.
  std::ifstream sf(dir + "/input_seq.bin", std::ios::binary);
  if (sf) {
    const std::uint32_t period = read_pod<std::uint32_t>(sf);
    const std::uint32_t n_seq = read_pod<std::uint32_t>(sf);
    for (std::uint32_t i = 0; i < n_seq; ++i) {
      const std::uint32_t pe = read_pod<std::uint32_t>(sf);
      const std::uint32_t cnt = read_pod<std::uint32_t>(sf);
      std::vector<std::vector<double>> frames(period, std::vector<double>(cnt));
      for (std::uint32_t f = 0; f < period; ++f) {
        sf.read(reinterpret_cast<char*>(frames[f].data()),
                static_cast<std::streamsize>(cnt * sizeof(double)));
      }
      runner.set_input_sequence(static_cast<PeId>(pe), std::move(frames));
    }
  }

  // Optional attention co-processors (Spikformer): one SSA block each.
  std::ifstream af(dir + "/attention.bin", std::ios::binary);
  if (af) {
    const std::uint32_t nblk = read_pod<std::uint32_t>(af);
    for (std::uint32_t b = 0; b < nblk; ++b) {
      AttnCoproc c;
      c.n_tok = static_cast<int>(read_pod<std::uint32_t>(af));
      c.embed = static_cast<int>(read_pod<std::uint32_t>(af));
      c.heads = static_cast<int>(read_pod<std::uint32_t>(af));
      c.scale = read_pod<double>(af);
      c.coproc_delay = static_cast<int>(read_pod<std::uint32_t>(af));
      auto read_pes = [&](std::vector<int>& v) {
        const std::uint32_t n = read_pod<std::uint32_t>(af);
        for (std::uint32_t i = 0; i < n; ++i) v.push_back(static_cast<int>(read_pod<std::uint32_t>(af)));
      };
      read_pes(c.q_pes); read_pes(c.k_pes); read_pes(c.v_pes); read_pes(c.proj_pes);
      runner.add_attn_coproc(std::move(c));
    }
  }

  runner.run(warmup);          // fill the pipeline (transient)
  runner.reset_counts();
  runner.run(measure);         // steady-state firing window

  std::ofstream out(dir + "/firing.bin", std::ios::binary);
  const std::uint32_t npe = static_cast<std::uint32_t>(runner.num_pe());
  out.write(reinterpret_cast<const char*>(&npe), sizeof(npe));
  for (std::uint32_t p = 0; p < npe; ++p) {
    const auto& fc = runner.fire_counts(static_cast<PeId>(p));
    const std::uint32_t cnt = static_cast<std::uint32_t>(fc.size());
    out.write(reinterpret_cast<const char*>(&p), sizeof(p));
    out.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
    std::vector<std::uint32_t> fc32(fc.begin(), fc.end());
    out.write(reinterpret_cast<const char*>(fc32.data()),
              static_cast<std::streamsize>(cnt * sizeof(std::uint32_t)));
  }
  std::printf("network run: %u PEs, %llu cycles, %u warmup + %u measure steps%s\n", npe,
              static_cast<unsigned long long>(runner.total_cycles()), warmup, measure,
              runner.deadlock() ? "  [DEADLOCK]" : "");

  // Soma-DNP metrics: per-PE virtual-memory occupancy/pruning. Written to dnp.bin (one record per
  // DNP-enabled PE) for the ratio-sweep harness, plus an aggregate line. Storage proxy = peak slots.
  std::uint64_t tot_log = 0, tot_phys = 0, tot_peak = 0, tot_prune = 0, tot_reject = 0;
  bool any_dnp = false;
  std::ofstream dout(dir + "/dnp.bin", std::ios::binary);
  std::uint32_t ndnp = 0;
  const std::streampos ndnp_pos = dout.tellp();
  dout.write(reinterpret_cast<const char*>(&ndnp), sizeof(ndnp));   // patched below
  for (std::uint32_t p = 0; p < npe; ++p) {
    const Dnp* d = runner.pe(static_cast<PeId>(p)).dnp();
    if (d == nullptr) continue;
    any_dnp = true;
    ++ndnp;
    const std::uint32_t rec[5] = {p, d->n_log(), d->n_phys(), d->peak_slots(),
                                  static_cast<std::uint32_t>(d->reject_count())};
    dout.write(reinterpret_cast<const char*>(rec), sizeof(rec));
    const std::uint64_t pr = d->prune_count();
    dout.write(reinterpret_cast<const char*>(&pr), sizeof(pr));
    tot_log += d->n_log();
    tot_phys += d->n_phys();
    tot_peak += d->peak_slots();
    tot_prune += d->prune_count();
    tot_reject += d->reject_count();
  }
  dout.seekp(ndnp_pos);
  dout.write(reinterpret_cast<const char*>(&ndnp), sizeof(ndnp));
  dout.close();
  if (any_dnp) {
    const double reduce = tot_peak ? static_cast<double>(tot_log) / static_cast<double>(tot_peak) : 0.0;
    std::printf("  DNP: logical=%llu phys=%llu peak=%llu (storage %.2fx) prunes=%llu rejects=%llu\n",
                static_cast<unsigned long long>(tot_log), static_cast<unsigned long long>(tot_phys),
                static_cast<unsigned long long>(tot_peak), reduce,
                static_cast<unsigned long long>(tot_prune),
                static_cast<unsigned long long>(tot_reject));
  }

  // Energy + latency over the steady-state MEASURE window (the action/latency counters + the cycle
  // boundary were reset at warmup->measure). Energy = Σ count[k]·pj[k] from the swappable 28nm table
  // (placeholder values); latency from the chip clock. Written to dir/energy.json.
  const ActionCounters ac = runner.stats().merge_counters();
  const std::uint64_t mcyc = static_cast<std::uint64_t>(runner.measure_cycles());
  constexpr double kFreqHz = 333.0e6;                          // 333 MHz chip clock (paper)
  const double tstep = measure > 0 ? static_cast<double>(mcyc) / measure : 0.0;   // cycles per SNN step
  std::printf("  latency: %llu cycles over %u steps = %.0f cyc/step, %.3f us/step @ 333MHz\n",
              static_cast<unsigned long long>(mcyc), measure, tstep, tstep / kFreqHz * 1e6);
  try {
    const EnergyModel em(EnergyTable::load_json(energy_table));
    const double pj = em.total_pj(ac);
    std::printf("  energy : %.3f uJ over %u steps = %.1f nJ/step  [PLACEHOLDER 28nm table]\n",
                pj / 1e6, measure, measure > 0 ? pj / measure / 1e3 : 0.0);
    // top action-energy contributors
    std::array<std::pair<double, ActionKind>, kNumActionKinds> by_e{};
    for (std::size_t i = 0; i < kNumActionKinds; ++i) {
      const ActionKind k = static_cast<ActionKind>(i);
      by_e[i] = {em.action_pj(ac, k), k};
    }
    std::sort(by_e.begin(), by_e.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::printf("           top:");
    for (int i = 0; i < 4 && by_e[i].first > 0; ++i) {
      std::printf(" %s=%.0f%%", to_cstr(by_e[i].second), 100.0 * by_e[i].first / (pj > 0 ? pj : 1));
    }
    std::printf("\n");
    // module-grouped breakdown: NoC (router+link+credit+spike+sync) / Compute (ALU primitives) /
    // Memory (SRAM/DRAM/scratchpad — weights + membrane state + axon/dendrite tables) / DNP.
    auto group_of = [](ActionKind k) -> int {
      switch (k) {
        case ActionKind::RouterRouteCompute: case ActionKind::RouterArbitrate:
        case ActionKind::RouterBufferWrite:  case ActionKind::RouterBufferRead:
        case ActionKind::LinkTraversal:      case ActionKind::CreditReturn:
        case ActionKind::SpikeInject:        case ActionKind::SpikeEject:
        case ActionKind::SyncFlitEmit:       return 0;  // NoC
        case ActionKind::SramAccess:         case ActionKind::DramAccess:
        case ActionKind::ScratchpadAccess:   return 2;  // Memory
        case ActionKind::MapTableRead:       case ActionKind::MapTableWrite:
        case ActionKind::FreeListPop:        case ActionKind::FreeListPush:
        case ActionKind::AgeTick:            case ActionKind::PruneScan:
        case ActionKind::ReclaimOp:          return 3;  // DNP
        default:                             return 1;  // Compute
      }
    };
    static const char* const kModName[4] = {"NoC", "Compute", "Memory", "DNP"};
    double mod_pj[4] = {0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < kNumActionKinds; ++i)
      mod_pj[group_of(static_cast<ActionKind>(i))] += em.action_pj(ac, static_cast<ActionKind>(i));
    std::printf("           modules:");
    for (int m = 0; m < 4; ++m)
      std::printf(" %s=%.1f%%", kModName[m], 100.0 * mod_pj[m] / (pj > 0 ? pj : 1));
    std::printf("\n");
    nlohmann::json j, counts, energy, modules;
    for (int m = 0; m < 4; ++m) modules[kModName[m]] = mod_pj[m];
    for (std::size_t i = 0; i < kNumActionKinds; ++i) {
      const ActionKind k = static_cast<ActionKind>(i);
      counts[to_cstr(k)] = ac.get(k);
      energy[to_cstr(k)] = em.action_pj(ac, k);
    }
    j["pes"] = npe;
    j["measure_steps"] = measure;
    j["measure_cycles"] = mcyc;
    j["freq_hz"] = kFreqHz;
    j["latency_s"] = static_cast<double>(mcyc) / kFreqHz;
    j["latency_per_step_s"] = tstep / kFreqHz;
    j["total_energy_pj"] = pj;
    j["energy_per_step_pj"] = measure > 0 ? pj / measure : 0.0;
    j["action_counts"] = counts;
    j["action_energy_pj"] = energy;
    j["module_energy_pj"] = modules;
    std::ofstream(dir + "/energy.json") << j.dump(2) << '\n';
  } catch (const std::exception& e) {
    std::printf("  energy : (skipped — %s)\n", e.what());
  }
  return runner.deadlock() ? 2 : 0;
}


std::unique_ptr<TrafficSource> make_traffic(const std::string& path, const Config& c) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open traffic file " + path);
  nlohmann::json j;
  f >> j;
  const std::string pattern = j.value("pattern", "uniform_random");
  const int w = c.noc.width, h = c.noc.height;
  const Timestep T = c.sim.num_timesteps;
  if (pattern == "none") return std::make_unique<NoTraffic>();
  if (pattern == "uniform_random") {
    return std::make_unique<UniformRandomTraffic>(c.sim.seed, w, h, j.value("spikes_per_pe", 2), T);
  }
  if (pattern == "hotspot") {
    const auto ctr = j.value("center", std::vector<int>{w / 2, h / 2});
    if (ctr.size() < 2) throw std::runtime_error("hotspot 'center' must be [x, y]");
    if (ctr[0] < 0 || ctr[0] >= w || ctr[1] < 0 || ctr[1] >= h) {
      throw std::runtime_error("hotspot 'center' is outside the mesh");
    }
    return std::make_unique<HotspotTraffic>(
        Coord{static_cast<std::uint8_t>(ctr[0]), static_cast<std::uint8_t>(ctr[1])}, w,
        j.value("spikes_per_pe", 2), T);
  }
  if (pattern == "neighbor_xy") return std::make_unique<NeighborXyTraffic>(w, h, T);
  throw std::runtime_error("unknown traffic pattern: " + pattern);
}

void write_report(const std::string& path, const BspEngine& eng, const EnergyModel& em) {
  const ActionCounters ac = eng.stats().merge_counters();
  const LatencyAccum lat = eng.stats().merge_latency();
  // NOTE: thread count is intentionally excluded — the dump must be byte-identical across thread
  // counts (the determinism acceptance gate diffs these files).
  nlohmann::json j;
  j["cycles"] = eng.total_cycles();
  j["total_energy_pj"] = em.total_pj(ac);
  nlohmann::json counts, energy;
  for (std::size_t i = 0; i < kNumActionKinds; ++i) {
    const ActionKind k = static_cast<ActionKind>(i);
    counts[to_cstr(k)] = ac.get(k);
    energy[to_cstr(k)] = em.action_pj(ac, k);
  }
  j["action_counts"] = counts;
  j["action_energy_pj"] = energy;
  j["latency"] = {{"delivered", lat.delivered},
                  {"total_latency", lat.total_latency},
                  {"mean_latency", lat.mean_latency()},
                  {"max_latency", lat.max_latency}};
  std::ofstream out(path);
  out << j.dump(2) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  std::string config_path = "configs/chip_default.json";
  std::string traffic_path;
  std::string dump_path;
  std::string network_dir;
  std::string energy_table = "configs/energy_table_28nm.json";   // for --network energy report
  int threads = -1;
  bool dnp_off = false, dnp_set = false;       // DNP override knobs for --network runs
  DnpConfig dnp_cfg;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--config" && i + 1 < argc) config_path = argv[++i];
      else if (a == "--traffic" && i + 1 < argc) traffic_path = argv[++i];
      else if (a == "--dump" && i + 1 < argc) dump_path = argv[++i];
      else if (a == "--network" && i + 1 < argc) network_dir = argv[++i];
      else if (a == "--energy-table" && i + 1 < argc) energy_table = argv[++i];
      else if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
      // Soma-DNP overrides (override the manifest's "dnp" block): --dnp-ratio R sets phys_ratio,
      // --dnp-age T / --dnp-pot P enable the prune thresholds, --dnp-off forces dense Soma.
      else if (a == "--dnp-ratio" && i + 1 < argc) {
        dnp_cfg.enabled = true; dnp_cfg.n_phys = 0; dnp_cfg.phys_ratio = std::stod(argv[++i]);
        dnp_set = true;
      } else if (a == "--dnp-age" && i + 1 < argc) {
        dnp_cfg.enabled = true; dnp_cfg.age_thresh = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        dnp_set = true;
      } else if (a == "--dnp-pot" && i + 1 < argc) {
        dnp_cfg.enabled = true; dnp_cfg.pot_thresh = std::stod(argv[++i]); dnp_set = true;
      } else if (a == "--dnp-off") {
        dnp_off = true;
      } else if (a == "--dnp-skip-pruned") {
        dnp_cfg.enabled = true; dnp_cfg.skip_pruned = true; dnp_set = true;
      }
    }

    if (!network_dir.empty()) {
      std::optional<DnpConfig> dnp_override;
      if (dnp_off) dnp_override = DnpConfig{};        // enabled=false => force dense Soma
      else if (dnp_set) dnp_override = dnp_cfg;        // else: fall back to the manifest's config
      return run_network(network_dir, dnp_override, energy_table);
    }

    Config cfg = Config::load(config_path);
    if (threads >= 0) cfg.sim.num_threads = threads;
    if (traffic_path.empty()) traffic_path = cfg.sim.traffic_file;

    BspEngine eng(cfg, make_traffic(traffic_path, cfg));
    eng.run();

    const EnergyModel em(EnergyTable::load_json(cfg.sim.energy_table_file));
    const ActionCounters ac = eng.stats().merge_counters();
    const LatencyAccum lat = eng.stats().merge_latency();

    std::printf("NeuroRT run: %dx%d mesh, %llu timesteps, %d threads%s\n", cfg.noc.width,
                cfg.noc.height, static_cast<unsigned long long>(cfg.sim.num_timesteps),
                eng.num_threads(), eng.deadlock_detected() ? "  [DEADLOCK]" : "");
    std::printf("  cycles            : %llu\n", static_cast<unsigned long long>(eng.total_cycles()));
    std::printf("  spikes injected   : %llu\n",
                static_cast<unsigned long long>(ac.get(ActionKind::SpikeInject)));
    std::printf("  spikes delivered  : %llu\n",
                static_cast<unsigned long long>(ac.get(ActionKind::SpikeEject)));
    std::printf("  inflight (drained): %llu\n", static_cast<unsigned long long>(eng.inflight()));
    std::printf("  mean spike latency: %.2f cycles\n", lat.mean_latency());
    std::printf("  total energy      : %.3f pJ (placeholder 28nm table)\n", em.total_pj(ac));

    if (!dump_path.empty()) {
      write_report(dump_path, eng, em);
      std::printf("  report written to : %s\n", dump_path.c_str());
    }
    return eng.deadlock_detected() ? 2 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
