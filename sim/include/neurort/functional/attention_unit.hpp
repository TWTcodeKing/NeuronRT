#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace neurort {

// Fused Spiking-Self-Attention unit (one per attention block) — the data-dependent matmul that
// additive dendrites cannot represent (additive gives |q|+|k|, not q·kᵀ). Each timestep it takes
// the binary spikes q,k,v in [N, embed] (heads of width dh = embed/heads) and computes, per head h:
//   score(h,i,j) = scale * Σ_d q[i, h·dh+d] · k[j, h·dh+d]      (binary coincidence count × scale)
//   out(i, h·dh+d) = Σ_j score(h,i,j) · v[j, h·dh+d]
// returning the analog pre-projection `out` [N·embed] that feeds proj. The scores stay internal
// (analog), so nothing analog crosses the spike NoC — only `out` leaves, as proj's input current.
class AttentionUnit {
 public:
  AttentionUnit(int n_tok, int embed, int heads, double scale)
      : n_(n_tok), e_(embed), h_(heads), dh_(embed / heads), scale_(scale),
        q_(static_cast<std::size_t>(n_tok) * embed, 0),
        k_(static_cast<std::size_t>(n_tok) * embed, 0),
        v_(static_cast<std::size_t>(n_tok) * embed, 0) {}

  void set_q(int i, int e) { q_[idx(i, e)] = 1; }   // mark q[i,e] fired this timestep
  void set_k(int j, int e) { k_[idx(j, e)] = 1; }
  void set_v(int j, int e) { v_[idx(j, e)] = 1; }
  void clear() {
    std::fill(q_.begin(), q_.end(), 0);
    std::fill(k_.begin(), k_.end(), 0);
    std::fill(v_.begin(), v_.end(), 0);
  }

  // Analog SSA output for this timestep, [N·embed] in (token i, dim e) row-major.
  std::vector<double> out() const {
    std::vector<double> o(static_cast<std::size_t>(n_) * e_, 0.0);
    for (int h = 0; h < h_; ++h) {
      const int base = h * dh_;
      for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
          int cnt = 0;
          for (int d = 0; d < dh_; ++d) cnt += q_[idx(i, base + d)] & k_[idx(j, base + d)];
          if (cnt == 0) continue;
          const double s = cnt * scale_;                       // score(h,i,j)
          for (int d = 0; d < dh_; ++d) {
            if (v_[idx(j, base + d)]) o[idx(i, base + d)] += s;  // out += score · v
          }
        }
      }
    }
    return o;
  }

  int n_tok() const { return n_; }
  int embed() const { return e_; }

 private:
  std::size_t idx(int tok, int e) const { return static_cast<std::size_t>(tok) * e_ + e; }
  int n_, e_, h_, dh_;
  double scale_;
  std::vector<std::uint8_t> q_, k_, v_;
};

}  // namespace neurort
