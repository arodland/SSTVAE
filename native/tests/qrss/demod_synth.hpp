// A deterministic synthetic track for the demodulator ctests
// (test_demod.cpp, test_ldemod.cpp), built from exact binary fractions so
// Python makes the same doubles. The Python twin that produced the pins:
//
//   def synth(spec, spread_rho=0.0, hdr_bits=None, with_meas=False):
//       n = spec.n_pos + 40; i = np.arange(n)
//       u = (0.75 + (i % 3) * 0.25) + 1j * (((i % 4) - 1.5) * 0.125)
//       m = (((i * 7) % 13 - 6) * 0.125) + 1j * (((i * 5) % 11 - 5) * 0.0625)
//       P = 0.001 * (1 + i % 5); N = 0.25 + 0.0625 * (i % 7)
//       psi = np.where(i % 97 == 0, 0.25, 1.0)
//       c = 0.5 + 1j * (((i % 2) * 2 - 1) * 0.375)
//       pos = np.arange(-40, spec.n_pos)
//       SimpleNamespace(spec=spec, pos=pos, m=m, u=u, P=P, N=N, psi=psi, c=c,
//           classes=SimpleNamespace(spread_rho=spread_rho, hdr_bits=hdr_bits),
//           meas=(i % 3 == 0) if with_meas else None, gi=lambda p: p + 40)
//
// and the known header bits ((np.arange(2474) * 7 + 3) % 5 < 2).astype(np.uint8).

#pragma once

#include <complex>
#include <cstdint>
#include <optional>
#include <vector>

#include "qrss/demod.hpp"
#include "qrss/frame.hpp"

namespace demod_synth {

using cd = std::complex<double>;

struct Synth {
    const sstvae::qrss::frame::FrameSpec* spec;
    std::vector<std::int64_t> pos;
    std::vector<cd> m, u, c;
    std::vector<double> P, N, psi;
    std::vector<std::uint8_t> meas;
    std::vector<std::uint8_t> bits;
    double spread_rho = 0.0;
    bool known = false;

    sstvae::qrss::demod::TrackView view() const {
        sstvae::qrss::demod::TrackView v;
        v.spec = spec;
        v.pos = pos;
        v.m = m;
        v.u = u;
        v.c = c;
        v.P = P;
        v.N = N;
        v.psi = psi;
        v.meas = meas;
        v.spread_rho = spread_rho;
        if (known) v.hdr_bits = std::span<const std::uint8_t>(bits);
        return v;
    }
};

inline Synth make(const sstvae::qrss::frame::FrameSpec& spec, double spread_rho = 0.0, bool known = false,
                  bool with_meas = false) {
    Synth s;
    s.spec = &spec;
    s.spread_rho = spread_rho;
    s.known = known;
    const std::int64_t n = spec.n_pos() + 40;
    for (std::int64_t i = 0; i < n; ++i) {
        s.pos.push_back(i - 40);
        s.u.emplace_back(0.75 + static_cast<double>(i % 3) * 0.25, (static_cast<double>(i % 4) - 1.5) * 0.125);
        s.m.emplace_back(static_cast<double>((i * 7) % 13 - 6) * 0.125, static_cast<double>((i * 5) % 11 - 5) * 0.0625);
        s.P.push_back(0.001 * static_cast<double>(1 + i % 5));
        s.N.push_back(0.25 + 0.0625 * static_cast<double>(i % 7));
        s.psi.push_back(i % 97 == 0 ? 0.25 : 1.0);
        s.c.emplace_back(0.5, static_cast<double>((i % 2) * 2 - 1) * 0.375);
        if (with_meas) s.meas.push_back(i % 3 == 0 ? 1 : 0);
    }
    for (int k = 0; k < 2474; ++k) s.bits.push_back((k * 7 + 3) % 5 < 2 ? 1 : 0);
    return s;
}

}  // namespace demod_synth
