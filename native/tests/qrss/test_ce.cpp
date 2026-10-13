// native/core/qrss/ce (the transmit side) against values pinned from the
// Python reference, sstvae/qrss/ce.py.
//
// Two kinds of pin, by whether the output goes through libm:
//   * digests (hashlib.sha256 of numpy's bytes) for what is pure arithmetic
//     on exact inputs -- phase_grid/var_grid on the frozen pulse table, the
//     OOK amplitude, at_positions -- which must match on every platform;
//   * hex values with a tolerance for what calls sin/cos (the closed-form
//     pulse, hann_int, the ramp, every phasor), since libm differs by an ulp
//     across glibc/macOS/MSVC.
// Inputs are integer closed forms both sides evaluate exactly:
//   sym_i = ((i * 7919) % 2001 - 1000) / 1000.0
//   var_i = ((i * 104729) % 1009) / 1009.0
//   z_i   = ((i*31337) % 1001 - 500)/500 + 1j*((i*7331) % 999 - 499)/499
//   keying = morse.keying_units("K1ABC/P")
// Every pin was printed by running the reference on these inputs, e.g.
//   hashlib.sha256(ce.phase_grid(8000, -1940, 480000, sym, SHORT).tobytes())
// tests/test_qrss_native_ce.py compares the two implementations directly.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/morse.hpp"
#include "qrss/npsum.hpp"
#include "qrss/tables.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace ce = sstvae::qrss::ce;
namespace fr = sstvae::qrss::frame;
namespace qc = sstvae::qrss::config;
using cplx = std::complex<double>;
using qrss_check::digest_is;

namespace {

std::vector<double> sym_of(const fr::FrameSpec& spec) {
    std::vector<double> s(static_cast<std::size_t>(spec.n_sym()));
    for (std::size_t i = 0; i < s.size(); ++i)
        s[i] = static_cast<double>((static_cast<std::int64_t>(i) * 7919) % 2001 - 1000) / 1000.0;
    return s;
}

std::vector<double> var_of(const fr::FrameSpec& spec) {
    std::vector<double> s(static_cast<std::size_t>(spec.n_sym()));
    for (std::size_t i = 0; i < s.size(); ++i)
        s[i] = static_cast<double>((static_cast<std::int64_t>(i) * 104729) % 1009) / 1009.0;
    return s;
}

std::vector<cplx> z_of(std::size_t n) {
    std::vector<cplx> z(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto k = static_cast<std::int64_t>(i);
        z[i] = cplx(static_cast<double>((k * 31337) % 1001 - 500) / 500,
                    static_cast<double>((k * 7331) % 999 - 499) / 499);
    }
    return z;
}

void near(double got, double want, double tol, const std::string& what) {
    ++check::checks;
    if (!(std::abs(got - want) <= tol)) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "got %a, want %a (|diff| %.3g > %.3g)", got, want, std::abs(got - want), tol);
        check::fail(what, buf);
    }
}

template <typename F>
std::string refusal(F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "(no refusal)";
}

bool bitwise(const std::vector<cplx>& a, const std::vector<cplx>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(cplx)) != 0) return false;
    return true;
}

bool bitwise(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// --- the pulse ---------------------------------------------------------------------

void test_pulse() {
    check::current_step = "pulse";
    // float(ce.G0).hex(): numpy's pairwise sum of the frozen table / 485, libm-free.
    check::equal(ce::g0(), 0x1.006d1ed38b121p+0, "G0 is the reference's, bit for bit");
    // float(ce.PULSE_NORM).hex(): through sin/cos, so to an ulp or two.
    near(ce::pulse_norm(), 0x1.fff4b07fc17c3p-1, 4e-16, "PULSE_NORM");
    check::equal(qrss_check::digest(std::vector<double>(ce::pulse_table().begin(), ce::pulse_table().end())),
                 std::string("7f69e973f1374d2d0b184b4b1798179e4c85068bbafbd50e8c85c234d0b0b96f"),
                 "PULSE_TABLE is the frozen table");
    // The closed form regenerates the frozen table: pulse(m/485), m = -3880..3880.
    double worst = 0.0;
    const auto table = ce::pulse_table();
    for (std::int64_t m = -ce::HALF; m <= ce::HALF; ++m)
        worst = std::max(worst, std::abs(ce::pulse(static_cast<double>(m) / qc::T_NUM) -
                                         table[static_cast<std::size_t>(m + ce::HALF)]));
    check::is_true(worst <= 1e-15, "pulse(m/485) regenerates PULSE_TABLE");
    // ce._rrc([0, 5/3, -5/3, 0.5, 2, 7.9]) -- the zero, both singularities, regular points
    const std::pair<double, double> rrc[] = {
        {0.0, 0x1.0a7e0dd44aae0p+0},         {5.0 / 3, -0x1.0c6a390299303p-3}, {-5.0 / 3, -0x1.0c6a390299303p-3},
        {0.5, 0x1.43344ccd26d83p-1},         {2.0, 0x1.3334d19f4a691p-5},      {7.9, 0x1.d260f594b777dp-8},
    };
    for (const auto& [t, want] : rrc) near(ce::rrc(t), want, 1e-15, "_rrc(" + std::to_string(t) + ")");
    // ce.pulse([0, 8, 8.0000001, -8.5, 1.25]): cut beyond |tau| = 8
    const std::pair<double, double> pul[] = {
        {0.0, 0x1.0a83f10c15a8bp+0}, {8.0, 0x1.85ee1cba358eap-8}, {8.0000001, 0.0}, {-8.5, 0.0}, {1.25, -0x1.97021fb0b1d87p-3},
    };
    for (const auto& [t, want] : pul) near(ce::pulse(t), want, 1e-15, "pulse(" + std::to_string(t) + ")");
    check::equal(ce::pulse(8.0000001), 0.0, "pulse is exactly zero past the cut");
}

// --- phase on the grid (exact) and in closed form ----------------------------------

void test_phase_grid() {
    check::current_step = "phase_grid";
    const auto& S = fr::get("short");
    const auto& T = fr::get("tiny");
    struct Pin {
        const fr::FrameSpec* spec;
        std::int64_t fs, n0, n;
        const char* digest;
    };
    // hashlib.sha256(ce.phase_grid(fs, n0, n, sym_of(spec), spec).tobytes())
    const Pin pins[] = {
        {&S, 8000, -1940, 480000, "8ab4899aac84d8fb46ab5f48d163d711eb9267e70122bb2b438dfa54dc5c382a"},
        {&S, 16000, -3880, 200000, "221de017bfae7075a6bdffde96af5160ed4b1b370cc2a0cda8ee80d309c209c8"},
        {&S, 250, -61, 9000, "f25da36d7fdc250360ac115ff966b7be0bc4b4f720f7e977780e5811287fa7e6"},
        {&T, 4000, 5, 70000, "be6d4dbd06290626c2bc9c547d81252dc03610530f6d7534e695d1b5f827fac3"},
        {&T, 8000, 123456, 5000, "292e5d6df73c00dc8096d3adf9b69b087a50d88c076b00188a76ba2c8694132f"},
    };
    for (const auto& p : pins) {
        const auto sym = sym_of(*p.spec);
        digest_is(ce::phase_grid(p.fs, p.n0, p.n, sym, *p.spec), p.digest,
                  "phase_grid " + p.spec->name() + " fs=" + std::to_string(p.fs) + " n0=" + std::to_string(p.n0));
    }
    digest_is(ce::var_grid(8000, -1940, 480000, var_of(S), S),
              "b643d37482d2608d8cce94473eb7a87d2126b95d46c832068583994227d26d08", "var_grid short 60 s");
    digest_is(ce::at_positions(sym_of(S), S), "6939f1a7c0631fa4ab2e4c8780a02512d85aa1cfee97c78d05ab6459eb88c439",
              "at_positions short");

    // Chunked: any split of [n0, n0 + n) gives the same samples, bit for bit.
    const auto sym = sym_of(S);
    const auto whole = ce::phase_grid(8000, -1940, 60000, sym, S);
    std::vector<double> parts;
    for (std::int64_t i0 = 0; i0 < 60000; i0 += 7777) {
        const auto m = std::min<std::int64_t>(7777, 60000 - i0);
        const auto part = ce::phase_grid(8000, -1940 + i0, m, sym, S);
        parts.insert(parts.end(), part.begin(), part.end());
    }
    check::is_true(bitwise(whole, parts), "phase_grid is chunk-invariant");
    check::equal(ce::phase_grid(8000, 0, 0, sym, S).size(), std::size_t(0), "n = 0 is empty");
    check::is_true(ce::phase_grid(8000, -100000, 1000, sym, S) == std::vector<double>(1000, 0.0),
                   "phase_grid is zero far before the frame");

    // phase_at (closed form) agrees with the table path at the grid times.
    std::vector<double> tau(20000);
    for (std::size_t i = 0; i < tau.size(); ++i)
        tau[i] = static_cast<double>((static_cast<std::int64_t>(i) * 7 - 1940) * 2) / qc::T_NUM;
    const auto var = var_of(S);
    const auto pg = ce::phase_grid(8000, -1940, 7 * 20000, sym, S);
    const auto vgrid = ce::var_grid(8000, -1940, 7 * 20000, var, S);
    std::vector<double> grid(tau.size()), vg(tau.size());
    for (std::size_t i = 0; i < tau.size(); ++i) {
        grid[i] = pg[7 * i];
        vg[i] = vgrid[7 * i];
    }
    const auto at = ce::phase_at(tau, sym, S);
    check::close(at, grid, 1e-12, "phase_at agrees with phase_grid");
    const auto [phi2, v2] = ce::phase_at(tau, sym, var, S);
    check::is_true(bitwise(phi2, at), "phase_at with var returns the same phi");
    check::close(v2, vg, 1e-12, "phase_at's v agrees with var_grid");
}

// --- the callsign windows ----------------------------------------------------------

void test_windows() {
    check::current_step = "windows";
    const auto& S = fr::get("short");
    const auto units = qrss::morse::keying_units("K1ABC/P");
    digest_is(units, "46c4fb4542fce939dffea24afa991d130d650e7bfca58a7e1735023750a83089", "keying_units(K1ABC/P)");
    const ce::Keying k = ce::window_keying(units, S);
    check::equal(k.size(), std::size_t(2 * 192), "window_keying broadcasts one mask to both windows");

    // ce._hann_int([-0.6, -0.5, 0, 0.25, 0.5, 0.7])
    const std::pair<double, double> hann[] = {
        {-0.6, 0.0}, {-0.5, 0.0}, {0.0, 0x1.307e84eee9c70p-4}, {0.25, 0x1.060fd09ddd38ep-2}, {0.5, 0x1.0p-1}, {0.7, 0x1.6666666666666p-1},
    };
    for (const auto& [y, want] : hann) near(ce::hann_int(y), want, 2e-16, "_hann_int(" + std::to_string(y) + ")");
    check::equal(ce::window_units(4999.5, S, 0), 0.0, "window_units at a window's start");
    check::equal(ce::window_units(5100.0, S, 0), 50.25, "window_units inside window 0");
    check::equal(ce::window_units(8053.5 + 384, S, 1), 192.0, "window_units at window 1's end");
    check::equal(ce::window_units(8053.5, S, -1), 0.0, "window_units takes a negative index, like a tuple");
    check::equal(refusal([&] { ce::window_units(0.0, S, 2); }), std::string("tuple index out of range"),
                 "window_units past the last window");

    // The 8 kHz grid across the whole SHORT frame, as the reference builds it.
    const std::int64_t n0 = -1940, n = S.n_pos() * 243 + 4000;
    std::vector<double> tau(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        tau[static_cast<std::size_t>(i)] = static_cast<double>((n0 + i) * 2) / qc::T_NUM;

    const auto ook = ce::cw_phase_turns(tau, S, &k, true);
    digest_is(ook.a, "7cc22a27bf72d812179aa19d9a87ff49632ff07f6f9a75be42c450b83312f517", "OOK amplitude, bit for bit");
    check::is_true(ook.theta == std::vector<double>(tau.size(), 0.0), "OOK keeps theta at 0");

    const auto fsk = ce::cw_phase_turns(tau, S, &k, false);
    check::is_true(fsk.a == std::vector<double>(tau.size(), 1.0), "FSK keeps a at 1");
    // th[i].hex() at these samples (np.searchsorted picks inside each window)
    const std::pair<std::int64_t, double> th[] = {
        {0, 0.0},
        {1214440, 0.0},
        {1216017, 0.0},
        {1218078, -0x1.8a205aa156154p-8},
        {1220000, -0x1.806a4a1363a45p+1},
        {1230000, -0x1.0551d07eae300p+4},
        {1233719, -0x1.41318d0ca7394p+4},
        {1260000, -0x1.918151d07eb00p+5},
        {1262965, -0x1.b26e8767ad8bdp+5},
        {1300000, -0x1.b800000000000p+5},
        {1955035, -0x1.b800000000000p+5},
        {1958673, -0x1.b80c5102d50abp+5},
        {1974314, -0x1.2c4c634329ce5p+6},
        {2003560, -0x1.b53743b3d6c5ep+6},
        {2047986, -0x1.b800000000000p+6},
        {2054433, -0x1.b800000000000p+6},
    };
    for (const auto& [i, want] : th)
        near(fsk.theta[static_cast<std::size_t>(i)], want, 1e-12, "FSK theta[" + std::to_string(i) + "]");
    // Between and after the windows theta is exactly minus the key-down units so far.
    check::equal(fsk.theta[1400000], -55.0, "theta after window 0 is -55 exactly");
    check::equal(fsk.theta.back(), -110.0, "theta after window 1 is -110 exactly");
    // np.sum(th): numpy's pairwise sum, so the order is the reference's too.
    near(qrss::npsum::sum(fsk.theta), -0x1.746dcd4000000p+25, 1e-6, "FSK theta summed");

    // Unordered times take the masked path and give the same values.
    std::vector<double> rev(tau.rbegin(), tau.rend());
    const auto back = ce::cw_phase_turns(rev, S, &k, false);
    std::vector<double> fwd(back.theta.rbegin(), back.theta.rend());
    check::is_true(bitwise(fwd, fsk.theta), "cw_phase_turns does not depend on time order");
    // No keying: FSK is the plain carrier, OOK is off in units 5..183.
    const auto none = ce::cw_phase_turns(tau, S, nullptr, false);
    check::is_true(none.theta == std::vector<double>(tau.size(), 0.0), "no keying: no FSK phase");

    // Refusals, str(e) of the reference's ValueError.
    const std::int64_t bad_len[1] = {191};
    check::equal(refusal([&] { ce::window_keying(std::vector<double>(191, 0.0), bad_len, S); }),
                 std::string("keying must be 0/1 of shape (192,) or (2, 192), not (2, 191)"), "keying too short");
    const std::int64_t one[1] = {192};
    check::equal(refusal([&] { ce::window_keying(std::vector<double>(192, 2.0), one, S); }),
                 std::string("keying must be 0/1 of shape (192,) or (2, 192), not (2, 192)"), "keying not 0/1");
    const std::int64_t three[2] = {3, 192};
    check::equal(refusal([&] { ce::window_keying(std::vector<double>(3 * 192, 0.0), three, S); }),
                 std::string("keying must be 0/1 of shape (192,) or (2, 192), not (3, 192)"), "keying, three windows");
    check::equal(ce::window_keying(units, fr::get("tiny")).size(), std::size_t(0), "no windows: (0, 192)");
}

// --- the keyed span and its ramp ---------------------------------------------------

void test_span_and_ramp() {
    check::current_step = "span";
    const auto& S = fr::get("short");
    const auto& T = fr::get("tiny");
    // float.hex of ce.keyed_span_pos(spec, lead)
    const struct {
        const fr::FrameSpec* spec;
        double lead, lo, hi;
    } spans[] = {
        {&S, 0.0, -0x1.0p+3, 0x1.07ac000000000p+13},
        {&T, 0.0, -0x1.0p+3, 0x1.b800000000000p+10},
        {&S, 2.5, -0x1.69e59bb61a645p+6, 0x1.07ac000000000p+13},
        {&T, 2.5, -0x1.69e59bb61a645p+6, 0x1.b800000000000p+10},
        {&S, 10.0, -0x1.51e59bb61a645p+8, 0x1.07ac000000000p+13},
        {&T, 10.0, -0x1.51e59bb61a645p+8, 0x1.b800000000000p+10},
    };
    for (const auto& s : spans) {
        const auto [lo, hi] = ce::keyed_span_pos(*s.spec, s.lead);
        check::equal(lo, s.lo, "keyed_span_pos start, " + s.spec->name() + " lead " + std::to_string(s.lead));
        check::equal(hi, s.hi, "keyed_span_pos end, " + s.spec->name());
    }
    check::equal(refusal([&] { ce::keyed_span_pos(S, 10.5); }), std::string("lead-in must be 0 to 10 s, not 10.5"),
                 "lead-in too long");
    check::equal(refusal([&] { ce::keyed_span_pos(S, -1.0); }), std::string("lead-in must be 0 to 10 s, not -1.0"),
                 "lead-in negative");
    check::equal(refusal([&] { ce::keyed_span_pos(S, std::nan("")); }), std::string("lead-in must be 0 to 10 s, not nan"),
                 "lead-in NaN");

    check::current_step = "ramp";
    const double ramp = 0.05 / qc::T_SYM;
    const auto [lo, hi] = ce::keyed_span_pos(S, 0.0);
    // ce._ramp_envelope(np.array([t]), lo, hi, ramp)[0].hex() at t.hex()
    const std::pair<double, double> env[] = {
        {-0x1.351a7f5056248p+3, 0.0},
        {-0x1.34c893cb376c3p+3, 0.0},
        {-0x1.27966ed869912p+3, 0x1.2bec33301886ep-3},
        {-0x1.1a6449e59bb62p+3, 0x1.fffffffffffe9p-2},
        {-0x1.00083126e978dp+3, 0x1.ffffe192148c3p-1},
        {-0x1.0p+3, 1.0},
        {0x1.07abffffffddap+13, 1.0},
        {0x1.07ac000000000p+13, 1.0},
        {0x1.07b0660c50ef5p+13, 0x1.7ffffffffe8f8p-1},
        {0x1.07b932246c962p+13, 0x1.fe88000000000p-41},
        {0x1.07b93224f2cdep+13, 0.0},
    };
    for (const auto& [t, want] : env) near(ce::ramp_envelope(t, lo, hi, ramp), want, 1e-15, "_ramp_envelope");
    check::equal(ce::ramp_envelope(-100.0, lo, hi, 0.0), 0.0, "no ramp: zero outside");
    check::equal(ce::ramp_envelope(lo, lo, hi, 0.0), 1.0, "no ramp: one inside");

    // ce._turns_to_phasor: reduced to [0, 1) before the trig
    const struct {
        double t, re, im;
    } ph[] = {
        {0.0, 1.0, 0.0},
        {0x1.0p-2, 0x1.1a62633145c07p-54, 1.0},
        {-0x1.0p-3, 0x1.6a09e667f3bcbp-1, -0x1.6a09e667f3bcep-1},
        {0x1.70ef54646d497p-57, 1.0, 0x1.21c2c25e4127dp-54},
        {0x1.e000000000000p+1, -0x1.a79394c9e8a0ap-53, -1.0},
        {0x1.e84809999999ap+19, -0x1.3c6ef377c665bp-2, 0x1.e6f0e1337e7ddp-1},
        {-0x1.e000000000000p+2, -1.0, 0x1.1a62633145c07p-53},
    };
    for (const auto& p : ph) {
        const cplx z = ce::turns_to_phasor(p.t);
        near(z.real(), p.re, 2e-16, "_turns_to_phasor real");
        near(z.imag(), p.im, 2e-16, "_turns_to_phasor imag");
    }
}

// --- baseband ---------------------------------------------------------------------

struct Sample {
    std::int64_t i;
    double re, im;
};

struct BbPin {
    const char* name;
    std::int64_t n, nonzero;
    double energy;
    std::vector<Sample> at;
};

void check_bb(const BbPin& p, const std::vector<cplx>& z) {
    check::equal(static_cast<std::int64_t>(z.size()), p.n, std::string(p.name) + ": length");
    // count_nonzero(z) and np.sum(np.abs(z) ** 2)
    std::int64_t nz = 0;
    std::vector<double> mag2(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) {
        nz += (z[i] != cplx(0.0, 0.0));
        const double m = std::abs(z[i]);
        mag2[i] = m * m;
    }
    const double e = qrss::npsum::sum(mag2);
    check::equal(nz, p.nonzero, std::string(p.name) + ": nonzero samples");
    near(e, p.energy, 1e-12 * p.energy, std::string(p.name) + ": energy");
    for (const auto& s : p.at) {
        near(z[static_cast<std::size_t>(s.i)].real(), s.re, 1e-12, std::string(p.name) + " re[" + std::to_string(s.i) + "]");
        near(z[static_cast<std::size_t>(s.i)].imag(), s.im, 1e-12, std::string(p.name) + " im[" + std::to_string(s.i) + "]");
    }
}

void test_baseband() {
    check::current_step = "baseband";
    const auto& S = fr::get("short");
    const auto& T = fr::get("tiny");
    const auto units = qrss::morse::keying_units("K1ABC/P");
    const ce::Keying k = ce::window_keying(units, S);
    const auto sym_s = sym_of(S), sym_t = sym_of(T);

    // ce.baseband(sym_of(spec), spec, fs, start_s, n, **kw): len, count_nonzero,
    // sum |z|^2, and samples (the last three: first nonzero, +17, last nonzero).
    ce::BasebandOptions plain;
    const BbPin tiny_plain{"tiny-plain", 440000, 429539, 0x1.a2fc000000000p+18,
                           {{0, 0.0, 0.0},
                            {62857, 0x1.f7efa9396b781p-1, -0x1.6a0bc52d7b02ap-3},
                            {146666, 0x1.e4b6b7f06a16dp-1, 0x1.49d3e76f582b9p-2},
                            {220000, 0x1.979ae1d8acb47p-1, 0x1.35d79709bbd76p-1},
                            {293333, 0x1.14466e5311966p-1, -0x1.af105998f57e9p-1},
                            {439999, 0.0, 0.0},
                            {5661, 0x1.02b9741744000p-16, 0.0},
                            {5678, 0x1.46e79feba8bc0p-8, 0.0},
                            {435199, 0x1.02b9741738000p-16, 0.0}}};
    const auto z_plain = ce::baseband(sym_t, T, 8000, -1.0, 440000, plain);
    check_bb(tiny_plain, z_plain);

    ce::BasebandOptions fsk;
    fsk.keying = &k;
    const BbPin short_fsk{"short-fsk", 120000, 120000, 0x1.d4c0000000000p+16,
                          {{0, 0x1.c2f3091a3b844p-1, 0x1.e4ed7278d04f2p-2},
                           {17142, 0x1.ba96b71f3904bp-2, 0x1.cdb4d6e71c2d4p-1},
                           {40000, 0x1.9be979d0d2073p-1, 0x1.3017c84d049e6p-1},
                           {60000, 0x1.8408db0f793cap-2, -0x1.d9d0cfb8aeea1p-1},
                           {80000, 1.0, 0.0},
                           {119999, 0x1.a5ccc8c5912f4p-1, -0x1.2238d54834771p-1},
                           {17, 0x1.c553f7f7999d2p-1, 0x1.dbf93f4adcbe7p-2}}};
    const auto z_fsk = ce::baseband(sym_s, S, 8000, 150.0, 120000, fsk);
    check_bb(short_fsk, z_fsk);

    ce::BasebandOptions ook = fsk;
    ook.ook = true;
    const BbPin short_ook{"short-ook", 120000, 59860, 0x1.d3a8000000000p+15,
                          {{0, 0x1.c2f3091a3b844p-1, 0x1.e4ed7278d04f2p-2},
                           {17142, 1.0, 0.0},
                           {40000, 1.0, 0.0},
                           {60000, 1.0, 0.0},
                           {80000, 0.0, 0.0},
                           {119999, 0x1.a5ccc8c59134fp-1, -0x1.2238d548346edp-1},
                           {17, 0x1.c553f7f7999d2p-1, 0x1.dbf93f4adcbe7p-2}}};
    const auto z_ook = ce::baseband(sym_s, S, 8000, 150.0, 120000, ook);
    check_bb(short_ook, z_ook);

    ce::BasebandOptions ook_none;
    ook_none.ook = true;
    const BbPin short_ook_none{"short-ook-none", 60000, 16592, 0x1.0340000000000p+14,
                               {{0, 0x1.c2f3091a3b844p-1, 0x1.e4ed7278d04f2p-2},
                                {8571, 0.0, 0.0},
                                {30000, 0.0, 0.0},
                                {59999, 0x1.a58937969448ap-1, -0x1.229af00ad1a27p-1},
                                {17, 0x1.c7c353c430d50p-1, 0x1.d294556313144p-2}}};
    check_bb(short_ook_none, ce::baseband(sym_s, S, 4000, 150.0, 60000, ook_none));

    ce::BasebandOptions lead;
    lead.lead_in_s = 10.0;
    const BbPin tiny_lead{"tiny-lead", 100000, 86339, 0x1.5049800000000p+16,
                          {{0, 0.0, 0.0},
                           {14285, 1.0, 0.0},
                           {50000, 1.0, 0.0},
                           {99999, 0x1.e6373c0bb4ecap-1, -0x1.40dce0b981758p-2},
                           {13661, 0x1.02b9741754000p-16, 0.0},
                           {13678, 0x1.46e79feba9440p-8, 0.0}}};
    check_bb(tiny_lead, ce::baseband(sym_t, T, 8000, -12.0, 100000, lead));

    ce::BasebandOptions scaled;
    scaled.time_scale = 1.0001;
    const BbPin tiny_scaled{"tiny-scaled", 4000, 4000, 0x1.f400000000000p+11,
                            {{0, 0x1.e68eb377a5632p-1, -0x1.3ec8cd8b8eda8p-2},
                             {571, 0x1.b014db16a3e86p-1, 0x1.12ae506fbc14ep-1},
                             {2000, 0x1.f43e8986d36b0p-1, 0x1.b45796f3f5985p-3},
                             {3999, 0x1.df07e65fcad0cp-1, -0x1.698411e5c4534p-2},
                             {17, 0x1.f7e263b8cf121p-1, -0x1.6b32d96e56d78p-3}}};
    check_bb(tiny_scaled, ce::baseband(sym_t, T, 8000, 0.01234, 4000, scaled));

    const BbPin tiny_250{"tiny-250", 2000, 1998, 0x1.f174051749bbep+10,
                         {{0, 0.0, 0.0},
                          {285, 0x1.bfe4ff8859366p-1, 0x1.f01f799efac88p-2},
                          {1000, 0x1.b98c43fa7cb92p-1, 0x1.0330a1f3dc98ep-1},
                          {1999, 0x1.c540ae5c57593p-1, -0x1.dc42b2d0020c4p-2},
                          {2, 0x1.02b458bbde800p-12, 0.0},
                          {19, 0x1.ffffdb7a04287p-1, 0x1.82c7f981bab6dp-10}}};
    check_bb(tiny_250, ce::baseband(sym_t, T, 250, -0.3, 2000, plain));

    const BbPin tiny_offgrid{"tiny-offgrid", 3000, 3000, 0x1.7700000000000p+11,
                             {{0, 0x1.52d2b5f805355p-1, -0x1.7fda4c2ad959bp-1},
                              {428, 0x1.68d151a96cabdp-1, 0x1.6b416e2d1b468p-1},
                              {1500, 0x1.dfa1bb0357e1cp-1, 0x1.66509716ee06bp-2},
                              {2999, 0x1.ff8ceca5a9f17p-1, -0x1.5732fe0eb1daep-5}}};
    check_bb(tiny_offgrid, ce::baseband(sym_t, T, 8000, 3.3e-5, 3000, plain));

    // The plan: the exact table path only on the grid of a rate dividing 16 kHz.
    check::is_true(ce::baseband_plan(T, 8000, -1.0).exact, "on the 8 kHz grid: exact");
    check::is_true(!ce::baseband_plan(T, 8000, 3.3e-5).exact, "off the grid: closed form");
    check::is_true(!ce::baseband_plan(T, 8000, 0.0, scaled).exact, "time_scale != 1: closed form");
    check::is_true(!ce::baseband_plan(T, 3000, 0.0).exact, "3 kHz does not divide 16 kHz: closed form");
    check::equal(ce::baseband_plan(T, 8000, -0.24254).n0, std::int64_t(-1940), "n0 rounds start_s * fs");

    // Far from t0. round(start_s * fs) past int64 is the reference's
    // OverflowError only where the exact path builds int64 sample numbers
    // (np.arange); the closed form and n = 0 never do. Below that, any frame
    // is out of reach: zeros (the reference's Python ints never overflow).
    {
        const auto far = ce::baseband_plan(T, 8000, 1e20);
        check::is_true(far.n0_overflows && far.exact, "1e20 s at 8 kHz: exact, n0 past int64");
        check::is_true(!ce::baseband_plan(T, 8000, -0.24254).n0_overflows, "n0 in range");
        check::equal(refusal([&] { ce::baseband(sym_t, T, 8000, 1e20, 10, plain); }),
                     std::string("Python int too large to convert to C long"), "exact path past int64");
        check::equal(ce::baseband(sym_t, T, 8000, 1e20, 0, plain).size(), std::size_t(0), "n = 0 builds nothing");
        ce::BasebandOptions o;
        o.time_scale = 1.0001;
        check::is_true(ce::baseband(sym_t, T, 8000, 1e20, 100, o) == std::vector<cplx>(100),
                       "the closed form far from t0 is silence, not a refusal");
        check::is_true(ce::baseband(sym_t, T, 8000, 1e15, 100, plain) == std::vector<cplx>(100),
                       "1e15 s on the exact path is silence");
        const auto vals = ce::at_positions(sym_t, T);
        for (const std::int64_t n0 : {std::int64_t(1) << 62, -(std::int64_t(1) << 62), std::int64_t(1000000000000000)})
            check::is_true(ce::grid_sum(16000, n0, 64, vals, ce::pulse_table()) == std::vector<double>(64, 0.0),
                           "grid_sum far from t0 is zeros, n0 = " + std::to_string(n0));
    }

    // Chunk invariance, bit for bit: the chunk only decides where the work splits.
    check::current_step = "baseband chunks";
    for (const std::int64_t chunk : {std::int64_t(1) << 12, std::int64_t(7), std::int64_t(100003)}) {
        ce::BasebandOptions o = fsk;
        o.chunk = chunk;
        check::is_true(bitwise(ce::baseband(sym_s, S, 8000, 150.0, 120000, o), z_fsk),
                       "FSK baseband, chunk " + std::to_string(chunk));
        o.ook = true;
        check::is_true(bitwise(ce::baseband(sym_s, S, 8000, 150.0, 120000, o), z_ook),
                       "OOK baseband, chunk " + std::to_string(chunk));
    }
    {
        ce::BasebandOptions o;
        o.chunk = 1;
        const auto one = ce::baseband(sym_t, T, 8000, -0.75, 3000, o);
        const auto ref = ce::baseband(sym_t, T, 8000, -0.75, 3000, plain);
        check::is_true(bitwise(one, ref), "baseband, chunk 1 across the ramp");
        o.chunk = 333;
        o.time_scale = 1.0001;
        scaled.chunk = 4000;
        check::is_true(bitwise(ce::baseband(sym_t, T, 8000, 0.01234, 4000, o),
                               ce::baseband(sym_t, T, 8000, 0.01234, 4000, scaled)),
                       "closed-form baseband, chunk 333");
    }
    // Splitting the stream into consecutive calls gives the same samples too.
    {
        const auto a = ce::baseband(sym_s, S, 8000, 150.0, 50000, fsk);
        const auto b = ce::baseband(sym_s, S, 8000, 150.0 + 50000.0 / 8000, 70000, fsk);
        std::vector<cplx> ab(a);
        ab.insert(ab.end(), b.begin(), b.end());
        check::is_true(bitwise(ab, z_fsk), "baseband in two calls equals one");
    }
    check::equal(ce::baseband(sym_t, T, 8000, 0.0, 0, plain).size(), std::size_t(0), "n = 0");
    check::equal(refusal([&] { ce::baseband(sym_t, T, 8000, 0.0, -1, plain); }),
                 std::string("negative dimensions are not allowed"), "n < 0");
    check::equal(refusal([&] { ce::baseband(std::vector<double>(5, 0.0), T, 8000, 0.0, 10, plain); }),
                 std::string("sym has shape (5,), expected (1753,)"), "baseband sym shape");
    ce::BasebandOptions too_long;
    too_long.lead_in_s = 11.0;
    check::equal(refusal([&] { ce::baseband(sym_t, T, 8000, 0.0, 10, too_long); }),
                 std::string("lead-in must be 0 to 10 s, not 11.0"), "baseband lead-in");
}

// --- to_audio ---------------------------------------------------------------------

void test_to_audio() {
    check::current_step = "to_audio";
    const auto z = z_of(20000);
    struct Pin {
        std::int64_t num, den, n0;
        std::vector<std::pair<std::int64_t, double>> at;
    };
    // ce.to_audio(z, 8000, num, den, 0.5, n0)[i].hex()
    const Pin pins[] = {
        {1200, 1, 0, {{0, -0x1.0p-1}, {1, 0x1.0e0c4a95aec09p-6}, {7, -0x1.350fc6c5c7a89p-2}, {999, -0x1.e106d762eefa2p-2}, {12345, 0x1.267a19a6bbc47p-2}, {19999, -0x1.45de49f929502p-7}}},
        {1200, 1, 123456789, {{0, 0x1.6594d5006a782p-1}, {1, 0x1.8d4fdf3b645a3p-3}, {7, 0x1.793ded40c34b8p-2}, {999, 0x1.c3d11baf713eep-2}, {12345, 0x1.23d2712a71625p-4}, {19999, 0x1.7ba940f86a61dp-4}}},
        {1500, 1, 0, {{0, -0x1.0p-1}, {1, 0x1.3263b0f5d4904p-4}, {7, 0x1.0940f11befc2ep-2}, {999, 0x1.024306f91d151p-1}, {12345, 0x1.375a5c0ebb9c7p-3}, {19999, -0x1.1c812133d94dcp-5}}},
        {1500, 1, 123456789, {{0, -0x1.4e7ae9144f0fep-1}, {1, -0x1.7a932da5c91c8p-6}, {7, 0x1.0cd35de23e96cp-3}, {999, 0x1.0p-1}, {12345, -0x1.b4c1a93fd0625p-8}, {19999, -0x1.2366179965110p-4}}},
        {2650, 1, 0, {{0, -0x1.0p-1}, {1, 0x1.e265e17c87cddp-3}, {7, 0x1.14669bdc2bfd5p-2}, {999, -0x1.5d58702f24cf1p-2}, {12345, -0x1.5c26adeac966dp-2}, {19999, -0x1.a05b6a24fdd95p-4}}},
        {2650, 1, 123456789, {{0, 0x1.678778b090ab2p-1}, {1, -0x1.3263b0f5d48fbp-4}, {7, 0x1.7cc554d829fb1p-5}, {999, 0x1.05bd1b5bcefe3p-1}, {12345, 0x1.a681d42080f6dp-6}, {19999, 0x1.7546ae9d4a07bp-4}}},
        {15003, 10, 0, {{0, -0x1.0p-1}, {1, 0x1.329f394da1687p-4}, {7, 0x1.09baef006947cp-2}, {999, 0x1.e11371a42bae2p-2}, {12345, -0x1.e503c24252c3ap-5}, {19999, -0x1.a20042c548d06p-4}}},
        {15003, 10, 123456789, {{0, 0x1.01b2a23697a04p-2}, {1, -0x1.55cb1ef848716p-3}, {7, -0x1.6897dfa5a94adp-2}, {999, -0x1.6180f92ecacadp-2}, {12345, 0x1.c57b9a7da670dp-3}, {19999, 0x1.b7a7aa8f6b4eap-4}}},
    };
    for (const auto& p : pins) {
        const auto y = ce::to_audio(z, 8000, p.num, p.den, 0.5, p.n0);
        check::equal(y.size(), z.size(), "to_audio length");
        const std::string tag = std::to_string(p.num) + "/" + std::to_string(p.den) + " n0=" + std::to_string(p.n0);
        for (const auto& [i, want] : p.at)
            near(y[static_cast<std::size_t>(i)], want, 1e-12, "to_audio " + tag + " [" + std::to_string(i) + "]");
        // In chunks with the running count: the same samples, bit for bit.
        std::vector<double> parts;
        for (std::size_t i0 = 0; i0 < z.size(); i0 += 3001) {
            const std::size_t m = std::min<std::size_t>(3001, z.size() - i0);
            const auto part = ce::to_audio(std::span<const cplx>(z).subspan(i0, m), 8000, p.num, p.den, 0.5,
                                           p.n0 + static_cast<std::int64_t>(i0));
            parts.insert(parts.end(), part.begin(), part.end());
        }
        check::is_true(bitwise(parts, y), "to_audio is chunk-invariant, " + tag);
    }
    // A whole carrier period later is the same phase: n0 counts mod den * 8000.
    check::is_true(bitwise(ce::to_audio(z, 8000, 15003, 10, 0.5, 80000 * 7), ce::to_audio(z, 8000, 15003, 10, 0.5, 0)),
                   "to_audio's phase is periodic in den * 8000");
    check::is_true(bitwise(ce::to_audio(z, 8000, 15003, 10, 0.5, -1), ce::to_audio(z, 8000, 15003, 10, 0.5, 79999)),
                   "a negative n0 counts mod the period (floor)");
    // ce.to_audio(z[:500], 250, 1500): resampled to 8 kHz first
    const auto r = ce::to_audio(std::span<const cplx>(z).subspan(0, 500), 250, 1500);
    check::equal(r.size(), std::size_t(16000), "250 Hz in, 8 kHz out");
    const std::pair<std::size_t, double> rs[] = {
        {0, -0x1.002c34e97147fp-1}, {100, -0x1.0c62e0ed10e53p-1}, {8000, -0x1.37818805568c9p-4}, {15999, 0x1.3d373172c7d27p-7},
    };
    for (const auto& [i, want] : rs) near(r[i], want, 1e-12, "to_audio from 250 Hz [" + std::to_string(i) + "]");
    check::equal(refusal([&] { ce::to_audio(z, 8000, 1500, 0); }), std::string("carrier_hz_den must be positive"),
                 "den = 0");

    // mulmod against Python's big ints, (a * b) % m.
    const struct {
        std::int64_t a, b, m, want;
    } mm[] = {
        {3, 5, 7, 1},
        {4611686018427387903LL, 4611686018427387901LL, 9223372036854775783LL, 2305843009213694055LL},
        {1000000000000000000LL, 999999999999999989LL, 1000000000000000009LL, 180},
        {123456789012345LL, 987654321098765LL, 8388608000LL, 1908549925LL},
        {9223372036854775782LL, 9223372036854775782LL, 9223372036854775783LL, 1},
    };
    for (const auto& c : mm) check::equal(ce::mulmod(c.a, c.b, c.m), c.want, "mulmod " + std::to_string(c.a));
}

// --- the carrier --------------------------------------------------------------------

void test_carrier() {
    check::current_step = "carrier";
    check::equal(ce::check_carrier(1500.0), 1500.0, "1500 Hz is in band");
    check::equal(ce::check_carrier(300.0), 300.0, "the band's low edge is in");
    check::equal(ce::check_carrier(2700.0), 2700.0, "the band's high edge is in");
    // str(e) of ce.check_carrier(x)
    const std::pair<double, const char*> refused[] = {
        {299.9, "carrier 299.9 Hz is outside the 300-2700 Hz audio band that receivers search"},
        {2700.5, "carrier 2700.5 Hz is outside the 300-2700 Hz audio band that receivers search"},
        {std::nan(""), "carrier nan Hz is outside the 300-2700 Hz audio band that receivers search"},
        {HUGE_VAL, "carrier inf Hz is outside the 300-2700 Hz audio band that receivers search"},
        {-1500.0, "carrier -1500.0 Hz is outside the 300-2700 Hz audio band that receivers search"},
        {1e300, "carrier 1e+300 Hz is outside the 300-2700 Hz audio band that receivers search"},
    };
    for (const auto& [f, want] : refused)
        check::equal(refusal([&] { ce::check_carrier(f); }), std::string(want), "check_carrier refuses");
    check::equal(refusal([&] { ce::check_carrier(3000.0, "3000"); }),
                 std::string("carrier 3000 Hz is outside the 300-2700 Hz audio band that receivers search"),
                 "check_carrier shows the value as given");

    // ce.carrier_fraction(x)
    const auto is = [](const qrss::Rational& r, std::int64_t num, std::int64_t den, const std::string& what) {
        check::equal(r.num(), num, what + " numerator");
        check::equal(r.den(), den, what + " denominator");
    };
    is(ce::carrier_fraction(std::int64_t(1200)), 1200, 1, "1200");
    is(ce::carrier_fraction(1500.3), 15003, 10, "1500.3");
    is(ce::carrier_fraction(std::string_view("1500.3")), 15003, 10, "'1500.3'");
    is(ce::carrier_fraction(std::string_view("2650")), 2650, 1, "'2650'");
    is(ce::carrier_fraction(std::string_view("3/7")), 3, 7, "'3/7'");
    is(ce::carrier_fraction(1234.56789), 123456789, 100000, "1234.56789");
    is(ce::carrier_fraction(1500.0000001), 1500, 1, "1500.0000001 (limited to 2^20)");
    is(ce::carrier_fraction(qrss::Rational(6, 4)), 3, 2, "a Fraction passes through");
}

// --- small refusals -----------------------------------------------------------------

void test_refusals() {
    check::current_step = "refusals";
    check::equal(ce::grid_step(8000), std::int64_t(2), "8 kHz: 2 grid steps per sample");
    check::equal(ce::grid_step(250), std::int64_t(64), "250 Hz: 64 grid steps per sample");
    for (const std::int64_t fs : {std::int64_t(3), std::int64_t(0), std::int64_t(-8000), std::int64_t(32000)})
        check::equal(refusal([&] { ce::grid_step(fs); }), "fs must divide 16000, not " + std::to_string(fs),
                     "_grid_step(" + std::to_string(fs) + ")");
    const std::int64_t five[1] = {5};
    check::equal(refusal([&] { ce::check_sym(fr::get("short"), five); }),
                 std::string("sym has shape (5,), expected (7670,)"), "_check_sym");
    const std::int64_t two_by_three[2] = {2, 3};
    check::equal(refusal([&] { ce::check_sym(fr::get("tiny"), two_by_three, "var"); }),
                 std::string("var has shape (2, 3), expected (1753,)"), "_check_sym named var");
    check::equal(refusal([&] { ce::phase_grid(3000, 0, 10, sym_of(fr::get("tiny")), fr::get("tiny")); }),
                 std::string("fs must divide 16000, not 3000"), "phase_grid at 3 kHz");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    try {
        test_pulse();
        test_phase_grid();
        test_windows();
        test_span_and_ramp();
        test_baseband();
        test_to_audio();
        test_carrier();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL in %s: %s\n", check::current_step.load(), e.what());
        return 1;
    }
    return check::report("qrss_ce");
}
