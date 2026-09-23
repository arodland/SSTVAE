#include "modem/modem.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "dsp/dsp.hpp"
#include "framing/framing.hpp"
#include "ofdm/ofdm.hpp"

namespace sstvae::modem {

namespace {

constexpr double FRAME_S = static_cast<double>(config::FRAME_SAMPLES) / config::FS;

// Second-order loop on the pilots' common phase. Port of
// sstvae/modem/modem.py's _DriftTracker -- read that docstring before
// changing anything here; three things in it are load-bearing and each
// was wrong first (the correction is a continuous ramp *within* the
// frame rather than one constant phase per frame; the measurement is the
// residual that survived the correction already applied, so it is
// integrated rather than chased; and the loop is second order because
// drift is a ramp).
class DriftTracker {
   public:
    DriftTracker(double alpha, double beta) : alpha_(alpha), beta_(beta) {}

    // This frame's samples, de-rotated by the running estimate. Zero
    // where the frame hangs off the buffer: a placed window can start a
    // frame's CP before the buffer does, and no demod window reads it.
    std::vector<cdouble> frame(std::span<const cdouble> z, std::int64_t p) const {
        std::vector<cdouble> out(static_cast<std::size_t>(config::FRAME_SAMPLES));
        for (int n = 0; n < config::FRAME_SAMPLES; ++n) {
            const std::int64_t i = p + n;
            if (i < 0 || i >= static_cast<std::int64_t>(z.size())) continue;
            const double cycles = dsp::wrap_cycles(
                phase_acc_ + f_est_ * static_cast<double>(n) / config::FS);
            const double theta = -2.0 * std::numbers::pi * cycles;
            out[static_cast<std::size_t>(n)] =
                z[static_cast<std::size_t>(i)] * cdouble{std::cos(theta), std::sin(theta)};
        }
        return out;
    }

    // `h_prev` empty means "no usable previous pilot": the loop coasts on
    // its rate estimate rather than integrating phase out of noise.
    void update(std::span<const cdouble> h_cur, std::span<const cdouble> h_prev) {
        if (!h_prev.empty()) {
            cdouble d{};
            for (std::size_t k = 0; k < h_cur.size(); ++k) d += h_cur[k] * std::conj(h_prev[k]);
            if (std::abs(d) > 0.0) {
                const double err = std::arg(d) / (2.0 * std::numbers::pi * FRAME_S);
                f_est_ += alpha_ * err;
                r_est_ += beta_ * err / FRAME_S;
            }
        }
        // Carry absolute phase across the boundary *before* stepping the
        // frequency, so the correction stays continuous frame to frame.
        phase_acc_ += f_est_ * FRAME_S;
        f_est_ += r_est_ * FRAME_S;
    }

   private:
    double alpha_;
    double beta_;
    double f_est_ = 0.0;    // residual CFO estimate, Hz
    double r_est_ = 0.0;    // drift rate estimate, Hz/s
    double phase_acc_ = 0.0;  // accumulated de-rotation, cycles
};

// Nullopt when tracking is off, so the default path never consults a
// tracker at all rather than running one with zero gains.
std::optional<DriftTracker> make_tracker(DriftTrack track) {
    switch (track) {
        case DriftTrack::Off:
            return std::nullopt;
        case DriftTrack::Slow:
            return DriftTracker(config::DRIFT_SLOW_ALPHA, config::DRIFT_SLOW_BETA);
        case DriftTrack::Fast:
            return DriftTracker(config::DRIFT_FAST_ALPHA, config::DRIFT_FAST_BETA);
    }
    return std::nullopt;
}

}  // namespace

DriftTrack drift_track_from_name(std::string_view name) {
    if (name == "off") return DriftTrack::Off;
    if (name == "slow") return DriftTrack::Slow;
    if (name == "fast") return DriftTrack::Fast;
    throw std::invalid_argument("unknown drift_track \"" + std::string(name) +
                                "\" (expected off, slow or fast)");
}

std::string_view drift_track_name(DriftTrack track) {
    switch (track) {
        case DriftTrack::Slow:
            return "slow";
        case DriftTrack::Fast:
            return "fast";
        case DriftTrack::Off:
            break;
    }
    return "off";
}

namespace {

using config::BEACON_CARRIER;
using config::CHIPS_PER_FRAME;
using config::CLIP_HEADROOM_DB;
using config::DATA_SYMS_PER_FRAME;
using config::DEMOD_BACKOFF;
using config::FRAME_SAMPLES;
using config::FRAMES_PER_GROUP;
using config::FS;
using config::HEADER_SAMPLES;
using config::LATENT_GROUPS;
using config::LATENTS_PER_FRAME;
using config::LEADIN_SAMPLES;
using config::LEADOUT_SAMPLES;
using config::M;
using config::NC;
using config::NC_LATENT;
using config::NCP;
using config::NSYM;
using config::PREAMBLE_CP;
using config::PREAMBLE_SAMPLES;
using config::RS;
using config::SNR_REF_BW_HZ;
using config::SYMS_PER_FRAME;

constexpr double PI = std::numbers::pi;

// np.median: for an even count, the mean of the two middle values.
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const std::size_t n = v.size();
    const std::size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    const double hi = v[mid];
    if (n % 2 == 1) return hi;
    const double lo =
        *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (lo + hi);
}

// Mean per-carrier phase increment of a gain vector (a timing proxy).
double bin_phase_step(std::span<const cdouble> h) {
    cdouble acc{0.0, 0.0};
    for (std::size_t i = 1; i < h.size(); ++i) acc += h[i] * std::conj(h[i - 1]);
    return std::arg(acc);
}

// --- Receive-side channel measurements over the whole transmission.
// Ports of sstvae/modem/modem.py's _time_shift_phase, _residual_cfo,
// _delay_support and _window_shift -- read those docstrings.

// exp(-2j*pi*f*s/FS) for integer f and s, reduced exactly first for the
// same reason ofdm's carrier_phasor is.
cdouble shift_phasor(std::int64_t f, std::int64_t s) {
    std::int64_t q = (f * s) % FS;
    if (q < 0) q += FS;
    return std::polar(1.0, -2.0 * PI * (static_cast<double>(q) / FS));
}

std::int64_t baseband_hz(int k) {
    return static_cast<std::int64_t>(std::llround(ofdm::baseband_freqs()[static_cast<std::size_t>(k)]));
}

// Rows of an (n, NC) gain array.
struct Rows {
    const cdouble* data;
    std::size_t n;
    std::span<const cdouble> row(std::size_t i) const { return {data + i * NC, NC}; }
};

double residual_cfo(Rows h, std::span<const char> received) {
    cdouble d{0.0, 0.0};
    for (std::size_t f = 1; f < h.n; ++f) {
        if (!received[f] || !received[f - 1]) continue;
        const auto a = h.row(f);
        const auto b = h.row(f - 1);
        for (int k = 0; k < NC; ++k) {
            const std::size_t i = static_cast<std::size_t>(k);
            d += a[i] * std::conj(b[i]);
        }
    }
    return std::abs(d) > 0 ? std::arg(d) / (2 * PI * FRAME_S) : 0.0;
}

std::pair<int, int> delay_support(const std::vector<cdouble>& h, double floor_db = -15.0) {
    constexpr int D = 4 * NCP + 1;  // delays -2*NCP .. 2*NCP
    const std::size_t n = h.size() / NC;
    std::array<double, NC> w{};
    for (int k = 0; k < NC; ++k)
        w[static_cast<std::size_t>(k)] =
            0.5 - 0.5 * std::cos(2 * PI * (k + 1) / static_cast<double>(NC + 1));
    std::array<double, D> prof{};
    for (int j = 0; j < D; ++j) {
        const std::int64_t d = j - 2 * NCP;
        std::array<cdouble, NC> steer{};
        for (int k = 0; k < NC; ++k)
            steer[static_cast<std::size_t>(k)] = w[static_cast<std::size_t>(k)] *
                                                 shift_phasor(baseband_hz(k), -d);
        double acc = 0.0;
        for (std::size_t f = 0; f < n; ++f) {
            cdouble g{0.0, 0.0};
            for (int k = 0; k < NC; ++k) {
                const std::size_t i = static_cast<std::size_t>(k);
                g += h[f * NC + i] * steer[i];
            }
            acc += std::norm(g);
        }
        prof[static_cast<std::size_t>(j)] = n ? acc / static_cast<double>(n) : 0.0;
    }
    // Gated on the noise floor as well; see _delay_support in modem.py.
    const double peak = *std::max_element(prof.begin(), prof.end());
    const double thr = std::max(peak * std::pow(10.0, floor_db / 10.0),
                                2.0 * median(std::vector<double>(prof.begin(), prof.end())));
    int first = -1;
    int last = -1;
    for (int j = 1; j < D - 1; ++j) {
        const std::size_t i = static_cast<std::size_t>(j);
        if (prof[i] >= thr && prof[i] >= prof[i - 1] && prof[i] >= prof[i + 1]) {
            if (first < 0) first = j;
            last = j;
        }
    }
    if (first < 0)
        first = last = static_cast<int>(std::max_element(prof.begin(), prof.end()) - prof.begin());
    return {first - 2 * NCP, last - 2 * NCP};
}

int window_shift(std::pair<int, int> support) {
    // round() half to even, like Python's.
    return static_cast<int>(std::nearbyint((support.first + support.second - NCP) / 2.0));
}

// One pass over the frames. Port of Modem._demod_frames in
// sstvae/modem/modem.py -- read its docstring, in particular for why
// `unstep` is false on the pass that places the window.
struct Frames {
    std::vector<cdouble> raw;      // (n_f, SYMS_PER_FRAME, NC)
    std::vector<cdouble> h_pilot;  // (n_f, NC)
    std::vector<char> received;    // (n_f)
    std::vector<std::int64_t> steps;  // (n_f) accumulated timing step
};

Frames demod_frames(std::span<const cdouble> z, std::int64_t p, int n_f, double phi_ref,
                    int shift, std::optional<DriftTracker> tracker,
                    std::span<const cdouble> pilot, bool unstep) {
    // The shift's own slope, or the loop would walk the window back.
    phi_ref += 2 * PI * RS * shift / FS;
    p += shift;
    Frames out{std::vector<cdouble>(static_cast<std::size_t>(n_f) * SYMS_PER_FRAME * NC),
               std::vector<cdouble>(static_cast<std::size_t>(n_f) * NC),
               std::vector<char>(static_cast<std::size_t>(n_f), 0)};
    std::vector<std::int64_t> steps(static_cast<std::size_t>(n_f), 0);
    std::vector<double> pilot_powers;
    std::vector<cdouble> h_prev;
    double tau_ema = 0.0;
    std::int64_t total = 0;
    for (int f = 0; f < n_f; ++f) {
        if (p + FRAME_SAMPLES > static_cast<std::int64_t>(z.size())) break;
        const std::size_t fbase = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
        std::vector<cdouble> zz;
        if (tracker) zz = tracker->frame(z, p);
        for (int s = 0; s < SYMS_PER_FRAME; ++s) {
            const auto sym = tracker
                ? ofdm::demod_window(zz, s * NSYM + NCP, DEMOD_BACKOFF)
                : ofdm::demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF);
            std::copy(sym.begin(), sym.end(),
                      out.raw.begin() + static_cast<std::ptrdiff_t>(
                                            fbase + static_cast<std::size_t>(s) * NC));
        }
        std::vector<cdouble> h(NC);
        for (int k = 0; k < NC; ++k) {
            const std::size_t i = static_cast<std::size_t>(k);
            h[i] = out.raw[fbase + i] / pilot[i];
        }
        out.received[static_cast<std::size_t>(f)] = 1;
        steps[static_cast<std::size_t>(f)] = total;
        p += FRAME_SAMPLES;

        double power = 0.0;
        for (const cdouble& v : h) power += std::norm(v);
        power /= NC;
        pilot_powers.push_back(power);
        const bool healthy = power > 0.1 * median(pilot_powers);
        if (tracker) {
            // A faded frame's pilot phase is noise; feed the loop nothing
            // rather than a bad measurement, but let it coast on its rate.
            // Its phase sees the step as well, so compare like with like.
            if (unstep && !h_prev.empty()) {
                const std::int64_t d = steps[static_cast<std::size_t>(f - 1)] -
                                       steps[static_cast<std::size_t>(f)];
                for (int k = 0; k < NC; ++k)
                    h_prev[static_cast<std::size_t>(k)] *= shift_phasor(baseband_hz(k), d);
            }
            tracker->update(h, healthy && !h_prev.empty()
                                   ? std::span<const cdouble>(h_prev)
                                   : std::span<const cdouble>{});
        }
        h_prev = h;
        if (healthy) {
            const double phi = bin_phase_step(h);
            // Wrap the difference to (-pi, pi] before scaling.
            const double d = std::arg(std::polar(1.0, phi - phi_ref));
            const double tau = -d * FS / (2 * PI * RS);
            tau_ema += 0.02 * (tau - tau_ema);
            if (std::abs(tau_ema) >= 2) {
                // np.clip(round(x), -2, 2); round() is half-to-even.
                const int step = static_cast<int>(std::clamp(std::nearbyint(tau_ema), -2.0, 2.0));
                p += step;
                total += step;
                tau_ema -= step;
            }
        }
    }
    out.steps = steps;
    for (int f = 0; f < n_f; ++f) {
        const std::size_t fi = static_cast<std::size_t>(f);
        std::array<cdouble, NC> ph{};
        for (int k = 0; k < NC; ++k)
            ph[static_cast<std::size_t>(k)] =
                unstep ? shift_phasor(baseband_hz(k), steps[fi]) : cdouble{1.0, 0.0};
        for (int s = 0; s < SYMS_PER_FRAME; ++s)
            for (int k = 0; k < NC; ++k)
                out.raw[(fi * SYMS_PER_FRAME + static_cast<std::size_t>(s)) * NC +
                        static_cast<std::size_t>(k)] *= ph[static_cast<std::size_t>(k)];
        for (int k = 0; k < NC; ++k) {
            const std::size_t i = static_cast<std::size_t>(k);
            out.h_pilot[fi * NC + i] = out.raw[fi * SYMS_PER_FRAME * NC + i] / pilot[i];
        }
    }
    return out;
}

// Received rows only, for the delay profile.
std::vector<cdouble> received_rows(const Frames& fr) {
    std::vector<cdouble> out;
    for (std::size_t f = 0; f < fr.received.size(); ++f)
        if (fr.received[f])
            out.insert(out.end(), fr.h_pilot.begin() + static_cast<std::ptrdiff_t>(f * NC),
                       fr.h_pilot.begin() + static_cast<std::ptrdiff_t>((f + 1) * NC));
    return out;
}

// exp(-2j*pi*f*s/FS) for a fractional shift s: not exact, but reduced
// before the transcendental. Mirrors _shift_phasor in modem.py.
cdouble frac_shift_phasor(std::int64_t f, double s) {
    const double c = dsp::wrap_cycles(s * static_cast<double>(f) / FS);
    return std::polar(1.0, -2.0 * PI * c);
}

// Eigenvectors of a real symmetric n x n matrix (row-major, destroyed),
// cyclic Jacobi. Returns eigenvalues; column j of `v` is vector j.
std::vector<double> jacobi_eigen(std::vector<double>& a, std::vector<double>& v, int n) {
    auto at = [n](std::vector<double>& m, int r, int c) -> double& {
        return m[static_cast<std::size_t>(r) * static_cast<std::size_t>(n) +
                 static_cast<std::size_t>(c)];
    };
    v.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) at(v, i, i) = 1.0;
    double norm = 0.0;
    for (double x : a) norm += x * x;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += at(a, p, q) * at(a, p, q);
        if (off <= 1e-30 * norm) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = at(a, p, q);
                if (apq == 0.0) continue;
                const double theta = (at(a, q, q) - at(a, p, p)) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double sn = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = at(a, k, p), akq = at(a, k, q);
                    at(a, k, p) = c * akp - sn * akq;
                    at(a, k, q) = sn * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = at(a, p, k), aqk = at(a, q, k);
                    at(a, p, k) = c * apk - sn * aqk;
                    at(a, q, k) = sn * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = at(v, k, p), vkq = at(v, k, q);
                    at(v, k, p) = c * vkp - sn * vkq;
                    at(v, k, q) = sn * vkp + c * vkq;
                }
            }
        }
    }
    std::vector<double> lam(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) lam[static_cast<std::size_t>(i)] = at(a, i, i);
    return lam;
}

// x = A^{-1} b for a real symmetric positive definite k x k A (Cholesky).
std::vector<double> spd_solve(const std::vector<double>& a, std::vector<double> b, int k) {
    std::vector<double> l(a.size(), 0.0);
    auto L = [&l, k](int r, int c) -> double& {
        return l[static_cast<std::size_t>(r * k + c)];
    };
    for (int r = 0; r < k; ++r)
        for (int c = 0; c <= r; ++c) {
            double sum = a[static_cast<std::size_t>(r * k + c)];
            for (int m = 0; m < c; ++m) sum -= L(r, m) * L(c, m);
            L(r, c) = r == c ? std::sqrt(sum) : sum / L(c, c);
        }
    for (int r = 0; r < k; ++r) {
        double sum = b[static_cast<std::size_t>(r)];
        for (int m = 0; m < r; ++m) sum -= L(r, m) * b[static_cast<std::size_t>(m)];
        b[static_cast<std::size_t>(r)] = sum / L(r, r);
    }
    for (int r = k - 1; r >= 0; --r) {
        double sum = b[static_cast<std::size_t>(r)];
        for (int m = r + 1; m < k; ++m) sum -= L(m, r) * b[static_cast<std::size_t>(m)];
        b[static_cast<std::size_t>(r)] = sum / L(r, r);
    }
    return b;
}

constexpr int LMMSE_TIME_TAPS = 4;
constexpr double LMMSE_DEFAULT_SPREAD_HZ = 2.0;
constexpr int DATA_SYMS = SYMS_PER_FRAME - 1;

// Port of _coherent_frames / _transmission_frames in modem.py -- read
// the latter's docstring. `h` is (n, NC).
constexpr double TX_COHERENCE = 0.5;

std::vector<char> coherent_frames(const std::vector<cdouble>& h) {
    const std::size_t n = h.size() / NC;
    std::vector<double> coh(n, 0.0);
    for (std::size_t f = 1; f < n; ++f) {
        cdouble acc{0.0, 0.0};
        double a = 0.0, b = 0.0;
        for (int k = 0; k < NC; ++k) {
            const cdouble cur = h[f * NC + static_cast<std::size_t>(k)];
            const cdouble prev = h[(f - 1) * NC + static_cast<std::size_t>(k)];
            acc += cur * std::conj(prev);
            a += std::norm(cur);
            b += std::norm(prev);
        }
        const double c = std::abs(acc) / std::sqrt(a * b + 1e-30);
        coh[f] = c;
        coh[f - 1] = std::max(coh[f - 1], c);
    }
    std::vector<char> ok(n, 0);
    bool any = false;
    for (std::size_t f = 0; f < n; ++f) any |= (ok[f] = coh[f] > TX_COHERENCE) != 0;
    if (!any) std::fill(ok.begin(), ok.end(), 1);
    return ok;
}

std::vector<double> frame_power(const std::vector<cdouble>& h) {
    std::vector<double> pw(h.size() / NC, 0.0);
    for (std::size_t f = 0; f < pw.size(); ++f) {
        for (int k = 0; k < NC; ++k) pw[f] += std::norm(h[f * NC + static_cast<std::size_t>(k)]);
        pw[f] /= NC;
    }
    return pw;
}

std::vector<char> transmission_frames(const std::vector<cdouble>& h) {
    const std::vector<double> pw = frame_power(h);
    std::vector<char> ok = coherent_frames(h);
    double top = 0.0;
    for (std::size_t f = 0; f < pw.size(); ++f)
        if (ok[f]) top = std::max(top, pw[f]);
    for (std::size_t f = 0; f < pw.size(); ++f) ok[f] = ok[f] && pw[f] > 0.1 * top;
    return ok;
}

// Port of _lmmse_channel in sstvae/modem/modem.py -- read its docstring.
// `h` is (n, NC); returns (n, DATA_SYMS, NC). The Hermitian projector is
// found through the real-symmetric 2*NC embedding [[Re, -Im], [Im, Re]],
// whose eigenvalues are the complex matrix's, each twice: the retained
// real eigenvectors span exactly the retained complex subspace.
std::vector<cdouble> lmmse_channel(const std::vector<cdouble>& h, std::span<const std::int64_t> steps,
                                   const std::vector<char>* plausible) {
    const std::size_t n = h.size() / NC;
    auto H = [&h](std::size_t f, int k) { return h[f * NC + static_cast<std::size_t>(k)]; };
    const std::vector<double> pw = frame_power(h);
    const std::vector<char> strong = plausible ? *plausible : std::vector<char>(n, 1);

    // Residual timing drift: a power-weighted line through each frame's
    // pilot phase slope, measured with the steps still in.
    std::vector<double> tau(n, 0.0);
    for (std::size_t f = 0; f < n; ++f) {
        const std::int64_t st = steps.empty() ? 0 : steps[f];
        cdouble acc{0.0, 0.0};
        cdouble prev{};
        for (int k = 0; k < NC; ++k) {
            const cdouble cur = H(f, k) * std::conj(shift_phasor(baseband_hz(k), st));
            if (k > 0) acc += cur * std::conj(prev);
            prev = cur;
        }
        tau[f] = -std::arg(acc) * FS / (2 * PI * RS);
    }
    std::vector<double> drift(n, 0.0);
    {
        double sw = 0.0, swf = 0.0, swt = 0.0;
        std::size_t nz = 0;
        for (std::size_t f = 0; f < n; ++f) {
            const double w = strong[f] ? pw[f] : 0.0;
            if (w != 0.0) ++nz;
            sw += w;
            swf += w * static_cast<double>(f);
            swt += w * tau[f];
        }
        if (nz >= 2) {
            const double fm = swf / sw;
            const double tm = swt / sw;
            double var = 0.0, cov = 0.0;
            for (std::size_t f = 0; f < n; ++f) {
                const double w = strong[f] ? pw[f] : 0.0;
                var += w * (static_cast<double>(f) - fm) * (static_cast<double>(f) - fm);
                cov += w * (static_cast<double>(f) - fm) * (tau[f] - tm);
            }
            if (var > 0)
                for (std::size_t f = 0; f < n; ++f)
                    drift[f] = cov / var * (static_cast<double>(f) - fm);
        }
    }
    std::vector<cdouble> undo(n * NC), aligned(n * NC);
    for (std::size_t f = 0; f < n; ++f)
        for (int k = 0; k < NC; ++k) {
            const std::size_t i = f * NC + static_cast<std::size_t>(k);
            const double total = static_cast<double>(steps.empty() ? 0 : steps[f]) + drift[f];
            undo[i] = frac_shift_phasor(baseband_hz(k), total);
            aligned[i] = h[i] * std::conj(undo[i]);
        }

    // Across carriers: projection onto the measured delay support.
    std::vector<cdouble> strong_rows;
    for (std::size_t f = 0; f < n; ++f)
        if (strong[f])
            strong_rows.insert(strong_rows.end(), aligned.begin() + static_cast<std::ptrdiff_t>(f * NC),
                               aligned.begin() + static_cast<std::ptrdiff_t>((f + 1) * NC));
    const auto [d0, d1] = delay_support(strong_rows);
    std::array<cdouble, NC * NC> g{};  // B B^H
    for (int d = d0 - 4; d <= d1 + 4; ++d)
        for (int r = 0; r < NC; ++r)
            for (int c = 0; c < NC; ++c)
                g[static_cast<std::size_t>(r * NC + c)] +=
                    shift_phasor(baseband_hz(r), d) * std::conj(shift_phasor(baseband_hz(c), d));
    constexpr int N2 = 2 * NC;
    std::vector<double> emb(static_cast<std::size_t>(N2 * N2));
    for (int r = 0; r < NC; ++r)
        for (int c = 0; c < NC; ++c) {
            const cdouble x = g[static_cast<std::size_t>(r * NC + c)];
            emb[static_cast<std::size_t>(r * N2 + c)] = x.real();
            emb[static_cast<std::size_t>(r * N2 + c + NC)] = -x.imag();
            emb[static_cast<std::size_t>((r + NC) * N2 + c)] = x.imag();
            emb[static_cast<std::size_t>((r + NC) * N2 + c + NC)] = x.real();
        }
    std::vector<double> vec;
    const std::vector<double> lam = jacobi_eigen(emb, vec, N2);
    const double lam_max = *std::max_element(lam.begin(), lam.end());
    std::array<cdouble, NC * NC> proj{};
    int kept = 0;
    for (int j = 0; j < N2; ++j) {
        if (!(lam[static_cast<std::size_t>(j)] > lam_max * 1e-4)) continue;
        ++kept;
        for (int r = 0; r < NC; ++r)
            for (int c = 0; c < NC; ++c) {
                // P_real = V V^T; P = P_real[top-left] + i P_real[bottom-left].
                const double vr = vec[static_cast<std::size_t>(r * N2 + j)];
                const double vi = vec[static_cast<std::size_t>((r + NC) * N2 + j)];
                const double vc = vec[static_cast<std::size_t>(c * N2 + j)];
                proj[static_cast<std::size_t>(r * NC + c)] += cdouble{vr * vc, vi * vc};
            }
    }
    const int rank = kept / 2;
    std::vector<cdouble> hs(n * NC);
    for (std::size_t f = 0; f < n; ++f)
        for (int k = 0; k < NC; ++k) {
            cdouble acc{0.0, 0.0};
            for (int m = 0; m < NC; ++m)
                acc += proj[static_cast<std::size_t>(k * NC + m)] * aligned[f * NC + static_cast<std::size_t>(m)];
            hs[f * NC + static_cast<std::size_t>(k)] = acc * undo[f * NC + static_cast<std::size_t>(k)];
        }

    double n0 = 0.0;
    for (std::size_t i = 0; i < hs.size(); ++i) n0 += std::norm(h[i] - hs[i]);
    n0 = n0 / static_cast<double>(hs.size()) * NC / std::max(NC - rank, 1);
    const double n0_s = n0 * rank / NC;
    std::vector<char> stats = strong;
    if (plausible) {
        const std::vector<char> coherent = coherent_frames(h);
        for (std::size_t f = 0; f < n; ++f)
            stats[f] = (coherent[f] && pw[f] > 2 * n0) || strong[f];
    }
    double p_acc = 0.0;
    std::size_t p_cnt = 0;
    for (std::size_t f = 0; f < n; ++f)
        if (stats[f])
            for (int k = 0; k < NC; ++k, ++p_cnt) p_acc += std::norm(hs[f * NC + static_cast<std::size_t>(k)]);
    const double p_sig = std::max(p_acc / static_cast<double>(p_cnt) - n0_s, 1e-12);
    cdouble lag1{0.0, 0.0};
    std::size_t n_pairs = 0;
    for (std::size_t f = 1; f < n; ++f) {
        if (!stats[f] || !stats[f - 1]) continue;
        ++n_pairs;
        for (int k = 0; k < NC; ++k)
            lag1 += hs[f * NC + static_cast<std::size_t>(k)] *
                    std::conj(hs[(f - 1) * NC + static_cast<std::size_t>(k)]);
    }
    if (n_pairs) lag1 /= static_cast<double>(n_pairs * NC);
    const double rot = std::arg(lag1) / (2 * PI);  // cycles per frame
    double spread = LMMSE_DEFAULT_SPREAD_HZ;
    if (n_pairs >= 8) {
        const double rho = std::clamp(std::abs(lag1) / p_sig, 1e-3, 0.9999);
        spread = std::clamp(2 * std::sqrt(-std::log(rho) / 2) / (PI * FRAME_S), 0.02, 4.0);
    }
    auto corr = [spread](double dt) {
        const double x = PI * spread / 2 * dt * FRAME_S;
        return std::exp(-2 * x * x);
    };
    auto turn = [rot](double t) { return std::polar(1.0, 2 * PI * dsp::wrap_cycles(rot * t)); };

    // In time: Wiener over the nearest pilots, rotation removed.
    const int k = static_cast<int>(std::min<std::size_t>(2 * LMMSE_TIME_TAPS, n));
    std::vector<double> rpp(static_cast<std::size_t>(k * k));
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c)
            rpp[static_cast<std::size_t>(r * k + c)] = p_sig * corr(r - c) + (r == c ? n0_s : 0.0);
    std::vector<cdouble> hd(n * NC);
    for (std::size_t f = 0; f < n; ++f) {
        const cdouble tr = std::conj(turn(static_cast<double>(f)));
        for (int m = 0; m < NC; ++m) hd[f * NC + static_cast<std::size_t>(m)] = hs[f * NC + static_cast<std::size_t>(m)] * tr;
    }
    std::vector<std::vector<double>> cache(static_cast<std::size_t>(k));  // W rows by f - lo
    std::vector<cdouble> out(n * DATA_SYMS * NC);
    for (std::size_t f = 0; f < n; ++f) {
        const std::int64_t lo = std::max<std::int64_t>(
            0, std::min<std::int64_t>(static_cast<std::int64_t>(f) - LMMSE_TIME_TAPS + 1,
                                      static_cast<std::int64_t>(n) - k));
        const int rel = static_cast<int>(static_cast<std::int64_t>(f) - lo);
        auto& w = cache[static_cast<std::size_t>(rel)];
        if (w.empty()) {  // evenly spaced pilots: few distinct W
            for (int s = 1; s <= DATA_SYMS; ++s) {
                std::vector<double> rdp(static_cast<std::size_t>(k));
                for (int j = 0; j < k; ++j)
                    rdp[static_cast<std::size_t>(j)] =
                        p_sig * corr(rel + static_cast<double>(s) / SYMS_PER_FRAME - j);
                const auto row = spd_solve(rpp, rdp, k);
                w.insert(w.end(), row.begin(), row.end());
            }
        }
        for (int s = 0; s < DATA_SYMS; ++s) {
            const cdouble tr = turn(static_cast<double>(f) + static_cast<double>(s + 1) / SYMS_PER_FRAME);
            for (int m = 0; m < NC; ++m) {
                cdouble acc{0.0, 0.0};
                for (int j = 0; j < k; ++j)
                    acc += w[static_cast<std::size_t>(s * k + j)] *
                           hd[(static_cast<std::size_t>(lo) + static_cast<std::size_t>(j)) * NC +
                              static_cast<std::size_t>(m)];
                out[(f * DATA_SYMS + static_cast<std::size_t>(s)) * NC + static_cast<std::size_t>(m)] = acc * tr;
            }
        }
    }
    return out;
}

// One data symbol's equalization: matched-filter combining, per-latent
// confidence, and the beacon chip. Shared between demodulate() and
// demodulate_blind(), which do the identical arithmetic.
struct EqualizedSymbol {
    std::array<double, NC_LATENT * 2> slots;
    std::array<double, NC_LATENT * 2> weights;
    double beacon_chip;
};

EqualizedSymbol equalize(std::span<const cdouble> raw_sym,
                         std::span<const cdouble> h, double floor, double med_h) {
    EqualizedSymbol out{};
    const double sqrt2 = std::sqrt(2.0);
    for (int k = 0; k < NC_LATENT; ++k) {
        const std::size_t i = static_cast<std::size_t>(k);
        const double mag = std::max(std::abs(h[i]), floor);
        const cdouble y = raw_sym[i] * std::conj(h[i]) / (mag * mag);
        const double w = std::min(std::abs(h[i]) / med_h, 1.0);
        out.slots[2 * i] = y.real() * sqrt2;
        out.slots[2 * i + 1] = y.imag() * sqrt2;
        out.weights[2 * i] = w;
        out.weights[2 * i + 1] = w;
    }
    // Maximal-ratio, not the equalized value: dividing by the channel
    // estimate turns a beacon carrier in a fade null into amplified
    // noise with a large magnitude, and Golay's soft ML decode reads
    // magnitude as confidence -- one nulled chip then outvotes the four
    // good ones in the same codeword. Weighting by |h|^2 (equivalently,
    // skipping the equalization) is the correct soft metric for BPSK
    // and needs no `floor`.
    const std::size_t b = static_cast<std::size_t>(BEACON_CARRIER);
    out.beacon_chip = (raw_sym[b] * std::conj(h[b])).real();
    return out;
}

}  // namespace

const ModeSpec& mode_by_name(std::string_view name) {
    for (const ModeSpec& m : config::MODES) {
        if (m.name == name) return m;
    }
    throw std::out_of_range("unknown mode \"" + std::string(name) + "\"");
}

double estimate_snr_db(std::span<const cdouble> h_pilot, int n_frames,
                       std::span<const char> received) {
    std::vector<int> idx;
    for (int f = 0; f < n_frames; ++f)
        if (received.empty() || received[static_cast<std::size_t>(f)]) idx.push_back(f);
    if (idx.size() < 2) return std::nan("");

    // Only *adjacent* received frames contribute a noise sample: the
    // estimator treats the frame-to-frame change in channel gain as
    // noise, which is only meaningful across consecutive frames.
    double noise_acc = 0.0;
    std::size_t noise_n = 0;
    for (std::size_t i = 0; i + 1 < idx.size(); ++i) {
        if (idx[i + 1] - idx[i] != 1) continue;
        for (int k = 0; k < NC; ++k) {
            const cdouble d =
                h_pilot[static_cast<std::size_t>(idx[i + 1]) * NC + static_cast<std::size_t>(k)] -
                h_pilot[static_cast<std::size_t>(idx[i]) * NC + static_cast<std::size_t>(k)];
            noise_acc += std::norm(d);
            ++noise_n;
        }
    }
    if (noise_n == 0) return std::nan("");

    double signal_acc = 0.0;
    for (int f : idx)
        for (int k = 0; k < NC; ++k)
            signal_acc += std::norm(
                h_pilot[static_cast<std::size_t>(f) * NC + static_cast<std::size_t>(k)]);

    const double noise_var = 0.5 * (noise_acc / static_cast<double>(noise_n));
    const double signal_var =
        signal_acc / static_cast<double>(idx.size() * static_cast<std::size_t>(NC));
    if (noise_var <= 0) return std::numeric_limits<double>::infinity();
    if (signal_var <= 0) return -std::numeric_limits<double>::infinity();

    // Per-carrier SNR is in a ~RS-wide (50 Hz) noise bandwidth -- the
    // DFT correlator's matched-filter bandwidth -- scaled to the
    // reference bandwidth assuming roughly even power across carriers.
    const double snr_50hz = signal_var / noise_var;
    return 10.0 * std::log10(snr_50hz * (NC * RS / SNR_REF_BW_HZ));
}

Modem::Modem() {
    const auto& p = ofdm::pilot_sequence();
    pilot_.assign(p.begin(), p.end());
}

std::vector<double> Modem::modulate(std::span<const double> latents,
                                    const ModeSpec& mode, bool normalize,
                                    const std::string& callsign,
                                    double clip_headroom_db) const {
    if (latents.size() != static_cast<std::size_t>(mode.n_latents))
        throw std::invalid_argument("modulate: wrong latent count for this mode");

    std::vector<double> lat(latents.begin(), latents.end());
    if (normalize) {
        double ms = 0.0;
        for (double v : lat) ms += v * v;
        const double rms = std::sqrt(ms / static_cast<double>(lat.size()));
        if (rms > 0)
            for (double& v : lat) v /= rms;
    }

    const std::vector<double> slots = framing::interleave(lat, mode);
    const int n_f = mode.n_frames;
    const std::vector<double> chips =
        beacon::chip_stream(0, n_f, callsign, mode.index);

    std::vector<cdouble> symbols(static_cast<std::size_t>(n_f) * SYMS_PER_FRAME * NC,
                                 cdouble{});
    for (int f = 0; f < n_f; ++f) {
        const std::size_t base = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
        for (int k = 0; k < NC; ++k)
            symbols[base + static_cast<std::size_t>(k)] =
                pilot_[static_cast<std::size_t>(k)];

        const std::span<const double> frame_slots(
            slots.data() + static_cast<std::size_t>(f) * LATENTS_PER_FRAME,
            LATENTS_PER_FRAME);
        const std::vector<cdouble> data = framing::slots_to_symbols(frame_slots);
        for (int s = 0; s < DATA_SYMS_PER_FRAME; ++s) {
            const std::size_t row = base + static_cast<std::size_t>(s + 1) * NC;
            for (int k = 0; k < NC_LATENT; ++k)
                symbols[row + static_cast<std::size_t>(k)] =
                    data[static_cast<std::size_t>(s) * NC_LATENT +
                         static_cast<std::size_t>(k)];
            symbols[row + static_cast<std::size_t>(BEACON_CARRIER)] =
                cdouble(chips[static_cast<std::size_t>(f) * CHIPS_PER_FRAME +
                              static_cast<std::size_t>(s)],
                        0.0);
        }
    }

    const std::vector<cdouble> hdr = framing::header_symbol(mode);
    std::vector<cdouble> hdr2(hdr.begin(), hdr.end());
    hdr2.insert(hdr2.end(), hdr.begin(), hdr.end());

    const std::vector<double> preamble = ofdm::preamble_waveform();
    const std::vector<double> hdr_wave = ofdm::modulate_symbols(hdr2, 2);
    const std::vector<double> body =
        ofdm::modulate_symbols(symbols, static_cast<std::size_t>(n_f) * SYMS_PER_FRAME);

    std::vector<double> x;
    x.reserve(LEADIN_SAMPLES + preamble.size() + hdr_wave.size() + body.size() +
              LEADOUT_SAMPLES);
    x.insert(x.end(), LEADIN_SAMPLES, 0.0);
    x.insert(x.end(), preamble.begin(), preamble.end());
    x.insert(x.end(), hdr_wave.begin(), hdr_wave.end());
    x.insert(x.end(), body.begin(), body.end());
    x.insert(x.end(), LEADOUT_SAMPLES, 0.0);
    return dsp::tx_condition(x, clip_headroom_db);
}

DemodResult Modem::demodulate(std::span<const double> x,
                              std::optional<std::pair<double, double>> search_s,
                              DriftTrack drift_track) const {
    std::vector<cdouble> z = dsp::to_baseband(x);

    std::optional<sync::SearchWindow> search;
    if (search_s)
        search = sync::SearchWindow{static_cast<std::int64_t>(search_s->first * FS),
                                    static_cast<std::int64_t>(search_s->second * FS)};
    const sync::Acquisition acq =
        sync::acquire(z, config::PREAMBLE_THRESHOLD, config::ACQUIRE_MAX_BINS, search);
    z = dsp::freq_correct(z, acq.freq_offset);

    // Channel reference from the preamble, averaged over every repeat.
    // Backing DEMOD_BACKOFF samples into the *previous* repeat is safe
    // for the same reason it is safe into the CP: the block is periodic
    // with M throughout.
    const std::int64_t u0 = acq.preamble_start + PREAMBLE_CP;
    std::vector<cdouble> h_pre(NC, cdouble{0.0, 0.0});
    for (int r = 0; r < config::PREAMBLE_REPEATS; ++r) {
        const auto w = ofdm::demod_window(z, u0 + r * M, DEMOD_BACKOFF);
        for (int k = 0; k < NC; ++k) {
            const std::size_t i = static_cast<std::size_t>(k);
            h_pre[i] += w[i];
        }
    }
    for (int k = 0; k < NC; ++k) {
        const std::size_t i = static_cast<std::size_t>(k);
        h_pre[i] /= static_cast<double>(config::PREAMBLE_REPEATS) * pilot_[i];
    }

    // Header: two identical BPSK symbols, matched-filter combined so
    // faded carriers contribute little instead of amplifying noise as
    // zero-forcing would.
    const std::int64_t h0 = acq.preamble_start + PREAMBLE_SAMPLES;
    std::vector<double> soft(NC, 0.0);
    for (int s = 0; s < 2; ++s) {
        const auto y = ofdm::demod_window(z, h0 + s * NSYM + NCP, DEMOD_BACKOFF);
        for (int k = 0; k < NC; ++k) {
            const std::size_t i = static_cast<std::size_t>(k);
            soft[i] += (y[i] * std::conj(h_pre[i])).real();
        }
    }
    const auto spec_opt = framing::decode_header(soft);
    if (!spec_opt) throw SyncError("header decode failed");
    const ModeSpec spec = *spec_opt;

    // Demodulate frames, tracking sample-clock drift via the phase slope
    // of the pilot across carriers (relative to the preamble). Pass 1
    // at acquisition timing only measures what the whole transmission
    // says about residual frequency and delay spread; see
    // Modem.demodulate in sstvae/modem/modem.py.
    const int n_f = spec.n_frames;
    const double phi_ref = bin_phase_step(h_pre);
    const std::int64_t p_frames = h0 + HEADER_SAMPLES;
    Frames pass1 = demod_frames(z, p_frames, n_f, phi_ref, 0, std::nullopt, pilot_, false);
    const double cfo_res = residual_cfo(Rows{pass1.h_pilot.data(), static_cast<std::size_t>(n_f)},
                                        pass1.received);
    z = dsp::freq_correct(z, cfo_res);
    pass1 = demod_frames(z, p_frames, n_f, phi_ref, 0, std::nullopt, pilot_, false);
    const std::vector<cdouble> rows1 = received_rows(pass1);
    const int shift = rows1.empty() ? 0 : window_shift(delay_support(rows1));
    const Frames fr = demod_frames(z, p_frames, n_f, phi_ref, shift, make_tracker(drift_track),
                                   pilot_, true);
    const std::vector<cdouble>& raw = fr.raw;
    const std::vector<cdouble>& h_pilot = fr.h_pilot;
    const std::vector<char>& received = fr.received;

    // Equalize data symbols with pilots interpolated across the frame.
    std::vector<double> latents(static_cast<std::size_t>(spec.n_tx_latents), 0.0);
    std::vector<double> weights(static_cast<std::size_t>(spec.n_tx_latents), 0.0);

    std::vector<double> mags;
    for (int f = 0; f < n_f; ++f)
        if (received[static_cast<std::size_t>(f)])
            for (int k = 0; k < NC; ++k)
                mags.push_back(std::abs(
                    h_pilot[static_cast<std::size_t>(f) * NC + static_cast<std::size_t>(k)]));
    const double med_h = mags.empty() ? 1.0 : median(mags);
    const double floor = std::max(0.05 * med_h, 1e-9);

    // Frames present are a prefix: the buffer can only run out.
    std::size_t n_rx = 0;
    while (n_rx < received.size() && received[n_rx]) ++n_rx;
    std::vector<cdouble> h_est;
    if (n_rx)
        h_est = lmmse_channel(
            std::vector<cdouble>(h_pilot.begin(), h_pilot.begin() + static_cast<std::ptrdiff_t>(n_rx * NC)),
            std::span<const std::int64_t>(fr.steps.data(), n_rx), nullptr);

    std::vector<double> beacon_soft(static_cast<std::size_t>(n_f) * CHIPS_PER_FRAME, 0.0);
    for (int f = 0; f < n_f; ++f) {
        if (!received[static_cast<std::size_t>(f)]) continue;
        const std::size_t fbase = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
        for (int s = 1; s < SYMS_PER_FRAME; ++s) {
            const std::span<const cdouble> h(
                h_est.data() + (static_cast<std::size_t>(f) * DATA_SYMS + static_cast<std::size_t>(s - 1)) * NC, NC);
            const std::span<const cdouble> raw_sym(
                raw.data() + fbase + static_cast<std::size_t>(s) * NC, NC);
            const EqualizedSymbol eq = equalize(raw_sym, h, floor, med_h);

            const std::size_t lo = static_cast<std::size_t>(f) * LATENTS_PER_FRAME +
                                   static_cast<std::size_t>(s - 1) * NC_LATENT * 2;
            for (std::size_t i = 0; i < eq.slots.size(); ++i) {
                latents[lo + i] = eq.slots[i];
                weights[lo + i] = eq.weights[i];
            }
            beacon_soft[static_cast<std::size_t>(f) * CHIPS_PER_FRAME +
                        static_cast<std::size_t>(s - 1)] = eq.beacon_chip;
        }
    }

    for (double& v : latents) v = std::clamp(v, -10.0, 10.0);
    const auto lat_full = framing::deinterleave(latents, spec);
    const auto w_full = framing::deinterleave(weights, spec);
    const auto beacon_result = beacon::decode(beacon_soft);

    int n_received = 0;
    for (char r : received) n_received += r;

    return DemodResult{lat_full.latents,
                       w_full.latents,
                       spec,
                       acq.freq_offset + cfo_res,
                       acq.metric,
                       n_received,
                       beacon_result,
                       beacon_result ? beacon_result->callsign : std::string(),
                       acq.preamble_start,
                       estimate_snr_db(h_pilot, n_f, received)};
}

BlindDemodResult Modem::demodulate_blind(
    std::span<const double> x,
    std::optional<std::pair<double, double>> search_s,
    std::optional<sync::BlindAcquisition> acquisition,
    DriftTrack drift_track) const {
    std::vector<cdouble> z = dsp::to_baseband(x);

    sync::BlindAcquisition ba{};
    if (acquisition) {
        ba = *acquisition;
    } else {
        std::optional<sync::SearchWindow> search;
        if (search_s)
            search = sync::SearchWindow{static_cast<std::int64_t>(search_s->first * FS),
                                        static_cast<std::int64_t>(search_s->second * FS)};
        ba = sync::acquire_blind(z, config::BLIND_MAX_OFFSET_HZ,
                                 config::BLIND_BIN_STEP_HZ, 8, 4.0, search);
    }
    z = dsp::freq_correct(z, ba.freq_offset);

    const std::int64_t p0 = ba.frame_start - NCP;  // CP-start of local frame 0
    const std::int64_t L_lo = static_cast<std::int64_t>(
        std::ceil(-static_cast<double>(p0) / FRAME_SAMPLES));
    const std::int64_t L_hi = static_cast<std::int64_t>(std::floor(
        (static_cast<double>(z.size()) - FRAME_SAMPLES - static_cast<double>(p0)) /
        FRAME_SAMPLES));
    if (L_lo > L_hi)
        throw SyncError("blind lock too close to buffer edge to demod any full frame");
    int n_f = static_cast<int>(L_hi - L_lo + 1);
    std::int64_t p_start = p0 + L_lo * FRAME_SAMPLES;

    auto frames = [&](std::int64_t p) {
        std::vector<cdouble> raw(static_cast<std::size_t>(n_f) * SYMS_PER_FRAME * NC,
                                 cdouble{});
        std::vector<cdouble> h_pilot(static_cast<std::size_t>(n_f) * NC, cdouble{});
        auto tracker = make_tracker(drift_track);
        std::vector<double> pilot_powers;
        for (int f = 0; f < n_f; ++f) {
            const std::size_t fbase = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
            if (!tracker) {
                for (int s = 0; s < SYMS_PER_FRAME; ++s) {
                    const auto sym = ofdm::demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF);
                    std::copy(sym.begin(), sym.end(),
                              raw.begin() + static_cast<std::ptrdiff_t>(
                                                fbase + static_cast<std::size_t>(s) * NC));
                }
            } else {
                const std::vector<cdouble> zz = tracker->frame(z, p);
                for (int s = 0; s < SYMS_PER_FRAME; ++s) {
                    const auto sym = ofdm::demod_window(zz, s * NSYM + NCP, DEMOD_BACKOFF);
                    std::copy(sym.begin(), sym.end(),
                              raw.begin() + static_cast<std::ptrdiff_t>(
                                                fbase + static_cast<std::size_t>(s) * NC));
                }
            }
            for (int k = 0; k < NC; ++k) {
                const std::size_t i = static_cast<std::size_t>(k);
                h_pilot[static_cast<std::size_t>(f) * NC + i] = raw[fbase + i] / pilot_[i];
            }
            if (tracker) {
                // Most of this range is usually not the transmission at all
                // (silence or noise around it -- see the med_h comment
                // below), so the loop must not integrate phase out of noise
                // frames. Same health test the preamble path uses.
                double power = 0.0;
                for (int k = 0; k < NC; ++k)
                    power += std::norm(raw[fbase + static_cast<std::size_t>(k)]);
                power /= NC;
                pilot_powers.push_back(power);
                const bool usable = f > 0 && power > 0.1 * median(pilot_powers);
                tracker->update(
                    std::span<const cdouble>(h_pilot.data() + static_cast<std::size_t>(f) * NC, NC),
                    usable ? std::span<const cdouble>(
                                 h_pilot.data() + static_cast<std::size_t>(f - 1) * NC, NC)
                           : std::span<const cdouble>{});
            }
            p += FRAME_SAMPLES;
        }
        return std::pair{std::move(raw), std::move(h_pilot)};
    };
    auto first_pass = frames(p_start);
    std::vector<cdouble> raw = std::move(first_pass.first);
    std::vector<cdouble> h_pilot = std::move(first_pass.second);
    {
        // Same placement as the preamble path, from the frames that are
        // plausibly the transmission. A frame the moved window would take
        // past either end of the buffer is dropped rather than the
        // placement skipped; see demodulate_blind in modem.py.
        const std::vector<double> pw = frame_power(h_pilot);
        const double pw_max = *std::max_element(pw.begin(), pw.end());
        if (pw_max > 0) {
            const std::vector<char> plausible = transmission_frames(h_pilot);
            std::vector<cdouble> rows;
            for (std::size_t f = 0; f < pw.size(); ++f)
                if (plausible[f])
                    rows.insert(rows.end(),
                                h_pilot.begin() + static_cast<std::ptrdiff_t>(f * NC),
                                h_pilot.begin() + static_cast<std::ptrdiff_t>((f + 1) * NC));
            const int shift = window_shift(delay_support(rows));
            std::int64_t lo = p_start;
            int n = n_f;
            if (lo + shift + NCP - DEMOD_BACKOFF < 0) {  // its first demod window
                lo += FRAME_SAMPLES;
                --n;
            }
            if (lo + shift + std::int64_t{n} * FRAME_SAMPLES > static_cast<std::int64_t>(z.size())) --n;
            if (shift != 0 && n > 0) {
                p_start = lo;
                n_f = n;
                std::tie(raw, h_pilot) = frames(p_start + shift);
            }
        }
    }

    // Blind demod always covers every frame the *whole current buffer*
    // can hold, since the transmission's true length is unknown until
    // the beacon resolves it -- unlike demodulate() above, which
    // restricts this same computation to the header's known real frame
    // count via `received`. Most of that range is often not the real
    // transmission at all (silence or noise before it starts, or
    // accumulating after it ends, while the caller waits to see whether
    // a longer mode is still arriving) -- a straight median over the
    // *whole* range describes "typical", which is the noise floor
    // whenever noise frames are the numerical majority, and noise then
    // reads as fully trustworthy (weight ~1) right alongside real
    // frames instead of being down-weighted. Anchoring instead on
    // frames within an order of magnitude of the strongest ones seen
    // needs only a few genuinely real frames to set the right
    // reference, regardless of how much silence surrounds them; a real
    // (even faded) frame is never excluded by this on its own account,
    // since a *minority* of low-|h| frames barely moves a median in the
    // first place. Mirrors sstvae/modem/modem.py's demodulate_blind.
    std::vector<double> mags;
    mags.reserve(h_pilot.size());
    for (const cdouble& v : h_pilot) mags.push_back(std::abs(v));
    const double peak_h = mags.empty()
        ? 0.0
        : *std::max_element(mags.begin(), mags.end());
    std::vector<double> plausible_mags;
    plausible_mags.reserve(mags.size());
    for (double m : mags)
        if (m > 0.1 * peak_h) plausible_mags.push_back(m);
    const double med_h = plausible_mags.empty() ? 1.0 : median(plausible_mags);
    const double floor = std::max(0.05 * med_h, 1e-9);

    // Channel statistics from the frames that are plausibly the
    // transmission; see demodulate_blind in modem.py.
    std::vector<cdouble> h_est;
    if (std::any_of(h_pilot.begin(), h_pilot.end(), [](cdouble v) { return v != cdouble{}; })) {
        const std::vector<char> plausible = transmission_frames(h_pilot);
        h_est = lmmse_channel(h_pilot, {}, &plausible);
    } else {
        h_est.assign(static_cast<std::size_t>(n_f) * DATA_SYMS * NC, cdouble{});
    }

    std::vector<double> beacon_soft(static_cast<std::size_t>(n_f) * CHIPS_PER_FRAME, 0.0);
    std::vector<double> slot_values(
        static_cast<std::size_t>(n_f) * LATENTS_PER_FRAME, 0.0);
    std::vector<double> slot_weights(
        static_cast<std::size_t>(n_f) * LATENTS_PER_FRAME, 0.0);
    for (int f = 0; f < n_f; ++f) {
        const std::size_t fbase = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
        for (int s = 1; s < SYMS_PER_FRAME; ++s) {
            const std::span<const cdouble> h(
                h_est.data() + (static_cast<std::size_t>(f) * DATA_SYMS + static_cast<std::size_t>(s - 1)) * NC, NC);
            const std::span<const cdouble> raw_sym(
                raw.data() + fbase + static_cast<std::size_t>(s) * NC, NC);
            const EqualizedSymbol eq = equalize(raw_sym, h, floor, med_h);
            const std::size_t lo = static_cast<std::size_t>(f) * LATENTS_PER_FRAME +
                                   static_cast<std::size_t>(s - 1) * NC_LATENT * 2;
            for (std::size_t i = 0; i < eq.slots.size(); ++i) {
                slot_values[lo + i] = eq.slots[i];
                slot_weights[lo + i] = eq.weights[i];
            }
            beacon_soft[static_cast<std::size_t>(f) * CHIPS_PER_FRAME +
                        static_cast<std::size_t>(s - 1)] = eq.beacon_chip;
        }
    }

    const auto beacon_result = beacon::decode(beacon_soft);
    const ModeSpec& mode_c = config::MODES[config::N_MODES - 1];
    std::vector<double> latents_full(static_cast<std::size_t>(mode_c.n_latents), 0.0);
    std::vector<double> weights_full(static_cast<std::size_t>(mode_c.n_latents), 0.0);

    std::optional<int> frame_offset;
    std::optional<std::int64_t> frame0_start;
    if (beacon_result) {
        frame_offset = beacon_result->frame_index -
                       static_cast<int>(beacon_result->chip_offset / CHIPS_PER_FRAME);
        // The beacon's mode field bounds which absolute frames can be
        // real: everything past the transmission's actual last frame is
        // post-transmission noise, and placing it would hand the decoder
        // garbage latents at nonzero weight where a true erasure
        // (weight 0) is what it was trained for. An unknown mode index
        // (a future mode) falls back to mode C's full range.
        int n_frames_limit = LATENT_GROUPS * FRAMES_PER_GROUP;
        if (beacon_result->mode_index >= 0 &&
            beacon_result->mode_index < config::N_MODES)
            n_frames_limit =
                config::MODES[static_cast<std::size_t>(beacon_result->mode_index)]
                    .n_frames;
        for (int f = 0; f < n_f; ++f) {
            const int abs_frame = *frame_offset + f;
            if (abs_frame < 0 || abs_frame >= n_frames_limit) continue;
            const auto fs = framing::slot_range_for_frame(abs_frame);
            for (int i = 0; i < LATENTS_PER_FRAME; ++i) {
                const std::size_t dst =
                    static_cast<std::size_t>(fs.indices[static_cast<std::size_t>(i)]);
                const std::size_t src = static_cast<std::size_t>(f) * LATENTS_PER_FRAME +
                                        static_cast<std::size_t>(i);
                latents_full[dst] = std::clamp(slot_values[src], -10.0, 10.0);
                weights_full[dst] = slot_weights[src];
            }
        }
        // Anchored on p_start, not p0: the demod loop (and so the beacon
        // chip stream that frame_offset indexes) starts at p_start,
        // which is L_lo frames away from p0 whenever the blind lock is
        // not already at the buffer start. Using p0 put absolute frame 0
        // off by L_lo frames -- tens of seconds for a mid-stream lock.
        frame0_start = p_start - static_cast<std::int64_t>(*frame_offset) * FRAME_SAMPLES;
    }

    return BlindDemodResult{latents_full,
                            weights_full,
                            ba.freq_offset,
                            beacon_result,
                            beacon_result ? beacon_result->callsign : std::string(),
                            frame_offset,
                            n_f,
                            frame0_start,
                            estimate_snr_db(h_pilot, n_f)};
}

}  // namespace sstvae::modem
