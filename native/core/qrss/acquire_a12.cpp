// acquire.py's A1 and A2 stages and the lead-in: see acquire_a12.hpp.

#include "qrss/acquire_a12.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "math/convolve.hpp"
#include "math/errors.hpp"
#include "math/npcomplex.hpp"
#include "math/nplog.hpp"
#include "math/npstats.hpp"
#include "qrss/ce.hpp"
#include "qrss/frontend.hpp"
#include "qrss/filters.hpp"
#include "qrss/lwave_rx.hpp"
#include "qrss/npsum.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/sequences.hpp"

namespace sstvae::qrss::acquire {
namespace {

namespace np = sstvae::math::np;
namespace sm = sstvae::math;

constexpr double PI = 3.141592653589793;   // math.pi / np.pi
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();
// Python's -2j * np.pi and 2j * np.pi: complex(-0.0, -2.0) * pi is
// (-0.0*pi - (-2.0)*0.0, ...) = (+0.0, -2 pi) in Python 3.13's complex
// product, and 2j * pi is (+0.0, 2 pi).
const cd MINUS_2J_PI{0.0, -2.0 * PI};
const cd PLUS_2J_PI{0.0, 2.0 * PI};

// --- Python's integer conversions -------------------------------------------------

[[noreturn]] void bad_float(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    throw std::overflow_error("cannot convert float infinity to integer");
}

std::int64_t to_int64(double r) {
    if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0))
        throw std::overflow_error("Python int too large to convert to C int64");
    return static_cast<std::int64_t>(r);
}

// int(x) for a float (toward zero).
std::int64_t py_int(double x) {
    if (!std::isfinite(x)) bad_float(x);
    return to_int64(std::trunc(x));
}
// math.floor / math.ceil / round(x) (half to even) to an int.
std::int64_t py_floor(double x) {
    if (!std::isfinite(x)) bad_float(x);
    return to_int64(std::floor(x));
}
std::int64_t py_ceil(double x) {
    if (!std::isfinite(x)) bad_float(x);
    return to_int64(std::ceil(x));
}
std::int64_t py_round(double x) {
    if (!std::isfinite(x)) bad_float(x);
    return to_int64(std::nearbyint(x));
}

// np.round(v).astype(np.int64): INT64_MIN for what x86-64 cannot convert.
std::int64_t np_round_int(double v) {
    const double r = std::nearbyint(v);
    if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0)) return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(r);
}

// Python's a % b and a // b on ints.
std::int64_t py_mod(std::int64_t a, std::int64_t b) {
    if (b == 0) throw ZeroDivision("integer modulo by zero");
    std::int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
std::int64_t py_div(std::int64_t a, std::int64_t b) {
    if (b == 0) throw ZeroDivision("integer division or modulo by zero");
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

// Python's float floor division, vx // wx (CPython's float_floor_div).
double py_floordiv(double vx, double wx) {
    if (wx == 0.0) throw ZeroDivision("float floor division by zero");
    double mod = std::fmod(vx, wx);
    double div = (vx - mod) / wx;
    if (mod != 0.0) {
        if ((wx < 0) != (mod < 0)) div -= 1.0;
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, vx / wx);
    }
    return floordiv;
}

// numpy's IndexError for a[i] on an axis of n.
std::size_t index(std::int64_t i, std::size_t n, int axis = 0) {
    const auto ns = static_cast<std::int64_t>(n);
    if (i < -ns || i >= ns)
        throw std::out_of_range("index " + std::to_string(i) + " is out of bounds for axis " + std::to_string(axis) +
                                " with size " + std::to_string(n));
    return static_cast<std::size_t>(i < 0 ? i + ns : i);
}

// A Python slice [a:b] (step 1) of a sequence of n: the clamped [start, stop).
std::pair<std::size_t, std::size_t> py_slice(std::int64_t a, std::int64_t b, std::size_t n) {
    const auto ns = static_cast<std::int64_t>(n);
    auto fix = [&](std::int64_t v) {
        if (v < 0) v = std::max<std::int64_t>(v + ns, 0);
        return std::min(v, ns);
    };
    const std::int64_t s = fix(a), e = std::max(fix(b), fix(a));
    return {static_cast<std::size_t>(s), static_cast<std::size_t>(e)};
}

// Python's max(a, b): a unless b > a (so a NaN first stays).
double py_max(double a, double b) { return b > a ? b : a; }

// np.max / np.argmax / np.argmin over a vector, numpy's NaN rules.
double np_max(std::span<const double> v, const char* what = "maximum") {
    if (v.empty())
        throw std::invalid_argument(std::string("zero-size array to reduction operation ") + what +
                                    " which has no identity");
    double m = v[0];
    for (const double x : v) {
        if (std::isnan(x)) return x;
        if (x > m) m = x;
    }
    return m;
}

template <typename T>
std::size_t np_argmax(std::span<const T> v) {
    if (v.empty()) throw std::invalid_argument("attempt to get argmax of an empty sequence");
    std::size_t b = 0;
    T m = v[0];
    if (std::isnan(m)) return 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (std::isnan(v[i])) return i;
        if (v[i] > m) {
            m = v[i];
            b = i;
        }
    }
    return b;
}

template <typename T>
std::size_t np_argmin(std::span<const T> v) {
    if (v.empty()) throw std::invalid_argument("attempt to get argmin of an empty sequence");
    std::size_t b = 0;
    T m = v[0];
    if (std::isnan(m)) return 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (std::isnan(v[i])) return i;
        if (v[i] < m) {
            m = v[i];
            b = i;
        }
    }
    return b;
}

// int(np.argmin(np.abs(drifts))).
std::size_t straight_drift(const std::vector<double>& drifts) {
    std::vector<double> a(drifts.size());
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = std::fabs(drifts[i]);
    return np_argmin(std::span<const double>(a));
}

// np.exp of a complex argument: glibc's cexp, which numpy calls.
cd np_exp(cd z) { return std::exp(z); }

// np.exp(c * x) for a Python complex constant c times a real x promoted to
// (x, 0), as numpy's multiply forms it.
cd exp_of(cd c, double x) { return np_exp(np::mul(c, cd(x, 0.0))); }

// numpy's np.float32 maximum with a weak Python float (NaN propagates).
float max32(float x, float c) {
    if (std::isnan(x)) return x;
    return x >= c ? x : c;
}

// The vertex offset on numpy's float32 log of the statistic at b-1, b, b+1.
double log_parabola(std::span<const float> row, std::size_t b) {
    const float floor32 = static_cast<float>(1e-9);
    const float ym = np::log_f32(max32(row[b - 1], floor32));
    const float y0 = np::log_f32(max32(row[b], floor32));
    const float yp = np::log_f32(max32(row[b + 1], floor32));
    return parabola(ym, y0, yp);
}

// float(cs.f_hz[1] - cs.f_hz[0]), numpy's IndexError when short.
double bin_width(const CarrierStats& cs) {
    const double f1 = cs.f_hz[index(1, cs.f_hz.size())];
    return f1 - cs.f_hz[0];
}

// The A1 statistic of one cell, refined as a line (_cell_line's body).
types::CarrierCandidate line_of(const CarrierStats& cs, std::int64_t b_in, std::int64_t d_in, double v,
                                double exc_rel) {
    const std::size_t nd = cs.stat.rows, nb = cs.stat.cols;
    const std::size_t d0 = straight_drift(cs.drifts);
    const std::size_t b = index(b_in, nb, 1);
    std::size_t d = index(d_in, nd);
    if (v - static_cast<double>(cs.stat(index(static_cast<std::int64_t>(d0), nd), b)) < exc_rel) d = d0;
    const std::span<const float> row = cs.stat.row(index(static_cast<std::int64_t>(d), nd));
    double off = 0.0;
    if (0 < b_in && b_in < static_cast<std::int64_t>(nb) - 1) off = log_parabola(row, static_cast<std::size_t>(b_in));
    types::CarrierCandidate c;
    const double df = bin_width(cs);
    c.f_hz = cs.f_hz[index(b_in, cs.f_hz.size())] + off * df;
    c.drift_hz_per_min = cs.drifts[index(static_cast<std::int64_t>(d), cs.drifts.size())];
    c.metric = v;
    return c;
}

// --- A1 ------------------------------------------------------------------------------

template <typename S>
CarrierStats carrier_line_stats_impl(std::span<const S> fe, double t0_index, double fs, std::array<double, 2> span_s,
                                     std::span<const double> drifts_in, std::array<double, 2> band_hz) {
    const std::int64_t L = frame_len(fs);
    const double hop = rxa::STFT_HOP_S;
    const std::int64_t M = py_int(py_floordiv(span_s[1] - span_s[0] - rxa::STFT_FRAME_S, hop)) + 1;
    const std::size_t Mn = M > 0 ? static_cast<std::size_t>(M) : 0;
    std::vector<double> t_start(Mn), t_mid(Mn);
    for (std::size_t k = 0; k < Mn; ++k) {
        t_start[k] = span_s[0] + hop * static_cast<double>(k);
        t_mid[k] = t_start[k] + rxa::STFT_FRAME_S / 2;
    }
    if (L == 0) throw ZeroDivision("float division by zero");
    const double df = fs / static_cast<double>(L);
    std::vector<double> drifts(drifts_in.begin(), drifts_in.end());
    // pad = int(ceil(max|drift| / 60 * max|t_mid| / df)) + 2
    std::vector<double> ad(drifts.size()), at(t_mid.size());
    for (std::size_t i = 0; i < ad.size(); ++i) ad[i] = std::fabs(drifts[i]);
    for (std::size_t i = 0; i < at.size(); ++i) at[i] = std::fabs(t_mid[i]);
    const double dmax = np_max(ad);
    const double tmax = np_max(at);
    const std::int64_t pad = py_ceil(dmax / 60.0 * tmax / df) + 2;
    const BandBins bb = band_bins(fs, band_hz);
    const std::size_t nsel = bb.sel.size();
    if (nsel == 0) throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    const std::int64_t lo = py_round((bb.f_hz.front() - config::FE_CENTER_HZ) / df) - pad;
    const std::int64_t hi = py_round((bb.f_hz.back() - config::FE_CENTER_HZ) / df) + pad;
    std::vector<std::int64_t> wide;
    for (std::int64_t v = lo; v <= hi; ++v) wide.push_back(py_mod(v, L));
    std::vector<double> starts(Mn);
    for (std::size_t k = 0; k < Mn; ++k) starts[k] = t0_index + t_start[k] * fs;
    const Grid32 P = spectrogram(fe, fs, starts, std::span<const std::int64_t>(wide));
    const std::size_t nw = wide.size();
    // med = np.median(P[:, inner], axis=1) / LN2, float32
    const auto [i0, i1] = py_slice(pad, pad + static_cast<std::int64_t>(nsel), nw);
    std::vector<float> med(Mn);
    std::vector<std::uint8_t> live(Mn);
    for (std::size_t k = 0; k < Mn; ++k) {
        const std::span<const float> row = P.row(k);
        med[k] = sm::median(row.subspan(i0, i1 - i0)) / static_cast<float>(rxa::LN2);
        live[k] = med[k] > 0;
    }
    Grid32 E(Mn, nw);
    for (std::size_t k = 0; k < Mn; ++k)
        for (std::size_t j = 0; j < nw; ++j) E(k, j) = live[k] ? P(k, j) / med[k] : 0.0f;
    const std::vector<double> sh = spectral_shape(E, rxa::CFAR_LOCAL_BINS, std::span<const std::uint8_t>(live));
    for (std::size_t k = 0; k < Mn; ++k)
        for (std::size_t j = 0; j < nw; ++j) E(k, j) = static_cast<float>(static_cast<double>(E(k, j)) / sh[j]);
    CarrierStats cs;
    cs.stat = Grid32(drifts.size(), nsel);
    for (std::size_t i = 0; i < drifts.size(); ++i) {
        const double d = drifts[i];
        float* out = cs.stat.data.data() + i * nsel;
        for (std::size_t k = 0; k < Mn; ++k) {
            const std::int64_t shift = np_round_int(d / 60.0 * t_mid[k] / df);
            const auto [a, b] = py_slice(pad + shift, pad + shift + static_cast<std::int64_t>(nsel), nw);
            const std::span<const float> src = E.row(k).subspan(a, b - a);
            if (src.size() == nsel) {
                for (std::size_t j = 0; j < nsel; ++j) out[j] = out[j] + src[j];
            } else if (src.size() == 1) {
                for (std::size_t j = 0; j < nsel; ++j) out[j] = out[j] + src[0];
            } else {
                throw std::invalid_argument("operands could not be broadcast together with shapes (" +
                                            std::to_string(nsel) + ",) (" + std::to_string(src.size()) + ",) (" +
                                            std::to_string(nsel) + ",) ");
            }
        }
    }
    cs.f_hz = bb.f_hz;
    cs.drifts = std::move(drifts);
    cs.M = M;
    return cs;
}

// --- A2: the channel -------------------------------------------------------------

// The FE samples [i_a, i_b) as complex64, zero outside fe (the reference's
// `seg`). n < 0 is numpy's refusal.
template <typename S>
std::vector<cf> segment(std::span<const S> fe, std::int64_t i_a, std::int64_t i_b) {
    const std::int64_t n = i_b - i_a;
    if (n < 0) throw std::invalid_argument("negative dimensions are not allowed");
    std::vector<cf> seg(static_cast<std::size_t>(n), cf(0.0f, 0.0f));
    const std::int64_t a = std::max<std::int64_t>(i_a, 0);
    const std::int64_t b = std::min<std::int64_t>(i_b, static_cast<std::int64_t>(fe.size()));
    for (std::int64_t i = a; i < b; ++i) {
        const S v = fe[static_cast<std::size_t>(i)];
        seg[static_cast<std::size_t>(i - i_a)] = cf(static_cast<float>(v.real()), static_cast<float>(v.imag()));
    }
    return seg;
}

std::int64_t decimation(double fs) { return py_div(py_int(fs), config::CH_FS); }

template <typename S>
A2Input a2_input_impl(std::span<const S> fe, double t0_index, double f_audio_hz, double drift, double fs) {
    const std::int64_t decim = decimation(fs);
    const double lo_s = -rxa::A2_TAU_S - rxa::A2_MARGIN_S_;
    const double hi_s = rxa::A2_TAU_S + config::N_PRE * config::T_SYM + rxa::A2_MARGIN_S_;
    std::int64_t i_a = py_floor(t0_index + lo_s * fs);
    i_a -= py_mod(i_a, decim);
    const std::int64_t i_b = py_ceil(t0_index + hi_s * fs);
    const std::vector<cf> seg = segment(fe, i_a, i_b);
    frontend::Channelised chd =
        frontend::channelise(std::span<const cf>(seg), py_int(fs), frontend::fe_hz(f_audio_hz), config::CH_FS, i_a);
    A2Input out;
    out.t0_ch = (t0_index - static_cast<double>(i_a)) / static_cast<double>(decim);
    out.f_mix = chd.f_mix;
    std::vector<cd> y(chd.ch.size());
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = cd(chd.ch[i].real(), chd.ch[i].imag());
    const double t0_ch = out.t0_ch;
    auto t_of = [&](std::size_t i) { return (static_cast<double>(i) - t0_ch) / config::CH_FS; };
    if (drift != 0.0 || std::isnan(drift)) {
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double t = t_of(i);
            const double c = drift / 120.0 * t * t;
            y[i] = np::mul(y[i], exp_of(MINUS_2J_PI, c - std::floor(c)));
        }
    }
    const double resid = frontend::fe_hz(f_audio_hz) - chd.f_mix.to_double();
    if (resid != 0.0 || std::isnan(resid)) {
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double c = resid * t_of(i);
            y[i] = np::mul(y[i], exp_of(MINUS_2J_PI, c - std::floor(c)));
        }
    }
    const std::vector<cd> lp = sm::oaconvolve(std::span<const cd>(y), std::span<const double>(notch_taps()),
                                              sm::ConvMode::Same);
    out.y.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) out.y[i] = y[i] - lp[i];
    return out;
}

// --- A2: the search ------------------------------------------------------------

// np.angle: arctan2(imag, real). numpy's own SIMD arctan2 rounds
// differently from libm's in the last bit for ~7% of arguments.
double np_angle(cd z) { return std::atan2(z.imag(), z.real()); }

// (k, Lambda_k, p, best) at the chosen cell, and the cell.
struct Cell {
    std::size_t i = 0, j = 0;
};

PreambleHit search(const A2Input& inp, const types::CarrierCandidate& cand, double pfa, const PreambleKind& kind) {
    const A2Channel chan = a2_mf(inp, kind);
    const Sums s = preamble_sums(chan);
    const PreambleP pp = preamble_p(s.S, s.n_df, s.n_tau, s.nc, s.var, rxa::A2_SCALES, SumOrder::Sequential);
    // pc.min(): NaN if any is NaN
    if (pp.p.empty()) throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
    double pmin = pp.p[0];
    for (const double v : pp.p) {
        if (std::isnan(v)) {
            pmin = v;
            break;
        }
        pmin = std::min(pmin, v);
    }
    std::size_t flat;
    if (pmin > 1e-250) {
        flat = np_argmin(std::span<const double>(pp.p));
    } else {   // p underflows on a strong signal
        flat = np_argmax(std::span<const double>(s.lam));
    }
    const std::size_t i = flat / s.n_tau, j = flat % s.n_tau;
    const Scale& sc = pp.st[static_cast<std::size_t>(pp.best[flat])];
    const std::vector<double>& lam = sc.lam;
    auto L = [&](std::size_t a, std::size_t b) { return lam[a * s.n_tau + b]; };
    const double cell = L(i, j);
    const auto& df_grid = rxa::A2_DF_HZ;
    const double ddf = df_grid[1] - df_grid[0];
    double df = df_grid[i];
    if (0 < i && i < df_grid.size() - 1) df += parabola(L(i - 1, j), L(i, j), L(i + 1, j)) * ddf;
    double tau = s.taus[j];
    if (0 < j && j < s.taus.size() - 1) tau += parabola(L(i, j - 1), L(i, j), L(i, j + 1));

    const auto nm = static_cast<std::int64_t>(chan.m.size());
    const double n_seq = static_cast<double>(kind.seq.size());
    const std::int64_t lo = std::max<std::int64_t>(0, py_floor(chan.t0_ch + tau - 80));
    const std::int64_t hi =
        std::min<std::int64_t>(nm, py_ceil(chan.t0_ch + tau + n_seq * kind.t_sym * config::CH_FS + 80));
    std::vector<double> xs;
    std::vector<cd> ys;
    for (std::int64_t k = lo; k < hi; ++k) {
        xs.push_back(static_cast<double>(k));
        ys.push_back(chan.m[static_cast<std::size_t>(k)]);
    }
    const sm::CubicSpline spline_obj(xs, ys);
    const SplineFn spline = spline_fn(spline_obj);
    const double dt = static_cast<double>(kind.chunk) * kind.t_sym;
    double rate = 0.0;
    for (int it = 0; it < 2; ++it) {
        const auto [Sc, n_] = fine_sums(spline, chan, tau, df, rate);
        (void)n_;
        const auto [a, b, sd_b] = phase_line(Sc, dt);
        if (std::fabs(b) > rxa::A2_DRIFT_SIGMAS * sd_b) {
            df += a;
            rate += b;
        } else {
            std::vector<cd> pr(Sc.size() > 0 ? Sc.size() - 1 : 0);
            for (std::size_t c = 0; c + 1 < Sc.size(); ++c) pr[c] = np::mul(Sc[c + 1], std::conj(Sc[c]));
            df += np_angle(npsum::sum(std::span<const cd>(pr))) / (2 * PI * dt);
        }
        for (const double d : {0.25, 0.1, 0.05}) {
            const double ym = coherent(spline, chan, tau + -d, df, rate);
            const double y0 = coherent(spline, chan, tau + 0.0, df, rate);
            const double yp = coherent(spline, chan, tau + d, df, rate);
            tau += parabola(ym, y0, yp) * d;
        }
    }
    const auto [Sc, nrm] = fine_sums(spline, chan, tau, df, rate);
    if (!chan.local) throw sm::TypeError("'NoneType' object is not subscriptable");
    const std::vector<double>& local = *chan.local;
    std::vector<double> terms(nrm.size());
    for (std::size_t c = 0; c < nrm.size(); ++c) {
        const double cen = chan.t0_ch + tau +
                           (static_cast<double>(c) + 0.5) * static_cast<double>(kind.chunk) * kind.t_sym * config::CH_FS;
        if (local.empty()) throw std::out_of_range("index -1 is out of bounds for axis 0 with size 0");
        std::int64_t k = np_round_int(cen);
        k = std::clamp<std::int64_t>(k, 0, static_cast<std::int64_t>(local.size()) - 1);
        const double sig = local[static_cast<std::size_t>(k)];
        const double a = np::abs(Sc[c]);
        terms[c] = (a * a) / (nrm[c] * sig);
    }
    const double lam_f = npsum::sum(std::span<const double>(terms));
    const cd total = npsum::sum(std::span<const cd>(Sc));
    double phase;
    if (kind.data_offset_hz == 0.0) {
        phase = np_angle(total) + kind.phase_offset;
    } else {
        const double tau_s = tau / config::CH_FS;
        const double turns = (df - kind.data_offset_hz) * tau_s - 0.5 * rate * tau_s * tau_s;
        const double w = turns - std::floor(turns);
        // Python complex (0, -2 pi) * w (the complex product, w promoted),
        // np.exp, then numpy's scalar (unfused) complex multiply.
        const cd arg{MINUS_2J_PI.real() * w - MINUS_2J_PI.imag() * 0.0,
                     MINUS_2J_PI.real() * 0.0 + MINUS_2J_PI.imag() * w};
        const cd e = np_exp(arg);
        const cd prod{total.real() * e.real() - total.imag() * e.imag(),
                      total.real() * e.imag() + total.imag() * e.real()};
        phase = np_angle(prod) + kind.phase_offset;
    }
    // timing sd from the coherent statistic's curvature
    const double y0 = coherent(spline, chan, tau + -0.5, df, rate);
    const double y1 = coherent(spline, chan, tau + 0.0, df, rate);
    const double y2 = coherent(spline, chan, tau + 0.5, df, rate);
    const double curv = -(y0 - 2 * y1 + y2) / 0.25;
    double sd = NaN;
    if (curv > 0) {
        const double q = 1.0 / curv;
        if (q < 0) throw std::invalid_argument("math domain error");
        sd = std::sqrt(q) / config::CH_FS;
    }
    const double p_cell = pp.p[flat];
    PreambleHit h;
    h.f_hz = cand.f_hz + df;
    h.drift_hz_per_min = cand.drift_hz_per_min + 60.0 * rate;
    h.tau0_s = tau / config::CH_FS;
    h.phase = phase;
    h.lam = lam_f;
    h.cell_lam = cell;
    h.p_value = p_cell;
    h.passed = p_cell < pfa;
    h.n_groups = sc.k;
    h.f_mix = chan.f_mix;
    h.tau0_sd_s = sd;
    h.chunk_sums = Sc;
    h.waveform = kind.waveform;
    return h;
}

// --- lead-in --------------------------------------------------------------------------

template <typename S>
double lead_in_impl(std::span<const S> fe, double t0_index, double f_hz, double drift, double tau0_s, double fs,
                    double z_min, double t_sym, const NoisePsdFn& noise_psd) {
    const double end = tau0_s - config::SPAN * t_sym;
    const double lo_s = end - config::LEAD_IN_MAX_S - 8.0;
    const std::int64_t decim = decimation(fs);
    std::int64_t i_a = py_floor(t0_index + lo_s * fs);
    i_a -= py_mod(i_a, decim);
    const std::int64_t i_b = py_ceil(t0_index + (end + 1.0) * fs);
    const std::vector<cf> seg = segment(fe, i_a, i_b);
    const frontend::Channelised chd =
        frontend::channelise(std::span<const cf>(seg), py_int(fs), frontend::fe_hz(f_hz), config::CH_FS, i_a);
    const std::vector<cf>& ch = chd.ch;
    const std::size_t n = ch.size();
    std::vector<double> t(n);
    for (std::size_t i = 0; i < n; ++i)
        t[i] = (static_cast<double>(static_cast<std::int64_t>(i) * decim + i_a) - t0_index) / fs;
    const double resid = frontend::fe_hz(f_hz) - chd.f_mix.to_double();
    std::vector<cd> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double turns = resid * t[i] + drift / 120.0 * t[i] * t[i];
        x[i] = np::mul(cd(ch[i].real(), ch[i].imag()), exp_of(MINUS_2J_PI, turns - std::floor(turns)));
    }
    const std::vector<double> taps = mf_taps();
    const std::vector<cd> m = sm::oaconvolve(std::span<const cd>(x), std::span<const double>(taps), sm::ConvMode::Same);
    const double sps = config::CH_FS * config::T_SYM;
    std::vector<cf> have;
    for (const cf& v : ch)
        if (np::abs(v) > 0.0f) have.push_back(v);
    double sigma2;
    if (static_cast<std::int64_t>(have.size()) > 2 * config::CH_FS) {
        const double audio_hz = config::FE_CENTER_HZ + chd.f_mix.to_double();
        std::vector<float> psd;
        if (noise_psd) {
            psd = noise_psd(std::span<const cf>(have), config::CH_FS, py_int(fs), audio_hz);
        } else {
            frontend::NoisePsdOptions opt;
            opt.fs_in = py_int(fs);
            opt.audio_hz = audio_hz;
            psd = frontend::noise_psd(std::span<const cf>(have), config::CH_FS, {}, 10.0, opt);
        }
        sigma2 = py_max(static_cast<double>(sm::median(std::span<const float>(psd))) / config::T_SYM, 1e-30);
    } else {
        sigma2 = py_max(1.0 / config::T_SYM, 1e-30);
    }
    double span = 0.0;
    const std::int64_t n_win = py_round(config::LEAD_IN_MAX_S / rxa::LEAD_WIN_S);
    std::vector<cd> sel;
    for (std::int64_t k = 0; k < n_win; ++k) {
        const double a = end - static_cast<double>(k + 1) * rxa::LEAD_WIN_S;
        const double b = end - static_cast<double>(k) * rxa::LEAD_WIN_S;
        sel.clear();
        for (std::size_t i = 0; i < n; ++i)
            if (t[i] >= a && t[i] < b) sel.push_back(m[i]);
        const double n_ind = static_cast<double>(sel.size()) / sps;
        const double q = py_max(n_ind, 1.0) * sigma2 / 2;
        if (q < 0) throw std::invalid_argument("math domain error");
        const double z = np::abs(npsum::sum(std::span<const cd>(sel))) / sps / std::sqrt(q);
        if (z <= z_min) break;
        span = static_cast<double>(k + 1) * rxa::LEAD_WIN_S;
    }
    return span;
}

}  // namespace

// --- A1 ---------------------------------------------------------------------------------

CarrierStats carrier_line_stats(std::span<const cf> fe, double t0_index, double fs, std::array<double, 2> span_s,
                                std::span<const double> drifts, std::array<double, 2> band_hz) {
    return carrier_line_stats_impl(fe, t0_index, fs, span_s, drifts, band_hz);
}
CarrierStats carrier_line_stats(std::span<const cd> fe, double t0_index, double fs, std::array<double, 2> span_s,
                                std::span<const double> drifts, std::array<double, 2> band_hz) {
    return carrier_line_stats_impl(fe, t0_index, fs, span_s, drifts, band_hz);
}

std::vector<types::CarrierCandidate> carrier_lines(const CarrierStats& cs, double fs, double pfa,
                                                   std::int64_t max_cands, double min_sep_hz) {
    const double thr = a1_threshold(cs.M, pfa, fs);
    const std::size_t nd = cs.stat.rows, nb = cs.stat.cols;
    if (nd == 0 && nb > 0) throw std::invalid_argument("attempt to get argmax of an empty sequence");
    std::vector<std::size_t> best_d(nb, 0);
    std::vector<float> best(nb);
    std::vector<float> col(nd);
    for (std::size_t b = 0; b < nb; ++b) {
        for (std::size_t d = 0; d < nd; ++d) col[d] = cs.stat(d, b);
        best_d[b] = np_argmax(std::span<const float>(col));
        best[b] = cs.stat(best_d[b], b);
    }
    const double df = bin_width(cs);
    // np.argsort(best)[::-1]: descending, NaN first (numpy sorts NaN last).
    // Ties (not stable in numpy) are taken highest bin first.
    std::vector<std::size_t> order(nb);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const float x = best[a], y = best[b];
        if (std::isnan(y)) return !std::isnan(x);
        if (std::isnan(x)) return false;
        return x < y;
    });
    std::reverse(order.begin(), order.end());
    std::vector<types::CarrierCandidate> out;
    std::vector<double> picked;
    for (const std::size_t b : order) {
        const double v = static_cast<double>(best[b]);
        if (v <= thr || static_cast<std::int64_t>(out.size()) >= max_cands) break;
        const double f0 = cs.f_hz[index(static_cast<std::int64_t>(b), cs.f_hz.size())];
        const double exc = v - static_cast<double>(cs.M);
        bool near = false;
        for (const double f : picked)
            if (std::fabs(f0 - f) < min_sep_hz) {
                near = true;
                break;
            }
        if (near) continue;
        std::size_t d = best_d[b];
        const std::size_t d0 = straight_drift(cs.drifts);
        if (v - static_cast<double>(cs.stat(index(static_cast<std::int64_t>(d0), nd), b)) < rxa::A1_DRIFT_MARGIN * exc)
            d = d0;
        const std::span<const float> row = cs.stat.row(d);
        double off = 0.0;
        if (0 < b && b < row.size() - 1) off = log_parabola(row, b);
        picked.push_back(f0);
        types::CarrierCandidate c;
        c.f_hz = f0 + off * df;
        c.drift_hz_per_min = cs.drifts[index(static_cast<std::int64_t>(d), cs.drifts.size())];
        c.metric = v;
        out.push_back(c);
    }
    return out;
}

types::CarrierCandidate hint_line(const CarrierStats& cs, double f_hz, double fs, double pfa, double span_hz,
                                  bool* refined) {
    if (refined != nullptr) *refined = false;
    std::vector<std::size_t> near;
    for (std::size_t i = 0; i < cs.f_hz.size(); ++i)
        if (std::fabs(cs.f_hz[i] - f_hz) <= span_hz) near.push_back(i);
    types::CarrierCandidate c;
    if (near.empty()) {
        c.f_hz = f_hz;
        return c;
    }
    const std::size_t nd = cs.stat.rows, nk = near.size();
    std::vector<float> sub(nd * nk);
    for (std::size_t d = 0; d < nd; ++d)
        for (std::size_t k = 0; k < nk; ++k) sub[d * nk + k] = cs.stat(d, index(static_cast<std::int64_t>(near[k]), cs.stat.cols, 1));
    const std::size_t flat = np_argmax(std::span<const float>(sub));
    const std::size_t d = flat / nk, k = flat % nk;
    const auto b = static_cast<std::int64_t>(near[k]);
    const double v = static_cast<double>(sub[flat]);
    if (v <= a1_threshold(cs.M, pfa, fs)) {
        c.f_hz = f_hz;
        c.metric = v;
        return c;
    }
    if (refined != nullptr) *refined = true;
    return cell_line(cs, b, static_cast<std::int64_t>(d), v);
}

types::CarrierCandidate cell_line(const CarrierStats& cs, std::int64_t b, std::int64_t d, double v) {
    return line_of(cs, b, d, v, rxa::A1_DRIFT_MARGIN * (v - static_cast<double>(cs.M)));
}

std::vector<types::CarrierCandidate> pilot_lines(const CarrierStats& cs, std::span<const types::CarrierCandidate> lines,
                                                 double sep_hz) {
    const double lo_hz = rxa::L_OWN_HZ[0], hi_hz = rxa::L_OWN_HZ[1];
    std::vector<double> taken;
    for (const auto& c : lines) taken.push_back(c.f_hz);
    std::vector<types::CarrierCandidate> out;
    const std::size_t nd = cs.stat.rows;
    for (const auto& c : lines) {
        std::vector<std::size_t> sel;
        for (std::size_t i = 0; i < cs.f_hz.size(); ++i)
            if (cs.f_hz[i] > c.f_hz + lo_hz && cs.f_hz[i] <= c.f_hz + hi_hz) sel.push_back(i);
        if (sel.empty()) continue;
        const std::size_t nk = sel.size();
        std::vector<float> sub(nd * nk);
        for (std::size_t d = 0; d < nd; ++d)
            for (std::size_t k = 0; k < nk; ++k) sub[d * nk + k] = cs.stat(d, index(static_cast<std::int64_t>(sel[k]), cs.stat.cols, 1));
        const std::size_t flat = np_argmax(std::span<const float>(sub));
        const std::size_t d = flat / nk, k = flat % nk;
        const types::CarrierCandidate h = cell_line(cs, static_cast<std::int64_t>(sel[k]), static_cast<std::int64_t>(d),
                                                    static_cast<double>(sub[flat]));
        bool dup = false;
        for (const double f : taken)
            if (std::fabs(h.f_hz - f) <= sep_hz) {
                dup = true;
                break;
            }
        if (dup) continue;
        taken.push_back(h.f_hz);
        out.push_back(h);
    }
    return out;
}

// --- A2: filters and kinds -------------------------------------------------------

const std::vector<double>& notch_taps() {
    static const std::vector<double> taps =
        filters::firwin(rxa::NOTCH_TAPS, rxa::NOTCH_HZ, filters::Window::hann(), filters::PassZero::Yes, true,
                        static_cast<double>(config::CH_FS));
    return taps;
}

namespace {

std::vector<double> compute_mf_taps(double fs) {
    const double sps = fs * config::T_SYM;
    const std::int64_t K = py_int(8 * sps);
    std::vector<double> out;
    for (std::int64_t k = -K; k <= K; ++k) out.push_back(ce::pulse(static_cast<double>(k) / sps) / sps);
    return out;
}

std::vector<double> compute_l_mf_taps(double fs) {
    const double sps = fs * config::L_T_SYM;
    const std::int64_t K = py_int(config::SPAN * sps);
    std::vector<double> out;
    for (std::int64_t k = -K; k <= K; ++k) {
        const double x = static_cast<double>(k) / sps;
        const double v = std::fabs(x) <= config::SPAN ? ce::rrc(x) / lwave_rx::PULSE_NORM : 0.0;
        out.push_back(v / sps);
    }
    return out;
}

}  // namespace

std::vector<double> mf_taps(double fs) {
    if (fs == static_cast<double>(config::CH_FS)) {
        static const std::vector<double> taps = compute_mf_taps(config::CH_FS);
        return taps;
    }
    return compute_mf_taps(fs);
}

std::vector<double> l_mf_taps(double fs) {
    if (fs == static_cast<double>(config::CH_FS)) {
        static const std::vector<double> taps = compute_l_mf_taps(config::CH_FS);
        return taps;
    }
    return compute_l_mf_taps(fs);
}

std::int64_t PreambleKind::n_chunks() const { return py_div(static_cast<std::int64_t>(seq.size()), chunk); }

std::string PreambleKind::name() const {
    if (waveform == config::WAVEFORM_CE) return "CE";
    if (waveform == config::WAVEFORM_L) return "L";
    throw pyjson::KeyError(std::to_string(waveform), "KeyError: " + std::to_string(waveform));
}

const PreambleKind& ce_pre() {
    static const PreambleKind k = [] {
        PreambleKind p;
        p.waveform = config::WAVEFORM_CE;
        for (const std::int8_t s : sequences::preamble_ce()) p.seq.emplace_back(static_cast<double>(s), 0.0);
        p.real = true;
        p.t_sym = config::T_SYM;
        p.chunk = rxa::A2_CHUNK;
        p.data_offset_hz = 0.0;
        p.phase_offset = -PI / 2;
        p.mf_taps = mf_taps();
        return p;
    }();
    return k;
}

const PreambleKind& l_pre() {
    static const PreambleKind k = [] {
        PreambleKind p;
        p.waveform = config::WAVEFORM_L;
        p.seq = sequences::preamble_l();
        p.real = false;
        p.t_sym = config::L_T_SYM;
        p.chunk = py_div(rxa::A2_CHUNK, 2);
        p.data_offset_hz = -config::L_PILOT_HZ.value();
        p.phase_offset = 0.0;
        p.mf_taps = l_mf_taps();
        return p;
    }();
    return k;
}

const PreambleKind& preamble_kind(std::int64_t waveform) {
    if (waveform == config::WAVEFORM_CE) return ce_pre();
    if (waveform == config::WAVEFORM_L) return l_pre();
    throw pyjson::KeyError(std::to_string(waveform), "KeyError: " + std::to_string(waveform));
}

types::Detection PreambleHit::to_detection(double duration_s) const {
    types::Detection x;
    const std::array<double, 3> t{rxa::TBD_START_S, 0.0, duration_s};
    x.path.t_s.assign(t.begin(), t.end());
    for (const double ti : t) x.path.f_hz.push_back(f_hz + drift_hz_per_min * ti / 60.0);
    x.path.weight.assign(3, 1.0);
    double sd = 1.0;
    if (std::isfinite(tau0_sd_s)) sd = tau0_sd_s * config::CH_FS;
    types::Timing tm;
    tm.tau0 = tau0_s * config::CH_FS;
    tm.ppm = 0.0;
    tm.gamma = 0.0;
    tm.z = lam;
    tm.cov = sm::Matrix(2, 2);
    volatile double two = 2.0;   // Python's float ** 2 is libm pow
    tm.cov(0, 0) = std::pow(sd, two);
    tm.cov(1, 1) = 0.0;
    x.timing = tm;
    x.f_hz = f_hz;
    x.z_ref = NaN;
    x.method = "preamble";
    x.lead_in_s = lead_in_s;
    x.waveform = waveform;
    return x;
}

// --- A2: the channel -------------------------------------------------------------

A2Input a2_input(std::span<const cf> fe, double t0_index, double f_audio_hz, double drift, double fs) {
    return a2_input_impl(fe, t0_index, f_audio_hz, drift, fs);
}
A2Input a2_input(std::span<const cd> fe, double t0_index, double f_audio_hz, double drift, double fs) {
    return a2_input_impl(fe, t0_index, f_audio_hz, drift, fs);
}

A2Channel a2_mf(const A2Input& inp, const PreambleKind& kind) {
    const double t0_ch = inp.t0_ch;
    std::vector<cd> y = inp.y;
    if (kind.data_offset_hz != 0.0 || std::isnan(kind.data_offset_hz)) {
        const double neg = -kind.data_offset_hz;
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double t = (static_cast<double>(i) - t0_ch) / config::CH_FS;
            const double c = neg * t;
            y[i] = np::mul(y[i], exp_of(PLUS_2J_PI, c - std::floor(c)));
        }
    }
    A2Channel ch;
    ch.m = sm::oaconvolve(std::span<const cd>(y), std::span<const double>(kind.mf_taps), sm::ConvMode::Same);
    ch.t0_ch = t0_ch;
    ch.f_mix = inp.f_mix;
    ch.kind = kind;
    const std::int64_t k0 = py_round(t0_ch - rxa::A2_TAU_S * config::CH_FS);
    const std::int64_t k1 = py_round(
        t0_ch + (rxa::A2_TAU_S + static_cast<double>(kind.seq.size()) * kind.t_sym) * config::CH_FS);
    const auto [a, b] = py_slice(std::max<std::int64_t>(k0, 0),
                                 std::min<std::int64_t>(k1, static_cast<std::int64_t>(ch.m.size())), ch.m.size());
    std::vector<double> m2(ch.m.size());
    for (std::size_t i = 0; i < ch.m.size(); ++i) {
        const double v = np::abs(ch.m[i]);
        m2[i] = v * v;
    }
    double s2 = 1.0;
    if (b > a) s2 = sm::median(std::span<const double>(m2).subspan(a, b - a)) / rxa::LN2;
    ch.sigma2 = py_max(s2, 1e-30);
    const std::int64_t w = py_round(rxa::A2_LOCAL_S * config::CH_FS);
    const std::vector<double> box(static_cast<std::size_t>(w), 1.0 / static_cast<double>(w));
    std::vector<double> loc = sm::oaconvolve(std::span<const double>(m2), std::span<const double>(box), sm::ConvMode::Same);
    for (double& v : loc) {
        // np.maximum(loc, sigma2): NaN from either side
        if (std::isnan(v)) continue;
        if (std::isnan(ch.sigma2) || ch.sigma2 > v) v = ch.sigma2;
    }
    ch.local = std::move(loc);
    return ch;
}

A2Channel preamble_channel(std::span<const cf> fe, double t0_index, double f_audio_hz, double drift, double fs,
                           const PreambleKind& kind) {
    return a2_mf(a2_input(fe, t0_index, f_audio_hz, drift, fs), kind);
}
A2Channel preamble_channel(std::span<const cd> fe, double t0_index, double f_audio_hz, double drift, double fs,
                           const PreambleKind& kind) {
    return a2_mf(a2_input(fe, t0_index, f_audio_hz, drift, fs), kind);
}

// --- A2: templates and sums ---------------------------------------------------------

Chunks chunks(const PreambleKind& kind) {
    if (kind.chunk <= 0) {
        if (kind.chunk == 0) throw ZeroDivision("integer division or modulo by zero");
        throw std::invalid_argument("_chunks: a chunk must be positive, not " + std::to_string(kind.chunk));
    }
    Chunks c;
    c.ck = static_cast<std::size_t>(kind.chunk);
    c.nc = kind.seq.size() / c.ck;
    c.real = kind.real;
    c.b.assign(kind.seq.begin(), kind.seq.begin() + static_cast<std::ptrdiff_t>(c.nc * c.ck));
    for (std::size_t r = 0; r < c.nc; ++r) {
        cd* row = c.b.data() + r * c.ck;
        if (c.real) {
            std::vector<double> re(c.ck);
            for (std::size_t s = 0; s < c.ck; ++s) re[s] = row[s].real();
            const double mean = npsum::sum(std::span<const double>(re)) / static_cast<double>(c.ck);
            for (std::size_t s = 0; s < c.ck; ++s) row[s] = cd(row[s].real() - mean, 0.0);
        } else {
            const cd sum = npsum::sum(std::span<const cd>(row, c.ck));
            const cd mean = np::div(sum, cd(static_cast<double>(c.ck), 0.0));
            for (std::size_t s = 0; s < c.ck; ++s) row[s] = row[s] - mean;
        }
    }
    return c;
}

std::vector<double> norm(const Chunks& b) {
    std::vector<double> out(b.nc);
    std::vector<double> sq(b.ck);
    for (std::size_t r = 0; r < b.nc; ++r) {
        for (std::size_t s = 0; s < b.ck; ++s) {
            const cd v = b.b[r * b.ck + s];
            sq[s] = b.real ? v.real() * v.real() : np::mul(v, std::conj(v)).real();
        }
        out[r] = npsum::sum(std::span<const double>(sq));
    }
    return out;
}

namespace {

// conj(b) as numpy forms it: a real template stays real and is promoted to
// (b, +0) by the product; a complex one is (br, -bi).
cd conj_template(const Chunks& b, std::size_t i) {
    const cd v = b.b[i];
    return b.real ? cd(v.real(), 0.0) : std::conj(v);
}

}  // namespace

Templates templates(std::span<const double> df_hz, const PreambleKind& kind) {
    Templates T;
    T.b = chunks(kind);
    const std::size_t nc = T.b.nc, ck = T.b.ck, n = nc * ck;
    std::vector<double> t(n);
    for (std::size_t i = 0; i < n; ++i) t[i] = static_cast<double>(i) * kind.t_sym;
    T.W.resize(df_hz.size() * n);
    for (std::size_t f = 0; f < df_hz.size(); ++f) {
        const cd a = np::mul(MINUS_2J_PI, cd(df_hz[f], 0.0));
        for (std::size_t i = 0; i < n; ++i) {
            const cd e = np_exp(np::mul(a, cd(t[i], 0.0)));
            T.W[f * n + i] = np::mul(conj_template(T.b, i), e);
        }
    }
    T.norm = norm(T.b);
    return T;
}

namespace {

Sums sums(const A2Channel& chan, std::span<const double> df_hz, double tau_s, bool want_var) {
    const PreambleKind& kind = chan.kind;
    const Templates T = templates(df_hz, kind);
    const std::size_t nc = T.b.nc, ck = T.b.ck, n = nc * ck;
    std::vector<std::int64_t> off(n);
    for (std::size_t i = 0; i < n; ++i) off[i] = np_round_int(static_cast<double>(i) * kind.t_sym * config::CH_FS);
    const std::int64_t k0 = py_round(chan.t0_ch);
    const std::int64_t R = py_round(tau_s * config::CH_FS);
    const std::int64_t r0 = -R;
    const std::size_t n_tau = R >= r0 ? static_cast<std::size_t>(R - r0 + 1) : 0;
    const std::size_t n_df = df_hz.size();
    const std::vector<cd>& m = chan.m;
    const auto nm = static_cast<std::int64_t>(m.size());
    Sums s;
    s.n_df = n_df;
    s.n_tau = n_tau;
    s.nc = nc;
    s.S.assign(n_df * n_tau * nc, cd(0.0, 0.0));
    s.taus.resize(n_tau);
    const double shift = static_cast<double>(k0) - chan.t0_ch;
    if (n_tau > 0 && n > 0 && nm == 0) throw std::out_of_range("index -1 is out of bounds for axis 0 with size 0");
    std::vector<cd> mx(ck);
    for (std::size_t ti = 0; ti < n_tau; ++ti) {
        const std::int64_t tau = r0 + static_cast<std::int64_t>(ti);
        s.taus[ti] = static_cast<double>(tau) + shift;
        for (std::size_t c = 0; c < nc; ++c) {
            for (std::size_t q = 0; q < ck; ++q) {
                const std::int64_t k = std::clamp<std::int64_t>(k0 + tau + off[c * ck + q], 0, nm - 1);
                mx[q] = m[static_cast<std::size_t>(k)];
            }
            for (std::size_t f = 0; f < n_df; ++f) {
                // the einsum's sum over symbols, in turn (the reference's is a
                // BLAS zgemm, whose order is the BLAS's: agrees to rounding)
                const cd* w = T.W.data() + f * n + c * ck;
                double ar = 0.0, ai = 0.0;
                for (std::size_t q = 0; q < ck; ++q) {
                    const double mr = mx[q].real(), mi = mx[q].imag(), wr = w[q].real(), wi = w[q].imag();
                    const double re = mr * wr - mi * wi;
                    const double im = mr * wi + mi * wr;
                    ar += re;
                    ai += im;
                }
                s.S[(f * n_tau + ti) * nc + c] = cd(0.0 + ar, 0.0 + ai);
            }
        }
    }
    // var = local[clip(k0 + tau + round(mean(off per chunk)))] * norm
    std::vector<std::int64_t> cmid(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        double sum = 0.0;   // exact: integers far below 2^53
        for (std::size_t q = 0; q < ck; ++q) sum += static_cast<double>(off[c * ck + q]);
        cmid[c] = np_round_int(sum / static_cast<double>(ck));
    }
    if (!chan.local) throw sm::TypeError("'NoneType' object is not subscriptable");
    const std::vector<double>& local = *chan.local;
    const auto nl = static_cast<std::int64_t>(local.size());
    s.var.resize(n_tau * nc);
    for (std::size_t ti = 0; ti < n_tau; ++ti) {
        const std::int64_t tau = r0 + static_cast<std::int64_t>(ti);
        for (std::size_t c = 0; c < nc; ++c) {
            const std::int64_t k = std::clamp<std::int64_t>(k0 + tau + cmid[c], 0, nm - 1);
            s.var[ti * nc + c] = local[index(k, static_cast<std::size_t>(nl))] * T.norm[c];
        }
    }
    // lam = sum over chunks of |S|^2 / var, added one chunk at a time (S is
    // Fortran-ordered in the reference, its chunk axis outermost)
    s.lam.assign(n_df * n_tau, 0.0);
    for (std::size_t f = 0; f < n_df; ++f)
        for (std::size_t ti = 0; ti < n_tau; ++ti) {
            double acc = 0.0;
            for (std::size_t c = 0; c < nc; ++c) {
                const double a = np::abs(s.S[(f * n_tau + ti) * nc + c]);
                acc = acc + (a * a) / s.var[ti * nc + c];
            }
            s.lam[f * n_tau + ti] = acc;
        }
    if (!want_var) s.var.clear();
    return s;
}

}  // namespace

Sums preamble_sums(const A2Channel& chan, std::span<const double> df_hz, double tau_s) {
    return sums(chan, df_hz, tau_s, true);
}

Sums preamble_lambda(const A2Channel& chan, std::span<const double> df_hz, double tau_s) {
    return sums(chan, df_hz, tau_s, false);
}

SplineFn spline_fn(const sm::CubicSpline& spline) {
    return [&spline](std::span<const double> x) { return spline(x); };
}

std::pair<std::vector<cd>, std::vector<double>> fine_sums(const SplineFn& spline, const A2Channel& chan, double tau,
                                                          double df, double rate) {
    const PreambleKind& kind = chan.kind;
    const Chunks b = chunks(kind);
    const std::size_t nc = b.nc, ck = b.ck, n = nc * ck;
    std::vector<double> x(n);
    std::vector<cd> W(n);
    const double half_rate = 0.5 * rate;
    const double base = chan.t0_ch + tau;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) * kind.t_sym;
        const double turns = df * t + half_rate * t * t;
        W[i] = np::mul(conj_template(b, i), exp_of(MINUS_2J_PI, turns));
        x[i] = base + t * config::CH_FS;
    }
    const std::vector<cd> mv = spline(x);
    if (mv.size() != n) throw std::invalid_argument("_fine_sums: the spline returned the wrong number of values");
    std::vector<cd> S(nc);
    std::vector<cd> prod(ck);
    for (std::size_t c = 0; c < nc; ++c) {
        for (std::size_t q = 0; q < ck; ++q) prod[q] = np::mul(W[c * ck + q], mv[c * ck + q]);
        S[c] = npsum::sum(std::span<const cd>(prod));
    }
    return {S, norm(b)};
}

double coherent(const SplineFn& spline, const A2Channel& chan, double tau, double df, double rate) {
    const auto [S, nrm] = fine_sums(spline, chan, tau, df, rate);
    const double a = np::abs(npsum::sum(std::span<const cd>(S)));
    volatile double two = 2.0;   // np.float64 ** 2: libm pow
    return std::pow(a, two) / (chan.sigma2 * npsum::sum(std::span<const double>(nrm)));
}

std::tuple<double, double, double> phase_line(std::span<const cd> S, double dt) {
    const std::size_t n = S.size() > 0 ? S.size() - 1 : 0;
    std::vector<double> g(n), w(n);
    const double two_pi_dt = 2 * PI * dt;
    for (std::size_t i = 0; i < n; ++i) {
        const cd pr = np::mul(S[i + 1], std::conj(S[i]));
        g[i] = np_angle(pr) / two_pi_dt;
        w[i] = np::abs(pr);
    }
    const double wsum = npsum::sum(std::span<const double>(w));
    if (wsum <= 0) return {0.0, 0.0, INF};
    std::vector<double> t(n);
    for (std::size_t i = 0; i < n; ++i) t[i] = (static_cast<double>(i) + 1.0) * dt;
    const double tb = sm::average(t, w);
    const double a = sm::average(g, w);
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double d = t[i] - tb;
        v[i] = w[i] * (d * d);
    }
    const double sxx = npsum::sum(std::span<const double>(v));
    for (std::size_t i = 0; i < n; ++i) v[i] = w[i] * (t[i] - tb) * (g[i] - a);
    const double b = npsum::sum(std::span<const double>(v)) / sxx;
    for (std::size_t i = 0; i < n; ++i) {
        const double res = g[i] - a - b * (t[i] - tb);
        v[i] = w[i] * (res * res);
    }
    const auto len = static_cast<std::int64_t>(n);
    const std::int64_t dof = std::max<std::int64_t>(len - 2, 1);
    const double s2 = npsum::sum(std::span<const double>(v)) / wsum * static_cast<double>(len) / static_cast<double>(dof);
    double sd_b = INF;
    if (sxx > 0) {
        const double q = s2 * wsum / sxx;
        if (q < 0) throw std::invalid_argument("math domain error");
        sd_b = std::sqrt(q);
    }
    return {a - b * tb, b, sd_b};
}

// --- A2: the search ------------------------------------------------------------------

PreambleHit preamble_search(const A2Input& inp, const types::CarrierCandidate& cand, double pfa,
                            const PreambleKind& kind) {
    return search(inp, cand, pfa, kind);
}
PreambleHit preamble_search(std::span<const cf> fe, double t0_index, const types::CarrierCandidate& cand, double fs,
                            double pfa, const PreambleKind& kind) {
    return search(a2_input(fe, t0_index, cand.f_hz, cand.drift_hz_per_min, fs), cand, pfa, kind);
}
PreambleHit preamble_search(std::span<const cd> fe, double t0_index, const types::CarrierCandidate& cand, double fs,
                            double pfa, const PreambleKind& kind) {
    return search(a2_input(fe, t0_index, cand.f_hz, cand.drift_hz_per_min, fs), cand, pfa, kind);
}

// --- lead-in -------------------------------------------------------------------------

double lead_in_span(std::span<const cf> fe, double t0_index, double f_hz, double drift, double tau0_s, double fs,
                    double z_min, double t_sym, const NoisePsdFn& noise_psd) {
    return lead_in_impl(fe, t0_index, f_hz, drift, tau0_s, fs, z_min, t_sym, noise_psd);
}
double lead_in_span(std::span<const cd> fe, double t0_index, double f_hz, double drift, double tau0_s, double fs,
                    double z_min, double t_sym, const NoisePsdFn& noise_psd) {
    return lead_in_impl(fe, t0_index, f_hz, drift, tau0_s, fs, z_min, t_sym, noise_psd);
}

}  // namespace sstvae::qrss::acquire
