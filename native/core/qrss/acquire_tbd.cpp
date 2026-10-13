// acquire.py's A3: see acquire_tbd.hpp.

#include "qrss/acquire_tbd.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>

#include "math/nplog.hpp"
#include "math/npstats.hpp"
#include "qrss/npsum.hpp"
#include "qrss/rational.hpp"

namespace sstvae::qrss::acquire {
namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();

// Python's round(x) to an int (half to even), with round()'s refusals.
std::int64_t py_round(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    const double r = std::nearbyint(x);
    if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0))
        throw std::overflow_error("Python int too large to convert to C int64");
    return static_cast<std::int64_t>(r);
}

// int(x) of a float: truncation, with int()'s refusals.
std::int64_t py_int(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    const double r = std::trunc(x);
    if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0))
        throw std::overflow_error("Python int too large to convert to C int64");
    return static_cast<std::int64_t>(r);
}

// math.floor / math.ceil of a float, as an int.
std::int64_t py_floor(double x) { return py_int(std::floor(x)); }
std::int64_t py_ceil(double x) { return py_int(std::ceil(x)); }

// np.round(v).astype(np.int64): x86-64's conversion of NaN, infinities and
// values past int64 is INT64_MIN.
std::int64_t round_to_int64(double v) {
    const double r = std::nearbyint(v);
    if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0)) return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(r);
}

// CPython's float floor division (float_floor_div / npy_divmod).
double py_floordiv(double a, double b) {
    if (b == 0.0) throw ZeroDivision("float floor division by zero");
    double mod = std::fmod(a, b);
    double div = (a - mod) / b;
    if (mod != 0.0) {
        if ((b < 0) != (mod < 0)) div -= 1.0;
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, a / b);
    }
    return floordiv;
}

// np.maximum(a, b): NaN-propagating, a where equal.
template <typename T>
T np_maximum(T a, T b) {
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return a >= b ? a : b;
}
template <typename T>
T np_minimum(T a, T b) {
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return a <= b ? a : b;
}

// np.argmax of a float row: the first maximum, or the first NaN.
std::size_t np_argmax(const double* x, std::size_t n) {
    std::size_t k = 0;
    double mp = x[0];
    if (std::isnan(mp)) return 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (!(x[i] <= mp)) {
            mp = x[i];
            k = i;
            if (std::isnan(mp)) break;
        }
    }
    return k;
}

// The log numpy computes for the element type: its float32 SIMD kernel,
// or (float64, where numpy calls SVML) libm's.
inline float elem_log(float x) { return math::np::log_f32(x); }
inline double elem_log(double x) { return std::log(x); }

// np.sum(rows of E gathered, dtype=np.float64) over a C-ordered 2-D
// gather: one coalesced reduction -- buffered through a cast for float32
// (8192 values a chunk), one pairwise sum for float64.
template <typename T>
double sum_f64(const std::vector<T>& v) {
    return npsum::sum_as<double, T>(std::span<const T>(v));
}

template <typename T>
TbdPath viterbi_grid(const Grid<T>& E, int max_step, double cost) {
    return tbd_viterbi<T>(E.data.data(), E.rows, E.cols, 0, E.cols, max_step, cost);
}

}  // namespace

// --- frames ---------------------------------------------------------------------

std::vector<double> tbd_frame_starts(const frame::FrameSpec& spec, std::optional<std::int64_t> n_fe, double t0_index,
                                     double fs) {
    const double end = spec.keyed_end_pos() * spec.t_sym() + rxa::TBD_END_PAD_S;
    const std::int64_t n =
        py_int(py_floordiv(end - rxa::TBD_START_S - rxa::STFT_FRAME_S, rxa::STFT_HOP_S)) + 1;
    std::vector<double> t;
    t.reserve(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
    for (std::int64_t k = 0; k < n; ++k) t.push_back(rxa::TBD_START_S + rxa::STFT_HOP_S * static_cast<double>(k));
    if (n_fe) {
        const double L = static_cast<double>(frame_len(fs));
        const double nf = static_cast<double>(*n_fe);
        std::vector<double> kept;
        kept.reserve(t.size());
        for (const double v : t) {
            const double i = t0_index + v * fs;
            if (i >= 0 && i + L <= nf) kept.push_back(v);
        }
        t = std::move(kept);
    }
    return t;
}

std::int64_t tbd_n_frames(const frame::FrameSpec& spec) {
    return static_cast<std::int64_t>(tbd_frame_starts(spec).size());
}

// --- the Viterbi ----------------------------------------------------------------

template <typename T>
TbdPath tbd_viterbi(const T* data, std::size_t rows, std::size_t stride, std::size_t col0, std::size_t B,
                    int max_step, double cost) {
    static_assert(std::is_floating_point_v<T>);
    if (rows == 0) throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    auto at = [&](std::size_t t, std::size_t b) { return static_cast<double>(data[t * stride + col0 + b]); };
    const long Bl = static_cast<long>(B);
    if (rows >= 2) {
        // the reference's refusals, in its order: per step, the slice
        // assignment (a step past the bins and under twice them), then np.int8(d)
        for (long d = -static_cast<long>(max_step); d <= static_cast<long>(max_step); ++d) {
            const long ad = d < 0 ? -d : d;
            if (ad > Bl && 2 * Bl - ad > 0)
                throw NotPorted("tbd_viterbi: a step of " + std::to_string(d) + " over " + std::to_string(B) +
                                " bins (numpy's broadcast refusal)");
            if (d < -128 || d > 127)
                throw std::overflow_error("Python integer " + std::to_string(d) + " out of bounds for int8");
        }
    }
    std::vector<double> score(B), next(B);
    for (std::size_t b = 0; b < B; ++b) score[b] = at(0, b);
    std::vector<std::int8_t> back(rows * B, 0);
    const int m = max_step;
    std::vector<double> cd;   // cost * |d| for d = -m..m
    for (long d = -static_cast<long>(m); d <= static_cast<long>(m); ++d)
        cd.push_back(cost * static_cast<double>(d < 0 ? -d : d));
    for (std::size_t t = 1; t < rows; ++t) {
        std::int8_t* bk = back.data() + t * B;
        for (long b = 0; b < Bl; ++b) {
            double best = -INF;
            std::int8_t arg = 0;
            for (long d = -static_cast<long>(m), i = 0; d <= static_cast<long>(m); ++d, ++i) {
                const long src = b - d;
                const double cand = (src >= 0 && src < Bl) ? score[static_cast<std::size_t>(src)] - cd[static_cast<std::size_t>(i)]
                                                           : -INF;
                if (cand > best) {
                    best = cand;
                    arg = static_cast<std::int8_t>(d);
                }
            }
            bk[b] = arg;
            next[static_cast<std::size_t>(b)] = best + at(t, static_cast<std::size_t>(b));
        }
        score.swap(next);
    }
    if (B == 0) throw std::invalid_argument("attempt to get argmax of an empty sequence");
    TbdPath out;
    out.path.assign(rows, 0);
    out.path[rows - 1] = static_cast<std::int64_t>(np_argmax(score.data(), B));
    for (std::size_t t = rows - 1; t > 0; --t) {
        const std::int64_t b = out.path[t];
        out.path[t - 1] = b - back[t * B + static_cast<std::size_t>(b)];
    }
    std::vector<double> em(rows);
    for (std::size_t t = 0; t < rows; ++t) em[t] = at(t, static_cast<std::size_t>(out.path[t]));
    out.z = npsum::sum(std::span<const double>(em)) / std::sqrt(static_cast<double>(rows));
    return out;
}

template TbdPath tbd_viterbi<float>(const float*, std::size_t, std::size_t, std::size_t, std::size_t, int, double);
template TbdPath tbd_viterbi<double>(const double*, std::size_t, std::size_t, std::size_t, std::size_t, int, double);

TbdPath tbd_viterbi(const Grid<float>& E, int max_step, double cost) { return viterbi_grid(E, max_step, cost); }
TbdPath tbd_viterbi(const Grid<double>& E, int max_step, double cost) { return viterbi_grid(E, max_step, cost); }

// --- hits -----------------------------------------------------------------------

std::vector<double> path_line(const types::FreqPath& path) {
    const std::vector<double>& t = path.t_s;
    const std::vector<double>& f = path.f_hz;
    const std::size_t n = t.size();
    if (f.size() != n || path.weight.size() != n)
        throw std::invalid_argument("path_line: t_s, f_hz and weight differ in length");
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) w[i] = path.weight[i] + 1e-9;
    std::vector<std::uint8_t> keep(n, 1);
    std::optional<std::vector<double>> c;
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<double> tk, fk, sw;
        for (std::size_t i = 0; i < n; ++i)
            if (keep[i]) {
                tk.push_back(t[i]);
                fk.push_back(f[i]);
                sw.push_back(std::sqrt(w[i]));
            }
        if (tk.size() < 2) break;
        c = math::polyfit(tk, fk, 1, std::span<const double>(sw));
        const std::vector<double> y = math::polyval(*c, t);
        for (std::size_t i = 0; i < n; ++i) keep[i] = std::fabs(f[i] - y[i]) < 1.0;
    }
    if (!c) throw UnboundLocal("cannot access local variable 'c' where it is not associated with a value");
    return *c;
}

double TbdHit::f_hz() const { return math::polyval(line(), 0.0); }

types::Detection TbdHit::to_detection(std::optional<std::int64_t> wf) const {
    const std::int64_t w = wf ? *wf : (waveform ? *waveform : config::WAVEFORM_CE);
    types::FreqPath p = path;
    double f = f_hz();
    if (w == config::WAVEFORM_L && l_path) {
        p = *l_path;
        f = math::polyval(path_line(p), 0.0);
    }
    types::Detection d;
    d.f_hz = f;
    d.path = std::move(p);
    d.timing = std::nullopt;
    d.z_ref = NaN;
    d.method = "tbd";
    d.lead_in_s = 0.0;
    d.waveform = w;
    d.a3_z = z;
    return d;
}

std::vector<double> tbd_channels(std::span<const double> cand_f_hz, std::array<double, 2> band_hz) {
    const double lo = band_hz[0] + rxa::TBD_HALF_HZ, hi = band_hz[1] - rxa::TBD_HALF_HZ;
    std::vector<double> cf;
    cf.reserve(cand_f_hz.size());
    for (const double c : cand_f_hz) {
        const double mx = (lo > c) ? lo : c;     // Python max(c, lo)
        cf.push_back((hi < mx) ? hi : mx);       // min(., hi)
    }
    // np.arange(ceil(lo / 50) * 50, hi + 1e-9, 50)
    const double start = static_cast<double>(py_ceil(lo / rxa::TBD_GRID_HZ)) * rxa::TBD_GRID_HZ;
    const double stop = hi + 1e-9;
    const double len = std::ceil((stop - start) / rxa::TBD_GRID_HZ);
    std::vector<double> out = cf;
    if (len > 0) {
        const double delta = (start + rxa::TBD_GRID_HZ) - start;
        const auto n = static_cast<std::int64_t>(len);
        for (std::int64_t i = 0; i < n; ++i) {
            const double g = i == 0 ? start : (i == 1 ? start + rxa::TBD_GRID_HZ : start + static_cast<double>(i) * delta);
            bool far = true;
            for (const double c : cf)
                if (!(std::fabs(g - c) >= rxa::TBD_GRID_HZ / 2)) {
                    far = false;
                    break;
                }
            if (far) out.push_back(g);
        }
    }
    std::stable_sort(out.begin(), out.end(), [](double a, double b) { return a < b; });
    return out;
}

namespace {

template <typename X>
Emissions emissions_of(std::span<const X> fe, double t0_index, const frame::FrameSpec& spec, double fs,
                       std::array<double, 2> band_hz) {
    Emissions out;
    out.t = tbd_frame_starts(spec, static_cast<std::int64_t>(fe.size()), t0_index, fs);
    BandBins bb = band_bins(fs, band_hz);
    std::vector<double> starts(out.t.size());
    for (std::size_t i = 0; i < out.t.size(); ++i) starts[i] = t0_index + out.t[i] * fs;
    {
        Grid32 P = spectrogram(fe, fs, starts, std::span<const std::int64_t>(bb.sel));
        out.E = cfar_normalise(P);
    }
    for (float& v : out.E.data) v = v - 1.0f;
    out.f = std::move(bb.f_hz);
    return out;
}

}  // namespace

Emissions tbd_emissions(std::span<const std::complex<float>> fe, double t0_index, const frame::FrameSpec& spec,
                        double fs, std::array<double, 2> band_hz) {
    return emissions_of(fe, t0_index, spec, fs, band_hz);
}
Emissions tbd_emissions(std::span<const std::complex<double>> fe, double t0_index, const frame::FrameSpec& spec,
                        double fs, std::array<double, 2> band_hz) {
    return emissions_of(fe, t0_index, spec, fs, band_hz);
}
Emissions tbd_emissions(std::span<const double> fe, double t0_index, const frame::FrameSpec& spec, double fs,
                        std::array<double, 2> band_hz) {
    return emissions_of(fe, t0_index, spec, fs, band_hz);
}

// --- the spectral shape ---------------------------------------------------------

std::vector<std::uint8_t> shape_frames(std::span<const double> t_s, std::span<const frame::FrameSpec> specs,
                                       double frame_s) {
    if (specs.empty()) throw std::invalid_argument("max() iterable argument is empty");
    double start = specs[0].keyed_start_pos() * specs[0].t_sym();
    for (std::size_t i = 1; i < specs.size(); ++i) {
        const double v = specs[i].keyed_start_pos() * specs[i].t_sym();
        if (v > start) start = v;
    }
    std::vector<std::uint8_t> keep(t_s.size());
    for (std::size_t i = 0; i < t_s.size(); ++i) keep[i] = t_s[i] >= start;
    for (const frame::FrameSpec& s : specs)
        for (const auto& [a, b] : s.window_spans_s())
            for (std::size_t i = 0; i < t_s.size(); ++i)
                keep[i] = keep[i] && !((t_s[i] < b) && (t_s[i] + frame_s > a));
    return keep;
}

namespace {

std::pair<std::int64_t, std::int64_t> l_band_bins(double df_hz) {
    return {py_ceil(rxa::L_BAND_HZ[0] / df_hz - 1e-9), py_floor(rxa::L_BAND_HZ[1] / df_hz + 1e-9)};
}

template <typename T>
std::pair<double, double> shape_z_impl(const Grid<T>& E, std::span<const std::int64_t> bins,
                                       std::optional<std::span<const std::uint8_t>> keep, double df_hz, double fs) {
    const std::size_t n = E.rows;
    const auto nb = static_cast<std::int64_t>(E.cols);
    const auto [lo, hi] = l_band_bins(df_hz);
    if (bins.size() != n) throw std::invalid_argument("shape_z: bins must have one entry per frame");
    if (keep && keep->size() != n) throw std::invalid_argument("shape_z: keep must have one entry per frame");
    std::vector<std::uint8_t> ok(n);
    for (std::size_t i = 0; i < n; ++i) {
        bool v = keep ? (*keep)[i] != 0 : true;
        const std::int64_t b = bins[i];
        v = v && (b - hi >= 0) && (b + hi < nb);
        if (nb == 0) throw std::out_of_range("index -1 is out of bounds for axis 1 with size 0");
        const std::int64_t bc = std::clamp<std::int64_t>(b, 0, nb - 1);
        v = v && (E(i, static_cast<std::size_t>(bc)) > T(-1.0));
        ok[i] = v;
    }
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < n; ++i)
        if (ok[i]) rows.push_back(i);
    if (rows.empty()) return {NaN, NaN};
    const std::int64_t n_off = hi - lo + 1;
    std::vector<T> below, above;
    below.reserve(rows.size() * static_cast<std::size_t>(std::max<std::int64_t>(n_off, 0)));
    above.reserve(below.capacity());
    for (const std::size_t r : rows) {
        const std::int64_t b = bins[r];
        for (std::int64_t o = lo; o <= hi; ++o) {
            below.push_back(E(r, static_cast<std::size_t>(b - o)));
            above.push_back(E(r, static_cast<std::size_t>(b + o)));
        }
    }
    const double sb = sum_f64(below);
    const double sa = sum_f64(above);
    const auto [v0, v1] = band_cov(n_off, fs);
    std::int64_t pairs = 0;
    for (std::size_t i = 1; i < n; ++i) pairs += (ok[i] && ok[i - 1]) ? 1 : 0;
    const double var = static_cast<double>(rows.size()) * v0 + static_cast<double>(2 * pairs) * v1;
    const double v2 = 2 * var;
    if (v2 < 0) throw std::invalid_argument("math domain error");
    const double s2 = std::sqrt(v2);
    if (s2 == 0.0) throw ZeroDivision("float division by zero");
    const double z1 = (sb - sa) / s2;
    const double s1 = std::sqrt(var);
    if (s1 == 0.0) throw ZeroDivision("float division by zero");
    return {z1, sa / s1};
}

}  // namespace

std::pair<double, double> shape_z(const Grid<float>& E, std::span<const std::int64_t> bins,
                                  std::optional<std::span<const std::uint8_t>> keep, double df_hz, double fs) {
    return shape_z_impl(E, bins, keep, df_hz, fs);
}
std::pair<double, double> shape_z(const Grid<double>& E, std::span<const std::int64_t> bins,
                                  std::optional<std::span<const std::uint8_t>> keep, double df_hz, double fs) {
    return shape_z_impl(E, bins, keep, df_hz, fs);
}

std::optional<std::int64_t> shape_waveform(double z1, double z_above) {
    if (z1 > rxa::SHAPE_L_Z && !(z_above > rxa::SHAPE_CE_ABOVE_Z)) return config::WAVEFORM_L;
    if (z_above > rxa::SHAPE_CE_ABOVE_Z && std::fabs(z1) < rxa::SHAPE_CE_MAX_Z1) return config::WAVEFORM_CE;
    return std::nullopt;
}

std::optional<std::int64_t> shape_verdict(std::optional<std::int64_t> v, std::optional<std::int64_t> w) {
    if (v == w || !w) return v;
    if (!v) return w;
    return std::nullopt;
}

// --- paths ----------------------------------------------------------------------

namespace {

std::int64_t wrap_index(std::int64_t i, std::int64_t n, int axis) {
    const std::int64_t j = i < 0 ? i + n : i;
    if (j < 0 || j >= n)
        throw std::out_of_range("index " + std::to_string(i) + " is out of bounds for axis " + std::to_string(axis) +
                                " with size " + std::to_string(n));
    return j;
}

template <typename T>
types::FreqPath bins_path_impl(const Grid<T>& E, std::span<const double> f, std::span<const double> t,
                               std::span<const std::int64_t> bins) {
    const std::size_t n = t.size();
    if (f.size() < 2) throw std::out_of_range("index 1 is out of bounds for axis 0 with size " + std::to_string(f.size()));
    const double df = f[1] - f[0];
    if (bins.size() != n) throw std::invalid_argument("bins_path: bins must have one entry per frame");
    if (n > E.rows) throw std::out_of_range("index " + std::to_string(E.rows) + " is out of bounds for axis 0 with size " +
                                            std::to_string(E.rows));
    const auto ncol = static_cast<std::int64_t>(E.cols);
    const auto nf = static_cast<std::int64_t>(f.size());
    types::FreqPath out;
    out.t_s.resize(n);
    out.f_hz.resize(n);
    out.weight.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t b = bins[i];
        const T em = E(i, static_cast<std::size_t>(wrap_index(b, ncol, 1)));
        double fk = f[static_cast<std::size_t>(wrap_index(b, nf, 0))];
        if (b > 0 && b < nf - 1) {
            T y[3];
            for (int k = 0; k < 3; ++k) {
                const T a = E(i, static_cast<std::size_t>(wrap_index(b + k - 1, ncol, 1))) + T(1);
                y[k] = elem_log(np_maximum(a, static_cast<T>(1e-6)));
            }
            const T den = (y[0] - T(2) * y[1]) + y[2];
            T off = T(0);
            if (den < T(0)) {
                const T v = (T(0.5) * (y[0] - y[2])) / den;
                off = np_minimum(np_maximum(v, T(-0.5)), T(0.5));
            }
            fk = fk + static_cast<double>(off) * df;
        }
        out.t_s[i] = t[i] + rxa::STFT_FRAME_S / 2;
        out.f_hz[i] = fk;
        out.weight[i] = static_cast<double>(np_maximum(em, T(0)));
    }
    return out;
}

template <typename T>
std::vector<std::int64_t> l_pilot_bins_impl(const Grid<T>& E, std::span<const double> f, std::span<const double> t,
                                            std::span<const double> line,
                                            std::optional<std::span<const std::uint8_t>> keep, double fs,
                                            std::array<double, 2> span_hz) {
    const std::size_t n = E.rows;
    const auto nb = static_cast<std::int64_t>(E.cols);
    if (f.size() < 2) throw std::out_of_range("index 1 is out of bounds for axis 0 with size " + std::to_string(f.size()));
    if (t.size() != n) throw std::invalid_argument("l_pilot_bins: t must have one entry per frame");
    if (keep && keep->size() != n) throw std::out_of_range("l_pilot_bins: keep must have one entry per frame");
    const double df = f[1] - f[0];
    const auto [lo, hi] = l_band_bins(df);
    const std::int64_t k_lo = py_floor(span_hz[0] / df);
    const std::int64_t k_hi = py_ceil(span_hz[1] / df);
    const std::int64_t g0 = k_lo - hi;
    const std::int64_t ncol = std::max<std::int64_t>(k_hi + 1 - g0, 0);
    const std::int64_t nj = std::max<std::int64_t>(ncol - hi, 0);
    if (nb == 0 && n > 0) throw std::out_of_range("index 0 is out of bounds for axis 1 with size 0");
    std::vector<double> tc(n);
    for (std::size_t i = 0; i < n; ++i) tc[i] = t[i] + rxa::STFT_FRAME_S / 2;
    const std::vector<double> pv = math::polyval(line, tc);
    const double sq = std::sqrt(band_cov(hi - lo + 1, fs).first);
    Grid<double> em(n, static_cast<std::size_t>(nj));
    std::vector<std::int64_t> c(n);
    std::vector<double> eg(static_cast<std::size_t>(ncol)), cs(static_cast<std::size_t>(ncol) + 1);
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = round_to_int64((pv[i] - f[0]) / df);
        for (std::int64_t k = 0; k < ncol; ++k) {
            const std::int64_t g = c[i] + g0 + k;
            const bool ok = g >= 0 && g < nb;
            eg[static_cast<std::size_t>(k)] =
                ok ? static_cast<double>(E(i, static_cast<std::size_t>(std::clamp<std::int64_t>(g, 0, nb - 1)))) : 0.0;
        }
        // np.concatenate([0, np.cumsum(Eg)]): the first element copied, then a running sum
        cs[0] = 0.0;
        if (ncol > 0) cs[1] = eg[0];
        for (std::int64_t k = 1; k < ncol; ++k)
            cs[static_cast<std::size_t>(k) + 1] = cs[static_cast<std::size_t>(k)] + eg[static_cast<std::size_t>(k)];
        const bool kept = !keep || (*keep)[i] != 0;
        for (std::int64_t jj = 0; jj < nj; ++jj) {
            const std::int64_t j = hi + jj;
            const std::int64_t g = c[i] + g0 + j;
            double v;
            if (!kept) {
                v = 0.0;
            } else if (g >= 0 && g < nb) {
                const double band = cs[static_cast<std::size_t>(j - lo + 1)] - cs[static_cast<std::size_t>(j - hi)];
                v = eg[static_cast<std::size_t>(j)] + band / sq;
            } else {
                v = -1e3;
            }
            em(i, static_cast<std::size_t>(jj)) = v;
        }
    }
    const TbdPath vp = tbd_viterbi(em);
    std::vector<std::int64_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t g = c[i] + g0 + hi + vp.path[i];
        out[i] = std::clamp<std::int64_t>(g, 0, nb - 1);
    }
    return out;
}

}  // namespace

types::FreqPath bins_path(const Grid<float>& E, std::span<const double> f, std::span<const double> t,
                          std::span<const std::int64_t> bins) {
    return bins_path_impl(E, f, t, bins);
}
types::FreqPath bins_path(const Grid<double>& E, std::span<const double> f, std::span<const double> t,
                          std::span<const std::int64_t> bins) {
    return bins_path_impl(E, f, t, bins);
}

std::vector<std::int64_t> l_pilot_bins(const Grid<float>& E, std::span<const double> f, std::span<const double> t,
                                       std::span<const double> line, std::optional<std::span<const std::uint8_t>> keep,
                                       double fs, std::array<double, 2> span_hz) {
    return l_pilot_bins_impl(E, f, t, line, keep, fs, span_hz);
}
std::vector<std::int64_t> l_pilot_bins(const Grid<double>& E, std::span<const double> f, std::span<const double> t,
                                       std::span<const double> line, std::optional<std::span<const std::uint8_t>> keep,
                                       double fs, std::array<double, 2> span_hz) {
    return l_pilot_bins_impl(E, f, t, line, keep, fs, span_hz);
}

// --- A3 -------------------------------------------------------------------------

std::vector<TbdHit> track_before_detect(const Emissions& emi, std::span<const double> cand_f_hz, double fs,
                                        const TbdOptions& opt) {
    const Grid<float>& E = emi.E;
    const std::vector<double>& f = emi.f;
    const std::vector<double>& t = emi.t;
    if (t.size() < 2) return {};
    if (f.size() < 2) throw std::out_of_range("index 1 is out of bounds for axis 0 with size " + std::to_string(f.size()));
    const double df = f[1] - f[0];
    const std::int64_t half = py_round(rxa::TBD_HALF_HZ / df);
    const std::vector<double> centres = tbd_channels(cand_f_hz, opt.band_hz);
    if (centres.empty()) throw std::invalid_argument("need at least one array to stack");
    const auto nf = static_cast<std::int64_t>(f.size());
    const std::int64_t c_hi = nf - 1 - half;
    if (c_hi < half || half < 0 || static_cast<std::int64_t>(E.cols) != nf)
        throw NotPorted("track_before_detect: a band narrower than one channel");
    const std::size_t n = t.size();
    const auto width = static_cast<std::size_t>(2 * half + 1);
    std::vector<std::vector<std::int64_t>> path_bins(centres.size());
    std::vector<TbdHit> hits(centres.size());
    for (std::size_t k = 0; k < centres.size(); ++k) {
        // np.clip(np.round(...).astype(int), half, len(f) - 1 - half)
        const std::int64_t ci = std::clamp<std::int64_t>(round_to_int64((centres[k] - f[0]) / df), half, c_hi);
        const TbdPath vp = tbd_viterbi<float>(E.data.data(), E.rows, E.cols, static_cast<std::size_t>(ci - half), width);
        std::vector<std::int64_t> bins(n);
        for (std::size_t i = 0; i < n; ++i) bins[i] = ci - half + vp.path[i];
        TbdHit& h = hits[k];
        h.path = bins_path(E, f, t, bins);
        h.z = vp.z;
        h.p_value = gumbel_p(vp.z, static_cast<std::int64_t>(n));
        h.n_frames = static_cast<std::int64_t>(n);
        h.centre_hz = centres[k];
        path_bins[k] = std::move(bins);
    }
    std::optional<std::vector<std::uint8_t>> keep;
    if (opt.shape_specs) {
        keep = opt.keep_fn ? opt.keep_fn(t) : shape_frames(t, *opt.shape_specs);
        if (keep->size() != n) throw std::invalid_argument("track_before_detect: shape frames of the wrong length");
    }
    bool with_l = false;
    if (opt.shape_specs)
        for (const frame::FrameSpec& sp : *opt.shape_specs) with_l = with_l || sp.waveform() == frame::Waveform::L;
    std::vector<std::size_t> order(hits.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return -hits[a].z < -hits[b].z; });
    std::vector<TbdHit> out;
    std::vector<double> out_f;
    for (const std::size_t k : order) {
        TbdHit& h = hits[k];
        if (!opt.all_hits && !(h.p_value < opt.pfa)) continue;
        double fh = NaN;
        if (!out.empty()) {
            fh = h.f_hz();
            bool dup = false;
            for (const double of : out_f)
                if (std::fabs(fh - of) < rxa::TBD_DEDUP_HZ) {
                    dup = true;
                    break;
                }
            if (dup) continue;
        } else {
            fh = h.f_hz();
        }
        if (keep) {
            const std::span<const std::uint8_t> ks(*keep);
            std::tie(h.shape_z, h.above_z) = shape_z(E, path_bins[k], ks, df, fs);
            h.waveform = shape_waveform(h.shape_z, h.above_z);
            if (with_l) {
                const std::vector<std::int64_t> lb = l_pilot_bins(E, f, t, h.line(), ks, fs);
                h.l_path = bins_path(E, f, t, lb);
                std::tie(h.l_shape_z, h.l_above_z) = shape_z(E, lb, ks, df, fs);
                h.waveform = shape_verdict(h.waveform, shape_waveform(h.l_shape_z, h.l_above_z));
            }
        }
        out_f.push_back(fh);
        out.push_back(std::move(h));
    }
    return out;
}

std::vector<TbdHit> track_before_detect(std::span<const std::complex<float>> fe, double t0_index,
                                        const frame::FrameSpec& spec, std::span<const double> cand_f_hz, double fs,
                                        const TbdOptions& opt) {
    return track_before_detect(tbd_emissions(fe, t0_index, spec, fs, opt.band_hz), cand_f_hz, fs, opt);
}
std::vector<TbdHit> track_before_detect(std::span<const std::complex<double>> fe, double t0_index,
                                        const frame::FrameSpec& spec, std::span<const double> cand_f_hz, double fs,
                                        const TbdOptions& opt) {
    return track_before_detect(tbd_emissions(fe, t0_index, spec, fs, opt.band_hz), cand_f_hz, fs, opt);
}

// --- acquire's helpers ----------------------------------------------------------

std::vector<std::pair<std::int64_t, std::size_t>> frame_map(std::span<const frame::FrameSpec> specs) {
    std::vector<std::pair<std::int64_t, std::size_t>> out;
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const auto w = static_cast<std::int64_t>(specs[i].waveform());
        for (const auto& [ow, oi] : out)
            if (ow == w)
                throw std::invalid_argument("acquire takes at most one frame per waveform, not " +
                                            frame::py_repr(specs[oi].name()) + " and " +
                                            frame::py_repr(specs[i].name()));
        out.emplace_back(w, i);
    }
    if (out.empty()) throw std::invalid_argument("acquire needs at least one frame");
    return out;
}

types::FreqPath path_of(double f_hz, double drift_hz_per_min, double dur) {
    types::FreqPath p;
    p.t_s = {rxa::TBD_START_S, 0.0, dur};
    for (const double v : p.t_s) p.f_hz.push_back(f_hz + drift_hz_per_min * v / 60.0);
    p.weight = {1.0, 1.0, 1.0};
    return p;
}

}  // namespace sstvae::qrss::acquire
