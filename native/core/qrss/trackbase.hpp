// The tracker's shared vocabulary: the part of sstvae/qrss/track.py (and
// ltrack.py's records) that every receive-side module uses -- the narrow
// channel record, the frequency path, the timing map, the matched-filter
// table type, the symbol classes and the tracker's result records -- ported
// first so the receiver's lanes (front end, acquisition, tracking,
// demodulation) can be built against one stable set of types.
//
// Not a format module: nothing here goes on the air. Constants are
// rx_config.hpp's (generated from track.py) and config.hpp's.
//
// What is here and what is not:
//
// * q_of_spread, spread_of_q, q_grid, Q_GRID (rx_config's track::Q_GRID),
//   rx_index and tau_of_rx: Python's arithmetic in Python's order, so they
//   agree to the bit on the same libm (q_of_spread's ** 3 and spread_of_q's
//   ** (1/3) are C pow, as Python's float power is).
// * ChanCapture: the fields, mix_audio_hz and t_s. noise_psd_at and
//   noise_floor are the front end's (they call frontend.noise_psd) and are
//   ported with it; this holds the slot their cache lives in (`psd`), safe
//   to fill from several threads.
// * PathFn and freq_path_spline in full: the weighted line (np.polyfit,
//   deg 1), the least-squares cubic spline with knots every knots_s
//   (scipy's make_lsq_spline), the fall back to the line when the spline
//   fit refuses (ValueError or LinAlgError) or comes out non-finite (a
//   knot interval with no data), and the linear continuation outside
//   [lo, hi] along the spline's end derivative. Inputs are sorted stably
//   (numpy's argsort is not stable for long arrays, so with repeated times
//   the two may order equal-time points differently -- a rounding-level
//   difference in the fit). t, f and w must be the same length
//   (std::invalid_argument; Python indexes f and w by t's order and so
//   silently truncates a longer one).
// * MfTable (the type and its ValueError), Known, Classes, LClasses,
//   TrackResult, LTrackResult: the records, with TrackResult's derived
//   accessors (p_lo, gi, doppler_hz, t_pos_s). The functions that build
//   them (mf_table, known_symbols, make_classes, track, ...) belong to the
//   lanes that port those modules.
//
// Errors are the reference's: std::invalid_argument for ValueError,
// math::LinAlgError, std::out_of_range for IndexError, ZeroDivision for
// ZeroDivisionError, std::overflow_error for OverflowError.

#ifndef SSTVAE_QRSS_TRACKBASE_HPP
#define SSTVAE_QRSS_TRACKBASE_HPP

#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "math/bspline.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/rational.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::track {

// --- process noise ----------------------------------------------------------

// Process-noise intensity per position for a 2-sigma Doppler spread:
// (2 pi sigma T)^3 with sigma = spread/2 and T the position's period.
double q_of_spread(double spread_hz, double t_sym = config::T_SYM);
// The inverse: the effective 2-sigma spread (Hz) of q.
double spread_of_q(double q, double t_sym = config::T_SYM);
// select_q's grid at period t_sym: q_of_spread of each Q_SPREADS_HZ.
std::array<double, rxconfig::track::Q_SPREADS_HZ.size()> q_grid(double t_sym = config::T_SYM);
// CE's grid, as Python computed it at import (generated).
inline constexpr const auto& Q_GRID = rxconfig::track::Q_GRID;

// --- timing -----------------------------------------------------------------

// CH index (relative to t0_index) at which sender time tau (symbols of
// period t_sym) is received: tau0 + tau*sps*(1 + ppm 1e-6) + gamma (tau/n_pos)^2.
double rx_index(const types::Timing& timing, double tau, std::int64_t n_pos, int fs = config::CH_FS,
                double t_sym = config::T_SYM);
std::vector<double> rx_index(const types::Timing& timing, std::span<const double> tau, std::int64_t n_pos,
                             int fs = config::CH_FS, double t_sym = config::T_SYM);
// _tau_of_rx: rx_index inverted (three fixed-point steps for gamma).
double tau_of_rx(const types::Timing& timing, double rel, std::int64_t n_pos, double t_sym = config::T_SYM);
std::vector<double> tau_of_rx(const types::Timing& timing, std::span<const double> rel, std::int64_t n_pos,
                              double t_sym = config::T_SYM);

// --- the narrow channel -----------------------------------------------------

// ChanCapture._psd: noise PSD (CH power per Hz) at window-centre receiver
// times.
struct NoisePsd {
    std::vector<double> t_s;
    std::vector<double> psd;
};

// The cache slot ChanCapture.noise_psd_at fills once. Thread-safe: one
// computation however many threads ask at once (get_or_compute holds the
// lock while computing). A copy holds the same (immutable) value with a
// lock of its own.
class NoisePsdSlot {
public:
    NoisePsdSlot() = default;
    NoisePsdSlot(const NoisePsdSlot& o) : value_(o.get()) {}
    NoisePsdSlot& operator=(const NoisePsdSlot& o);
    std::shared_ptr<const NoisePsd> get() const;
    void set(std::shared_ptr<const NoisePsd> v);
    std::shared_ptr<const NoisePsd> get_or_compute(const std::function<NoisePsd()>& compute) const;

private:
    mutable std::mutex mutex_;
    mutable std::shared_ptr<const NoisePsd> value_;
};

// One signal's 250 Hz channel: CH = FE' mixed down by f_mix, decimated.
// t0_index: CH index of the nominal t0 on the receiver's clock. f_mix: the
// channeliser's exact mix, Hz inside FE (audio Hz - 1500). keep: the
// blanker's keep indicator through the same decimation (nullopt: none).
// exclude_hz: neighbour frequencies (audio Hz) left out of the noise floor.
struct ChanCapture {
    std::vector<std::complex<float>> ch;
    int fs = config::CH_FS;
    double t0_index = 0.0;
    Rational f_mix;
    std::optional<std::vector<float>> keep;
    std::vector<double> exclude_hz;
    NoisePsdSlot psd;   // Python's _psd

    double mix_audio_hz() const;   // FE_CENTER_HZ + float(f_mix)
    double t_s(double n) const;    // receiver time (s after t0) of CH index n
    std::vector<double> t_s(std::span<const double> n) const;
};

// --- the frequency path -----------------------------------------------------

// A frequency trajectory f(t) (audio Hz at receiver time t, s after t0).
class PathFn {
public:
    // PathFn(t, f, w=None, knots_s=KNOTS_S). Throws as the reference does:
    // std::out_of_range for no points, LinAlgError from the line fit (a
    // NaN or infinite t or w with two or more points), std::invalid_argument
    // for int() of a NaN span (one NaN point), ZeroDivision for knots_s 0.
    PathFn(std::span<const double> t, std::span<const double> f,
           std::optional<std::span<const double>> w = std::nullopt,
           double knots_s = rxconfig::track::KNOTS_S);
    // An already fitted path (no refit): lo, hi, the line's polyfit
    // coefficients (highest power first) and the spline, if any.
    static PathFn from_parts(double lo, double hi, std::vector<double> line, std::optional<math::BSpline> spline);

    double lo() const { return lo_; }
    double hi() const { return hi_; }
    const std::vector<double>& line() const { return line_; }
    const std::optional<math::BSpline>& spline() const { return spl_; }

    double operator()(double t) const;
    std::vector<double> operator()(std::span<const double> t) const;

private:
    PathFn() = default;
    void prepare_ends();
    double inside(double tc) const;

    double lo_ = 0.0, hi_ = 0.0;
    std::vector<double> line_;
    std::optional<math::BSpline> spl_;
    // spl(lo), spl'(lo), spl(hi), spl'(hi): the linear continuations
    double s_lo_ = 0.0, d_lo_ = 0.0, s_hi_ = 0.0, d_hi_ = 0.0;
};

// The frequency path f_A(t) of a detection (design 6.5 step 1).
PathFn freq_path_spline(const types::FreqPath& path, double knots_s = rxconfig::track::KNOTS_S);

// --- matched-filter tables, known symbols, classes --------------------------

// A position matched filter for symbol_mf and psi_at: taps at offsets
// d = n - t_p every 1/MF_RES CH sample from -half to +half (2 half MF_RES
// + 1 values), real or complex; g0 is the output for a constant 1. The
// tap storage is shared between copies (tables are immutable and large).
class MfTable {
public:
    // std::invalid_argument "tab must hold <2 half MF_RES + 1> taps, not <len>".
    MfTable(double t_sym, double sps, int half, std::vector<double> tab, double g0);
    MfTable(double t_sym, double sps, int half, std::vector<std::complex<double>> tab, double g0);

    double t_sym() const { return t_sym_; }
    double sps() const { return sps_; }
    int half() const { return half_; }
    double g0() const { return g0_; }
    bool is_complex() const { return static_cast<bool>(ctab_); }
    std::size_t size() const;
    // The taps; std::logic_error when asking a complex table for real ones
    // or the reverse.
    std::span<const double> tab() const;
    std::span<const std::complex<double>> ctab() const;

private:
    double t_sym_, sps_;
    int half_;
    double g0_;
    std::shared_ptr<const std::vector<double>> tab_;
    std::shared_ptr<const std::vector<std::complex<double>>> ctab_;
};

// track._Known: known stream symbols as positions and values -- real (CE's
// +-1) or complex (L's QPSK and references).
struct Known {
    std::vector<std::int64_t> pos;
    std::variant<std::vector<double>, std::vector<std::complex<double>>> val;

    bool is_complex() const { return val.index() == 1; }
    const std::vector<double>& real() const { return std::get<0>(val); }
    const std::vector<std::complex<double>>& cplx() const { return std::get<1>(val); }
};

// track.Classes: per stream symbol, template mean and variance and which
// are known (CE). known is 0/1 per symbol.
struct Classes {
    std::vector<double> mu;
    std::vector<double> nu;
    std::vector<std::uint8_t> known;
    bool hdr_known = false;
    std::optional<std::vector<double>> spread;   // float64[n_sym], a known spread header
    double spread_rho = 0.0;                     // unknown spread-header variance on the data
};

// ltrack.LClasses: L's classes, complex means and a class gain. Its
// `spread` is always None in the reference and has no field here.
struct LClasses {
    std::vector<std::complex<double>> mu;
    std::vector<double> nu;
    std::vector<std::uint8_t> known;
    std::vector<double> gain;
    std::optional<std::vector<std::uint8_t>> hdr_bits;   // uint8[2474] once the header is known
    bool hdr_known = false;
    double spread_rho = 0.0;
};

// --- the tracker's result ---------------------------------------------------

// track.TrackResult: everything one tracker run produced, on its grid of
// positions. A per-position array the reference leaves None until the
// tracker fills it (m, u, P, N, psi, pu) is empty here until then. The
// channel and z_d are shared, not copied, between the results of the
// tracker runs over one capture.
struct TrackResult {
    frame::FrameSpec spec;
    std::shared_ptr<const ChanCapture> chan;
    types::Timing timing;
    PathFn freq;
    std::vector<std::int64_t> pos;                                    // grid positions
    std::shared_ptr<const std::vector<std::complex<double>>> z_d;
    std::vector<std::complex<double>> m;
    std::vector<std::complex<double>> c;
    std::vector<std::complex<double>> u;
    std::vector<double> P;
    std::vector<double> N;
    std::vector<double> psi;
    std::vector<std::uint8_t> meas;
    std::vector<double> pu;
    double q = 0.0;
    double loglik = 0.0;
    std::variant<std::monostate, Classes, LClasses> classes;          // monostate: None
    std::optional<std::vector<std::uint8_t>> keying;
    bool ook = false;
    double lead_in_s = 0.0;
    double z_ref = std::numeric_limits<double>::quiet_NaN();
    double df_max = 0.0;
    std::optional<types::TrackReport> report;

    TrackResult(frame::FrameSpec s, std::shared_ptr<const ChanCapture> ch, types::Timing tm, PathFn f)
        : spec(std::move(s)), chan(std::move(ch)), timing(std::move(tm)), freq(std::move(f)) {}
    virtual ~TrackResult() = default;
    TrackResult(const TrackResult&) = default;
    TrackResult& operator=(const TrackResult&) = default;
    TrackResult(TrackResult&&) = default;
    TrackResult& operator=(TrackResult&&) = default;

    // int(pos[0]); std::out_of_range on an empty grid (Python's IndexError).
    std::int64_t p_lo() const;
    // Grid index of position(s) p.
    std::int64_t gi(std::int64_t p) const { return p - p_lo(); }
    std::vector<std::int64_t> gi(std::span<const std::int64_t> p) const;
    // spread_of_q(q, spec.t_sym)
    double doppler_hz() const;
    // Receiver time (s after t0) of positions p (default: the grid).
    std::vector<double> t_pos_s() const;
    std::vector<double> t_pos_s(std::span<const double> p) const;

    const Classes* ce_classes() const { return std::get_if<Classes>(&classes); }
    const LClasses* l_classes() const { return std::get_if<LClasses>(&classes); }
};

// ltrack.LTrackResult: the inherited per-position fields are the data
// band's; the pilot stream's sit beside them (nullopt: None).
struct LTrackResult : TrackResult {
    using TrackResult::TrackResult;
    std::optional<std::vector<std::complex<double>>> m_pl;
    std::optional<std::vector<std::complex<double>>> c_pl;
    std::optional<std::vector<double>> N_pl;
    std::optional<std::vector<double>> psi_pl;
    std::optional<std::vector<std::uint8_t>> meas_pl;
    std::complex<double> delta{1.0, 0.0};
    std::optional<std::vector<std::complex<double>>> u_pl;
    std::optional<std::vector<double>> P_pl;
    double resid_ratio = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace sstvae::qrss::track

#endif  // SSTVAE_QRSS_TRACKBASE_HPP
