// The QRSSTVAE receiver's A3, track-before-detect over a stored slot: the
// "--- A3 ---" part of sstvae/qrss/acquire.py with what acquire() needs
// around it (frame_map, _path_of). The statistics A3 shares with A1/A2
// (spectrogram, cfar_normalise, tbd_gumbel, gumbel_p, band_cov) are
// acquire_stats.hpp's.
//
//   acquire.py               here
//   tbd_frame_starts         tbd_frame_starts
//   tbd_n_frames             tbd_n_frames
//   tbd_viterbi              tbd_viterbi (one channel at a time)
//   TbdHit                   TbdHit (f_hz, line, passed, to_detection)
//   _path_line               path_line
//   tbd_channels             tbd_channels
//   tbd_emissions            tbd_emissions
//   shape_frames             shape_frames
//   shape_z                  shape_z
//   shape_waveform           shape_waveform
//   track_before_detect      track_before_detect
//   _shape_verdict           shape_verdict
//   _bins_path               bins_path
//   l_pilot_bins             l_pilot_bins
//   frame_map                frame_map
//   _path_of                 path_of
//
// The reference stacks every channel's emissions into one float64 array
// (n_ch, n_frames, 2 half + 1) and runs its Viterbi on all of them at
// once; here each channel's runs on its own, reading the float32
// emissions in place (numpy widens them exactly), so the arithmetic is the
// reference's, bit for bit, at a fraction of its memory (on a FULL slot:
// the reference's float64 stack alone is ~170 MB).
//
// Exactness. Bit-exact: the frame starts, every Viterbi path and Z, the
// channel centres, shape_frames, shape_z and l_pilot_bins on float32
// emissions (numpy's pairwise and buffered sums), and bins_path on float32
// emissions -- its log is numpy's float32 SIMD log, reproduced in
// math/nplog.hpp (np::log_f32). To a tolerance: path_line, and so TbdHit.f_hz and the
// pilot line l_pilot_bins follows, go through math::polyfit (an SVD that
// is not LAPACK's: ~1e-15 relative); gumbel_p calls numpy's SIMD exp and
// expm1 (acquire_stats.hpp); bins_path on float64 emissions takes libm's
// log where numpy calls SVML's (an ulp). The discrete decisions downstream
// (dedup within 10 Hz, p < pfa, the line's 1 Hz cut) match unless a value
// sits within those rounding errors of a boundary.
//
// Errors are the reference's: ValueError (std::invalid_argument),
// IndexError (std::out_of_range), OverflowError (std::overflow_error), and
// UnboundLocal for path_line's UnboundLocalError (a path of under two
// points). Shapes the reference only reaches by accident -- a band
// narrower than one channel, a Viterbi step past twice the bins -- are
// NotPorted (the bindings raise NotImplementedError).
//
// Thread-safe: everything is pure (band_cov's cache is acquire_stats'
// mutex-protected one).

#ifndef SSTVAE_QRSS_ACQUIRE_TBD_HPP
#define SSTVAE_QRSS_ACQUIRE_TBD_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/acquire_stats.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::acquire {

// Python's UnboundLocalError (a NameError).
struct UnboundLocal : std::logic_error {
    using std::logic_error::logic_error;
};

// A shape the reference handles only by accident (NotImplementedError).
struct NotPorted : std::logic_error {
    using std::logic_error::logic_error;
};

// Frame start times (s after t0) covering [t0 - 10 s, end of keying + 2 s],
// limited, given n_fe, to frames lying wholly inside a capture of n_fe
// samples whose t0 is at index t0_index.
std::vector<double> tbd_frame_starts(const frame::FrameSpec& spec, std::optional<std::int64_t> n_fe = std::nullopt,
                                     double t0_index = 0.0, double fs = config::FE_FS);

std::int64_t tbd_n_frames(const frame::FrameSpec& spec);

// tbd_viterbi on one channel: the best path (bin per frame) and Z = the
// sum of the emissions on it / sqrt(n_frames).
struct TbdPath {
    std::vector<std::int64_t> path;
    double z = 0.0;
};

// One channel's emissions: rows frames of n_bins values, frame t's at
// data[t * stride + col0 .. + n_bins), widened to float64 as numpy does.
// rows == 0 is IndexError, n_bins == 0 ValueError, |max_step| past int8
// (with two frames or more) OverflowError, all in numpy's words; a step
// past n_bins and under 2 n_bins (numpy's broadcast refusal, worded with
// the array's lead dimensions) is NotPorted.
template <typename T>
TbdPath tbd_viterbi(const T* data, std::size_t rows, std::size_t stride, std::size_t col0, std::size_t n_bins,
                    int max_step = rxconfig::acquire::TBD_MAX_STEP, double cost = rxconfig::acquire::TBD_COST);
TbdPath tbd_viterbi(const Grid<float>& E, int max_step = rxconfig::acquire::TBD_MAX_STEP,
                    double cost = rxconfig::acquire::TBD_COST);
TbdPath tbd_viterbi(const Grid<double>& E, int max_step = rxconfig::acquire::TBD_MAX_STEP,
                    double cost = rxconfig::acquire::TBD_COST);

// _path_line: (slope Hz/s, intercept Hz at t0) of the emission-weighted
// line through a path, refitted once without points over 1 Hz off it.
std::vector<double> path_line(const types::FreqPath& path);

// TbdHit: A3's best path in one channel.
struct TbdHit {
    types::FreqPath path;
    double z = 0.0;
    double p_value = 1.0;
    std::int64_t n_frames = 0;
    double centre_hz = 0.0;
    double shape_z = std::numeric_limits<double>::quiet_NaN();
    double above_z = std::numeric_limits<double>::quiet_NaN();
    std::optional<std::int64_t> waveform;
    std::optional<types::FreqPath> l_path;
    double l_shape_z = std::numeric_limits<double>::quiet_NaN();
    double l_above_z = std::numeric_limits<double>::quiet_NaN();

    std::vector<double> line() const { return path_line(path); }
    double f_hz() const;                 // polyval(line(), 0)
    bool passed() const { return p_value < rxconfig::acquire::TBD_PFA; }
    // A detection with no timing, gated as `waveform` (default: the
    // shape's, else CE); as L on the pilot-anchored path when there is one.
    types::Detection to_detection(std::optional<std::int64_t> waveform = std::nullopt) const;
};

// tbd_channels: channel centres, every candidate frequency plus a 50 Hz
// grid (a grid centre within 25 Hz of a candidate dropped), all kept 40 Hz
// inside the band, ascending.
std::vector<double> tbd_channels(std::span<const double> cand_f_hz,
                                 std::array<double, 2> band_hz = rxconfig::acquire::BAND_HZ);

// tbd_emissions: E (float32 normalised power - 1, frames x band bins),
// the band's audio frequencies and the frame starts.
struct Emissions {
    Grid<float> E;
    std::vector<double> f;
    std::vector<double> t;
};
Emissions tbd_emissions(std::span<const std::complex<float>> fe, double t0_index, const frame::FrameSpec& spec,
                        double fs = config::FE_FS, std::array<double, 2> band_hz = rxconfig::acquire::BAND_HZ);
Emissions tbd_emissions(std::span<const std::complex<double>> fe, double t0_index, const frame::FrameSpec& spec,
                        double fs = config::FE_FS, std::array<double, 2> band_hz = rxconfig::acquire::BAND_HZ);
Emissions tbd_emissions(std::span<const double> fe, double t0_index, const frame::FrameSpec& spec,
                        double fs = config::FE_FS, std::array<double, 2> band_hz = rxconfig::acquire::BAND_HZ);

// shape_frames: 0/1 per frame start, 1 for a frame overlapping no
// callsign window of any spec and starting no earlier than the latest
// keyed start. An empty specs is max()'s ValueError.
std::vector<std::uint8_t> shape_frames(std::span<const double> t_s, std::span<const frame::FrameSpec> specs,
                                       double frame_s = rxconfig::acquire::STFT_FRAME_S);

// shape_z: (z1, z_above) along a path (bins per frame), over the frames in
// keep (all when absent); (NaN, NaN) with none usable. bins and keep must
// have E.rows entries (std::invalid_argument otherwise, numpy's broadcast
// refusal).
std::pair<double, double> shape_z(const Grid<float>& E, std::span<const std::int64_t> bins,
                                  std::optional<std::span<const std::uint8_t>> keep = std::nullopt,
                                  double df_hz = 1.0 / rxconfig::acquire::STFT_FRAME_S, double fs = config::FE_FS);
std::pair<double, double> shape_z(const Grid<double>& E, std::span<const std::int64_t> bins,
                                  std::optional<std::span<const std::uint8_t>> keep = std::nullopt,
                                  double df_hz = 1.0 / rxconfig::acquire::STFT_FRAME_S, double fs = config::FE_FS);

// shape_waveform: L, CE or nullopt (ambiguous).
std::optional<std::int64_t> shape_waveform(double z1, double z_above);

// _shape_verdict(v, w).
std::optional<std::int64_t> shape_verdict(std::optional<std::int64_t> v, std::optional<std::int64_t> w);

// _bins_path: the FreqPath of a path through E (bin per frame starting at
// t): frame centres, the bins' frequencies refined by a parabola on the
// log power, weights the emissions floored at 0.
types::FreqPath bins_path(const Grid<float>& E, std::span<const double> f, std::span<const double> t,
                          std::span<const std::int64_t> bins);
types::FreqPath bins_path(const Grid<double>& E, std::span<const double> f, std::span<const double> t,
                          std::span<const std::int64_t> bins);

// l_pilot_bins: an L pilot's bins near a path whose line is `line`.
std::vector<std::int64_t> l_pilot_bins(const Grid<float>& E, std::span<const double> f, std::span<const double> t,
                                       std::span<const double> line,
                                       std::optional<std::span<const std::uint8_t>> keep = std::nullopt,
                                       double fs = config::FE_FS,
                                       std::array<double, 2> span_hz = rxconfig::acquire::L_ANCHOR_HZ);
std::vector<std::int64_t> l_pilot_bins(const Grid<double>& E, std::span<const double> f, std::span<const double> t,
                                       std::span<const double> line,
                                       std::optional<std::span<const std::uint8_t>> keep = std::nullopt,
                                       double fs = config::FE_FS,
                                       std::array<double, 2> span_hz = rxconfig::acquire::L_ANCHOR_HZ);

// track_before_detect's keyword arguments. shape_specs: the frames of a
// waveform-aware acquisition (shape test on each kept hit; with L among
// them the pilot-anchored path too). keep_fn: how the frames the shape
// test uses are found from the frame starts (default shape_frames(t,
// shape_specs)); the binding passes the Python attribute through it.
struct TbdOptions {
    double pfa = rxconfig::acquire::TBD_PFA;
    std::array<double, 2> band_hz = rxconfig::acquire::BAND_HZ;
    bool all_hits = false;
    std::optional<std::vector<frame::FrameSpec>> shape_specs;
    std::function<std::vector<std::uint8_t>(std::span<const double> t)> keep_fn;
};

// track_before_detect: one Viterbi path per channel, hits at p < pfa (all
// with all_hits), deduplicated within 10 Hz, best first.
std::vector<TbdHit> track_before_detect(const Emissions& em, std::span<const double> cand_f_hz,
                                        double fs = config::FE_FS, const TbdOptions& opt = {});
std::vector<TbdHit> track_before_detect(std::span<const std::complex<float>> fe, double t0_index,
                                        const frame::FrameSpec& spec, std::span<const double> cand_f_hz = {},
                                        double fs = config::FE_FS, const TbdOptions& opt = {});
std::vector<TbdHit> track_before_detect(std::span<const std::complex<double>> fe, double t0_index,
                                        const frame::FrameSpec& spec, std::span<const double> cand_f_hz = {},
                                        double fs = config::FE_FS, const TbdOptions& opt = {});

// frame_map: (waveform, index into specs), one per waveform, in first-seen
// order (a dict's). Two frames of one waveform, or none, is ValueError.
std::vector<std::pair<std::int64_t, std::size_t>> frame_map(std::span<const frame::FrameSpec> specs);

// _path_of: the straight path f_hz + drift t / 60 at t = -10, 0, dur.
types::FreqPath path_of(double f_hz, double drift_hz_per_min, double dur);

}  // namespace sstvae::qrss::acquire

#endif  // SSTVAE_QRSS_ACQUIRE_TBD_HPP
