// The QRSSTVAE receiver's acquisition as a whole: the "--- everything at
// once ---" part of sstvae/qrss/acquire.py (the reference), arbitrating
// A1 (carrier lines), A2 (preamble search) and A3 (track-before-detect)
// of acquire_a12.hpp and acquire_tbd.hpp into the candidates the
// verification gate takes.
//
//   acquire.py                      here
//   Acquisition                     Acquisition
//   Acquisition.detections(spec)    Acquisition::detections(spec)
//   Acquisition.detections(tuple)   Acquisition::detections(specs)
//   Acquisition._joint_detections   joint_detections(acq, frame map)
//   drop_symbol_rate_aliases        drop_symbol_rate_aliases(_idx)
//   drop_band_owned                 drop_band_owned(_idx)
//   merge_hits                      merge_hits(_idx)
//   live_end_s                      live_end_s
//   acquire(cap, spec)              acquire(fe, t0_index, fs, spec, ...)
//   acquire(cap, (spec, ...))       acquire(fe, t0_index, fs, specs, ...)
//   _hint_items                     hint_items
//   _acquire_joint                  acquire_joint
//   (frame_map, _path_of: acquire_tbd.hpp)
//
// Exactness. The arbitration itself (sorts, the alias, band-ownership and
// merge rules, the hint logic, the detections' order and dedup) is the
// reference's arithmetic in the reference's order: given the same hits it
// keeps the same hits, in the same order. Python's sorted() is stable and
// so is every sort here (std::stable_sort on the same keys); a NaN key,
// which no passed hit carries, could order differently. What the stages
// underneath agree to is theirs: A1 bit for bit, A2 to rounding
// (acquire_a12.hpp), A3 bit for bit but for p_value and the path lines
// (acquire_tbd.hpp) -- so a decision here can differ only where an A2 or
// A3 value sits within that rounding of one of the rules' thresholds.
//
// Seams. acquire() reaches every stage through AcquireSeams: an empty
// std::function is the native stage; a filled one replaces it. The
// Python binding fills one wherever acquire's module attribute is not the
// native row (a test that monkeypatches acquire.carrier_lines, say), which
// keeps such tests meaningful with the C++ substituted. With no seams set
// nothing here needs Python.
//
// Errors are the reference's: std::invalid_argument for ValueError
// (frame_map's refusals), math::TypeError for TypeError (a waveform hint
// to a one-frame acquisition, where the reference's float() of a tuple
// fails); the stages' own errors pass through.
//
// Thread-safe: pure functions of their arguments (the stages' tables are
// built once, read-only).

#ifndef SSTVAE_QRSS_ACQUIRE_HPP
#define SSTVAE_QRSS_ACQUIRE_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "qrss/acquire_a12.hpp"
#include "qrss/acquire_tbd.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::acquire {

// {waveform: frame} in the order the frames were given (a dict's), at most
// one per waveform: frame_map's result with the frames themselves.
using FrameMap = std::vector<std::pair<std::int64_t, frame::FrameSpec>>;

// frame_map(specs) as a FrameMap (ValueError texts acquire_tbd's).
FrameMap frame_map_of(std::span<const frame::FrameSpec> specs);

// What the acquisition found (acquire.Acquisition).
struct Acquisition {
    std::vector<types::CarrierCandidate> lines;    // A1's lines
    std::vector<PreambleHit> preamble;              // A2 hits that passed, arbitrated
    std::vector<TbdHit> tbd;                        // A3 hits that passed
    std::vector<types::CarrierCandidate> hinted;    // hints A2 did not pass
    std::vector<types::CarrierCandidate> pilots;    // L pilot hypotheses searched

    // detections(spec), one frame: preamble hits as stored, the hints (no
    // timing, waveform CE), then A3's -- the reference's single-frame order.
    std::vector<types::Detection> detections(const frame::FrameSpec& spec) const;
    // detections((spec, ...)): the waveform-aware order (_joint_detections).
    std::vector<types::Detection> detections(std::span<const frame::FrameSpec> specs) const;
};

// Acquisition._joint_detections(specs) with the frame map built.
std::vector<types::Detection> joint_detections(const Acquisition& acq, const FrameMap& specs);

// --- arbitration ------------------------------------------------------------------------
//
// Each takes the hits and returns the indices of the ones kept, in the
// order the reference returns them (so a caller holding its own objects
// keeps them); the hit-returning forms copy those hits out.

// drop_symbol_rate_aliases(hits, t_sym): strongest first, without the hits
// that are a stronger one's alias at f +- k/T, with its timing.
std::vector<std::size_t> drop_symbol_rate_aliases_idx(std::span<const PreambleHit> hits,
                                                      double t_sym = config::T_SYM);
std::vector<PreambleHit> drop_symbol_rate_aliases(std::span<const PreambleHit> hits, double t_sym = config::T_SYM);

// drop_band_owned(hits): strongest first, without the L hits in a stronger
// L hit's data band (3.1-22.1 Hz below its pilot) with its timing.
std::vector<std::size_t> drop_band_owned_idx(std::span<const PreambleHit> hits);
std::vector<PreambleHit> drop_band_owned(std::span<const PreambleHit> hits);

// merge_hits(hits): one hit per signal (within 5 Hz and T_L/2), the
// smaller p (then the larger Lambda) kept; strongest first.
std::vector<std::size_t> merge_hits_idx(std::span<const PreambleHit> hits);
std::vector<PreambleHit> merge_hits(std::span<const PreambleHit> hits);

// live_end_s(): how far past t0 a live acquisition reads FE.
double live_end_s();

// --- hints --------------------------------------------------------------------------------

// A hint: an audio frequency where a signal is expected, of one waveform
// (the reference's (f, waveform) pair) or of every enabled one (a bare f).
struct Hint {
    double f_hz = 0.0;
    std::optional<std::int64_t> waveform;
};

// _hint_items(hints, specs): (f, waveforms) per hint -- every enabled
// waveform for a bare f, in the map's order; [w] for (f, w) when w is
// enabled, else nothing.
std::vector<std::pair<double, std::vector<std::int64_t>>> hint_items(std::span<const Hint> hints,
                                                                     const FrameMap& specs);
// The same over the enabled waveforms alone (the frame map's keys, in order).
std::vector<std::pair<double, std::vector<std::int64_t>>> hint_items(std::span<const Hint> hints,
                                                                     std::span<const std::int64_t> waveforms);

// --- the stages, replaceable ----------------------------------------------------------------

using cf = std::complex<float>;
using cd = std::complex<double>;

// The front end as acquire() holds it: the caller's complex64 or
// complex128 samples, or (preprocessed = false) the blanked, normalised
// complex64 copy.
using FeView = std::variant<std::span<const cf>, std::span<const cd>>;

// Every stage acquire() calls, as the reference names it; empty is the
// native one. Arguments are those the reference passes; where it passes a
// keyword only sometimes, the pointer / optional is null when it does not.
struct AcquireSeams {
    // fe, keep = frontend.blank(fe, fs=fs); fe, _ = frontend.normalise(fe, keep, fs=fs)
    std::function<std::vector<cf>(const FeView& fe, double fs)> prepare;
    // carrier_line_stats(fe, t0_index, fs)
    std::function<CarrierStats(const FeView& fe, double t0_index, double fs)> carrier_line_stats;
    // carrier_lines(fe, t0_index, fs, stats=cs)
    std::function<std::vector<types::CarrierCandidate>(const FeView& fe, double t0_index, double fs,
                                                       const CarrierStats& cs)>
        carrier_lines;
    // pilot_lines(cs, lines)
    std::function<std::vector<types::CarrierCandidate>(const CarrierStats& cs,
                                                       std::span<const types::CarrierCandidate> lines)>
        pilot_lines;
    // hint_line(cs, f_hz, fs)
    std::function<types::CarrierCandidate(const CarrierStats& cs, double f_hz, double fs)> hint_line;
    // a1_threshold(M, A1_PFA, fs)
    std::function<double(std::int64_t M, double pfa, double fs)> a1_threshold;
    // One channelisation at an A1 line feeding A2 of each kind:
    // inp = _a2_input(fe, t0_index, line.f_hz, line.drift_hz_per_min, fs), then
    // preamble_search(fe, t0_index, line, fs, kind=k, inp=inp) for k in kinds.
    std::function<std::vector<PreambleHit>(const FeView& fe, double t0_index, const types::CarrierCandidate& line,
                                           double fs, std::span<const PreambleKind* const> kinds)>
        line_search;
    // preamble_search(fe, t0_index, cand, fs[, kind=kind]); kind null: none passed (CE's).
    std::function<PreambleHit(const FeView& fe, double t0_index, const types::CarrierCandidate& cand, double fs,
                              const PreambleKind* kind)>
        search;
    // lead_in_span(fe, t0_index, f_hz, drift, tau0_s, fs[, t_sym=t_sym]).
    std::function<double(const FeView& fe, double t0_index, double f_hz, double drift_hz_per_min, double tau0_s,
                         double fs, std::optional<double> t_sym)>
        lead_in_span;
    // frontend.noise_psd for the native lead_in_span (acquire_a12's own seam).
    NoisePsdFn noise_psd;
    // track_before_detect(fe, t0_index, spec, lines, fs[, shape_specs=...]).
    std::function<std::vector<TbdHit>(const FeView& fe, double t0_index, const frame::FrameSpec& spec,
                                      std::span<const types::CarrierCandidate> lines, double fs,
                                      const std::vector<frame::FrameSpec>* shape_specs)>
        track_before_detect;
    // shape_frames(t, shape_specs) for the native track_before_detect.
    std::function<std::vector<std::uint8_t>(std::span<const double> t, std::span<const frame::FrameSpec> specs)>
        shape_frames;
};

// acquire()'s keyword arguments.
struct AcquireOptions {
    bool live_only = false;       // A1 -> A2 only, no A3
    bool preprocessed = false;    // fe is blanked and normalised already
    std::vector<Hint> hints;
};

// acquire(cap, spec): one frame -- CE's acquisition whatever the frame's
// waveform (only A3's span and the hints' paths follow the frame). A hint
// with a waveform is the reference's TypeError here (its float() of a
// tuple).
Acquisition acquire(std::span<const cf> fe, double t0_index, double fs, const frame::FrameSpec& spec,
                    const AcquireOptions& opt = {}, const AcquireSeams& seams = {});
Acquisition acquire(std::span<const cd> fe, double t0_index, double fs, const frame::FrameSpec& spec,
                    const AcquireOptions& opt = {}, const AcquireSeams& seams = {});
// acquire(cap, (spec, ...)): the waveform-aware acquisition (frame_map,
// then acquire_joint).
Acquisition acquire(std::span<const cf> fe, double t0_index, double fs, std::span<const frame::FrameSpec> specs,
                    const AcquireOptions& opt = {}, const AcquireSeams& seams = {});
Acquisition acquire(std::span<const cd> fe, double t0_index, double fs, std::span<const frame::FrameSpec> specs,
                    const AcquireOptions& opt = {}, const AcquireSeams& seams = {});
// The same on a Capture (cap.fe, cap.t0_index, cap.fs).
Acquisition acquire(const frontend::Capture& cap, const frame::FrameSpec& spec, const AcquireOptions& opt = {},
                    const AcquireSeams& seams = {});
Acquisition acquire(const frontend::Capture& cap, std::span<const frame::FrameSpec> specs,
                    const AcquireOptions& opt = {}, const AcquireSeams& seams = {});

// _acquire_joint(cap, specs, ...): with the frame map already built.
Acquisition acquire_joint(std::span<const cf> fe, double t0_index, double fs, const FrameMap& specs,
                          const AcquireOptions& opt = {}, const AcquireSeams& seams = {});
Acquisition acquire_joint(std::span<const cd> fe, double t0_index, double fs, const FrameMap& specs,
                          const AcquireOptions& opt = {}, const AcquireSeams& seams = {});

}  // namespace sstvae::qrss::acquire

#endif  // SSTVAE_QRSS_ACQUIRE_HPP
