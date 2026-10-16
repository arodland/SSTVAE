// The slot receiver: the slot half of sstvae/qrss/receiver.py, the
// reference (design 6.1, 6.8; WP6), on the per-pass half (receiver.hpp).
//
// Not a format module: nothing here goes on the air. Constants are
// rx_config.hpp's (receiver.py's, generated) and config.hpp's.
//
//   receiver.py              here
//   _ghost_of                ghost_of
//   _aliases, _ghosts        aliases / aliases_idx, ghosts / ghosts_idx
//   _suspect_neighbour,      suspect_neighbour, suspect_beside
//   _suspect_beside
//   _a3_lines, _owned        a3_lines, owned
//   candidates               candidates
//   detect, _detect_joint    detect (a selection holding an L frame is
//                            _detect_joint, as the reference's), detect_joint
//   _gate                    gate
//   _ce_resid_ratio          ce_resid_ratio
//   receive_slot,            receive_slot (likewise), receive_slot_joint
//   _receive_slot_joint
//   capture_from_wav         capture_from_wav
//   receive_wav              receive_wav
//
// The orchestration itself is receiver_slot_flow.hpp's, one template the
// binding also runs over Python objects; these functions are it over the
// C++ records and stages. rx_api.hpp is the application's entry.
//
// A selection (`sel`) is the reference's `spec`: a span of frames read as
// a tuple of them (frames(): CE's first, at most one per waveform). One CE
// frame is the CE receiver, call for call; a selection holding an L frame
// is the joint receiver.
//
// Seams. Every stage the reference calls through a module attribute a
// test may replace is a std::function in SlotSeams (receive_pass's own
// in SlotSeams::pass); an empty one is the C++ here. `now` is the clock
// behind budget_s (time.monotonic: empty is std::chrono::steady_clock),
// and `old_gating` the rules.GATING switch (nullopt: the environment, read
// where the reference reads it). The noise tables every stage fills go
// through SlotSeams::pass.noise_psd (channels) and acquire.noise_psd (the
// lead-in), which rx_api.hpp sets to take the EDGE_NOISE rule per call.
//
// Exactness. Everything decided here -- which candidates go to gate V and
// in which order, what each is handed (the noise exclusions as
// tuple(sorted(set))), the merges, the alias and ghost rules, the budget's
// clock readings, max_candidates' count -- is the reference's arithmetic
// in the reference's order, so with the same stages underneath the
// detections and passes are the reference's to the bit. ce_resid_ratio is
// bitwise (numpy's pairwise sums, np.abs as hypot, numpy's complex
// product). What the stages agree to is theirs (acquire, track, ltrack,
// receive_pass). One difference: Python sorts a NaN Z_ref where its
// timsort leaves it, std::stable_sort elsewhere; no stage makes one.
//
// Errors are the reference's where it has one (frames' refusals as
// std::invalid_argument; a detection without timing where a rule needs
// tau0 as std::invalid_argument naming it), and whatever the stages raise.
//
// Thread-safe: no state of its own; the seams are called on the caller's
// thread. With old_gating nullopt or an empty noise_psd the environment is
// read (rules.hpp: set the rules before any receiver runs, or pass them).

#ifndef SSTVAE_QRSS_RECEIVER_SLOT_HPP
#define SSTVAE_QRSS_RECEIVER_SLOT_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/acquire.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/receiver.hpp"
#include "qrss/receiver_slot_flow.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::receiver {

inline constexpr double GHOST_HZ = rxconfig::receiver::GHOST_HZ;
inline constexpr double GHOST_T = rxconfig::receiver::GHOST_T;
inline constexpr double GHOST_Z_RATIO = rxconfig::receiver::GHOST_Z_RATIO;
inline constexpr double ALIAS_HZ = rxconfig::receiver::ALIAS_HZ;
inline constexpr double ALIAS_RATIO = rxconfig::receiver::ALIAS_RATIO;
inline constexpr double L_OWN_BELOW_HZ = rxconfig::receiver::L_OWN_BELOW_HZ;
inline constexpr double L_OWN_ABOVE_HZ = rxconfig::receiver::L_OWN_ABOVE_HZ;
inline constexpr double CE_RESID_MIN = rxconfig::receiver::CE_RESID_MIN;

using Erase = std::vector<std::pair<double, double>>;
using flow::A3Line;

// --- the rules ----------------------------------------------------------------------------------

// _ghost_of: whether a detection at (f_hz, tau0) is a ghost of a stronger
// one at (g_f_hz, g_tau0): within GHOST_HZ and GHOST_T symbols of t_sym.
bool ghost_of(double f_hz, double tau0, double g_f_hz, double g_tau0, double t_sym = config::T_SYM);

// _aliases(dets, t_sym): strongest first, without a detection at a
// stronger one's f +- k/T (k = 1, 2) with its timing within k + 1/2
// symbols and under 1/ALIAS_RATIO of its Z_ref. _idx: the indices kept,
// in that order. std::invalid_argument for a detection without timing
// where the rule reads its tau0.
std::vector<std::size_t> aliases_idx(std::span<const types::Detection> dets, double t_sym = config::T_SYM);
std::vector<types::Detection> aliases(std::span<const types::Detection> dets, double t_sym = config::T_SYM);

// _ghosts(dets, t_sym): strongest first, without the ghosts of a stronger one.
std::vector<std::size_t> ghosts_idx(std::span<const types::Detection> dets, double t_sym = config::T_SYM);
std::vector<types::Detection> ghosts(std::span<const types::Detection> dets, double t_sym = config::T_SYM);

// _suspect_neighbour(p, out): p SUSPECT, within GHOST_HZ of a sound pass
// in out and under GHOST_Z_RATIO of its Z_ref.
bool suspect_neighbour(const types::PassResult& p, std::span<const types::PassResult> out);
// _suspect_beside(p, out): the same by band centre (the joint receiver).
bool suspect_beside(const types::PassResult& p, std::span<const types::PassResult> out);

// _a3_lines(cands): (line, centre, Z) of each whole-slot (A3) candidate.
std::vector<A3Line> a3_lines(std::span<const types::Detection> cands);

// _owned(d, found, a3): whether candidate d is a whole-slot line in a
// preamble-found signal's own band and no stronger in A3 than its own line
// (so not gated). old_gating nullopt: the environment (rules.GATING).
bool owned(const types::Detection& d, std::span<const types::Detection> found, std::span<const A3Line> a3,
           std::optional<bool> old_gating = std::nullopt);

// _ce_resid_ratio(tr): R of a CE gate-V track, the known symbols' residual
// |m - u c|^2 over N + |c|^2 P weighted as Z_ref weighs them; 1.0 when no
// known symbol is usable. std::invalid_argument for a track without CE
// classes.
double ce_resid_ratio(const track::TrackResult& tr);

// --- the stages, replaceable ------------------------------------------------------------------

// What a selection is: the reference's `spec`, a frame or a tuple of frames.
using Selection = std::span<const frame::FrameSpec>;

// Every stage the slot receiver calls through a module attribute in the
// reference; empty is the C++ here. Arguments are those the reference
// passes.
struct SlotSeams {
    // receive_pass's own (prepare, channel_for, track.verify,
    // ltrack.verify_l, frontend.noise_psd), which detect and _gate also
    // call: prepare(cap), channel_for(prep, spec, d, others),
    // track.verify(chan, CH_FS, spec, d, erase_s=), ltrack.verify_l(...).
    ReceiverSeams pass;
    // The acquisition's stages, for the C++ candidates (its noise_psd is
    // also the lead-in's in detect and _gate).
    acquire::AcquireSeams acquire;
    // receiver.candidates(prep, spec, live_only, hints): `sel` one frame
    // (the CE receiver's) or the selection's frames (the joint's).
    std::function<std::vector<types::Detection>(const Prepared& prep, Selection sel, bool joint, bool live_only,
                                                std::span<const acquire::Hint> hints)>
        candidates;
    // track.verify_track(chan, spec, v, fit=False, erase_s=): null is None.
    std::function<std::shared_ptr<track::TrackResult>(std::shared_ptr<const track::ChanCapture> chan,
                                                      const frame::FrameSpec& spec, const types::Detection& v,
                                                      std::span<const std::pair<double, double>> erase_s)>
        verify_track;
    // receiver._ce_resid_ratio(tr)
    std::function<double(const track::TrackResult& tr)> ce_resid_ratio;
    // acquire.lead_in_span(prep.fe, prep.t0_index, f_hz, 0.0, tau0_s, prep.fs[, t_sym=])
    std::function<double(const Prepared& prep, double f_hz, double tau0_s, std::optional<double> t_sym)>
        lead_in_span;
    // receiver._gate(prep, spec, d, others, erase_s)
    std::function<std::optional<types::Detection>(const Prepared& prep, const frame::FrameSpec& spec,
                                                  const types::Detection& d, std::span<const double> others,
                                                  std::span<const std::pair<double, double>> erase_s)>
        gate;
    // receiver.detect(prep, spec, live_only, max_candidates, budget_s,
    // erase_s=, hints=), as receive_slot calls it
    std::function<std::vector<types::Detection>(
        const Prepared& prep, Selection sel, bool joint, bool live_only, std::optional<std::int64_t> max_candidates,
        std::optional<double> budget_s, std::span<const std::pair<double, double>> erase_s,
        std::span<const acquire::Hint> hints)>
        detect;
    // receiver.receive_pass(prep, spec, d, estimator=, exclude_hz=, round_b=, erase_s=)
    std::function<std::optional<types::PassResult>(const Prepared& prep, const frame::FrameSpec& spec,
                                                   const types::Detection& d, const PassArgs& args)>
        receive_pass;
    // time.monotonic(), seconds
    std::function<double()> now;
    // rules.old(rules.GATING); nullopt: the environment
    std::optional<bool> old_gating;
};

// --- acquisition and gate V ---------------------------------------------------------------------

// candidates(prep, spec, live_only, hints): the candidates for gate V,
// preamble hits first (A1 -> A2, and A3 unless live_only), with the hints
// A2 did not pass after them. Joint: the waveform-aware acquisition and
// the hints as given; else CE's, with ce_hints(hints).
std::vector<types::Detection> candidates(const Prepared& prep, Selection sel, bool live_only = false,
                                         std::span<const acquire::Hint> hints = {}, const SlotSeams& seams = {});

// detect's keyword arguments. skip_hz: frequencies (CE) or band centres
// (joint) of signals known already.
struct DetectOptions {
    bool live_only = false;
    std::optional<std::int64_t> max_candidates;
    std::optional<double> budget_s;
    Erase erase_s;
    std::optional<std::vector<types::Detection>> cands;
    std::vector<double> skip_hz;
    std::vector<acquire::Hint> hints;
};

// detect(prep, spec, ...): acquisition plus gate V, verified detections
// strongest first (CE: merged within MERGE_HZ, then aliases and ghosts;
// joint: by band centre, aliases and ghosts per waveform at its symbol).
std::vector<types::Detection> detect(const Prepared& prep, Selection sel, const DetectOptions& opt = {},
                                     const SlotSeams& seams = {});
// _detect_joint(prep, specs, ...): the joint receiver's detect whatever the
// frames are (specs as frames() gives them; a CE-only tuple still gates
// through _gate, after CE's acquisition, as the reference's does).
std::vector<types::Detection> detect_joint(const Prepared& prep, Selection specs, const DetectOptions& opt = {},
                                           const SlotSeams& seams = {});

// _gate(prep, spec, d, others, erase_s): gate V of one joint-receiver
// candidate with its waveform's frame (a CE accept rechecked on the
// residual ratio), its lead-in measured at the frame's symbol.
std::optional<types::Detection> gate(const Prepared& prep, const frame::FrameSpec& spec, const types::Detection& d,
                                     std::span<const double> others,
                                     std::span<const std::pair<double, double>> erase_s = {},
                                     const SlotSeams& seams = {});

// --- the slot -------------------------------------------------------------------------------------

// receive_slot's keyword arguments. dets: receive these instead of
// detecting (in that order; each is its own object).
struct SlotOptions {
    bool live_only = false;
    std::string estimator = "joint";
    Erase erase_s;
    bool round_b = true;
    std::optional<std::vector<types::Detection>> dets;
    std::optional<std::int64_t> max_candidates;
    std::optional<double> budget_s;
    std::vector<acquire::Hint> hints;
};

// receive_slot(cap, spec, ...): A1 -> A2 (+ A3) -> V -> receive_pass for
// every signal in a stored slot, after tracking without ghosts, weak
// SUSPECT neighbours and second passes of one signal.
std::vector<types::PassResult> receive_slot(CaptureRef cap, Selection sel, const SlotOptions& opt = {},
                                            const SlotSeams& seams = {});
// _receive_slot_joint(prep, specs, ...): the joint receiver's slot over a
// prepared capture whatever the frames are.
std::vector<types::PassResult> receive_slot_joint(const Prepared& prep, Selection specs, const SlotOptions& opt = {},
                                                  const SlotSeams& seams = {});

// --- a WAV file ---------------------------------------------------------------------------------------

// capture_from_wav's capture from audio already read (float64 at
// config::FS): frontend::audio_to_fe, t0 = 900 q + 1, the audio starting
// start_unix (nullopt: 12 s before t0, as qrss_transmit.py writes it).
frontend::Capture capture_from_audio(std::span<const double> x8k, std::int64_t q,
                                     std::optional<double> start_unix = std::nullopt);

// capture_from_wav(path, q, start_unix): a slot capture from a WAV, read as
// sstvae.wavio.read_wav reads it (wavread.hpp; wav::ReadError when the file
// cannot be read, wav::FormatError for a file the reference refuses).
frontend::Capture capture_from_wav(const std::string& path, std::int64_t q,
                                   std::optional<double> start_unix = std::nullopt);

// receive_wav(path, q, start_unix, spec, estimator)
std::vector<types::PassResult> receive_wav(const std::string& path, std::int64_t q,
                                           std::optional<double> start_unix, Selection sel,
                                           const std::string& estimator = "joint", const SlotSeams& seams = {});

}  // namespace sstvae::qrss::receiver

#endif  // SSTVAE_QRSS_RECEIVER_SLOT_HPP
