// The known-waveform (template) search and retroactive detection: port of
// the template-search half of sstvae/qrss/em.py (the reference; its module
// docstring, "Template search", is the design: spec 7 step 3, design 8.3).
// em.hpp holds the pieces this stands on (the references, known_header,
// known_keying, template_z, the hooks).
//
// Not a format module: nothing here goes on the air. Constants are em.py's
// (em.hpp, rxconfig::em) and config.hpp's.
//
//   em.py                    here
//   _candidates              candidates
//   _chunk_template          chunk_template
//   _search_chan             search_chan
//   _search_input            search_input
//   _from_passband           from_passband
//   template_search          template_search (return_stats: the whole
//                            SearchResult; without it, SearchResult::det)
//   receive_template         receive_template
//   retro_detect             retro_detect (and retro_detect_steps, the same
//                            loop over replaceable steps: the binding's seams)
//
// THE STATISTIC. A capture is correlated with the picture's expected
// waveform in 2 s chunks, coherent within a chunk and summed in power over
// the frame: Lambda(lag, m) = sum_k |C_k(lag, m)|^2 / (N0 E_k), with C_k the
// circular cross-correlation (one 2048-point FFT of the capture's chunk
// window, one of the template chunk shifted by m bins, one inverse FFT per
// (k, m)). The best cell over every segment x candidate x clock step is
// Bonferroni-corrected into a normal Z (em::template_z); a detection needs Z
// > z_accept.
//
// EXACTNESS. The search is the reference's arithmetic in its order: numpy's
// FFTs are pocketfft's own plans called as numpy calls them (math/npfft),
// numpy's complex multiply and absolute value (math/npcomplex: the fused
// products, the SIMD hypot), x**2 as x*x, np.exp(-2j pi x) as cos/sin of
// -2 pi x (x already reduced to [0, 1)), np.sum along axis 1 as numpy's
// pairwise sum (npsum) and along axis 0 as numpy's row-by-row accumulation,
// np.argmax's first maximum (a NaN counting as the maximum) and the strict
// `>` between cells. So with the same inputs the winning cell is the
// reference's and Lambda agrees to the bit wherever its pieces do; the
// pieces that do not are ce_rx::mean_template (numpy's SIMD exp(-v/2) is 1
// ulp from libm's on a few percent of arguments: the template to ~1e-16
// relative where v != 0) and the noise floor (track::noise_floor, rtol
// ~1e-15). Measured against the reference: see the parity file
// (tests/test_qrss_native_em_search.py) and the port's insights.
//
// What is precomputed that the reference recomputes (with identical
// results, so no departure): a candidate's channel, its derotated capture
// and N0 are computed once and reused for every segment (the reference
// rebuilds them per segment); a 250 Hz capture's channel and noise floor
// once for all candidates (they do not depend on the candidate); the
// capture's FFT per chunk start once per (segment, candidate) instead of per
// clock step; and the FFT-domain roll of the template is an index offset,
// not a copied array. retro_detect prepares a slot's capture once for its
// search and its receive (the reference prepares it twice, identically).
//
// ERRORS are the reference's, texts included: std::invalid_argument
// (ValueError) for a CE-only refusal (em::ce_only, before anything else is
// read), "pass <uid> keeps no capture", "a 250 Hz capture needs the
// frequency it was mixed at (f_mix_hz)", "searching a PassbandStore needs
// the slot: pass q= (or use em.retro_detect to search every stored slot)",
// "the passband store holds nothing for slot q=<q>", prior_from's negative
// q; std::out_of_range "index 0 is out of bounds for axis 0 with size 0"
// (IndexError) where the reference indexes an empty chunk grid; the store's
// (KeyError "no accumulator <key>", FileError ...) passed through. One
// refusal is this port's own: a negative tau_s (std::invalid_argument
// "tau_s must be >= 0, not <tau_s>"), which the reference turns into a
// negative slice of the correlation; NaN and infinite tau_s / df_hz are
// int(math.ceil())'s ValueError / OverflowError ("cannot convert float NaN
// to integer" -> std::invalid_argument, "cannot convert float infinity to
// integer" -> std::overflow_error).
//
// THREADS. Every function is safe to call from any thread at once: no
// state of its own, no mutable globals; the FFT plans are pocketfft's
// mutex-guarded cache. The ones taking a Store read (and, for
// receive_template's caller, retro_detect, write) through it, and a Store
// serialises its own calls (store.hpp): nothing here holds its mutex across
// slow work -- each store call takes and releases it, and associate holds
// it only for its own transaction. A PassbandStore's capture() reads its
// files outside its lock (passband.hpp). All results are values.
//
// What blocks, and for how long (single-threaded, measured 2026-10-14 on
// this container, a shared 4-CPU x86-64 with AVX2; one segment, one
// candidate and the default 5 clock steps unless said otherwise -- the
// cost scales with segments x candidates x (2 ppm_steps + 1)):
//   candidates        loads every member's pass file (~5 ms each for FULL)
//   template_search   a raw FE Capture: TINY 0.17 s, SHORT 0.88 s, FULL
//                     6.3 s (receiver::prepare's blanking and normalising
//                     is ~0.08 / 0.25 / 2.5 s of that); a Prepared: 0.10 /
//                     0.62 / 3.8 s. The reference: 0.33 / 1.9 / 16.6 s on a
//                     Capture. The FFTs are ~55% of what is left (pocketfft
//                     at the library's baseline SIMD width, batched 32
//                     rows), the rolled products and |C|^2 the rest
//   receive_template  one tracker run and one demodulation: about what
//                     receiver::receive_pass's round B costs (0.1 s TINY,
//                     ~0.5 s SHORT, a few s FULL), plus the search when no
//                     det is given
//   retro_detect      one template_search per searchable slot plus one
//                     receive_template and one associate per find: 4.9 s
//                     over seven SHORT slots (five searched, two received;
//                     the reference 12.1 s), minutes over a full 48 h
//                     passband store. It polls hooks.cancel before every
//                     slot and between candidate hypotheses (template_
//                     search's poll), so a cancel is honoured within one
//                     (segment, candidate) search.

#ifndef SSTVAE_QRSS_EM_SEARCH_HPP
#define SSTVAE_QRSS_EM_SEARCH_HPP

#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "qrss/associate.hpp"
#include "qrss/ce.hpp"
#include "qrss/em.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/passband.hpp"
#include "qrss/rational.hpp"
#include "qrss/receiver.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::em {

// --- the hypotheses ------------------------------------------------------------------------

// One (frequency, drift, clock) hypothesis: (f_hz audio Hz, drift Hz/min, ppm).
struct Hypothesis {
    double f_hz = 0.0;
    double drift_hz_per_min = 0.0;
    double ppm = 0.0;
    bool operator==(const Hypothesis&) const = default;
};

// _candidates(store, acc, extra_hz): every member's (report.offset_hz,
// report.drift_hz_per_min, timing.ppm), in member order, skipping a member
// whose pass cannot be loaded (the reference's `except (OSError,
// KeyError)`: store::FileError and pyjson::KeyError -- a missing file, an
// archive member or meta key missing, a member without "uid") and every L
// member; then (f, 0, 0) for each of `extra_hz`; then deduplicated in
// order, a hypothesis dropped when an earlier kept one is within 0.25 Hz,
// 0.05 Hz/min and 1 ppm of it. Other load errors (a bad uid's ValueError, a
// corrupt file's) propagate.
std::vector<Hypothesis> candidates(store::Store& store, const store::Accumulator& acc,
                                   std::span<const double> extra_hz = {});

// --- the chunked template ------------------------------------------------------------------

// _chunk_template's (starts, chunks, energies): the mean waveform on the CH
// grid in `chunk`-sample chunks (E row-major, size() x chunk), the chunks
// whose energy is not above 1e-6 of the largest dropped.
struct ChunkTemplate {
    std::vector<std::int64_t> starts;
    std::vector<std::complex<double>> E;
    std::vector<double> en;
    std::int64_t chunk = 0;
    std::size_t size() const { return starts.size(); }
};

// _chunk_template(spec, mu, nu, keying, ook, t0_index, ppm, n_ch, chunk):
// sample n of the channel at tau = (n - t0_index)/(T fs (1 + ppm 1e-6))
// symbols, n over the keyed span inside [0, n_ch). `keying` a valid keying
// (ce::window_keying) or null for None. std::out_of_range "index 0 is out of
// bounds for axis 0 with size 0" when that span holds no sample; mu / nu
// refused as ce_rx::mean_template refuses them.
ChunkTemplate chunk_template(const frame::FrameSpec& spec, std::span<const double> mu, std::span<const double> nu,
                             const ce::Keying* keying, bool ook, double t0_index, double ppm, std::int64_t n_ch,
                             std::int64_t chunk);

// --- one channel's search ------------------------------------------------------------------

// _search_chan's best cell: Lambda, its lag (CH samples, -lag_max ..
// lag_max) and frequency bin m (-m_max .. m_max). lam -inf (lag 0, m 0) when
// no cell beat -inf (m_max < 0, or every Lambda NaN).
struct ChanPeak {
    double lam = -std::numeric_limits<double>::infinity();
    std::int64_t lag = 0;
    std::int64_t m = 0;
    bool operator==(const ChanPeak&) const = default;
};

// _search_chan(z, N0, starts, E, en, lag_max, m_max, drift_bins): the
// maximum over (lag, m) of Lambda, the first maximum in lag of each m (as
// np.argmax) and the first m to reach it (strict >). drift_bins: one int per
// chunk (empty: zeros, as template_search passes; otherwise
// std::invalid_argument unless one per chunk). The FFT length is TS_NFFT,
// doubled while shorter than chunk + 2 lag_max. No chunk: every Lambda 0,
// so (0, -lag_max, -m_max). std::invalid_argument for lag_max < 0.
ChanPeak search_chan(std::span<const std::complex<double>> z, double N0, const ChunkTemplate& t,
                     std::int64_t lag_max, std::int64_t m_max, std::span<const std::int64_t> drift_bins = {});

// --- the search's input ----------------------------------------------------------------------

// What template_search and receive_template take as `cap`: a raw
// frontend.Capture (blanked and normalised here), a receiver.Prepared, a
// stored PassResult (its 250 Hz capture and mix) or a PassbandStore (with
// SearchOptions::q, the slot to read). Not owned: each must outlive the call.
using SearchCap = std::variant<const frontend::Capture*, const receiver::Prepared*, const types::PassResult*,
                               const passband::PassbandStore*>;

// _search_input(cap, f_mix_hz): the prepared capture and the mix it was made
// at -- a pass's own (its ch, which must not be empty: "pass <uid> keeps no
// capture"), f_mix_hz for any other 250 Hz capture (required: "a 250 Hz
// capture needs the frequency it was mixed at (f_mix_hz)"), none at the FE
// rate. A Prepared passes through (the result then points at it: it must
// outlive the result). A PassbandStore is from_passband's, not this.
struct SearchInput {
    std::shared_ptr<const receiver::Prepared> prep;
    std::optional<Rational> f_mix;
};
SearchInput search_input(const types::PassResult& p);
SearchInput search_input(const frontend::Capture& cap, const std::optional<Rational>& f_mix_hz);
SearchInput search_input(const receiver::Prepared& prep, const std::optional<Rational>& f_mix_hz);

// _from_passband(pb, q, spec): slot q's capture over the frame's keyed span
// (pb.capture(q, keyed_end_pos * T_SYM), the store's default lead and
// tail). Refusals: no q ("searching a PassbandStore needs the slot: ..."),
// nothing stored ("the passband store holds nothing for slot q=<q>").
frontend::Capture from_passband(const passband::PassbandStore& pb, std::optional<std::int64_t> q,
                                const frame::FrameSpec& spec);

// --- the search -------------------------------------------------------------------------------

// template_search's keyword arguments (the reference's defaults).
struct SearchOptions {
    std::vector<double> freqs;                       // extra frequencies, audio Hz
    std::optional<std::vector<std::int64_t>> segments;   // none: every segment with data
    double tau_s = TS_TAU_S;
    double df_hz = TS_DF_HZ;
    std::int64_t ppm_steps = TS_PPM_STEPS;
    double z_accept = config::Z_ACCEPT;
    std::optional<Rational> f_mix_hz;                 // a 250 Hz capture's mix (Hz inside FE)
    std::optional<std::int64_t> q;                    // a PassbandStore's slot
};

// return_stats' info dict.
struct SearchInfo {
    double lam = 0.0;
    std::int64_t chunks = 0;
    double cells = 0.0;
    std::int64_t segment = 0;
    double f_hz = 0.0;
    double ppm = 0.0;
    double tau0 = 0.0;
    double drift = 0.0;
    bool operator==(const SearchInfo&) const = default;
};

// template_search(..., return_stats=True): (Detection or none, Z, info). info
// nullopt is the reference's {} (no segment or candidate to search, or no
// finite Lambda), with Z 0.
struct SearchResult {
    std::optional<types::Detection> det;
    double z = 0.0;
    std::optional<SearchInfo> info;
};

// template_search(store, key, cap, spec, ...): search `cap` for a pass of
// picture `key` over every segment with data (or opt.segments) x the
// members' hypotheses (candidates, plus opt.freqs) +- df_hz x timing within
// +- tau_s of t0 x 2 ppm_steps + 1 clock steps about each candidate's ppm
// (each step TS_SLIP_SAMPLES of slip over the frame). A 250 Hz capture
// skips a candidate whose band reaches past TS_CH_REACH_HZ of its mix. The
// detection (Z > z_accept): method "template", z_ref = timing.z = Z, the path
// the winning frequency plus its candidate's drift at t = -20, 0, dur + 10 s,
// timing (lag, ppm, gamma 0, cov diag(1, 1, 0)), lead-in 0. CE frames only.
// hooks: only hooks.cancel is read (polled before each (segment, candidate)
// hypothesis: em::Cancelled).
SearchResult template_search(store::Store& store, const store::Key& key, const SearchCap& cap,
                             const frame::FrameSpec& spec, const SearchOptions& opt = {}, const Hooks& hooks = {});

// receive_template(store, key, cap, spec, det, estimator, segment, **search):
// template-search `cap` (unless det is given; then `segment`, default 0, is
// the segment it was found as) and receive what is found: tracked with the
// accumulator's soft data of that segment as its template and the picture's
// header known (and the callsign keying, once a member's windows matched),
// demodulated, its header decoded. The PassResult is the reference's field
// for field (uid from (q, f_hz, z); frame the plain frame when a version 1
// header was read; waveform CE; the channel's capture and mix; em_round 0).
// Not associated. nullopt when the search finds nothing. search.q /
// search.f_mix_hz select the capture as template_search's do; the rest of
// `search` reaches the search only when det is not given. CE frames only.
std::optional<types::PassResult> receive_template(store::Store& store, const store::Key& key, const SearchCap& cap,
                                                  const frame::FrameSpec& spec,
                                                  const std::optional<types::Detection>& det = std::nullopt,
                                                  const std::string& estimator = "joint",
                                                  std::optional<std::int64_t> segment = std::nullopt,
                                                  const SearchOptions& search = {}, const Hooks& hooks = {});

// --- retroactive detection -----------------------------------------------------------------------

// One find: the pass received and what association made of it (key none
// for a provisional pass; rule "header" | "soft-header" | "corr" | "prov" |
// "" for a pass that was a member already).
struct RetroFind {
    types::PassResult pass;
    std::optional<store::Key> key;
    std::string rule;
};

// retro_detect's keyword arguments: `slots` (none: every slot the passband
// store holds), the estimator, and the search's options (passed to every
// template_search; receive_template takes none of them, as the reference's
// call passes none).
struct RetroOptions {
    std::optional<std::vector<std::int64_t>> slots;
    std::string estimator = "joint";
    SearchOptions search;
};

// retro_detect(store, key, pb, spec, slots, estimator, log, **search): every
// slot of `pb` (or opt.slots) that holds no member of picture `key` and
// whose frame is at least RETRO_MIN_COVERAGE on disk is template-searched;
// a find within RETRO_SAME_HZ of a pass already stored for that slot (an L
// pass by its band centre, pilot + L_EXCLUDE_OFFSET_HZ) is skipped; any
// other is received (receive_template, the search's segment) and associated
// (associate::associate). Returns the finds in slot order. hooks.log
// receives exactly the reference's `log=` lines ("slot q=<q>: nothing (Z
// <Z:.1f>)", "slot q=<q>: Z <Z:.1f> at <f:.3f> Hz, already stored", "slot
// q=<q>: Z <Z:.1f> at <f:.3f> Hz, segment <g>, attached to <CALL> <pid:08x>
// by <rule or 'existing membership'>" / "..., stored provisional");
// hooks.progress (done, total, "slot q=<q>") after each slot of the list
// (total: the slots considered, held and short ones included); hooks.cancel
// is polled before each slot and inside each search. CE frames only.
std::vector<RetroFind> retro_detect(store::Store& store, const store::Key& key, const passband::PassbandStore& pb,
                                    const frame::FrameSpec& spec, const RetroOptions& opt = {},
                                    const Hooks& hooks = {});

// --- retro_detect's steps, replaceable (the binding's seams) ---------------------------------------

// One slot's capture as the steps hand it on: the passband store's Capture
// (what the C++ steps read; `prep` caches its prepared form for the search
// and the receive, and a caller may set it beforehand instead of
// `capture`), and/or `other`, an object a replacement capture step produced
// (opaque here: the binding's Python capture).
struct SlotInput {
    std::optional<frontend::Capture> capture;
    std::shared_ptr<const void> other;
    mutable std::shared_ptr<const receiver::Prepared> prep;
};
using SlotInputPtr = std::shared_ptr<const SlotInput>;

// One slot's search: the detection (none: nothing), Z, and info["segment"]
// (nullopt when the info holds none: reading it is the reference's
// KeyError 'segment', raised where the reference reads it).
struct SlotSearch {
    std::optional<types::Detection> det;
    double z = 0.0;
    std::optional<std::int64_t> segment;
};

// The steps of retro_detect's loop. Every one must be set (retro_steps
// gives the C++ ones).
struct RetroSteps {
    std::function<std::vector<std::int64_t>(double dur_s)> slots;                             // pb.slots(dur)
    std::function<std::pair<SlotInputPtr, double>(std::int64_t q, double dur_s)> capture;     // pb.capture(q, dur)
    std::function<SlotSearch(const SlotInputPtr& cap)> search;                                // template_search
    std::function<std::optional<types::PassResult>(const SlotInputPtr& cap, const types::Detection& det,
                                                   std::int64_t segment)>
        receive;                                                                               // receive_template
    std::function<associate::Result(const types::PassResult& p)> associate;                   // associate
};

// The C++ steps over `pb` (which must outlive them): slots and capture
// PassbandStore's, search template_search(store, key, cap, spec, opt.search,
// hooks) on the slot's prepared capture (its `prep`, else `capture`
// prepared; a SlotInput holding only `other` is std::invalid_argument
// here), receive receive_template(store, key, cap,
// spec, det, opt.estimator, segment) on that same prepared capture,
// associate associate::associate(p, store, soft) (soft empty: the C++ rule
// 2). A null `pb` leaves slots and capture empty, for a caller that
// replaces both.
RetroSteps retro_steps(store::Store& store, const store::Key& key, const passband::PassbandStore* pb,
                       const frame::FrameSpec& spec, const RetroOptions& opt = {}, const Hooks& hooks = {},
                       associate::SoftHeaderFn soft = {});

// retro_detect over `steps`: ce_only, the accumulator, the slots held and
// the passes stored (store reads), then the loop, as retro_detect.
// opt.search and opt.estimator are the steps' business (not read here).
std::vector<RetroFind> retro_detect_steps(store::Store& store, const store::Key& key, const frame::FrameSpec& spec,
                                          const RetroSteps& steps,
                                          const std::optional<std::vector<std::int64_t>>& slots = std::nullopt,
                                          const Hooks& hooks = {});

// The log line of a find (the reference's f-string), for a caller that
// formats one itself: "slot q=<q>: Z <z:.1f> at <f_hz:.3f> Hz, segment <g>,
// attached to <CALL> <pid:08x> by <rule or 'existing membership'>" or
// "..., stored provisional".
std::string retro_find_line(std::int64_t q, double z, double f_hz, std::int64_t segment,
                            const std::optional<store::Key>& key, const std::string& rule);

}  // namespace sstvae::qrss::em

#endif  // SSTVAE_QRSS_EM_SEARCH_HPP
