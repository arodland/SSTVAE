// Leave-one-out EM, the loop: port of em.py's pass_path, rereceive and
// em_refine (sstvae/qrss/em.py, the reference; its module docstring is the
// design, design 8.3). The shared pieces they stand on (the references, what
// every member is known to have sent, a stored pass's frame, the hooks) are
// em.hpp's; the template search and retro_detect are em_search.hpp's.
//
// Not a format module: nothing here goes on the air. Constants are em.hpp's
// (rx_config.hpp's rxconfig::em, generated from em.py).
//
//   em.py         here
//   pass_path     pass_path (an optional PassPathTrace says which path won)
//   rereceive     rereceive (always returns the final path beside the pass:
//                 Python's return_path=True; an optional Hooks for cancel)
//   em_refine     em_refine (Hooks: log, progress, cancel; EmSeams for a
//                 caller that replaces rereceive, as a test's spy does)
//
// Exactness. These are compositions: everything they compute themselves
// (which path wins, which branch runs, the block split's parities and
// templates, the PassResult's fields, the loop's order, skips, history and
// stop) is the reference's arithmetic in the reference's order, so with the
// same primitives underneath they are the reference to the bit. What the
// primitives agree to is theirs: track_timing.hpp's line_path / refine_path /
// verify_track and track.hpp's track / report agree with Python to rounding
// (the stage 2 bars), demod's extract likewise, and receiver.hpp's
// receive_pass to its parity file's bars. So rereceive agrees with Python to
// those bars and every decision is the reference's on inputs whose deciding
// values are not within rounding of a threshold (pass_path's choice compares
// two Z_ref values with `>`: a tie within rounding could go either way).
// em_refine's history agrees to the same bars (acc_mean_w_db of stores that
// agree to them) and its log lines are Python's text, the numbers printed
// "%+.2f" as Python's format does (frame::py_format).
//
// Errors are the reference's: std::invalid_argument (ValueError) for an L
// frame given to pass_path ("em.pass_path is CE-only (design 8.3): frame
// 'l-short' is waveform L"), the store's KeyError for a key with no
// accumulator ("no accumulator <key_name>") or a member whose pass file is
// gone, math::TypeError "cannot unpack non-iterable NoneType object" where
// the reference unpacks a receive_pass that refused the pass (a stored pass
// whose z_ref is not finite or whose timing is missing, so receive_pass runs
// gate V and it fails: unreachable from passes the receiver stores), and
// whatever the primitives raise (an unknown estimator, a frame name no
// preset has). em::Cancelled when a cancel is seen (Hooks).
//
// THREADS. Every function here is safe to call from any thread at once on
// different stores, and on one store from several threads as far as the
// store's own guarantee goes (store.hpp: each store call holds the store's
// mutex for that one call, never across two). Nothing here holds any lock
// across slow work: em_refine reads the accumulator, re-receives with no
// lock held, then stores the result with one update_pass call (which holds
// the store's mutex while it rewrites the pass file and rebuilds the
// accumulators holding it). A listener writing the same store at the same
// time (listen_api.hpp) is safe: a pass it attaches meanwhile is seen from the
// next round on (the members re-received in a round are the accumulator's at
// the round's start, as the reference's `list(acc.uids)` is). Results are
// values, never references into the store. The hooks and seams are called on
// the calling thread, never with a lock held.
//
// What blocks, and for how long (measured 2026-10-14, RelWithDebInfo, one
// thread, simulated passes at -6 dB; Python's time in brackets): pass_path
// runs line_path, refine_path and two verify_tracks on the pass's capture
// (TINY 0.05 s [0.21], SHORT 0.19 s [0.94], FULL 1.1 s [6.1]); rereceive
// adds two tracker runs and extractions (the receive_pass branch, or no
// split) or three (split): with the path given TINY 0.06 s [0.36], SHORT
// 0.3-1.0 s [2.6-4.9], FULL 3.2-4.2 s [17-23], measuring its own SHORT
// 1.2 s [6.7], FULL 5.0 s [29]; em_refine is up to rounds x (members with
// a capture) of those plus one store update per pass (a pass file rewrite
// and an accumulator rebuild, milliseconds): one round over three TINY
// passes 0.31 s [1.7], over two SHORT 2.0 s [10.2]. Run em_refine on a
// worker thread, never the GUI's.

#ifndef SSTVAE_QRSS_EM_REFINE_HPP
#define SSTVAE_QRSS_EM_REFINE_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qrss/em.hpp"
#include "qrss/header.hpp"
#include "qrss/receiver.hpp"
#include "qrss/store.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::em {

// What pass_path measured: the two candidate paths' Z_ref with the pass's own
// timing (NaN where verify_track gave NaN) and which one won (0: the line,
// line_path's; 1: refine_path's refinement of it).
struct PassPathTrace {
    types::FreqPath line;
    types::FreqPath refined;
    double z_line = 0.0;
    double z_refined = 0.0;
    int choice = 0;
};

// pass_path(p, chan): a frequency path for re-receiving pass p, measured on
// its own capture -- track::line_path from a flat path at p.f_hz, and that
// line refined segment by segment (track::refine_path), whichever gives the
// known symbols the larger Z_ref (verify_track with the pass's timing, fit
// off: known symbols withheld, so the choice does not look at the data);
// the line on a tie, as Python's `>` keeps the first. chan null: the pass's
// own channel (receiver::channel_for of capture_from_pass at p.f_mix_hz).
// CE only: ce_only's refusal "em.pass_path is CE-only ..." for an L pass.
// Blocks for line_path, refine_path and two tracker runs (see above).
types::FreqPath pass_path(const types::PassResult& p, std::shared_ptr<const track::ChanCapture> chan = nullptr,
                          PassPathTrace* trace = nullptr);

// What rereceive made: the pass (nullopt: an L pass, or one without a
// capture -- Python's None) and the frequency path its final track ran on
// (Python's return_path=True second element; nullopt with the pass).
struct Rereceived {
    std::optional<types::PassResult> pass;
    std::optional<types::FreqPath> path;
};

// rereceive's arguments past the pass and its prior (Python's keywords).
struct RereceiveArgs {
    std::optional<header::HeaderFields> hdr;          // the header p was sent with, or none
    std::optional<receiver::CwKnown> cw_known;        // the callsign windows' keying, if known
    std::optional<std::string> estimator;             // none (or ""): p.estimator
    std::optional<std::int64_t> em_round;             // none: p.em_round
    std::optional<types::FreqPath> path;              // none: pass_path(p)
    bool split = true;                                // the block split (with a prior)
};

// rereceive(p, prior, hdr, cw_known, estimator, em_round, path, split):
// pass p re-tracked and re-demodulated from its stored 250 Hz capture
// (design 8.3 step 3), every branch of the reference:
//   - an L pass, or one without a capture: nothing (both nullopt);
//   - hdr's version is taken from p's own header when both have one;
//   - without hdr, or for a frame without a header: receiver::receive_pass on
//     the capture (prior, the estimator, p.f_mix_hz) from p's detection with
//     the path, keeping p's uid, its callsign reading, its header (or the one
//     the re-receive decoded when p had none) and setting em_round;
//   - with hdr: the header symbols known from the start and two tracker
//     rounds as receive_pass's (the first from p's timing, the second from
//     the first's timing and re-centred path); with a prior and `split`, the
//     second is two runs, each demodulating the precoder blocks of one parity
//     with a track whose templates leave that parity's reference out (the
//     report, z_ref, header LLRs and psi from the even-parity run). z_ref is
//     p's own, hdr_llr p's when the frame carries a spread header copy.
// Hooks: a cancel poll before the path is measured (when it is not given),
// after it, and before each tracker run of the two-round branch; the
// receive_pass branch is polled before it starts, not between its own two
// rounds (em::Cancelled; nothing is written here, so nothing is left half
// done). Blocks for the path (unless given) and two or three tracker runs.
Rereceived rereceive(const types::PassResult& p, const std::optional<types::EmPrior>& prior,
                     const RereceiveArgs& args = {}, const Hooks& hooks = {});

// What em_refine calls that a caller may replace. Empty: the C++.
struct EmSeams {
    // rereceive(p, prior, hdr, cw_known, estimator, em_round, path) as
    // em_refine calls it (Python's `rereceive(p, prior, hdr, cw_known,
    // estimator, r, path=..., return_path=True)`).
    std::function<Rereceived(const types::PassResult& p, const types::EmPrior& prior,
                             const std::optional<header::HeaderFields>& hdr,
                             const std::optional<receiver::CwKnown>& cw_known,
                             const std::optional<std::string>& estimator, std::int64_t em_round,
                             const std::optional<types::FreqPath>& path)>
        rereceive;
};

// em_refine's keyword arguments.
struct EmRefineArgs {
    std::int64_t rounds = EM_ROUNDS;
    double tol_db = EM_TOL_DB;
    std::optional<std::string> estimator;   // none: each pass's own
    bool use_header = true;
};

// em_refine(store, key, rounds, tol_db, estimator, use_header, log):
// leave-one-out EM over accumulator `key`'s members (design 8.3). Returns the
// accumulator's mean W in dB (acc_mean_w_db) before EM and after each round
// run (so its length is the rounds run + 1); rounds stop once a round gains
// less than tol_db (a NaN gain, -inf to -inf, does not stop them, as
// Python's `<`). In each round every member of the accumulator at the
// round's start, in member order: a member whose pass keeps no capture is
// skipped silently (it still takes part in every reference); an L member is
// skipped, logged in round 1 ("pass <uid>: waveform L, not re-received (EM is
// CE-only)"); every other is re-received (rereceive) with its leave-one-out
// prior (em_prior), its segment's known header (known_header, with
// use_header) and the callsign keying (known_keying of the first known
// header's call), from the path it ended on last round, stored
// (Store::update_pass) and logged ("round <r> pass <uid>: mean W <before>
// -> <after> dB", "%+.2f"), and the accumulator is read again.
//
// Hooks:
//   log       the reference's lines, as above;
//   progress  (done, total, "round <r> pass <uid>") after each pass is
//             re-received and stored; total = rounds x the members with a
//             capture and a CE frame (a pass that cannot be read counts, as
//             the round will meet its error), counted again at each round's
//             start as done + the rounds left x the members then, and never
//             lowered: a member attached meanwhile (a listener writing the
//             same store) raises it, so done never runs past it; when the
//             rounds stop early (tol_db) one last call (done, done,
//             "converged after round <r>") says the rest will not run; with
//             no eligible member, or rounds <= 0, no call at all;
//   cancel    polled before each member is loaded, and inside rereceive
//             (around the path, before each tracker run): em::Cancelled. Every
//             pass re-received before it is stored and its accumulators
//             rebuilt; the one in flight is not stored.
// The store's KeyError for a key with no accumulator, before anything else.
std::vector<double> em_refine(store::Store& store, const store::Key& key, const EmRefineArgs& args = {},
                              const Hooks& hooks = {}, const EmSeams& seams = {});

}  // namespace sstvae::qrss::em

#endif  // SSTVAE_QRSS_EM_REFINE_HPP
