// Leave-one-out EM and the template search, the shared half: port of the
// pieces of sstvae/qrss/em.py (the reference; its module docstring is the
// design) that the EM loop (pass_path, rereceive, em_refine) and the
// retroactive search (template_search, receive_template, retro_detect)
// both stand on -- the references, what every member is known to have
// sent, the frame a stored pass was sent in, the tail statistics -- and
// the hooks every long call of theirs takes (log, progress, cancel).
//
// Not a format module: nothing here goes on the air. Constants are the
// generated rx_config.hpp's (em.py's own) and config.hpp's.
//
//   em.py                    here
//   EM_ROUNDS, EM_TOL_DB,    the constants below (rxconfig::em)
//   TS_*, RETRO_*
//   lmmse_latents            lmmse_latents
//   prior_from               prior_from
//   em_prior                 em_prior
//   acc_mean_w_db            acc_mean_w_db (moved here from decode.hpp,
//                            which re-exports it)
//   known_header             known_header
//   known_keying             known_keying
//   _psi_frame               psi_frame (receiver::psi_frame, the same
//                            function: receiver.py's _psi_frame)
//   pass_spec, is_l          pass_spec, is_l
//   _ce_only                 ce_only
//   _path_of                 path_of
//   _flat                    flat
//   _gauss_z                 gauss_z
//   template_z               template_z
//
// Exactness. lmmse_latents, prior_from, em_prior, acc_mean_w_db,
// known_header, known_keying, pass_spec, is_l, path_of, flat and gauss_z
// are the reference's arithmetic in its order and agree with it to the
// bit (gauss_z through math::norm_isf = scipy's cephes ndtri, and libm's
// exp/log/sqrt as Python's math module calls them). acc_mean_w_db's
// 10 log10 is the C library's where numpy's may be its own SIMD one (the
// last place, as decode.hpp has always said). template_z's
// gamma.logsf takes its logs from the C library where numpy uses its own
// SIMD log/log1p on an AVX-512 machine: a few ulp of log_p, far inside
// the 1e-12 the gate needs (see math/special.hpp).
//
// Errors are the reference's: std::invalid_argument (ValueError) with its
// texts -- em_prior's length refusal, ce_only's refusal, a negative q
// ("q must fit in uint64, not <q>", sequences.scrambler's), template_z's
// "math domain error" where Python's math.log / math.lgamma raise one --
// and the store's (store.hpp: KeyError for a uid that is not a member,
// FileError for a pass file that cannot be read, ...) passed through.
// path_of on a tracker with no grid is std::out_of_range (IndexError).
//
// THREADS. Every function here is safe to call from any thread at once:
// no state of its own, no mutable globals. The ones that take a Store
// read through it, and a Store serialises its own calls (store.hpp):
// each store call holds the store's mutex for that one call, never across
// two, and nothing here holds it across slow work of its own. What blocks,
// and for how long (measured 2026-10-14 on six FULL members, warm page
// cache): em_prior reads every other member's (z, w) (0.4 MB each for
// FULL; from the store's cache after the first read) and sums them, about
// 3 ms in all; known_header reads each member's light metadata (cached
// after the first read) and may run one polar list decode per segment
// (~1.3 ms each), about 1.5 ms; known_keying loads every member's whole
// pass file, its 250 Hz capture included (~3.6 MB for FULL, ~5 ms each),
// about 30 ms. The rest are arithmetic on their arguments (microseconds;
// template_z's gamma tail well under a millisecond). Every result is a
// value, never a reference into the store.
//
// THE HOOKS (Hooks, Cancelled, check_cancel) are what the long calls built
// on this file take -- em_refine and retro_detect, which run for minutes on
// an app worker thread while the GUI thread and possibly the in-process
// listener (listen_api.hpp, writing the same store and passband
// directory) run beside them:
//
//   log       receives exactly the lines Python's `log=` callable receives,
//             without a prefix (retro's "retro: " is the CLI's, added
//             there); empty: nothing is logged.
//   progress  (done, total, what), an addition Python does not have;
//             called on the calling thread only, never with a lock held
//             (the store's or any other), so it may block or call back
//             into anything but the call it reports on; empty: none.
//   cancel    an addition too: when *cancel is set the running call stops
//             at its next poll (check_cancel), between two units of work
//             -- a pass re-received and stored, a slot searched -- and
//             throws Cancelled. Every file it wrote before that is
//             complete: the store writes each file atomically (a temporary
//             file renamed into place), and a poll sits only where no write
//             is half done, so the store is always left consistent (the
//             passes re-received so far keep their new (z, w) and the
//             accumulators holding them were rebuilt). null: never
//             cancelled. Read with a relaxed load; set it from any thread.
//
// The hooks are called on the thread that made the call. A hook that
// throws ends the call with its exception, at the same consistent point.

#ifndef SSTVAE_QRSS_EM_HPP
#define SSTVAE_QRSS_EM_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/receiver.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/store.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::em {

// --- constants (em.py's, generated) -----------------------------------------------------

inline constexpr int EM_ROUNDS = rxconfig::em::EM_ROUNDS;                       // 3
inline constexpr double EM_TOL_DB = rxconfig::em::EM_TOL_DB;                    // 0.05
inline constexpr double TS_CHUNK_S = rxconfig::em::TS_CHUNK_S;                  // coherent chunk (s)
inline constexpr int TS_NFFT = rxconfig::em::TS_NFFT;                           // FFT per chunk at 250 Hz
inline constexpr double TS_DF_HZ = rxconfig::em::TS_DF_HZ;                      // +- frequency per candidate
inline constexpr double TS_TAU_S = rxconfig::em::TS_TAU_S;                      // +- timing about t0
inline constexpr int TS_PPM_STEPS = rxconfig::em::TS_PPM_STEPS;                 // clock steps either side
inline constexpr double TS_SLIP_SAMPLES = rxconfig::em::TS_SLIP_SAMPLES;        // CH samples of slip per step
inline constexpr double TS_CH_REACH_HZ = rxconfig::em::TS_CH_REACH_HZ;          // a 250 Hz capture's reach
inline constexpr double RETRO_MIN_COVERAGE = rxconfig::em::RETRO_MIN_COVERAGE;  // 0.9
inline constexpr double RETRO_SAME_HZ = rxconfig::em::RETRO_SAME_HZ;            // 1.0

// --- the hooks ----------------------------------------------------------------------------

// What a long EM or retro call reports through and is stopped by (see the
// header comment). Default-constructed: silent, never cancelled.
struct Hooks {
    std::function<void(const std::string&)> log;
    std::function<void(std::int64_t done, std::int64_t total, const std::string& what)> progress;
    const std::atomic<bool>* cancel = nullptr;
};

// Thrown by check_cancel once *hooks.cancel is set: the call stopped at a
// consistent point (see the header comment). what() is "cancelled".
class Cancelled : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Throws Cancelled when hooks.cancel is set and true; otherwise nothing.
void check_cancel(const Hooks& hooks);
// hooks.log(line) when there is a log; otherwise nothing.
void log_line(const Hooks& hooks, const std::string& line);
// hooks.progress(done, total, what) when there is one; otherwise nothing.
void report_progress(const Hooks& hooks, std::int64_t done, std::int64_t total, const std::string& what);

// --- references ---------------------------------------------------------------------------

// lmmse_latents(S, W): a = S/(1 + W'), r = 1/(1 + W'), W' = np.maximum(W, 0)
// (a NaN W stays NaN). std::invalid_argument with numpy's broadcast text
// when S and W differ in length.
struct Lmmse {
    std::vector<double> a;
    std::vector<double> r;
};
Lmmse lmmse_latents(std::span<const double> S, std::span<const double> W);

// prior_from(S, W, q): EmPrior(x_hat = precode(a, q), v = block_mean(r)).
// q < 0 is sequences.scrambler's ValueError (after lmmse_latents' own
// refusal, which the reference meets first).
types::EmPrior prior_from(std::span<const double> S, std::span<const double> W, std::int64_t q);
// The same for any slot number scrambler takes (0 .. 2^64 - 1): a stored
// pass's q is an int64, so this is for a caller holding a q past it.
types::EmPrior prior_from_u64(std::span<const double> S, std::span<const double> W, std::uint64_t q);

// em_prior(acc, uid, store, q, spec): pass `uid`'s soft data symbols from
// the accumulator without it (acc.loo). std::invalid_argument "pass <uid>
// holds <n> latents; frame '<name>' sends <n_data>" when the pass's length
// is not the frame's; the store's KeyError for a uid that is not a member.
types::EmPrior em_prior(const store::Accumulator& acc, const std::string& uid, store::Store& store,
                        std::int64_t q, const frame::FrameSpec& spec);

// acc_mean_w_db(acc): 10 log10 of the mean W over the latents heard (W > 0,
// numpy's pairwise mean in float64), -inf when none is.
double acc_mean_w_db(const store::Accumulator& acc);

// --- what every member pass is known to have sent ------------------------------------------

// known_header(store, acc, segment): the header the accumulator's passes of
// `segment` were sent with -- the first member (in member order) whose
// own decoded header matches the key, else the first segment's summed
// header LLRs (members' hdr_llr of 2474 values, not all zero, summed per
// segment in member order, the segments in order of first appearance)
// that decode to the key -- with its segment set to `segment`; nullopt
// when neither, or when `segment` is outside 0..mode of that header.
std::optional<header::HeaderFields> known_header(store::Store& store, const store::Accumulator& acc,
                                                 std::int64_t segment);

// known_keying(store, acc, call, waveform): the callsign windows' keying
// (keying_units(call)) and whether it is OOK, if a member of `waveform`
// read its windows as agreeing with the header (cw.agrees, z_match >
// Z_ACCEPT): OOK when more of those voted OOK than FSK, and never for L.
// nullopt without a call, for an invalid call (check_callsign's refusal is
// swallowed, as the reference's `except ValueError`), or with no vote.
std::optional<receiver::CwKnown> known_keying(store::Store& store, const store::Accumulator& acc,
                                              const std::optional<std::string>& call,
                                              std::int64_t waveform = config::WAVEFORM_CE);

// --- a stored pass ------------------------------------------------------------------------

// pass_spec(p): the frame p was sent in -- lframe.get_any(p.frame), without
// the spread copy (frame::plain) for a CE pass whose header says version
// 1. get_any's refusal for an unknown frame name.
frame::FrameSpec pass_spec(const types::PassResult& p);
// is_l(p): pass_spec(p) is waveform L.
bool is_l(const types::PassResult& p);

// _ce_only(spec, what): std::invalid_argument "<what> is CE-only (design
// 8.3): frame '<name>' is waveform L" for any frame not CE.
void ce_only(const frame::FrameSpec& spec, std::string_view what);

// receiver._psi_frame, which em.py's _psi_frame is.
using receiver::psi_frame;

// _path_of(tr): the tracker's frequency path tr.freq sampled every second
// from 10 s before its first grid position to 10 s after its last
// (np.arange's points: start, start + 1, then start + i ((start + 1) -
// start)), weight 1. std::out_of_range "index 0 is out of bounds for axis
// 0 with size 0" for an empty grid.
types::FreqPath path_of(const track::TrackResult& tr);

// _flat(f_hz): the constant path f_hz at t = -20, 0, 1800 s, weight 1.
types::FreqPath flat(double f_hz);

// --- the template search's statistic -------------------------------------------------------

// _gauss_z(log_p): the normal Z with upper tail exp(log_p): 0 for log_p >=
// log(0.5); norm.isf(exp(log_p)) above -700; below, four Mills-ratio
// iterations (to well under 0.01 in Z). NaN for NaN.
double gauss_z(double log_p);

// template_z(lam, n_chunks, n_cells): Z of a template-search peak --
// gamma.logsf(lam, n_chunks) (the Gamma(K, 1) tail), where that is not
// finite the log of its leading term (K - 1) log(lam) - lam - lgamma(K)
// (Python's math.log and CPython's math.lgamma), plus log(max(n_cells, 1))
// (Bonferroni), capped at 0, through gauss_z. std::invalid_argument
// "math domain error" where Python's math module raises one there (that
// fallback with lam <= 0, or with n_chunks <= 0).
double template_z(double lam, std::int64_t n_chunks, double n_cells);

}  // namespace sstvae::qrss::em

#endif  // SSTVAE_QRSS_EM_HPP
