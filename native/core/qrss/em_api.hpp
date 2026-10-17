// QRSSTVAE leave-one-out EM and the retroactive search, in process: what an
// application wires to two buttons on a received picture ("refine" and
// "search the recording") without running qrss_decode.py --em / --retro.
// Qt-free and free of onnxruntime, like rx_api.hpp and listen_api.hpp: what
// changes is the store (passes re-received or added, accumulators rebuilt),
// and turning an accumulator into a picture is the codec's, as before.
//
// The calls, as `sstvae-qrss decode` (and qrss_decode.py) makes them:
//
//   check_picture(store, key)                    NoPicture
//   refine(store, key, RefineOptions, hooks)     -> RefineResult
//       (em.em_refine: --em N)
//   retro(store, key, Passband, frame, RetroOptions, hooks) -> RetroResult
//       (em.retro_detect over PassbandStore(--passband or STORE/passband),
//       frame.get(--frame): --retro)
//
// Both are the C++ ports underneath (em_refine.hpp, em_search.hpp), so
// what they write to the store and what they log is the reference's (the
// log lines exactly the reference's `log=` lines, without the CLI's
// "retro: " prefix); this file adds the inputs' checks, a typed result and
// a cancel that returns rather than throws.
//
// INPUTS are checked before anything runs, with the CLI's refusal texts, as
// Refusal subclasses (never an exit): NoPicture "no picture <CALL> <pid:08x>
// in <root>"; for retro, in the CLI's order after that, NoPassbandStore
// "--retro: no passband store at <dir>" (the directory is not a directory),
// NotCeFrame "--frame '<name>': --retro searches for CE passes only" (an
// "l-..." name), UnknownFrame "--frame '<name>': not one of full, medium,
// short, tiny" (sorted(frame.PRESETS); frame.get also takes the version 1
// frames and the prototypes by name). Beyond the CLI's, std::invalid_argument
// with the receiver's text for an estimator that is not "joint" or "plain"
// ("estimator must be 'joint' or 'plain', not '<name>'"), which the
// reference meets only at the first re-receive or the first find (minutes
// in). Everything else is what the ports throw (em_refine.hpp,
// em_search.hpp: the store's KeyError and FileError, std::invalid_argument
// for a search option they refuse, ...), at the point they meet it.
//
// THREADS. Every call runs on the caller's thread and blocks until done; it
// keeps no state between calls and holds no lock across slow work (the
// store's mutex is taken per store call, the passband store's only to read
// an hour's index), so run it on a worker thread, never the GUI's. Several
// calls may run at once on different pictures; two on one picture are safe
// (the store serialises each write) but pointless. The hooks are called on
// the calling thread with no lock held: log with each reference line,
// progress (done, total, what) after each pass re-received or slot
// examined (em_refine.hpp, em_search.hpp say exactly when: done rises by
// one, total never falls except on refine's last call when EM converges
// early, (done, done, "converged after round <r>")), and cancel polled
// between those units and inside them.
// What blocks, and for how long (measured 2026-10-14, RelWithDebInfo, one
// thread of a shared 4-CPU x86-64; em_refine.hpp and em_search.hpp have
// the pieces):
//   refine   rounds x the members with a capture, each one re-receive and
//            one store update: per pass TINY ~0.1 s, SHORT ~0.3-1.2 s, FULL
//            ~3-5 s, so one round over a FULL picture of four passes is
//            ~15-20 s and the default three rounds about a minute; a
//            cancel lands within one re-receive's tracker run.
//   retro    one template search per slot of the passband store not held
//            by a member and on disk to RETRO_MIN_COVERAGE: TINY ~0.2 s,
//            SHORT ~0.9 s, FULL ~6 s per slot (plus reading it: a fraction
//            of a second), and one receive and one association per find
//            (about a receive's round B: 0.1 s TINY, ~0.5 s SHORT, a few s
//            FULL). A full 48 h store is 192 slots: ~3 min for SHORT, ~20
//            min for FULL. A cancel lands within one (segment, candidate)
//            search.
//   check_picture, the refusals: a stat or two (microseconds).
//
// SHARING THE STORE WITH AN IN-PROCESS LISTENER (listen_api.hpp). A
// listener session holds the store's directory lock (one listener per
// store); these calls take no lock of that kind, as qrss_decode.py takes
// none, so they run beside a session on the same store. Prefer the
// session's own objects -- Session::store() and Session::passband(), null
// when the session does not hold them -- which share its caches and its
// in-process locks: refine(*session->store(), ...),
// retro(*session->store(), key, Passband::of(session->passband()), ...).
// A second Store on the same root is safe too (every Store on one root in a
// process shares one mutex, and reads check each file's signature), so a
// UI without a session, or with one that holds no store, opens
// store::Store(root). The passband store: a separate PassbandStore on the
// directory a session is writing reads what is on disk (each hour's index
// is written after its data, a slot not yet on disk to
// RETRO_MIN_COVERAGE is skipped, and a raw hour packed meanwhile is read
// from its FLAC); the session's own instance also coordinates with its
// packer in-process, so prefer it when there is one. What the two write
// meanwhile is each a whole transaction of the store's: a pass the
// listener attaches during a refine is re-received from the next round on
// (em_refine.hpp), and a find the listener had stored first is "already
// stored" or folded into its twin by the store (Store::attach).
//
// The receiver rules (rules.hpp: QRSS_OLD_EDGE_NOISE) are the process
// environment's, read where the receiver reaches them, as the reference
// reads them: set them once at startup (rules::set_old), never per call.
//
// Qt-free; in sstvae_qrss. No mutable globals.

#ifndef SSTVAE_QRSS_EM_API_HPP
#define SSTVAE_QRSS_EM_API_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "qrss/em.hpp"
#include "qrss/em_search.hpp"
#include "qrss/frame.hpp"
#include "qrss/passband.hpp"
#include "qrss/store.hpp"

namespace sstvae::qrss::em_api {

// --- refusals ----------------------------------------------------------------------------------

// The CLI's refusals (its SystemExit texts), as types a UI can tell apart.
class Refusal : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
// "no picture <CALL> <pid:08x> in <root>": the store holds no accumulator
// for the key.
class NoPicture : public Refusal {
public:
    using Refusal::Refusal;
};
// "--retro: no passband store at <dir>": the directory is not one.
class NoPassbandStore : public Refusal {
public:
    using Refusal::Refusal;
};
// "--frame '<name>': --retro searches for CE passes only".
class NotCeFrame : public Refusal {
public:
    using Refusal::Refusal;
};
// "--frame '<name>': not one of full, medium, short, tiny".
class UnknownFrame : public Refusal {
public:
    using Refusal::Refusal;
};

// NoPicture unless the store holds an accumulator for `key`.
void check_picture(const store::Store& store, const store::Key& key);
// The passband store a listener keeps beside `store`: STORE/passband.
std::string default_passband_dir(const store::Store& store);
// The frame --retro searches with, by name: NotCeFrame for an "l-..."
// name, UnknownFrame for one frame.get does not know (`name` shown as
// Python's repr shows it).
const frame::FrameSpec& retro_frame(std::string_view name);
// NoPassbandStore unless `dir` names a directory, as Path.is_dir() decides
// (false for a missing path or a name the OS cannot spell);
// std::filesystem::filesystem_error for a stat that fails otherwise (a name
// too long), where Path.is_dir() raises its OSError.
void check_passband_dir(const std::string& dir);

// --- refine (--em N) -------------------------------------------------------------------------

// em_refine's arguments. rounds <= 0 runs nothing (history: the mean W
// before); rounds past MAX_ROUNDS are MAX_ROUNDS (the rounds stop once one
// gains less than tol_db, long before). estimator: none or "" for each
// pass's own.
inline constexpr std::int64_t MAX_ROUNDS = std::int64_t{1} << 31;
struct RefineOptions {
    std::int64_t rounds = em::EM_ROUNDS;
    double tol_db = em::EM_TOL_DB;
    std::optional<std::string> estimator;
    bool use_header = true;
};

// One pass re-received and stored: the round, its uid and its own mean W
// (receiver::mean_w_db) before and after -- the numbers of the log line
// "round <r> pass <uid>: mean W <before> -> <after> dB".
struct RefinedPass {
    std::int64_t round = 0;
    std::string uid;
    double mean_w_db_before = 0.0;
    double mean_w_db_after = 0.0;
};

struct RefineResult {
    // The accumulator's mean W in dB (em::acc_mean_w_db) before EM and after
    // each round run: em_refine's return value. When cancelled: the value
    // before and after each round that a later round had started after (a
    // round in progress, or just ended, when the cancel came shows only in
    // mean_w_db).
    std::vector<double> history;
    // Every pass re-received and stored, in order.
    std::vector<RefinedPass> passes;
    // The rounds run to their end (history.size() - 1 when not cancelled).
    std::int64_t rounds_run = 0;
    // The accumulator's mean W at return (history.back() when not
    // cancelled; read again when cancelled).
    double mean_w_db = 0.0;
    // Stopped by hooks.cancel: every pass in `passes` is stored and the
    // accumulators holding it rebuilt; the one in flight is not stored.
    bool cancelled = false;
};

// check_picture, the estimator, then em::em_refine(store, key, ...): log,
// progress and cancel as em_refine.hpp says. Never throws em::Cancelled.
RefineResult refine(store::Store& store, const store::Key& key, const RefineOptions& opt = {},
                    const em::Hooks& hooks = {});

// --- retro (--retro) -----------------------------------------------------------------------------

// Where retro reads the passband store: a directory (none or "": the
// store's own, STORE/passband) or an open one (a session's: it must stay
// open for the call, which the shared_ptr sees to).
struct Passband {
    std::optional<std::string> dir;
    std::shared_ptr<const passband::PassbandStore> open;
    static Passband at(std::string dir) { return Passband{std::move(dir), nullptr}; }
    static Passband of(std::shared_ptr<const passband::PassbandStore> pb) {
        return Passband{std::nullopt, std::move(pb)};
    }
};

// retro_detect's arguments (em_search.hpp's RetroOptions: the slots, none
// for every slot the passband store holds; the estimator; the template
// search's options).
using RetroOptions = em::RetroOptions;

// One find: what the log line says, typed. f_hz is the received pass's,
// z the search's Z; key none for a pass stored provisional; rule "header" |
// "soft-header" | "corr" | "prov", or "" for a pass that was a member
// already (the log's "existing membership").
struct RetroEntry {
    std::int64_t q = 0;
    double f_hz = 0.0;
    double z = 0.0;
    std::int64_t segment = 0;
    std::optional<store::Key> key;
    std::string rule;
    std::string uid;          // the stored pass
    double mean_w_db = 0.0;   // its own mean W (receiver::mean_w_db)
};

struct RetroResult {
    std::vector<RetroEntry> finds;   // in slot order
    std::int64_t slots_total = 0;    // the slots considered (held and short ones included)
    std::int64_t slots_done = 0;     // examined (== slots_total unless cancelled)
    std::int64_t slots_searched = 0; // template-searched (not held, on disk)
    bool cancelled = false;          // stopped by hooks.cancel; every find listed is stored
};

// check_picture, the passband directory (for Passband::dir), the frame,
// the estimator, then em::retro_detect(store, key, pb, frame, opt):
// log, progress and cancel as em_search.hpp says. Never throws
// em::Cancelled.
RetroResult retro(store::Store& store, const store::Key& key, const Passband& pb, std::string_view frame,
                  const RetroOptions& opt = {}, const em::Hooks& hooks = {});

}  // namespace sstvae::qrss::em_api

#endif  // SSTVAE_QRSS_EM_API_HPP
