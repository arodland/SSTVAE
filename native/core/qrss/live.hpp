// The live QRSSTVAE listener's pieces below the listener itself:
// sstvae/qrss/live.py (the reference) up to `class LiveListener`, without
// StreamFE (frontend.hpp has it) and the end-of-slot child process
// (_FinishProcess, which a C++ listener runs as a thread instead).
//
//   live.py              here
//   utc_iso              utc_iso
//   frame_seconds        frame_seconds
//   in_band, centre_of   in_band, centre_of
//   _waveform_id         waveform_id
//   _due_waveform        due_waveform
//   FeRing               FeRing (and FeReader: anything read like one, such
//                        as the passband store)
//   _spans               spans
//   capture_read         capture_span (the indices), capture_read
//   capture_prepare      capture_prepare
//   slot_capture         slot_capture
//   finish_receive       finish_receive; finish_gate and finish_branch are
//                        its decisions, shared with the binding's way over
//                        Python objects
//   DialLog              DialLog (parse_dial_text: its parse)
//   same_dial, restrict  same_dial, restrict (restrict_ranges: the ranges kept)
//   Tile, SlotState,     Tile, SlotState, LiveConfig
//   LiveConfig
//   frontend.slot_t0     slot_t0
//
// Not a format module: nothing here goes on the air. Constants are
// rx_config.hpp's (live.py's, generated).
//
// Exactness. FeRing, spans, capture_read, restrict and the DialLog's
// arithmetic are the reference's integer and float arithmetic in its order:
// the same samples, masks, spans and times, to the bit. capture_prepare is
// bitwise too (frontend::blank and normalise agree with numpy to the bit),
// given the same noise: the reference fills what was not heard with
// numpy's default_rng(q) normals, which is a generator this code does not
// carry. The noise is a seam (NoiseFn); the default is default_noise, a
// generator of the same distribution, so a capture prepared here is the
// reference's in everything but the noise samples themselves, and the
// parity tests hand the C++ numpy's noise and compare bit for bit.
// finish_receive's decisions (k1, the shares heard, which receive runs) are
// the reference's; its passes are api::receive_slot's (rx_api.hpp), which
// agree with the reference's receiver to its stages' tolerances.
//
// Python's ints are unbounded and these are int64: an FE index, a dial or a
// time past 2^63 does not arise from a clock or a radio, and where one could
// arrive from a file (a dial line) it is refused rather than wrapped.
//
// Threads. FeRing and DialLog are safe from several threads at once (each
// holds a mutex; every accessor returns a copy). Everything else is a pure
// function, or a plain record (Tile, SlotState, LiveConfig) that its owner
// guards, as the reference's listener guards its own under one lock.
// What blocks, and for how long:
//   FeRing::write       an exclusive lock for the copy in (a chunk: microseconds)
//   FeRing::read        a shared lock for the copy out: writers wait that
//                       long (a FULL slot's 6M samples: ~10-20 ms)
//   DialLog::refresh    a stat, and a read of the file when it changed
//                       (milliseconds; never while holding the points' lock)
//   capture_prepare     the blanker and AGC over the heard part (about a
//                       tenth of a second per ten minutes heard) and the noise
//   finish_receive      a slot's receive: seconds (TINY) to minutes (FULL);
//                       the whole-slot search stops gating new candidates
//                       after FinishOptions::budget_s (FINISH_BUDGET_S)
// None of them holds a lock while it calls a seam.
//
// Errors are the reference's where it has one: std::invalid_argument for
// ValueError (a NaN where an int is wanted, a negative array length, an
// unknown frame), std::overflow_error for OverflowError (an infinite one,
// a time outside time_t), std::out_of_range for IndexError,
// math::TypeError for TypeError, std::system_error for the OSError gmtime
// gives past its range.

#ifndef SSTVAE_QRSS_LIVE_HPP
#define SSTVAE_QRSS_LIVE_HPP

#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/acquire.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/receiver.hpp"
#include "qrss/rx_api.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::live {

using cf = std::complex<float>;
using Erase = std::vector<std::pair<double, double>>;
// What the listener hands the receiver: one frame (CE alone) or the frames
// listened for (receiver::frames' order).
using Selection = std::span<const frame::FrameSpec>;

inline constexpr double RING_S = rxconfig::live::RING_S;
inline constexpr double MERGE_HZ = rxconfig::live::MERGE_HZ;
inline constexpr int PREAMBLE_S = rxconfig::live::PREAMBLE_S;
inline constexpr double CAPTURE_MARGIN_S = rxconfig::live::CAPTURE_MARGIN_S;
inline constexpr double FINISH_SEARCH_MIN_HEARD = rxconfig::live::FINISH_SEARCH_MIN_HEARD;
inline constexpr std::int64_t FINISH_MAX_CANDIDATES = rxconfig::live::FINISH_MAX_CANDIDATES;
inline constexpr double FINISH_BUDGET_S = rxconfig::live::FINISH_BUDGET_S;
inline constexpr double HEADERLESS_MAX_SNR_DB = rxconfig::live::HEADERLESS_MAX_SNR_DB;
inline constexpr double RECOVER_HOURS = rxconfig::live::RECOVER_HOURS;
inline constexpr std::string_view GUESS_NOTE = rxconfig::live::GUESS_NOTE;
inline constexpr std::string_view DIAL_FILE = rxconfig::live::DIAL_FILE;
inline constexpr std::string_view LOG_FILE = rxconfig::live::LOG_FILE;
inline constexpr std::int64_t LOG_FILE_BYTES = rxconfig::live::LOG_FILE_BYTES;
inline constexpr std::array<double, 2> DIAL_GUARD_S = rxconfig::live::DIAL_GUARD_S;
inline constexpr double L_BELOW_HZ = rxconfig::live::L_BELOW_HZ;
inline constexpr double PB_LEAD_S = rxconfig::frontend::PB_LEAD_S;
inline constexpr double PB_TAIL_S = rxconfig::frontend::PB_TAIL_S;

// --- small things ---------------------------------------------------------------------------

// frontend.slot_t0: unix time of slot q's nominal t0, the quarter hour plus 1 s.
std::int64_t slot_t0(std::int64_t q);

// utc_iso(t): "%Y-%m-%dT%H:%M:%SZ" of datetime.fromtimestamp(t, utc) -- the
// microseconds rounded half to even first, so 0.9999995 is the next second;
// the year unpadded below 1000, as glibc's strftime writes it ("1-01-01").
// std::invalid_argument "Invalid value NaN (not a number)" or "year N is out
// of range" (outside 1..9999), std::overflow_error "timestamp out of range
// for platform time_t", std::system_error (EOVERFLOW) where gmtime fails.
std::string utc_iso(double t);
// The same of an int timestamp (fromtimestamp's int path: whole seconds,
// nothing rounded), for one past 2^53 that a double would round.
std::string utc_iso_int(std::int64_t t);

// frame_seconds: t0 to the end of the frame's last keyed symbol; for
// several frames the longest's (how long the listener's slot lasts).
// std::invalid_argument "max() iterable argument is empty" for none.
double frame_seconds(const frame::FrameSpec& spec);
double frame_seconds(Selection specs);

// in_band: a CE carrier at f_hz, or an L pilot at f_hz with its data
// L_BELOW_HZ under it, inside CARRIER_BAND_HZ.
bool in_band(double f_hz, int waveform = config::WAVEFORM_CE);

// centre_of: receiver.centre_hz for what is not a detection or a pass (a
// tile, a hint): f_hz + L_EXCLUDE_OFFSET_HZ for L, f_hz for anything else.
double centre_of(double f_hz, int waveform);

// _waveform_id of a name: "CE" 0, "L" 1, any other name CE (a tile without one).
int waveform_id(std::string_view name);
// _waveform_id of a JSON value (a tile's or a stored pass's "waveform"): a
// name as above, else int() of it (a float truncated; std::invalid_argument
// for NaN, std::overflow_error for an infinity, math::TypeError for null, a
// list or a dict, as int() raises).
std::int64_t waveform_id(const pyjson::Value& w);

// _due_waveform(d): the waveform of a stored pass (`store.due`'s record): its
// frame's when "frame" names one ("l-..." is L), else waveform_id of its
// "waveform", else CE (a pass stored before L).
std::int64_t due_waveform(const pyjson::Value& d);
// Its first half alone: the waveform of the frame `name` names (nullopt: no frame).
std::optional<int> frame_waveform(std::string_view name);

// --- the FE ring ----------------------------------------------------------------------------

// FE samples [a, b) as read from the ring (or anything read like it): the
// samples (zero where not held) and which were written (1) or not (0).
struct FeSpan {
    std::vector<cf> x;
    std::vector<std::uint8_t> ok;
};

// Anything read like an FeRing: (x, written) for FE indices [a, b) (the
// passband store's read, for a slot that ended while no listener ran).
using FeReader = std::function<FeSpan(std::int64_t a, std::int64_t b)>;

// FeRing: the last span_s seconds of FE in memory, addressed by absolute FE
// index (t * 4000). The reference's rules exactly: a write longer than the
// ring keeps its last n samples; a write starting past the end clears the
// hole between (what was there is older than the ring); end becomes
// max(end or 0, n0 + len), and whatever of the write lies before the new
// end - n, behind the ring's window, is dropped (its slots hold newer
// samples: it overwrote them before 2026-10-14's fix); the rest lands at
// its index mod n. A read is zero and unwritten wherever the ring does not
// hold the index (before end - n, at or after end).
//
// Thread-safe: write and clear take an exclusive lock, read a shared one;
// each copies under it and returns a copy.
class FeRing {
public:
    // n = int(span_s * FE_FS) samples. std::invalid_argument for NaN or a
    // negative n ("negative dimensions are not allowed"), std::overflow_error
    // for an infinity.
    explicit FeRing(double span_s = RING_S);

    std::int64_t n() const { return n_; }
    // One past the newest index written (nullopt: nothing written yet).
    std::optional<std::int64_t> end() const;

    void write(std::int64_t n0, std::span<const cf> x);
    // _clear(a, b): [a, min(b, a + n)) unwritten and zero.
    void clear(std::int64_t a, std::int64_t b);
    // (x, written) for [a, b); std::invalid_argument for b < a.
    FeSpan read(std::int64_t a, std::int64_t b) const;
    // The same into the caller's buffers, [a, a + x.size()) (no allocation,
    // the shared lock held for the copy alone); std::invalid_argument when
    // x and ok differ in length.
    void read_into(std::int64_t a, std::span<cf> x, std::span<std::uint8_t> ok) const;

    // Copies of the ring's storage, by ring position (index mod n).
    std::vector<cf> samples() const;
    std::vector<std::uint8_t> written() const;

private:
    std::int64_t n_;
    mutable std::shared_mutex mu_;
    std::vector<cf> x_;
    std::vector<std::uint8_t> ok_;
    std::optional<std::int64_t> end_;

    void clear_locked(std::int64_t a, std::int64_t b);
};

// _spans(mask, fs, origin): (start, end) seconds of the False runs of mask,
// sample k at (k - origin) / fs.
Erase spans(std::span<const std::uint8_t> mask, double fs, double origin);

// --- a slot's capture -------------------------------------------------------------------------

// capture_read's FE indices: [a, b) and the index of t0 in them (origin):
// a = t0 FE_FS - round(lead_s FE_FS), b = t0 FE_FS + round((frame_s +
// tail_s) FE_FS), half to even. std::invalid_argument / std::overflow_error
// for a NaN or infinite length, as round() raises them.
struct CaptureSpan {
    std::int64_t a = 0;
    std::int64_t b = 0;
    std::int64_t origin = 0;
};
CaptureSpan capture_span(std::int64_t q, double frame_s, double lead_s = PB_LEAD_S, double tail_s = PB_TAIL_S);

// (FE, written, index of t0 in them): slot q as read from the ring or a reader.
struct SlotRead {
    std::vector<cf> x;
    std::vector<std::uint8_t> ok;
    std::int64_t origin = 0;
};
SlotRead capture_read(const FeRing& ring, std::int64_t q, Selection sel, double lead_s = PB_LEAD_S,
                      double tail_s = PB_TAIL_S);
SlotRead capture_read(const FeReader& reader, std::int64_t q, Selection sel, double lead_s = PB_LEAD_S,
                      double tail_s = PB_TAIL_S);

// The noise that fills what was not heard: n complex64 samples, drawn from
// `seed`. It must be a circular complex Gaussian whose real and imaginary
// parts are independent N(0, s^2) with s = sqrt(0.5 / ln 2): E|n|^2 =
// 1 / ln 2, so the median of |n|^2 is exactly 1, the level normalise leaves
// the heard samples at. The reference draws it as
//     s * (rng.standard_normal(n) + 1j * rng.standard_normal(n)), complex64,
//     rng = np.random.default_rng(seed)
// and a seam returning those values makes the capture the reference's bit
// for bit. Returning other than n samples is std::invalid_argument.
using NoiseFn = std::function<std::vector<cf>(std::int64_t seed, std::size_t n)>;

// The default: that distribution from xoshiro256** seeded by splitmix64(seed),
// Box-Muller pairs (one complex sample each), s * (re, im) in double rounded
// to float. Deterministic for a seed on one platform (libm's log, sqrt,
// cos and sin underneath), but not numpy's samples.
std::vector<cf> default_noise(std::int64_t seed, std::size_t n);

// The front end's two stages capture_prepare runs, replaceable: empty is
// frontend::blank / frontend::normalise with their defaults (the
// reference's frontend.blank(x[:m]) and frontend.normalise(fe[:m],
// keep[:m])).
using BlankFn = std::function<frontend::Blanked(std::span<const cf> fe)>;
using NormaliseFn = std::function<frontend::Normalised(std::span<const cf> fe, std::span<const float> keep)>;
struct PrepareSeams {
    NoiseFn noise;           // empty: default_noise
    BlankFn blank;           // empty: frontend::blank
    NormaliseFn normalise;   // empty: frontend::normalise
};

// capture_prepare(x, ok, origin, q, spec, seed): the Prepared capture (fe
// blanked and normalised where heard, up to CAPTURE_MARGIN_S past the
// newest heard sample, noise where not heard; keep the blanker's gain where
// heard and 1 where filled; q, t0_index = origin, fs = FE_FS), the spans
// to erase (`spans` of ok), and the share of the frame heard
// (mean(ok[origin:k1]) with k1 = min(len, origin + round(frame_s FE_FS)),
// Python's slice rules; 0.0 when k1 <= origin or the slice is empty, as the
// reference's _share_heard). seed nullopt: q.
// std::invalid_argument when x and ok differ in length.
struct SlotCapture {
    receiver::Prepared prep;
    Erase erase;
    double heard = 0.0;
};
SlotCapture capture_prepare(std::span<const cf> x, std::span<const std::uint8_t> ok, std::int64_t origin,
                            std::int64_t q, double frame_s, std::optional<std::int64_t> seed = std::nullopt,
                            const PrepareSeams& seams = {});
SlotCapture capture_prepare(std::span<const cf> x, std::span<const std::uint8_t> ok, std::int64_t origin,
                            std::int64_t q, Selection sel, std::optional<std::int64_t> seed = std::nullopt,
                            const PrepareSeams& seams = {});

// slot_capture(ring, q, spec, seed, lead_s, tail_s): capture_read then capture_prepare.
SlotCapture slot_capture(const FeRing& ring, std::int64_t q, Selection sel, std::optional<std::int64_t> seed = {},
                         double lead_s = PB_LEAD_S, double tail_s = PB_TAIL_S, const PrepareSeams& seams = {});

// --- the end-of-slot receive ----------------------------------------------------------------------

// Python's slice ok[start:stop] as [lo, hi) of a sequence of length n
// (negative indices from the end, clamped).
std::pair<std::int64_t, std::int64_t> py_slice(std::int64_t n, std::int64_t start, std::int64_t stop);

// finish_receive's first decision, before anything is prepared: k1, and
// whether to receive at all. receive false: nothing heard of the frame, or
// no dets and no hints with less than min_heard of it heard; `heard` is then
// what the reference returns (mean(ok[origin:k1]), 0.0 when k1 <= origin or
// the slice is empty: it was NaN before 2026-10-14's fix).
struct FinishGate {
    bool receive = false;
    std::int64_t k1 = 0;
    double heard = 0.0;
};
FinishGate finish_gate(std::span<const std::uint8_t> ok, std::int64_t origin, double frame_s, bool have_dets,
                       bool have_hints, double min_heard = FINISH_SEARCH_MIN_HEARD);

// Its second, on the share heard capture_prepare measured: Search (the
// whole-slot search, with FinishOptions' candidate and time budgets) when at
// least min_heard was heard; else Known (the dets, plus a live-only detect
// at the hints when there are any) when there are dets or hints and
// anything was heard; else Nothing.
enum class FinishBranch { Search, Known, Nothing };
FinishBranch finish_branch(double heard, bool have_dets, bool have_hints, double min_heard = FINISH_SEARCH_MIN_HEARD);

// A ReceiveConfig for finish_receive from the listener's: frames (the
// selection) and estimator (the rules, old_edge_noise and old_gating, are
// the receiver's defaults: set them on the result). The search's budgets
// are not the config's: they are FinishOptions'.
api::ReceiveConfig finish_config(std::vector<frame::FrameSpec> frames, std::string estimator = "joint");

// finish_receive's own knobs, the reference's constants by default. The
// candidate and time budgets are here rather than in the ReceiveConfig
// because the reference's finish_receive always applies them, whatever the
// listener's receives use: a config passed in with none (the default
// ReceiveConfig, or the live receives') must not turn the end-of-slot
// search on a busy band into an unbounded one, which ran for longer than a
// slot before FINISH_MAX_CANDIDATES existed. nullopt: no limit.
struct FinishOptions {
    double min_heard = FINISH_SEARCH_MIN_HEARD;
    std::optional<std::int64_t> max_candidates = FINISH_MAX_CANDIDATES;
    std::optional<double> budget_s = FINISH_BUDGET_S;
    std::optional<std::int64_t> seed;   // the noise's seed; nullopt: q
    PrepareSeams seams;                 // the noise and the front end (empty: the C++)
};

struct FinishResult {
    std::vector<types::PassResult> passes;
    double heard = 0.0;
};

// finish_receive(x, ok, origin, q, spec, estimator, dets, hints): the
// end-of-slot receive of slot q from its capture_read (`ok` already
// restricted to one dial). `config` is the listener's: its frames are the
// selection, its estimator the passes', its old_edge_noise and old_gating
// the receiver's rules (finish_config); live_only, erase_s, round_b, hints,
// max_candidates and budget_s are set here per call, as the reference
// passes them (the budgets opt's, whatever the config holds):
//   Search: receive_slot(prep, sel, estimator, erase_s=erase,
//           max_candidates=opt, budget_s=opt, hints=hints)
//   Known:  more = detect(prep, sel, live_only=True, erase_s=erase,
//           hints=hints, skip_hz=known, max_candidates=opt, budget_s=opt) when
//           there are hints (known: the dets' f_hz, their band centres for
//           a joint selection); receive_slot(prep, sel, estimator,
//           erase_s=erase, dets=dets + more)
std::vector<double> known_hz(std::span<const types::Detection> dets, bool joint);
FinishResult finish_receive(std::span<const cf> x, std::span<const std::uint8_t> ok, std::int64_t origin,
                            std::int64_t q, const api::ReceiveConfig& config,
                            std::span<const types::Detection> dets = {},
                            std::span<const acquire::Hint> hints = {}, const FinishOptions& opt = {});

// --- the radio's dial ---------------------------------------------------------------------------------

// (time, dial) at the start of each dial epoch.
struct DialPoint {
    double t = 0.0;
    std::int64_t dial_hz = 0;
    bool operator==(const DialPoint&) const = default;
};
// (lo, hi, dial): heard on `dial` from lo to hi, unix seconds (-inf / inf
// where no change bounds it); dial nullopt: unknown.
struct DialSpan {
    double lo = 0.0;
    double hi = 0.0;
    std::optional<std::int64_t> dial;
};
using DialGuard = std::array<double, 2>;
using DialEpochs = std::vector<std::pair<std::optional<std::int64_t>, std::vector<std::pair<double, double>>>>;

// DialLog.refresh's parse of the file's bytes: decoded as UTF-8 with
// invalid sequences replaced (U+FFFD for each maximal invalid subpart, as
// Python's "replace"), split as str.splitlines splits (\n, \r, \r\n, \v,
// \f, \x1c-\x1e, U+0085, U+2028, U+2029); each line json.loads'd; its "t"
// and "dial_hz" taken as float() takes them (a number, a bool, or a
// numeric string: pynum::py_float), the dial rounded half to even; a line
// that does not parse, lacks either, or is not a dict, skipped, and so is
// one whose t is not finite or whose dial is not positive, and one whose
// dial is infinite or whose t is past float's range (OverflowError, which
// the reference skips since 2026-10-14's fix); stably sorted by t, a
// repeated dial (no change) dropped. A dial past int64 is skipped too, where
// the reference keeps it as a Python int (a deviation).
std::vector<DialPoint> parse_dial_text(std::string_view bytes);

// DialLog: the radio's dial frequency over time, from a DIAL_FILE the app
// writes (one {"t": unix seconds, "dial_hz": Hz} per line). refresh
// re-reads the file when its signature (modification time, size) changed;
// a file that cannot be stat'ed means no dial (no points); one that cannot
// be read keeps the points it had. Every query takes the guard (unheard
// seconds before and after a change), DIAL_GUARD_S by default.
//
// Thread-safe: refresh holds a lock of its own across the read, and the
// points' lock only to swap them in; the queries take the points' lock
// for a copy.
class DialLog {
public:
    explicit DialLog(std::string path);

    const std::string& path() const { return path_; }

    // Re-read the file if it changed (blocks on the file system).
    void refresh();

    std::vector<DialPoint> points() const;
    void set_points(std::vector<DialPoint> points);

    // (lo, hi, dial) for each dial in force at some time in [a, b), in time
    // order, the guard taken off around each change; [(-inf, inf,
    // unknown)] when the dial is unknown.
    std::vector<DialSpan> spans(double a, double b, const DialGuard& guard = DIAL_GUARD_S) const;
    // spans gathered by dial, in order of first use.
    DialEpochs by_dial(double a, double b, const DialGuard& guard = DIAL_GUARD_S) const;
    // The dial in force at time t (nullopt: unknown).
    std::optional<std::int64_t> dial_at(double t, const DialGuard& guard = DIAL_GUARD_S) const;
    // spans for the dial in force at time t (lo can lie after t just after a change).
    DialSpan span_at(double t, const DialGuard& guard = DIAL_GUARD_S) const;

private:
    struct Signature {
        std::int64_t mtime = 0;
        std::uintmax_t size = 0;
        bool operator==(const Signature&) const = default;
    };

    std::string path_;
    std::mutex refresh_mu_;
    mutable std::mutex mu_;
    std::optional<Signature> sig_;
    std::vector<DialPoint> points_;
};

// The same functions over points already read (DialLog's own, and the
// binding's for a points list a caller set).
std::vector<DialSpan> dial_spans(std::span<const DialPoint> pts, double a, double b,
                                 const DialGuard& guard = DIAL_GUARD_S);
DialEpochs dial_by_dial(std::span<const DialPoint> pts, double a, double b, const DialGuard& guard = DIAL_GUARD_S);
DialSpan dial_span_at(std::span<const DialPoint> pts, double t, const DialGuard& guard = DIAL_GUARD_S);

// same_dial: equal, or either unknown.
bool same_dial(std::optional<std::int64_t> a, std::optional<std::int64_t> b);

// restrict(ok, origin, q, spans)'s ranges: the [a, b) of ok kept for each
// span (lo, hi) in unix seconds, in the spans' order (empty ones dropped);
// nullopt when a span is (-inf, inf), where restrict returns ok itself.
// A span bound of +inf as lo, or -inf as hi, is std::overflow_error, and
// NaN std::invalid_argument, as math.ceil raises them.
std::optional<std::vector<std::pair<std::int64_t, std::int64_t>>> restrict_ranges(
    std::int64_t n, std::int64_t origin, std::int64_t q, std::span<const std::pair<double, double>> spans);
// restrict: ok less everything outside the spans.
std::vector<std::uint8_t> restrict(std::span<const std::uint8_t> ok, std::int64_t origin, std::int64_t q,
                                   std::span<const std::pair<double, double>> spans);

// --- tiles ------------------------------------------------------------------------------------

// What the window shows for one signal in one slot (live.Tile, its fields
// in its order and with its defaults).
struct Tile {
    std::string id;
    std::int64_t q = 0;
    std::string slot_utc;
    std::string frame;
    double f_hz = 0.0;                          // audio Hz at t0 (an L tile's pilot)
    std::optional<std::int64_t> dial_hz;        // the radio's dial while it was heard
    std::string status = "receiving";           // receiving | finishing | complete | lost
    std::optional<double> snr_db;               // SNR in 2500 Hz
    std::optional<double> z_ref;
    std::optional<std::string> callsign;
    std::optional<std::string> grid;
    std::optional<std::string> picture_id;      // hexadecimal
    std::optional<std::string> mode;            // A, B, C
    std::optional<std::int64_t> segment;
    std::optional<std::string> cw_text;
    double progress = 0.0;                      // share of the frame's time elapsed when read
    double heard = 0.0;                         // share of the frame's audio heard
    double received = 0.0;                      // share of the data latents with W > 0
    std::optional<double> mean_w_db;
    std::vector<std::optional<double>> confidence;   // per-latent SNR (dB) over time, None: nothing to say
    std::int64_t passes = 1;                    // passes of this picture in the picture shown
    std::optional<std::string> image;           // path relative to the state directory
    std::int64_t image_rev = 0;                 // bumped every time the picture changes
    double updated = 0.0;
    std::string note;
    std::optional<std::string> uid;             // the pass as filed in the multi-pass store
    std::string waveform = "CE";                // "CE" or "L"

    // Tile.to_json(): asdict(tile) as a JSON object, the fields in the
    // reference's order, floats as floats, a float that is not finite null
    // (in confidence too), so state.json stays strict JSON.
    pyjson::Value to_json() const;
    // Tile(**{k: t[k] for k in fields if k in t}), the listener's reading
    // of an earlier state.json: unknown keys ignored, absent ones their
    // defaults. nullopt where the reference raises TypeError (id, q,
    // slot_utc, frame or f_hz missing) or where a value is not of its
    // field's JSON type (an int is taken where a float is, as Python
    // would; the reference keeps whatever it finds).
    static std::optional<Tile> from_json(const pyjson::Value& v);
};

// One slot the listener is receiving. Not synchronised: the listener guards
// its slots with its own lock.
struct SlotState {
    std::int64_t q = 0;
    std::vector<types::Detection> dets;
    std::vector<std::pair<std::string, Tile>> tiles;   // id -> tile, in insertion order
    std::optional<double> last_refresh;
    bool done = false;
    bool finishing = false;              // the end-of-slot receive is queued or running
    FeReader reader;                     // where its audio is read from (empty: the ring)
    std::optional<std::vector<types::Detection>> cands;   // its candidates, once they can no longer change
    std::vector<acquire::Hint> hints;    // ...and the hints they were found with

    Tile* tile(std::string_view id);
    const Tile* tile(std::string_view id) const;
};

// The listener's configuration (live.LiveConfig), its defaults.
struct LiveConfig {
    frame::FrameSpec spec = frame::get("full");
    // The waveforms listened for, each in spec's shape; nullopt is CE alone.
    std::optional<std::vector<std::int64_t>> waveforms;
    double refresh_s = 120.0;
    double first_s = 25.0;
    double detect_until_s = 300.0;
    double recheck_until_s = 900.0;
    double empty_refresh_s = 300.0;
    double keep_done_s = 6 * 3600.0;
    std::string estimator = "joint";
    std::string precision = "fp32";
    std::optional<std::string> model;
    bool render = true;
    bool background = false;
    bool finish_process = false;

    // The frames listened for, CE's first: spec with no waveforms, else
    // its twin in each. std::invalid_argument "waveforms (2,): one or more
    // of 0, 1" for none or an unknown one (the waveforms shown as a tuple),
    // and twin's for a shape the other waveform has no frame of (full-v1).
    std::vector<frame::FrameSpec> specs() const;
    // What the listener hands the receiver: specs() when it holds an L
    // frame (receiver::is_joint), else its CE frame alone.
    std::vector<frame::FrameSpec> selection() const;
    bool joint() const;
};

}  // namespace sstvae::qrss::live

#endif  // SSTVAE_QRSS_LIVE_HPP
