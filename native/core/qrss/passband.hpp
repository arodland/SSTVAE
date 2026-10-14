// frontend.PassbandStore (sstvae/qrss/frontend.py, the reference): the
// whole FE stream on disk for 48 h, so that passes too weak to detect when
// they arrived can be found later by template search (spec 7, design 6.9).
//
// The files are the reference's, byte for byte, so a store either
// implementation wrote is read and extended by the other:
//
//   meta.json          {"fs": <int>, "scale": <float>}   (json.dumps)
//   fe_<hour>.i16      hour = floor(t/3600): 3600*fs samples of interleaved
//                      int16 I/Q, a fixed size, created sparse (POSIX
//                      ftruncate; Windows FSCTL_SET_SPARSE), sample k at
//                      unix time 3600*hour + k/fs
//   fe_<hour>.json     the written sample ranges, [[lo, hi], ...] merged
//                      as _merge_ranges merges them (json.dumps)
//   fe_<hour>.flac     a packed hour: I and Q as two 16-bit channels,
//                      lossless; written by ours (flac.hpp) or soundfile
//
// Every file is written to a temporary sibling and moved into place
// (os.replace), with the reference's temporary names, "<name>.<pid>.
// <thread ident>.tmp" (the JSON files; fe_<hour>.flac while packing,
// fe_<hour>.i16 while unpacking), so no two writers -- processes, or
// threads of one -- share one. expire() also deletes the store's
// temporaries whose mtime is more than an hour old by the wall clock
// (PB_TMP_AGE_S; never the store's clock, which may be a replay's): what a
// process killed mid-pack or mid-write leaves. That sweep is best effort:
// one it cannot stat or remove is left for the next expire, never thrown
// out of the write that called it. Raw hours are written and read in place
// by positioned I/O.
//
// Samples are quantised exactly as the reference does it: x as complex64,
// then per part rint(float32(part) * float32(scale)) in float32 (NEP 50),
// clipped to +-32767, cast to int16 (NaN to 0, as numpy's cast does on
// every platform this runs on); read back as float32(int16) /
// float32(scale). Times: a write or read at unix time t starts at sample
// round(t*fs), half to even; a NaN time is ValueError("cannot convert
// float NaN to integer") and an infinite one OverflowError, as round()
// raises them.
//
// Packing (`pack`): an hour is packed PB_PACK_AFTER_S after it ends, by the
// stream's own time (a write's end), on a thread of its own, at most one
// at a time and checked for at most every 60 s; pack_hour encodes, syncs
// the result to the device, decodes it back and compares it sample for
// sample with the raw file, and only then (under the store's lock)
// renames it into place and deletes the raw file -- refusing if the raw
// file's mtime moved, or an in-process write reached the hour (a count
// per hour, which a file system's coarse mtime cannot hide), meanwhile;
// all as the reference does it. Unlike the reference, packing needs
// nothing installed (the encoder is ours), so `pack` is honoured as given.
//
// Durability differs from the reference's in one place, on purpose: a raw
// write is not synced (the reference's mm.flush() is msync(MS_SYNC), a
// device write per call on POSIX, which the listener's audio feed should
// not wait on), so after a power cut the index may name samples that read
// back as zeros. A raw hour found shorter than an hour (a crash between
// its creation and its sizing) reads as unwritten past its end, and the
// next write into it grows it back to size, as numpy's memmap "r+" does.
// A store refuses fs <= 0 (ValueError), from its arguments or meta.json.
//
// Threads: every public call may be made from any thread. A recursive
// mutex covers each write into an hour (data and index) against a pack's
// swap, an unpack and expiry, as the reference's RLock does; reads take it
// only to read an hour's index and open its file, and read the samples
// outside it, so a reader never holds up the writer for longer than a
// JSON read. Nothing returned refers into guarded state. No mutable
// globals. What blocks, and for how long:
//   - write: the positioned write and an index rewrite (milliseconds),
//     once per hour touched; with expire, a directory listing. A write
//     into a packed hour unpacks it first (decoding a whole FLAC hour:
//     about a quarter of a second) under the lock.
//   - read / capture: the I/O (a slot is ~30 MB of raw hours or a FLAC
//     decode of the same, a fraction of a second), outside the lock.
//   - pack_hour: 1.4 s for a full hour of band noise on a desktop (encode
//     1.0, decode and compare the rest; soundfile's path takes 1.0 s), a
//     third of a second for a mostly unwritten one, outside the lock, then
//     a rename under it. Two pack_hour calls run one after the other.
//   - stop_packing: a flag, nothing waited for.
//   - ~PassbandStore: asks a running packer to stop (it checks once per
//     FLAC frame) and joins it: a few milliseconds.

#ifndef SSTVAE_QRSS_PASSBAND_HPP
#define SSTVAE_QRSS_PASSBAND_HPP

#include <array>
#include <atomic>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "qrss/frontend.hpp"
#include "qrss/rawfile.hpp"
#include "qrss/rx_config.hpp"

namespace sstvae::qrss::passband {

using cf = std::complex<float>;

inline constexpr double PB_PACK_AFTER_S = rxconfig::frontend::PB_PACK_AFTER_S;
inline constexpr double PB_LEAD_S = rxconfig::frontend::PB_LEAD_S;
inline constexpr double PB_TAIL_S = rxconfig::frontend::PB_TAIL_S;
inline constexpr double PB_PACK_CHECK_S = 60.0;   // _maybe_pack's "at most every 60 s"
inline constexpr double PB_TMP_AGE_S = 3600.0;    // expire's "temporaries older than an hour"

// frontend.slot_t0: unix time of slot q's nominal t0, 900 q + 1.
// std::overflow_error past int64.
std::int64_t slot_t0(std::int64_t q);

// frontend._merge_ranges: sorted, overlapping or touching ranges joined.
using Range = std::array<std::int64_t, 2>;
std::vector<Range> merge_ranges(std::vector<Range> rs);

// The hour a file name names (_HOUR_RE: "fe_<ASCII digits>.i16" or
// ".flac", and nothing after); nullopt for any other name. No hour past
// HOUR_MAX (2^62 / 3600, some 10^11 years), so that hour arithmetic
// cannot overflow; the reference's int would take one.
inline constexpr std::int64_t HOUR_MAX = (std::int64_t{1} << 62) / 3600;
std::optional<std::int64_t> hour_of_name(std::string_view name);

// An OSError naming its file (rawfile.hpp).
using FileError = rawfile::FileError;

// round(x) as Python's round() of a float gives an int (half to even);
// std::invalid_argument / std::overflow_error for NaN / infinity, in its words.
std::int64_t py_round(double x);

struct Masked {
    std::vector<cf> x;
    std::vector<std::uint8_t> mask;   // 1 where samples were written
};

struct SlotCapture {
    std::optional<frontend::Capture> capture;   // none when nothing in the span was written
    double coverage = 0.0;
};

struct Written {
    double t_first = 0.0;   // unix time of the capture's first sample
    double gain = 1.0;      // what it was stored with
};

using Clock = std::function<double()>;   // unix seconds

struct StoreOptions {
    int fs = config::FE_FS;
    double scale = 8192.0;
    double keep_hours = 48.0;
    bool pack = false;
    Clock clock;   // for expire() without a time; empty: the system clock
    // Test seam: called by pack_hour (on its thread, no lock held) once the
    // packed file has decoded to the raw hour, just before the swap's own
    // checks -- where a write that must stop the swap can be placed
    // deterministically. Empty: nothing.
    std::function<void(std::int64_t hour)> before_swap;
};

class PassbandStore {
public:
    // Makes `root` (mkdir -p); reads meta.json when there is one (fs and
    // scale from it, as the files were written), else writes it from
    // `opt`. FileError for a root that cannot be made, std::invalid_argument
    // (ValueError) for unparsable JSON or an fs <= 0 (before meta.json is
    // written), pyjson::KeyError for a key missing.
    explicit PassbandStore(std::string root, StoreOptions opt = {});
    ~PassbandStore();
    PassbandStore(const PassbandStore&) = delete;
    PassbandStore& operator=(const PassbandStore&) = delete;

    const std::string& root() const { return root_; }
    int fs() const { return fs_; }
    double scale() const { return scale_; }
    double keep_hours() const { return keep_hours_; }
    std::int64_t per_hour() const { return per_hour_; }
    bool pack() const { return pack_; }

    // Store complex samples x, the first at unix time t0 (sample
    // round(t0*fs)). With `expire`, hours older than keep_hours before
    // the end of x are deleted and, with pack, finished hours queued.
    void write(double t0, std::span<const cf> x, bool expire = true);
    void write_at(std::int64_t n0, std::span<const cf> x, bool expire = true);

    // complex64 samples [round(t_start fs), round(t_end fs)); zeros, with
    // mask 0, where nothing was written (past the end of a short file too).
    Masked read(double t_start, double t_end) const;

    // A slot capture's raw FE stream: (unix time of its first sample, gain).
    // A capture whose peak (over its finite samples) would use more than
    // half the int16 range is stored attenuated by a power of 2; a
    // non-finite sample is stored as write() stores it. std::invalid_argument
    // when its fs is not the store's.
    Written write_capture(const frontend::Capture& cap, bool expire = true);
    // Slot q read back, t0 - lead_s to t0 + dur_s + tail_s; coverage is the
    // fraction of [t0, t0 + dur_s) written. Unwritten ends are trimmed.
    SlotCapture capture(std::int64_t q, double dur_s, double lead_s = PB_LEAD_S, double tail_s = PB_TAIL_S) const;
    // Every q whose frame [t0, t0 + dur_s) overlaps a stored hour, ascending.
    std::vector<std::int64_t> slots(double dur_s) const;

    // Hours with a file on disk, ascending; and those still raw.
    std::vector<std::int64_t> hours() const;
    std::vector<std::int64_t> raw_hours() const;
    // Delete hours that ended more than keep_hours before `now` (the
    // clock's time when absent); the hours deleted. Also deletes the
    // store's temporaries older than PB_TMP_AGE_S by the wall clock, best
    // effort (one that cannot be stat'ed or removed is skipped silently).
    std::vector<std::int64_t> expire(std::optional<double> now = std::nullopt);

    // Pack raw hour `hour` into FLAC; false (raw file kept) when it could
    // not be (no raw file, an I/O error, a mismatch on decoding, a write
    // meanwhile, or the store closing).
    bool pack_hour(std::int64_t hour);
    // Start packing, on the packer thread, the raw hours that ended
    // PB_PACK_AFTER_S before `now` (frontend's _maybe_pack).
    void maybe_pack(double now);

    // The packer thread: started at all; running now; wait for it (up to
    // `timeout_s`, forever when absent): true once it is not running.
    bool packer_started() const;
    bool packer_alive() const;
    bool join_packer(std::optional<double> timeout_s = std::nullopt);
    // Pack nothing more (the flag the destructor sets before it joins the
    // packer): the packer starts no further hour, and gives up the one in
    // hand at its next FLAC frame, unless that hour has passed its checks
    // and is being swapped into place, which then completes (the rename
    // under the store's lock). From then on maybe_pack starts no packer
    // and pack_hour returns false; writes, reads and expiry go on.
    // Non-blocking (it waits for nothing: join_packer does) and idempotent.
    void stop_packing();

    // File names (UTF-8), as the reference spells them.
    std::string raw_path(std::int64_t hour) const;
    std::string index_path(std::int64_t hour) const;
    std::string flac_path(std::int64_t hour) const;

private:
    struct Hour;   // an opened hour (raw file or FLAC reader)

    std::vector<Range> ranges_locked(std::int64_t hour) const;
    void write_hour_locked(std::int64_t hour, std::int64_t k, const std::vector<std::int16_t>& q);
    void unpack_locked(std::int64_t hour);
    std::shared_ptr<Hour> open_hour_locked(std::int64_t hour) const;
    void quantise(std::span<const cf> x, std::vector<std::int16_t>& out) const;
    double now() const;

    std::string root_;
    int fs_;
    double scale_;
    double keep_hours_;
    std::int64_t per_hour_;
    bool pack_;
    Clock clock_;
    std::function<void(std::int64_t)> before_swap_;

    mutable std::recursive_mutex mu_;           // writes, swaps, expiry; index reads
    std::map<std::int64_t, std::uint64_t> gen_; // in-process writes per hour (guarded by mu_)
    std::mutex pack_mu_;                        // one pack_hour at a time (they share temporary names)

    mutable std::mutex packer_mu_;              // the packer's bookkeeping
    std::condition_variable packer_cv_;
    std::thread packer_;
    bool packer_started_ = false;
    bool packer_running_ = false;
    double pack_check_ = 0.0;
    std::atomic<bool> closing_{false};
};

}  // namespace sstvae::qrss::passband

#endif  // SSTVAE_QRSS_PASSBAND_HPP
