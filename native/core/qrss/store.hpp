// The multi-pass store: port of sstvae/qrss/store.py (design 8.1, spec
// section 8) -- per-pass files, per-picture accumulators, provisional
// clusters, expiry and merging.
//
// Not a format module: nothing here goes on the air. What is a contract is
// the accumulator's meaning (S = S_ext + sum over members of w*z, W = W_ext
// + sum of w, canonical [3, 52800], float32) and, because the store is
// shared with the Python tools (qrss_decode.py, qrss_receive.py, a Python
// listener), its files. The layout under the root is store.py's:
//
//     passes/<uid>.npz           PassResult::to_npz (byte for byte PassResult.save)
//     acc/<CALL>_<pid:08x>.json  json.dumps(meta, indent=1, sort_keys=True)
//     acc/<CALL>_<pid:08x>.npz   S, W, S_ext, W_ext float32[3, 52800], then each
//                                foreign merge's own Sx_<sha>, Wx_<sha>
//     prov/<cluster>.json        a provisional cluster of unassociated pass uids
//     log.jsonl                  association events, one json.dumps(event) a line
//
// and every file is written as store.py writes it, so a store either
// implementation wrote is the same bytes (tests/test_qrss_native_store.py
// diffs two trees): the npz through npz.hpp (np.savez's archive, 1980
// timestamps), the JSON through pyjson.hpp (Python's float repr, ensure_ascii,
// indent=1 and sorted keys), and each write to the temporary sibling
// ".<name>.<pid>.tmp" (store.py's name, so a Python process sharing the store
// meets the same names), flushed (and fsync'd where store.py fsyncs: the
// accumulators and clusters, not the pass files) and moved into place by the
// replace seam (os.replace). A failure removes the temporary file and
// rethrows. One documented difference in the bytes: a member's mean_w_db is
// 10*log10(mean w), which numpy computes with its own SIMD log10 on x86-64
// machines with AVX-512 (SVML) and this computes with the C library's; the
// two differ in the last place for a few percent of arguments (5 to 9% of
// 10*log10(x), x uniform on [0.01, 100] or [0.5, 20], measured against
// glibc on an AVX-512 machine, 2026-10-14), which then shows
// in the JSON text (and nowhere else: no decision here reads it but `twin`'s
// choice between two copies of one transmission).
//
// The JSON a file holds is kept as pyjson::Value (members, foreign entries,
// clusters, a pass's header), so keys and types this code does not know
// survive a rewrite exactly as Python's dicts carry them.
//
// Errors are Python's: a missing accumulator or pass is KeyError (pyjson's,
// which the binding raises as KeyError), a refusal std::invalid_argument
// (ValueError) with store.py's text, a file that cannot be opened, read,
// written, replaced or removed npz::FileError (OSError: verb, path, errno),
// malformed JSON std::invalid_argument in json's words. A JSON value of the
// wrong type where store.py calls int() or float() on it converts as
// Python does (a float truncated, a bool as 0/1, a decimal string parsed);
// anything else is math::TypeError (TypeError).
//
// Deliberate differences from store.py, all in what the reference would
// accept from a damaged file and this refuses: accumulator arrays must hold
// 3 x 52800 values (Python loads any shape and fails later), a foreign
// part's arrays likewise when they are summed; a pass's z and w are read with
// every other member of its npz (np.load reads lazily, so a damaged member
// it never reads goes unnoticed there).
//
// THREADS. A Store is safe to call from any number of threads: one internal
// recursive mutex guards its caches and serialises its file operations, and
// every call returns values (Accumulator, PassInfo, a shared_ptr to an
// immutable ZW), never a reference into guarded state. The mutex is held
// for the whole of a call, file I/O included: a call blocks while another
// thread's call runs. What a call costs: reading a pass is ~1 ms per MB
// (a full pass is about 1 MB, with its narrow capture), an accumulator
// rebuild ~1 ms per member once its passes are cached (ZW_CACHE of them),
// writing an accumulator ~2 MB of npz plus fsync (milliseconds on a local
// disk, longer on a slow card). `hold()` takes the mutex for a whole
// transaction (associate() holds it for one association), so a call made
// while another thread holds it waits until that transaction ends. The
// seams (clock, replace, zw_cache) are called with the mutex held: they must
// not wait for another thread that is itself waiting on this Store.
// An Accumulator is a plain value: safe to read from several threads,
// not to write from several at once (copy it instead).

#ifndef SSTVAE_QRSS_STORE_HPP
#define SSTVAE_QRSS_STORE_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qrss/npz.hpp"
#include "qrss/picture.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::store {

inline constexpr int N_GROUPS = rxconfig::store::N_GROUPS;             // 3
inline constexpr int GROUP_LATENTS = rxconfig::store::GROUP_LATENTS;   // 52,800
inline constexpr int SENT = rxconfig::store::SENT;                     // 50,600
inline constexpr std::size_t PLANE = static_cast<std::size_t>(N_GROUPS) * GROUP_LATENTS;
inline constexpr double DAY_S = rxconfig::store::DAY_S;
inline constexpr std::int64_t SLOT_S = rxconfig::store::SLOT_S;
inline constexpr std::size_t ZW_CACHE = rxconfig::store::ZW_CACHE;
inline constexpr double SAME_PASS_HZ = rxconfig::store::SAME_PASS_HZ;
static_assert(N_GROUPS == picture::N_GROUPS && GROUP_LATENTS == picture::GROUP_LATENTS && SENT == picture::SENT,
              "store.py's sizes are picture.py's");

// store.py's KeyError (a missing pass, accumulator or member).
using KeyError = pyjson::KeyError;
// A file that could not be opened, read, written, replaced or removed.
using FileError = npz::FileError;

// An accumulator's key: (callsign, picture ID).
struct Key {
    std::string callsign;
    std::int64_t picture_id = 0;
    bool operator==(const Key&) const = default;
};

// $QRSSTVAE_HOME (when set and not empty), else Path.home()/.local/share/
// qrsstvae, as str(Path(...)) spells it (UTF-8). Path.home() is
// os.path.expanduser("~"): $HOME (else the password database) on POSIX,
// %USERPROFILE% (else %HOMEDRIVE%%HOMEPATH%) on Windows, where the
// separators are backslashes as a WindowsPath prints them.
std::string default_root();

// "<CALL>_<pid:08x>", '/' -> '_' (a negative ID as Python formats one,
// "-0000005").
std::string key_name(const Key& key);

// Unix time of slot q's quarter hour: float(q * 900).
double slot_time(std::int64_t q);

// 10 log10 of a pass's mean weight; -99 when there are none or the mean is
// not positive (NaN included). The mean is np.mean's (pairwise sum).
double mean_w_db(std::span<const double> w);
double mean_w_db(std::span<const float> w);

// Canonical offsets (within the group) of a pass's n air-order latents:
// air_to_canonical(segment)[:n]. std::invalid_argument "a pass carries
// 1..50600 latents, not <n>" unless 0 < n <= SENT, then picture's refusal
// of a segment outside 0..2.
std::span<const std::uint16_t> canonical_index(std::int64_t segment, std::int64_t n);

// A pass's latents and weights as float64 (what accumulators are built from).
struct ZW {
    std::vector<double> z;
    std::vector<double> w;
};
using ZWPtr = std::shared_ptr<const ZW>;

// Store.pass_info's light metadata of a pass.
struct PassInfo {
    std::string uid;
    std::int64_t q = 0;
    double f_hz = 0.0;
    pyjson::Value header;            // the header dict, or null
    std::vector<double> hdr_llr;     // float64, flattened (empty for a headerless frame)
    double z_ref = 0.0;
    std::int64_t em_round = 0;
    std::int64_t waveform = 0;       // 0 = CE, 1 = L; 0 for a pass from before L
    pyjson::Value frame;             // meta.get("frame"): the preset's name, or null
};

// A foreign merge's own (S, W), as the npz holds them (Sx_<sha>, Wx_<sha>):
// kept in the file's dtype and shape, so a rewrite is byte-identical.
struct Part {
    std::string sha;
    npz::Array S;
    npz::Array W;
};

// One element of `due`.
struct Due {
    Key key;
    std::int64_t segment = 0;
    double f_hz = 0.0;
    std::int64_t q = 0;
    std::int64_t waveform = 0;
    pyjson::Value frame;
};

// What `expire` removed.
struct Expired {
    std::vector<Key> acc;
    std::vector<pyjson::Value> prov;      // the clusters' "cluster" values
    std::vector<std::string> passes;
};

class Store;

// One picture's accumulated latents, canonical [3, 52800] float32 row-major.
// members and foreign are the JSON's dicts (see Store::attach, Store::merge);
// parts is a dict in insertion order (sha -> foreign (S, W)).
struct Accumulator {
    Key key;
    std::int64_t mode = 0;
    std::int64_t codec_id = 0;
    std::vector<float> S, W, S_ext, W_ext;   // PLANE values each
    std::vector<pyjson::Value> members;
    std::vector<pyjson::Value> foreign;
    std::vector<Part> parts;
    double created = 0.0;
    double updated = 0.0;

    // Zero arrays, no members, created = updated = now.
    static Accumulator empty(Key key, std::int64_t mode, std::int64_t codec_id, double now);

    // [m["uid"] for m in members]
    std::vector<std::string> uids() const;
    // The first member whose "uid" is `uid` (a copy), or nullopt.
    std::optional<pyjson::Value> member(std::string_view uid) const;
    // Its position in `members`.
    std::optional<std::size_t> member_index(std::string_view uid) const;
    // The part for `sha`, or null.
    const Part* part(std::string_view sha) const;

    // float64 (S, W): S_ext/W_ext (if `ext`) plus every member but `skip`, in
    // member order (reads each member's pass through store.pass_zw). A null
    // `skip` skips a member whose "uid" is JSON null, as Python's `== None`.
    struct Sums {
        std::vector<double> S, W;
    };
    Sums sums(Store& store, std::optional<std::string_view> skip = std::nullopt, bool ext = true) const;
    // S, W = sums() as float32.
    void rebuild(Store& store);
    // Leave-one-out (S, W) for member `uid`, float64, in that pass's air
    // order; KeyError "<uid> is not a member of <key_name>" otherwise.
    std::pair<std::vector<double>, std::vector<double>> loo(std::string_view uid, Store& store) const;
    // np.any(W[g] > 0); std::out_of_range (IndexError) for g outside 0..2.
    bool has_data(std::int64_t g) const;
    // np.median of W over group g's 50,600 sent latents (float32 arithmetic).
    double median_w(std::int64_t g) const;
    // 10 log10 median W per segment 0..mode; nullopt without data.
    std::vector<std::optional<double>> seg_snr_db() const;
    // The JSON index store.py writes.
    pyjson::Value meta() const;
};

// The on-disk store. See the header comment for threads and blocking.
class Store {
public:
    // Unix seconds (created/updated stamps, expire's now, log times).
    using Clock = std::function<double()>;
    // os.replace(from, to); throws on failure (FileError by default).
    using Replace = std::function<void(const std::string& from, const std::string& to)>;
    // How many passes' (z, w) to keep in memory, asked at each read.
    using CacheSize = std::function<std::size_t()>;
    struct Seams {
        Clock clock;           // empty: the system clock (time.time())
        Replace replace;       // empty: util::replace_file
        CacheSize zw_cache;    // empty: ZW_CACHE
    };

    // Opens (creating passes/, acc/ and prov/ as needed) the store at
    // `root` (UTF-8), or default_root(). FileError when a directory
    // cannot be made.
    explicit Store(std::optional<std::string> root = std::nullopt, Seams seams = {});

    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;

    const std::string& root() const { return root_; }
    double now() const;

    // A lock on the store for a transaction of several calls (the calls
    // themselves lock it too; the mutex is recursive).
    std::unique_lock<std::recursive_mutex> hold() const { return std::unique_lock<std::recursive_mutex>(mu_); }

    // --- passes ---
    // root/passes/<uid>.npz; std::invalid_argument "bad pass uid '<uid>'"
    // for a uid holding '/' or starting with '.'.
    std::string pass_path(std::string_view uid) const;
    // Write (or overwrite) a pass file; forgets its cached (z, w) and info.
    void add_pass(const types::PassResult& p);
    bool has_pass(std::string_view uid) const;
    types::PassResult load_pass(std::string_view uid) const;
    // Sorted uids of passes/*.npz.
    std::vector<std::string> pass_uids() const;
    // A pass's (z, w), float64, from the LRU cache or the file.
    // std::invalid_argument "pass <uid>: z(<shape>) and w(<shape>) do not
    // match" when they are not one 1-D length.
    ZWPtr pass_zw(const std::string& uid);
    // The uids in the (z, w) cache, least recently used first.
    std::vector<std::string> zw_cached() const;
    // The cache itself, in that order (store.py's _zw), without touching it.
    std::vector<std::pair<std::string, ZWPtr>> zw_items() const;
    PassInfo pass_info(const std::string& uid);
    // Replace a stored pass and rebuild every accumulator holding it (its
    // member's em_round and mean_w_db updated); the keys rebuilt.
    std::vector<Key> update_pass(const types::PassResult& p);

    // --- accumulators ---
    std::pair<std::string, std::string> acc_paths(const Key& key) const;   // (json, npz)
    bool has_accumulator(const Key& key) const;
    // KeyError "no accumulator <key_name>" when there is none.
    Accumulator accumulator(const Key& key) const;
    // A new, empty accumulator (saved), or the existing one for `key`.
    Accumulator create(const Key& key, std::int64_t mode, std::int64_t codec_id);
    // The arrays first, then the JSON index, each atomically.
    void save_accumulator(const Accumulator& acc);
    // rebuild, updated = clock(), save.
    void commit(Accumulator& acc);
    // Every accumulator's key, in file-name order.
    std::vector<Key> open_keys() const;
    // The later passes of pictures partly received that are due in slot q
    // (store.py's `due`; an accumulator that cannot be read is skipped, the
    // rows it gave before failing kept).
    std::vector<Due> due(std::int64_t q);
    // The key of the accumulator holding pass `uid`.
    std::optional<Key> find_member(std::string_view uid) const;
    // Make pass `uid` a member of `key` as `segment`, and rebuild (see
    // store.py's attach: mode and codec ID for a new accumulator from the
    // arguments, else the pass's header, else (segment, 0); the pass moved
    // out of any other accumulator and the provisional clusters; a twin
    // replaced when this copy carries more, else this one left out).
    Accumulator attach(const Key& key, const std::string& uid, std::int64_t segment, const std::string& method,
                       std::optional<std::int64_t> mode = std::nullopt,
                       std::optional<std::int64_t> codec_id = std::nullopt);
    // The member of `acc` that is the transmission at (q, f_hz) of `segment`
    // received again, or nullopt (store.py's twin, waveform-aware).
    std::optional<pyjson::Value> twin(const Accumulator& acc, std::int64_t q, double f_hz, std::int64_t segment,
                                      std::optional<std::string_view> uid = std::nullopt,
                                      std::optional<std::int64_t> waveform = std::nullopt);
    Accumulator detach(const Key& key, std::string_view uid);
    void delete_accumulator(const Key& key);
    // Add a foreign (S, W) (PLANE values each) if it verifies; false when
    // it is not finite, has a negative weight, was merged already, or does
    // not correlate with the members. std::invalid_argument when S or W
    // does not hold PLANE values (numpy's reshape refusal).
    bool merge(const Key& key, std::span<const float> S, std::span<const float> W, const std::string& source = "");

    // --- provisional clusters ---
    std::string prov_path(std::string_view cid) const;
    // Every cluster's dict, in file-name order.
    std::vector<pyjson::Value> prov_clusters() const;
    std::optional<pyjson::Value> prov_cluster_of(std::string_view uid) const;
    // Put pass `uid` in cluster `cluster` (a new "c_<uid>" when nullopt);
    // a no-op when it is there already. Returns the cluster.
    std::string prov_add(const std::string& uid, std::optional<std::string> cluster = std::nullopt);
    void prov_remove(std::string_view uid);

    // --- housekeeping ---
    // Append {"time": clock(), **event} to log.jsonl (event: a dict).
    void log(const pyjson::Value& event);
    // Drop accumulators and clusters not updated for `days`, and orphan
    // passes whose slot is older than the cutoff.
    Expired expire(std::optional<double> now = std::nullopt, double days = 7.0);

private:
    void write_atomic(const std::string& path, std::span<const std::uint8_t> data, bool sync) const;
    void write_json(const std::string& path, const pyjson::Value& v) const;
    void forget(const std::string& uid);
    std::string dir(const char* name) const;

    std::string root_;
    Seams seams_;
    mutable std::recursive_mutex mu_;
    // (z, w) LRU: front is the least recently used.
    std::list<std::pair<std::string, ZWPtr>> zw_;
    std::unordered_map<std::string, std::list<std::pair<std::string, ZWPtr>>::iterator> zw_index_;
    std::unordered_map<std::string, PassInfo> light_;
};

// --- helpers shared with associate ---------------------------------------------------

// Python's int() / float() of a JSON value (see the header comment).
std::int64_t py_int(const pyjson::Value& v);
double py_float(const pyjson::Value& v);
// Python's truth value of a JSON value.
bool truthy(const pyjson::Value& v);
// Python's == between a JSON value and a str.
bool equals(const pyjson::Value& v, std::string_view s);
// `uid in v` for a JSON list (TypeError for anything else).
bool contains(const pyjson::Value& list, std::string_view s);

}  // namespace sstvae::qrss::store

#endif  // SSTVAE_QRSS_STORE_HPP
