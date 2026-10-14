// The live listener (sstvae/qrss/live.py's LiveListener and run()). See
// listener.hpp for what is the reference's and what is not.

#include "qrss/listener.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include "images/images.hpp"
#include "math/errors.hpp"
#include "qrss/picture.hpp"
#include "qrss/pyjson.hpp"
#include "util/fileio.hpp"

#if defined(_WIN32)
#include <process.h>
// Declared by hand rather than through <windows.h>, whose min and max
// macros break every std::min/std::max after it (CLAUDE.md).
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread(void);
extern "C" __declspec(dllimport) int __stdcall SetThreadPriority(void* thread, int priority);
#else
#include <sys/resource.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif
#endif

namespace sstvae::qrss::live {

Renderer::~Renderer() = default;

namespace {

namespace qcfg = sstvae::qrss::config;
using pyjson::Value;
using types::Detection;
using types::PassResult;

constexpr double FS = qcfg::FS;
constexpr double FE_FS_D = qcfg::FE_FS;
constexpr std::int64_t FE_FS_I = qcfg::FE_FS;
constexpr std::int64_t FE_DECIM = rxconfig::live::FE_DECIM;
constexpr double RESYNC_S = rxconfig::live::RESYNC_S;
constexpr double EARLY_WINDOW_S = rxconfig::live::EARLY_WINDOW_S;
constexpr double SILENCE_LEVEL = rxconfig::live::SILENCE_LEVEL;
constexpr double SILENCE_S = rxconfig::live::SILENCE_S;

// --- Python's arithmetic --------------------------------------------------------------------

constexpr double INT64_EDGE = 9223372036854775808.0;   // 2^63

// int(math.floor(x)): ValueError for NaN, OverflowError for an infinity
// (and here for anything past int64).
std::int64_t py_floor(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    const double f = std::floor(x);
    if (f < -INT64_EDGE || f >= INT64_EDGE) throw std::overflow_error("Python int too large to convert to C long");
    return static_cast<std::int64_t>(f);
}

// int(round(x)): half to even.
std::int64_t py_round_int(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    const double r = std::nearbyint(x);
    if (r < -INT64_EDGE || r >= INT64_EDGE) throw std::overflow_error("Python int too large to convert to C long");
    return static_cast<std::int64_t>(r);
}

// a // b for b > 0 (floor).
std::int64_t floor_div(std::int64_t a, std::int64_t b) {
    const std::int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// -(-a // b): ceil.
std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return -floor_div(-a, b); }

// round(x, 2): the correctly rounded two-decimal value, ties to even on the
// exact binary value (what Python's dtoa-based round does).
double py_round2(double x) {
    if (!std::isfinite(x)) return x;
    std::array<char, 400> buf{};
    const auto r = std::to_chars(buf.data(), buf.data() + buf.size(), x, std::chars_format::fixed, 2);
    if (r.ec != std::errc()) return x;
    double y = x;
    std::from_chars(buf.data(), r.ptr, y);
    return y;
}

// Python's max(a, b) and min(a, b) (the first unless the second compares past it).
double py_max(double a, double b) { return b > a ? b : a; }
double py_min(double a, double b) { return b < a ? b : a; }

std::string fmt(double v, const char* spec) { return frame::py_format(v, spec); }
std::string f1(double v) { return fmt(v, "%.1f"); }

// f"{v:08x}" (a negative value as Python writes it: "-0000005").
std::string hex08(std::int64_t v) {
    const bool neg = v < 0;
    const std::uint64_t m = neg ? (~static_cast<std::uint64_t>(v) + 1) : static_cast<std::uint64_t>(v);
    char buf[32];
    std::snprintf(buf, sizeof buf, "%llx", static_cast<unsigned long long>(m));
    std::string digits = buf;
    const std::size_t width = neg ? 7 : 8;
    if (digits.size() < width) digits.insert(0, width - digits.size(), '0');
    return neg ? "-" + digits : digits;
}

// "ABC"[i], Python's indexing (negative from the end); IndexError otherwise.
std::string abc(std::int64_t i) {
    static const char* const s = "ABC";
    if (i < 0) i += 3;
    if (i < 0 || i >= 3) throw std::out_of_range("string index out of range");
    return std::string(1, s[i]);
}

// WAVEFORM_NAMES[w] (KeyError otherwise).
std::string wf_name(std::int64_t w) {
    if (w == qcfg::WAVEFORM_CE) return "CE";
    if (w == qcfg::WAVEFORM_L) return "L";
    throw pyjson::KeyError(std::to_string(w), "KeyError: " + std::to_string(w));
}

std::string slot_utc(std::int64_t q) { return utc_iso_int(slot_t0(q) - 1); }

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    std::string out;
    std::size_t pos = 0;
    for (;;) {
        const std::size_t k = s.find(from, pos);
        if (k == std::string::npos) break;
        out.append(s, pos, k - pos);
        out += to;
        pos = k + from.size();
    }
    out.append(s, pos, std::string::npos);
    return out;
}

// What a text-mode write makes of "\n" (Python's open(..., "w"/"a") and
// write_text: "\r\n" on Windows).
std::string text_mode(const std::string& s) {
#if defined(_WIN32)
    std::string out;
    out.reserve(s.size() + s.size() / 16);
    for (const char c : s) {
        if (c == '\n') out += '\r';
        out += c;
    }
    return out;
#else
    return s;
#endif
}

long process_id() {
#if defined(_WIN32)
    return static_cast<long>(::_getpid());
#else
    return static_cast<long>(::getpid());
#endif
}

// Write `bytes` to `path` (created or truncated); std::system_error on failure.
void write_file(const std::string& path, const std::string& bytes) {
    util::File f = util::open_file(path, "wb");
    if (!f) throw std::system_error(errno, std::generic_category(), "cannot write " + path);
    if (!bytes.empty() && std::fwrite(bytes.data(), 1, bytes.size(), f.get()) != bytes.size())
        throw std::system_error(errno, std::generic_category(), "cannot write " + path);
    std::FILE* raw = f.release();
    if (std::fclose(raw) != 0) throw std::system_error(errno, std::generic_category(), "cannot write " + path);
}

void replace_or_throw(const std::string& from, const std::string& to) {
    std::error_code ec;
    if (!util::replace_file(from, to, ec)) throw std::system_error(ec, "cannot replace " + to);
}

// The JSON value of a float field: null when not finite (strict JSON).
Value num(double v) { return std::isfinite(v) ? Value::number(v) : Value::null(); }
Value opt_str(const std::optional<std::string>& s) { return s ? Value::string(*s) : Value::null(); }

// bool(v) of a JSON value, as Python's truthiness.
bool truthy(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Null: return false;
        case Value::Kind::Bool: return v.as_bool();
        case Value::Kind::Int: return v.as_int() != 0;
        case Value::Kind::Float: return v.as_double() != 0.0;   // NaN is true
        case Value::Kind::String: return !v.as_string().empty();
        case Value::Kind::Array: return !v.items().empty();
        case Value::Kind::Object: return !v.keys().empty();
    }
    return false;
}

// int(v) of a JSON value; std::invalid_argument / std::overflow_error /
// math::TypeError where int() raises.
std::int64_t py_int(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Int: return v.as_int();
        case Value::Kind::Bool: return v.as_bool() ? 1 : 0;
        case Value::Kind::Float: {
            const double f = v.as_double();
            if (std::isnan(f)) throw std::invalid_argument("cannot convert float NaN to integer");
            if (std::isinf(f)) throw std::overflow_error("cannot convert float infinity to integer");
            const double t = std::trunc(f);
            if (t < -INT64_EDGE || t >= INT64_EDGE) throw std::overflow_error("int too large");
            return static_cast<std::int64_t>(t);
        }
        case Value::Kind::String: {
            std::string s = v.as_string();
            const auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
            while (!s.empty() && ws(s.front())) s.erase(s.begin());
            while (!s.empty() && ws(s.back())) s.pop_back();
            std::string digits;
            std::size_t k = 0;
            bool neg = false;
            if (k < s.size() && (s[k] == '+' || s[k] == '-')) neg = s[k++] == '-';
            bool last_us = true;
            for (; k < s.size(); ++k) {
                if (s[k] == '_' && !last_us) {
                    last_us = true;
                    continue;
                }
                if (s[k] < '0' || s[k] > '9') throw std::invalid_argument("invalid literal for int(): '" + s + "'");
                digits += s[k];
                last_us = false;
            }
            if (digits.empty() || last_us) throw std::invalid_argument("invalid literal for int(): '" + s + "'");
            std::int64_t out = 0;
            const auto r = std::from_chars(digits.data(), digits.data() + digits.size(), out);
            if (r.ec != std::errc()) throw std::overflow_error("int too large");
            return neg ? -out : out;
        }
        default:
            throw math::TypeError("int() argument must be a string, a bytes-like object or a real number");
    }
}

// str(v) as an f-string shows a JSON value.
std::string py_str(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Null: return "None";
        case Value::Kind::Bool: return v.as_bool() ? "True" : "False";
        case Value::Kind::String: return v.as_string();
        default: return pyjson::dumps(v);
    }
}

bool hint_eq(const acquire::Hint& a, const acquire::Hint& b) { return a.f_hz == b.f_hz && a.waveform == b.waveform; }

bool contains(std::span<const acquire::Hint> hs, const acquire::Hint& h) {
    return std::any_of(hs.begin(), hs.end(), [&](const acquire::Hint& g) { return hint_eq(g, h); });
}

// set(a) <= set(b), set(a) == set(b)
bool subset(std::span<const acquire::Hint> a, std::span<const acquire::Hint> b) {
    return std::all_of(a.begin(), a.end(), [&](const acquire::Hint& h) { return contains(b, h); });
}
bool set_equal(std::span<const acquire::Hint> a, std::span<const acquire::Hint> b) { return subset(a, b) && subset(b, a); }

// Python's sorted(xs, key=...) for a key compared with `less`: stable, and
// well defined whatever `less` does (a merge sort that takes from the right
// only when less(right, left)).
template <typename T, typename Less>
void py_sort(std::vector<T>& xs, Less less) {
    const std::size_t n = xs.size();
    if (n < 2) return;
    std::vector<T> tmp;
    tmp.reserve(n);
    for (std::size_t width = 1; width < n; width *= 2) {
        tmp.clear();
        for (std::size_t lo = 0; lo < n; lo += 2 * width) {
            const std::size_t mid = std::min(lo + width, n), hi = std::min(lo + 2 * width, n);
            std::size_t i = lo, j = mid;
            while (i < mid && j < hi) {
                if (less(xs[j], xs[i]))
                    tmp.push_back(std::move(xs[j++]));
                else
                    tmp.push_back(std::move(xs[i++]));
            }
            while (i < mid) tmp.push_back(std::move(xs[i++]));
            while (j < hi) tmp.push_back(std::move(xs[j++]));
        }
        xs.swap(tmp);
    }
}

// Runs [lo, hi) of true in a mask.
std::vector<std::pair<std::int64_t, std::int64_t>> runs_of(std::span<const std::uint8_t> m) {
    std::vector<std::pair<std::int64_t, std::int64_t>> out;
    const std::int64_t n = static_cast<std::int64_t>(m.size());
    std::int64_t k = 0;
    while (k < n) {
        if (!m[k]) {
            ++k;
            continue;
        }
        const std::int64_t lo = k;
        while (k < n && m[k]) ++k;
        out.emplace_back(lo, k);
    }
    return out;
}

// np.mean(mask) (NaN for an empty one).
double mask_mean(std::span<const std::uint8_t> m) {
    if (m.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::size_t c = 0;
    for (const std::uint8_t b : m) c += b ? 1 : 0;
    return static_cast<double>(c) / static_cast<double>(m.size());
}

double system_time() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

void lower_priority() {
#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), -1);   // THREAD_PRIORITY_BELOW_NORMAL
#elif defined(__linux__)
    const auto tid = static_cast<id_t>(::syscall(SYS_gettid));
    errno = 0;
    int cur = ::getpriority(PRIO_PROCESS, tid);
    if (errno != 0) cur = 0;
    ::setpriority(PRIO_PROCESS, tid, std::min(cur + 10, 19));
#elif defined(__APPLE__)
    ::pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
}

}  // namespace

// --- errors ---------------------------------------------------------------------------------

std::string error_name(const std::exception& e) {
    if (const auto* s = dynamic_cast<const SeamError*>(&e)) return s->type_name();
    if (dynamic_cast<const pyjson::KeyError*>(&e)) return "KeyError";
    if (dynamic_cast<const math::TypeError*>(&e)) return "TypeError";
    if (dynamic_cast<const std::invalid_argument*>(&e)) return "ValueError";
    if (dynamic_cast<const std::domain_error*>(&e)) return "ValueError";
    if (dynamic_cast<const std::length_error*>(&e)) return "ValueError";
    if (dynamic_cast<const std::out_of_range*>(&e)) return "IndexError";
    if (dynamic_cast<const std::overflow_error*>(&e)) return "OverflowError";
    if (dynamic_cast<const std::system_error*>(&e)) return "OSError";
    if (dynamic_cast<const npz::FileError*>(&e)) return "OSError";
    if (dynamic_cast<const std::bad_alloc*>(&e)) return "MemoryError";
    return "RuntimeError";
}

std::string error_text(const std::exception& e) {
    std::string what = e.what();
    // pyjson's KeyError says "KeyError: '<key>'" itself
    if (dynamic_cast<const pyjson::KeyError*>(&e) && what.rfind("KeyError: ", 0) == 0) what.erase(0, 10);
    return error_name(e) + ": " + what;
}

// --- the listener's records -------------------------------------------------------------------

struct LiveListener::Rec {
    explicit Rec(std::int64_t q) { s.q = q; }
    SlotState s;
    // Its end-of-slot receive is being applied: the reference's s.done,
    // which it sets as that starts (under a lock it then holds to the end;
    // this one is released for the store and the pictures, so `done` waits
    // until the slot's tiles are final and step() may retire it).
    bool applying = false;
};

struct LiveListener::RenderJob {
    std::string tile_id;
    std::uint64_t gen = 0;
    std::shared_ptr<const PassResult> p;   // what _render reads of the pass
    std::optional<store::Key> key;
};

// --- construction -----------------------------------------------------------------------------

LiveListener::LiveListener(std::string state_dir, LiveConfig cfg, std::shared_ptr<store::Store> store,
                           std::shared_ptr<passband::PassbandStore> passband, Seams seams)
    : dir_(std::move(state_dir)),
      cfg_(std::move(cfg)),
      specs_(cfg_.specs()),
      sel_(cfg_.selection()),
      joint_(receiver::is_joint(sel_)),
      dur_(frame_seconds(specs_)),
      store_(std::move(store)),
      passband_(std::move(passband)),
      seams_(std::move(seams)),
      ring_(dur_ + PB_LEAD_S + PB_TAIL_S + 300.0),
      dial_(dir_ + "/" + std::string(DIAL_FILE)) {
    const std::optional<std::filesystem::path> tiles = util::fs_path(dir_ + "/tiles");
    if (!tiles) throw std::invalid_argument("embedded null byte");
    std::filesystem::create_directories(*tiles);

    rx_.frames = sel_;
    rx_.estimator = cfg_.estimator;
    rx_.old_edge_noise = seams_.old_edge_noise;
    rx_.old_gating = seams_.old_gating;
    finish_rx_ = finish_config(sel_, cfg_.estimator);
    finish_rx_.old_edge_noise = seams_.old_edge_noise;
    finish_rx_.old_gating = seams_.old_gating;

    if (!seams_.clock) seams_.clock = system_time;
    if (!seams_.log)
        seams_.log = [](const std::string& line) {
            std::fputs((line + "\n").c_str(), stderr);
            std::fflush(stderr);
        };
    const api::ReceiveConfig rx = rx_;
    if (!seams_.candidates)
        seams_.candidates = [rx](const receiver::Prepared& prep, Selection sel, std::span<const acquire::Hint> hints) {
            return receiver::candidates(prep, sel, true, hints, api::slot_seams(rx));
        };
    if (!seams_.detect)
        seams_.detect = [rx](const receiver::Prepared& prep, Selection sel, const receiver::DetectOptions& opt) {
            return receiver::detect(prep, sel, opt, api::slot_seams(rx));
        };
    if (!seams_.receive_slot)
        seams_.receive_slot = [rx](const receiver::Prepared& prep, Selection sel, const receiver::SlotOptions& opt) {
            return receiver::receive_slot(receiver::CaptureRef(&prep), sel, opt, api::slot_seams(rx));
        };
    if (!seams_.finish) {
        const NoiseFn noise = seams_.noise;
        seams_.finish = [noise](std::span<const cf> x, std::span<const std::uint8_t> ok, std::int64_t origin,
                                std::int64_t q, const api::ReceiveConfig& config,
                                std::span<const Detection> dets, std::span<const acquire::Hint> hints) {
            FinishOptions opt;
            opt.seams.noise = noise;
            return finish_receive(x, ok, origin, q, config, dets, hints, opt);
        };
    }
    if (!seams_.associate)
        seams_.associate = [](const PassResult& p, store::Store& st) { return associate::associate(p, st); };
    if (!cfg_.render) seams_.renderer.reset();

    load_previous();
    if (cfg_.background) {
        live_thread_ = std::thread([this] { worker(false); });
        finish_thread_ = std::thread([this] { worker(true); });
    }
}

LiveListener::~LiveListener() {
    try {
        close();
    } catch (...) {
    }
}

void LiveListener::close() {
    std::lock_guard<std::mutex> c(close_mu_);
    {
        Lock lk(mu_);
        if (closed_) return;
        closed_ = true;
        stopping_ = true;
    }
    work_cv_.notify_all();
    idle_cv_.notify_all();
    if (live_thread_.joinable()) live_thread_.join();
    if (finish_thread_.joinable()) finish_thread_.join();
    deliver();
    if (passband_) {
        try {
            passband_->join_packer(CLOSE_PACKER_S);
        } catch (...) {
        }
    }
}

// --- logging ----------------------------------------------------------------------------------

double LiveListener::clock_locked() const { return seams_.clock(); }

double LiveListener::clock() const {
    Lock lk(mu_);
    return clock_locked();
}

void LiveListener::log_locked(const std::string& msg) {
    const std::string line = utc_iso(now_ != 0.0 ? now_ : clock_locked()) + " " + msg;
    lines_.push_back(line);
    if (lines_.size() > 50) lines_.erase(lines_.begin(), lines_.end() - 50);
    sink_queue_.push_back(line);
    // and to a file (a log that cannot be written must not stop the listener)
    try {
        const std::string path = dir_ + "/" + std::string(LOG_FILE);
        if (const auto p = util::fs_path(path)) {
            std::error_code ec;
            if (std::filesystem::exists(*p, ec)) {
                const std::uintmax_t size = std::filesystem::file_size(*p, ec);
                if (!ec && size > static_cast<std::uintmax_t>(LOG_FILE_BYTES)) util::replace_file(path, path + ".1", ec);
            }
        }
        util::File f = util::open_file(path, "ab");
        if (f) {
            const std::string out = text_mode(line + "\n");
            std::fwrite(out.data(), 1, out.size(), f.get());
        }
    } catch (...) {
    }
}

void LiveListener::log(const std::string& msg) {
    {
        Lock lk(mu_);
        log_locked(msg);
    }
    deliver();
}

void LiveListener::deliver() {
    for (;;) {
        {
            std::unique_lock<std::mutex> s(sink_mu_, std::try_to_lock);
            if (!s.owns_lock()) return;   // whoever holds it hands ours over too
            std::vector<std::string> lines;
            {
                Lock lk(mu_);
                lines.swap(sink_queue_);
            }
            for (const std::string& line : lines) {
                try {
                    seams_.log(line);
                } catch (...) {
                }
            }
        }
        Lock lk(mu_);
        if (sink_queue_.empty()) return;
    }
}

// --- the previous run's tiles ---------------------------------------------------------------

void LiveListener::load_previous() {
    std::string text;
    {
        util::File f = util::open_file(dir_ + "/state.json", "rb");
        if (!f) return;
        char buf[65536];
        for (;;) {
            const std::size_t got = std::fread(buf, 1, sizeof buf, f.get());
            text.append(buf, got);
            if (got < sizeof buf) {
                if (std::ferror(f.get())) return;
                break;
            }
        }
    }
    Value st;
    try {
        st = pyjson::loads(text);
    } catch (const std::exception&) {
        return;   // json's ValueError: not read
    }
    // Anything but the reference's structure starts the listener fresh
    // (the reference raised from its constructor).
    std::set<std::int64_t> fin;
    std::vector<Tile> done;
    try {
        if (st.kind() != Value::Kind::Object) return;
        if (const Value* f = st.find("finished")) {
            if (f->kind() != Value::Kind::Array) return;
            for (const Value& q : f->items()) fin.insert(py_int(q));
        }
        if (const Value* ts = st.find("tiles")) {
            if (ts->kind() != Value::Kind::Array) return;
            for (const Value& t : ts->items()) {
                if (t.kind() != Value::Kind::Object) return;
                const Value* status = t.find("status");
                const bool is_str = status && status->kind() == Value::Kind::String;
                const std::string s = is_str ? status->as_string() : std::string();
                if (is_str && (s == "complete" || s == "lost"))
                    if (const Value* q = t.find("q")) fin.insert(py_int(*q));   // (a state file from before "finished")
                if (is_str && (s == "receiving" || s == "finishing")) continue;
                const Value* call = t.find("callsign");
                const Value* cw = t.find("cw_text");
                if (!(call && truthy(*call)) && !(cw && truthy(*cw))) {
                    const Value* snr = t.find("snr_db");
                    double v = -99.0;
                    if (snr && truthy(*snr)) {
                        switch (snr->kind()) {
                            case Value::Kind::Int: v = static_cast<double>(snr->as_int()); break;
                            case Value::Kind::Float: v = snr->as_double(); break;
                            case Value::Kind::Bool: v = 1.0; break;
                            default: return;   // TypeError comparing it
                        }
                    }
                    if (v > HEADERLESS_MAX_SNR_DB) continue;   // kept by an older listener; see plausible
                }
                if (std::optional<Tile> tile = Tile::from_json(t)) done.push_back(std::move(*tile));
            }
        }
    } catch (const std::exception&) {
        return;
    }
    Lock lk(mu_);
    finished_.insert(fin.begin(), fin.end());
    for (Tile& t : done) done_tiles_.push_back(std::move(t));
}

// --- state --------------------------------------------------------------------------------------

std::optional<std::string> LiveListener::busy() const {
    Lock lk(mu_);
    std::string out;
    for (const auto* b : {&busy_live_, &busy_finish_}) {
        if (!*b || (*b)->empty()) continue;
        if (!out.empty()) out += "; ";
        out += **b;
    }
    if (out.empty()) return std::nullopt;
    return out;
}

double LiveListener::now() const {
    Lock lk(mu_);
    return now_;
}

std::optional<double> LiveListener::audio_end() const {
    Lock lk(mu_);
    return audio_end_;
}

std::optional<std::int64_t> LiveListener::next_index() const {
    Lock lk(mu_);
    return fe_.next_index();
}

std::string LiveListener::source() const {
    Lock lk(mu_);
    return source_;
}

void LiveListener::set_source(std::string name) {
    Lock lk(mu_);
    source_ = std::move(name);
}

std::vector<std::string> LiveListener::lines() const {
    Lock lk(mu_);
    return lines_;
}

std::vector<std::int64_t> LiveListener::finished() const {
    Lock lk(mu_);
    return {finished_.begin(), finished_.end()};
}

std::vector<SlotState> LiveListener::slots() const {
    Lock lk(mu_);
    std::vector<SlotState> out;
    for (const auto& r : slots_) out.push_back(r->s);
    py_sort(out, [](const SlotState& a, const SlotState& b) { return a.q < b.q; });
    return out;
}

std::vector<Tile> LiveListener::done_tiles() const {
    Lock lk(mu_);
    return done_tiles_;
}

std::optional<std::string> LiveListener::codec_error() const {
    Lock lk(mu_);
    return codec_error_;
}

std::vector<std::int64_t> LiveListener::recovering() const {
    Lock lk(mu_);
    std::vector<std::int64_t> out;
    for (const auto& r : recover_) out.push_back(r->s.q);
    return out;
}

std::vector<Tile> LiveListener::tiles_locked() const {
    std::vector<Tile> out;
    for (const auto& r : slots_)
        for (const auto& [id, t] : r->s.tiles) out.push_back(t);
    out.insert(out.end(), done_tiles_.begin(), done_tiles_.end());
    py_sort(out, [](const Tile& a, const Tile& b) { return a.q > b.q || (a.q == b.q && a.f_hz < b.f_hz); });
    return out;
}

std::vector<Tile> LiveListener::tiles() const {
    Lock lk(mu_);
    return tiles_locked();
}

void LiveListener::write_state(bool listening) {
    std::lock_guard<std::mutex> w(state_mu_);
    std::string text;
    {
        Lock lk(mu_);
        Value st = Value::object();
        st.set("version", Value::integer(1));
        st.set("updated", num(clock_locked()));
        st.set("now", num(now_));
        st.set("now_utc", Value::string(utc_iso(now_ != 0.0 ? now_ : clock_locked())));
        st.set("listening", Value::boolean(listening));
        st.set("source", Value::string(source_));
        st.set("frame", Value::string(cfg_.spec.name()));
        Value wfs = Value::array();
        for (const frame::FrameSpec& s : specs_) wfs.push_back(Value::string(wf_name(static_cast<int>(s.waveform()))));
        st.set("waveforms", std::move(wfs));
        st.set("audio_utc", audio_end_ && *audio_end_ != 0.0 ? Value::string(utc_iso(*audio_end_)) : Value::null());
        std::vector<std::shared_ptr<Rec>> by_q = slots_;
        py_sort(by_q, [](const std::shared_ptr<Rec>& a, const std::shared_ptr<Rec>& b) { return a->s.q < b->s.q; });
        Value slots = Value::array();
        for (const auto& r : by_q) {
            Value o = Value::object();
            o.set("q", Value::integer(r->s.q));
            o.set("slot_utc", Value::string(slot_utc(r->s.q)));
            o.set("progress", num(progress_locked(r->s.q)));
            o.set("signals", Value::integer(static_cast<std::int64_t>(r->s.tiles.size())));
            slots.push_back(std::move(o));
        }
        st.set("slots", std::move(slots));
        st.set("codec_error", opt_str(codec_error_));
        std::string b;
        for (const auto* x : {&busy_live_, &busy_finish_}) {
            if (!*x || (*x)->empty()) continue;
            if (!b.empty()) b += "; ";
            b += **x;
        }
        st.set("busy", b.empty() ? Value::null() : Value::string(b));
        Value tiles = Value::array();
        for (const Tile& t : tiles_locked()) tiles.push_back(t.to_json());
        st.set("tiles", std::move(tiles));
        Value lg = Value::array();
        const std::size_t from = lines_.size() > 20 ? lines_.size() - 20 : 0;
        for (std::size_t k = from; k < lines_.size(); ++k) lg.push_back(Value::string(lines_[k]));
        st.set("log", std::move(lg));
        const double cut = (now_ != 0.0 ? now_ : clock_locked()) - 48 * 3600;
        Value fin = Value::array();
        for (const std::int64_t q : finished_)
            if (static_cast<double>(slot_t0(q)) > cut) fin.push_back(Value::integer(q));
        st.set("finished", std::move(fin));
        text = pyjson::dumps(st, 1, false);
    }
    const std::string tmp = dir_ + "/state.json." + std::to_string(process_id()) + ".tmp";
    write_file(tmp, text_mode(text));
    replace_or_throw(tmp, dir_ + "/state.json");
}

// --- audio ----------------------------------------------------------------------------------------

void LiveListener::feed(std::span<const double> x, double t_end) {
    if (x.empty()) return;
    {
        Lock lk(mu_);
        const std::int64_t n = static_cast<std::int64_t>(x.size());
        std::int64_t a = py_round_int(t_end * FS) - n + 1;
        if (const std::optional<std::int64_t> nxt = fe_.next_index()) {
            const double d = static_cast<double>(a - *nxt) / FS;   // > 0: later than the samples say
            if (d > RESYNC_S) {
                log_locked(fmt(d, "%.1f") + " s of audio missing; the stream restarts there");
                early_.clear();
            } else {
                // Audio only ever arrives late, never early: the stream
                // follows on unless every chunk for EARLY_WINDOW_S came early.
                early_.emplace_back(t_end, d);
                while (!early_.empty() && t_end - early_.front().first > EARLY_WINDOW_S) early_.pop_front();
                bool restart = false;
                if (!early_.empty() && t_end - early_.front().first >= 0.9 * EARLY_WINDOW_S) {
                    double mx = early_.front().second;
                    for (const auto& e : early_) mx = py_max(mx, e.second);
                    restart = mx < -RESYNC_S;
                }
                if (restart) {
                    log_locked("audio running " + fmt(-d, "%.1f") + " s ahead of the clock; "
                               "the stream restarts at the clock");
                    early_.clear();
                } else {
                    a = *nxt;
                }
            }
        }
        note_silence(a, x);
        for (const frontend::StreamFE::Chunk& c : fe_.push(a, x)) {
            for (const auto& [lo, hi] : heard_runs(c.index, static_cast<std::int64_t>(c.fe.size()))) {
                const std::span<const cf> part(c.fe.data() + lo, static_cast<std::size_t>(hi - lo));
                ring_.write(c.index + lo, part);
                if (passband_) passband_->write(static_cast<double>(c.index + lo) / FE_FS_D, part);
            }
        }
        audio_end_ = static_cast<double>(a + n - 1) / FS;
    }
    deliver();
}

void LiveListener::note_silence(std::int64_t a, std::span<const double> x) {
    const std::int64_t n_min = py_round_int(SILENCE_S * FS);
    std::optional<std::pair<std::int64_t, std::int64_t>> run;
    if (quiet_ && quiet_->second == a) run = quiet_;
    quiet_.reset();
    std::vector<std::uint8_t> quiet(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) quiet[k] = std::fabs(x[k]) < SILENCE_LEVEL ? 1 : 0;
    const auto runs = runs_of(quiet);
    if (silent_from_ && (runs.empty() || runs.front().first != 0)) sound_back(a);   // the silence ended where this chunk began
    const std::int64_t n = static_cast<std::int64_t>(x.size());
    for (const auto& [s, e] : runs) {
        const std::int64_t lo = (s == 0 && run) ? run->first : a + s;
        const std::int64_t hi = a + e;
        if (hi - lo >= n_min) {
            if (!silent_.empty() && lo <= silent_.back().second)
                silent_.back().second = std::max(silent_.back().second, hi);
            else
                silent_.emplace_back(lo, hi);
            if (!silent_from_) {
                silent_from_ = lo;
                log_locked("the audio is digital silence from " + utc_iso(static_cast<double>(lo) / FS) +
                           ": not counted as heard");
            }
        }
        if (e == n)
            quiet_ = std::make_pair(lo, hi);   // it may go on in the next chunk
        else if (silent_from_)
            sound_back(hi);
    }
}

void LiveListener::sound_back(std::int64_t a) {
    log_locked("audio back after " + fmt(static_cast<double>(a - *silent_from_) / FS, "%.0f") +
               " s of digital silence");
    silent_from_.reset();
}

std::vector<std::pair<std::int64_t, std::int64_t>> LiveListener::heard_runs(std::int64_t n0, std::int64_t n) {
    std::vector<std::uint8_t> ok(static_cast<std::size_t>(n), 1);
    std::vector<std::pair<std::int64_t, std::int64_t>> keep;
    for (const auto& [lo, hi] : silent_) {
        const std::int64_t ka = ceil_div(lo, FE_DECIM) - n0;
        const std::int64_t kb = ceil_div(hi, FE_DECIM) - n0;
        if (kb > -10 * FE_FS_I) keep.emplace_back(lo, hi);   // (kept a while: the stream can step back)
        const std::int64_t s = std::max<std::int64_t>(ka, 0), e = std::max<std::int64_t>(std::min(kb, n), 0);
        for (std::int64_t k = s; k < e; ++k) ok[static_cast<std::size_t>(k)] = 0;
    }
    silent_ = std::move(keep);
    return runs_of(ok);
}

double LiveListener::backfill(double now) {
    if (!passband_) return 0.0;
    const std::int64_t a = py_round_int(now * FE_FS_D) - ring_.n();
    const double t_a = static_cast<double>(a) / FE_FS_D;
    const passband::Masked m = passband_->read(t_a, t_a + static_cast<double>(ring_.n()) / FE_FS_D);
    const auto runs = runs_of(m.mask);
    if (runs.empty()) return 0.0;
    std::size_t count = 0;
    for (const std::uint8_t b : m.mask) count += b ? 1 : 0;
    for (const auto& [lo, hi] : runs)
        ring_.write(a + lo, std::span<const cf>(m.x.data() + lo, static_cast<std::size_t>(hi - lo)));
    const double got = static_cast<double>(count) / FE_FS_D;
    log(fmt(got, "%.0f") + " s of earlier audio read back from the passband store");
    return got;
}

std::vector<std::int64_t> LiveListener::recover(double now, double hours) {
    if (!passband_ || hours <= 0) return {};
    const std::shared_ptr<passband::PassbandStore> pb = passband_;
    const FeReader reader = [pb](std::int64_t a, std::int64_t b) {
        passband::Masked m = pb->read(static_cast<double>(a) / FE_FS_D, static_cast<double>(b) / FE_FS_D);
        return FeSpan{std::move(m.x), std::move(m.mask)};
    };
    // A slot whose passes the store already holds was finished by the earlier
    // listener, which stopped before it recorded that: receiving it again
    // would file a second copy of each pass.
    std::set<std::int64_t> stored;
    if (store_) {
        for (const std::string& u : store_->pass_uids()) {   // pass_uid: "<q>_<mHz>_<sha8>"
            const std::string head = u.substr(0, u.find('_'));
            std::size_t k = 0;
            while (k < head.size() && head[k] == '-') ++k;
            const std::string_view digits(head.data() + k, head.size() - k);
            if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
                continue;
            if (k > 1) continue;   // ("--5": the reference's int() would raise; never a pass_uid)
            std::int64_t v = 0;
            if (std::from_chars(digits.data(), digits.data() + digits.size(), v).ec != std::errc()) continue;
            stored.insert(k ? -v : v);
        }
    }
    std::vector<std::int64_t> out;
    const std::int64_t q_hi = py_floor(now / 900.0);
    const std::int64_t q_lo = py_floor((now - hours * 3600.0 - dur_) / 900.0) - 1;
    for (std::int64_t q = q_hi; q > q_lo; --q) {
        const double end = static_cast<double>(slot_t0(q)) + dur_ + PB_TAIL_S;
        const std::int64_t t0 = slot_t0(q);
        {
            Lock lk(mu_);
            const bool queued = std::any_of(recover_.begin(), recover_.end(),
                                            [&](const std::shared_ptr<Rec>& r) { return r->s.q == q; });
            if (!(now - hours * 3600.0 <= end && end <= now) || finished_.count(q) || slot_locked(q) || queued)
                continue;
            if (stored.count(q)) {
                finished_.insert(q);
                log_locked("slot " + slot_utc(q) + ": its passes are in the store already; not received again");
                continue;
            }
        }
        const FeSpan sp = reader(t0 * FE_FS_I, py_round_int((static_cast<double>(t0) + dur_) * FE_FS_D));
        if (mask_mean(sp.ok) < FINISH_SEARCH_MIN_HEARD) continue;   // not heard (enough) by an earlier listener
        Lock lk(mu_);
        auto r = std::make_shared<Rec>(q);
        r->s.finishing = true;
        r->s.reader = reader;
        recover_.push_back(r);
        out.push_back(q);
        log_locked("slot " + slot_utc(q) + ": ended while no listener finished it; receiving it from the passband store");
    }
    deliver();
    return out;
}

// --- slots ------------------------------------------------------------------------------------

std::shared_ptr<LiveListener::Rec> LiveListener::slot_locked(std::int64_t q) const {
    for (const auto& r : slots_)
        if (r->s.q == q) return r;
    return nullptr;
}

double LiveListener::progress_locked(std::int64_t q) const {
    return py_min(py_max((now_ - static_cast<double>(slot_t0(q))) / dur_, 0.0), 1.0);
}

double LiveListener::progress(std::int64_t q) const {
    Lock lk(mu_);
    return progress_locked(q);
}

std::vector<std::int64_t> LiveListener::active_slots_locked(double now) const {
    const std::int64_t lo = py_floor((now - dur_ - PB_TAIL_S - 1.0) / 900.0);
    const std::int64_t hi = py_floor((now - 1.0) / 900.0);
    std::vector<std::int64_t> out;
    for (std::int64_t q = lo; q <= hi; ++q) {
        const double t0 = static_cast<double>(slot_t0(q));
        if (t0 <= now && now < t0 + dur_ + PB_TAIL_S) out.push_back(q);
    }
    return out;
}

std::vector<std::int64_t> LiveListener::active_slots(double now) const { return active_slots_locked(now); }

double LiveListener::interval_locked(const Rec& s, double now) const {
    if (s.s.dets.empty() && now - static_cast<double>(slot_t0(s.s.q)) >= cfg_.detect_until_s)
        return py_max(cfg_.refresh_s, cfg_.empty_refresh_s);
    return cfg_.refresh_s;
}

double LiveListener::interval(std::int64_t q, double now) const {
    Lock lk(mu_);
    const std::shared_ptr<Rec> r = slot_locked(q);
    if (r) return interval_locked(*r, now);
    return interval_locked(Rec(q), now);
}

bool LiveListener::step(double now) {
    bool changed = false;
    {
        Lock lk(mu_);
        now_ = now;
        for (const std::int64_t q : active_slots_locked(now)) {
            if (!slot_locked(q) && !finished_.count(q)) {
                // a slot whose start we never heard is still received live
                slots_.push_back(std::make_shared<Rec>(q));
                log_locked("slot " + slot_utc(q) + ": frame started");
                changed = true;
            }
        }
        std::vector<std::shared_ptr<Rec>> snap = slots_;
        py_sort(snap, [](const std::shared_ptr<Rec>& a, const std::shared_ptr<Rec>& b) { return a->s.q < b->s.q; });
        for (const std::shared_ptr<Rec>& r : snap) {
            SlotState& s = r->s;
            const double t0 = static_cast<double>(slot_t0(s.q));
            if (s.done) continue;
            if (now >= t0 + dur_ + PB_TAIL_S) {
                if (!s.finishing) {
                    s.finishing = true;
                    for (auto& [id, t] : s.tiles) t.status = "finishing";
                    // One more live receive, of the whole frame, shows the
                    // pass to its end while the end-of-slot receive runs.
                    if (cfg_.background && !s.dets.empty() && (!s.last_refresh || *s.last_refresh < t0 + dur_)) {
                        s.last_refresh = now;
                        submit(lk, false, r);
                    }
                    submit(lk, true, r);
                    changed = true;
                }
            } else if (now >= t0 + cfg_.first_s &&
                       (!s.last_refresh || now - *s.last_refresh >= interval_locked(*r, now))) {
                if (!cfg_.background || (!busy_live_ && live_jobs_.empty())) {
                    s.last_refresh = now;   // (when busy, it is still due next time)
                    submit(lk, false, r);
                    changed = true;
                }
            }
        }
        if (!recover_.empty() && finish_idle_locked()) {
            const std::shared_ptr<Rec> r = recover_.front();
            recover_.erase(recover_.begin());
            bool placed = false;
            for (auto& x : slots_)
                if (x->s.q == r->s.q) {
                    x = r;
                    placed = true;
                }
            if (!placed) slots_.push_back(r);
            submit(lk, true, r);
            changed = true;
        }
        std::vector<std::shared_ptr<Rec>> keep_slots;
        for (const auto& r : slots_) {
            if (!r->s.done) {
                keep_slots.push_back(r);
                continue;
            }
            for (const auto& [id, t] : r->s.tiles) headers_.erase(id);
            for (const auto& [id, t] : r->s.tiles) done_tiles_.push_back(t);
            finished_.insert(r->s.q);
        }
        slots_ = std::move(keep_slots);
        std::vector<Tile> keep;
        for (Tile& t : done_tiles_)
            if (now - t.updated < cfg_.keep_done_s) keep.push_back(std::move(t));
        changed = changed || keep.size() != done_tiles_.size();
        done_tiles_ = std::move(keep);
    }
    deliver();
    return changed;
}

// --- the workers --------------------------------------------------------------------------------

void LiveListener::submit(Lock& lk, bool finish, const std::shared_ptr<Rec>& s) {
    if (!cfg_.background) {
        lk.unlock();
        run_job(Job{finish, s});
        lk.lock();
        return;
    }
    if (stopping_) return;   // closed: nothing more is received
    (finish ? finish_jobs_ : live_jobs_).push_back(Job{finish, s});
    work_cv_.notify_all();
}

void LiveListener::worker(bool finish) {
    if (seams_.low_priority) lower_priority();
    std::deque<Job>& jobs = finish ? finish_jobs_ : live_jobs_;
    std::optional<std::string>& busy = finish ? busy_finish_ : busy_live_;
    Lock lk(mu_);
    for (;;) {
        work_cv_.wait(lk, [&] { return stopping_ || !jobs.empty(); });
        if (stopping_) break;
        const Job j = jobs.front();
        jobs.pop_front();
        // taken and busy in one step, so idle() never reads true in between
        try {
            busy = std::string(finish ? "finish " : "refresh ") + slot_utc(j.s->s.q);
        } catch (const std::exception&) {
            busy = finish ? "finish" : "refresh";
        }
        lk.unlock();
        run_job(j);
        lk.lock();
        busy.reset();
        idle_cv_.notify_all();
    }
    idle_cv_.notify_all();
}

void LiveListener::run_job(const Job& j) {
    try {
        if (j.finish)
            finish(j.s);
        else
            refresh(j.s);
    } catch (const std::exception& e) {
        job_failed(j, error_text(e));
    } catch (...) {
        job_failed(j, "RuntimeError: unknown exception");
    }
    deliver();
}

void LiveListener::job_failed(const Job& j, const std::string& text) {
    Lock lk(mu_);
    std::string line;
    try {
        line = "slot " + slot_utc(j.s->s.q) + ": " + text;
    } catch (const std::exception&) {
        line = "slot " + std::to_string(j.s->s.q) + ": " + text;
    }
    if (!j.finish) {
        // One bad live receive must not lose the slot: said once, and the
        // slot is received again at its next refresh.
        if (refresh_errors_.insert({j.s->s.q, text}).second) log_locked(line);
        return;
    }
    log_locked(line);
    j.s->applying = false;
    j.s->s.done = true;
}

bool LiveListener::finish_idle_locked() const {
    if (!cfg_.background || stopping_) return true;
    return !busy_finish_ && finish_jobs_.empty();
}

bool LiveListener::idle_locked() const {
    if (!finish_idle_locked()) return false;
    if (!cfg_.background || stopping_) return true;
    return !busy_live_ && live_jobs_.empty();
}

bool LiveListener::finish_idle() const {
    Lock lk(mu_);
    return finish_idle_locked();
}

bool LiveListener::idle() const {
    Lock lk(mu_);
    return idle_locked();
}

bool LiveListener::wait_idle(std::optional<double> timeout_s) const {
    Lock lk(mu_);
    if (!timeout_s) {
        idle_cv_.wait(lk, [&] { return idle_locked(); });
        return true;
    }
    return idle_cv_.wait_for(lk, std::chrono::duration<double>(*timeout_s), [&] { return idle_locked(); });
}

std::optional<double> LiveListener::replay_end_locked() const {
    if (slots_.empty()) return std::nullopt;
    std::optional<double> mx;
    for (const auto& r : slots_) {
        const double e = static_cast<double>(slot_t0(r->s.q)) + dur_ + PB_TAIL_S + 1.0;
        mx = mx ? py_max(*mx, e) : e;
    }
    if (audio_end_) return py_max(*mx, *audio_end_);
    return mx;
}

std::optional<double> LiveListener::replay_end() const {
    Lock lk(mu_);
    return replay_end_locked();
}

// --- which signal is which ------------------------------------------------------------------------

double LiveListener::centre(const Detection& d) const { return joint_ ? receiver::centre_hz(d) : d.f_hz; }
double LiveListener::centre(const PassResult& p) const { return joint_ ? receiver::centre_hz(p) : p.f_hz; }
std::int64_t LiveListener::waveform_of(const PassResult& p) const { return joint_ ? p.waveform : qcfg::WAVEFORM_CE; }

std::pair<std::string, std::string> LiveListener::kind(const PassResult& p) const {
    if (!joint_) return {"CE", cfg_.spec.name()};
    return {wf_name(waveform_of(p)), p.frame};
}

const frame::FrameSpec& LiveListener::spec_of(const PassResult& p) const {
    return joint_ ? frame::get_any(p.frame) : cfg_.spec;
}

Tile* LiveListener::match_locked(Rec& s, double f_hz, std::optional<std::int64_t> dial, const std::string& waveform) {
    const double c = centre_of(f_hz, waveform_id(waveform));
    Tile* best = nullptr;
    double gap = MERGE_HZ;
    for (auto& [id, t] : s.s.tiles) {
        const double d = std::fabs(centre_of(t.f_hz, waveform_id(t.waveform)) - c);
        if (same_dial(t.dial_hz, dial) && d < gap) {
            best = &t;
            gap = d;
        }
    }
    return best;
}

std::optional<Tile> LiveListener::match(std::int64_t q, double f_hz, std::optional<std::int64_t> dial,
                                        const std::string& waveform) const {
    Lock lk(mu_);
    const std::shared_ptr<Rec> r = slot_locked(q);
    if (!r) return std::nullopt;
    Tile* t = const_cast<LiveListener*>(this)->match_locked(*r, f_hz, dial, waveform);
    if (!t) return std::nullopt;
    return *t;
}

Tile& LiveListener::tile_for(Rec& r, double f_hz, std::optional<std::int64_t> dial, const std::string& waveform,
                             const std::optional<std::string>& frame) {
    SlotState& s = r.s;
    Tile* t = match_locked(r, f_hz, dial, waveform);
    if (!t) {
        std::string tid = std::to_string(s.q) + "-" + std::to_string(py_round_int(f_hz));
        while (s.tile(tid)) tid += "b";
        Tile nt;
        nt.id = tid;
        nt.q = s.q;
        nt.slot_utc = slot_utc(s.q);
        nt.frame = (frame && !frame->empty()) ? *frame : cfg_.spec.name();
        nt.f_hz = f_hz;
        nt.dial_hz = dial;
        nt.updated = now_;
        nt.waveform = waveform;
        s.tiles.emplace_back(tid, std::move(nt));
        t = &s.tiles.back().second;
        const std::string on = dial ? " (dial " + fmt(static_cast<double>(*dial) / 1e6, "%.6f") + " MHz)" : "";
        const std::string what = joint_ ? ", waveform " + waveform : "";
        log_locked("slot " + t->slot_utc + ": signal at " + f1(f_hz) + " Hz" + what + on);
    } else if (!t->dial_hz) {
        t->dial_hz = dial;   // found before the dial was known (rig control connected)
    }
    return *t;
}

bool LiveListener::takes(Tile& t, const PassResult& p) {
    const auto [wf, fr] = kind(p);
    if (t.waveform != wf) {
        const double z = p.report.z_ref;
        if (t.z_ref && !(z > *t.z_ref)) return false;
        const std::string was = t.z_ref ? " against " + f1(*t.z_ref) + " as " + t.waveform : "";
        log_locked("slot " + t.slot_utc + ": the signal at " + f1(t.f_hz) + " Hz is waveform " + wf + " (Z_ref " +
                   f1(z) + was + ")");
        t.waveform = wf;
    }
    t.frame = fr;
    return true;
}

void LiveListener::keep(Rec& r, const Detection& d) {
    std::vector<Detection>& dets = r.s.dets;
    const double c = centre(d);
    if (joint_ && std::any_of(dets.begin(), dets.end(), [&](const Detection& g) {
            return std::fabs(centre(g) - c) < MERGE_HZ && g.waveform != d.waveform && !(d.z_ref > g.z_ref);
        }))
        return;
    std::vector<Detection> out;
    for (const Detection& g : dets)
        if (std::fabs(centre(g) - c) >= MERGE_HZ) out.push_back(g);
    out.push_back(d);
    dets = std::move(out);
}

std::vector<acquire::Hint> LiveListener::hints_locked(std::int64_t q, std::optional<std::int64_t> dial) {
    struct Due {
        double f;
        std::int64_t w;
        std::optional<std::int64_t> was;
    };
    std::vector<Due> due;   // (f_hz, waveform, the dial it was heard on)
    if (store_) {
        try {
            std::vector<Due> got;
            for (const store::Due& d : store_->due(q)) {
                Value rec = Value::object();
                rec.set("frame", d.frame);
                rec.set("waveform", Value::integer(d.waveform));
                got.push_back({d.f_hz, due_waveform(rec), dial_.dial_at(static_cast<double>(slot_t0(d.q)))});
            }
            due = std::move(got);
        } catch (const std::exception& e) {   // never lose the slot over the store
            log_locked(std::string("hints: ") + e.what());
        }
    }
    for (const Tile& t : tiles_locked()) {
        if (!t.mode || !(*t.mode == "A" || *t.mode == "B" || *t.mode == "C") || !t.segment || t.q >= q) continue;
        const std::int64_t n = (*t.mode == "A" ? 0 : *t.mode == "B" ? 1 : 2) + 1;
        bool hit = false;
        for (std::int64_t g = *t.segment + 1; g < n; ++g)
            if (t.q + 2 * (g - *t.segment) == q) hit = true;
        if (hit)
            due.push_back({t.f_hz, waveform_id(t.waveform),
                           t.dial_hz ? t.dial_hz : dial_.dial_at(static_cast<double>(slot_t0(t.q)))});
    }
    std::set<std::int64_t> heard_as;
    for (const frame::FrameSpec& s : specs_) heard_as.insert(static_cast<std::int64_t>(s.waveform()));
    std::vector<std::pair<double, std::int64_t>> out;
    for (const Due& d : due) {
        if (!heard_as.count(d.w)) continue;
        double f = d.f;
        if (d.was && dial) f += static_cast<double>(*d.was - *dial);
        if (!in_band(f, static_cast<int>(d.w))) continue;
        const bool apart = std::all_of(out.begin(), out.end(), [&](const std::pair<double, std::int64_t>& o) {
            return std::fabs(centre_of(f, static_cast<int>(d.w)) - centre_of(o.first, static_cast<int>(o.second))) >=
                   MERGE_HZ;
        });
        if (apart) out.emplace_back(py_round2(f), d.w);
    }
    std::vector<acquire::Hint> hs;
    if (joint_) {
        py_sort(out, [](const auto& a, const auto& b) { return a.first < b.first || (!(b.first < a.first) && a.second < b.second); });
        for (const auto& [f, w] : out) hs.push_back({f, w});
    } else {
        std::vector<double> fs;
        for (const auto& [f, w] : out) fs.push_back(f);
        py_sort(fs, [](double a, double b) { return a < b; });
        for (const double f : fs) hs.push_back({f, std::nullopt});
    }
    return hs;
}

std::vector<acquire::Hint> LiveListener::hints(std::int64_t q, std::optional<std::int64_t> dial) {
    std::vector<acquire::Hint> out;
    {
        Lock lk(mu_);
        out = hints_locked(q, dial);
    }
    deliver();
    return out;
}

void LiveListener::known_header_in_place(Rec& s, PassResult& p, std::optional<std::int64_t> dial) {
    if (p.header) return;
    const Tile* t = match_locked(s, p.f_hz, dial, kind(p).first);
    if (!t) return;
    const auto it = headers_.find(t->id);
    if (it == headers_.end()) return;
    p.header = it->second;
}

bool LiveListener::plausible(const Rec& s, const PassResult& p, std::optional<std::int64_t> dial) const {
    const double f = p.f_hz;
    if (!(std::isfinite(f) && in_band(f, static_cast<int>(waveform_of(p))))) return false;
    if (p.header) return true;
    // A header decodes far below this SNR, so a pass this strong without one
    // is something else locked on, not a QRSSTVAE signal.
    const double snr = p.report.snr2500_db;
    const bool called = p.cw && p.cw->text && !p.cw->text->empty();
    if (!called && (!std::isfinite(snr) || snr > HEADERLESS_MAX_SNR_DB)) return false;
    const double c = centre(p);
    auto ghost = [&](const Tile& t) {
        return t.callsign && !t.callsign->empty() && same_dial(t.dial_hz, dial) &&
               std::fabs(centre_of(t.f_hz, waveform_id(t.waveform)) - c) < receiver::GHOST_HZ;
    };
    for (const auto& r : slots_) {
        if (std::llabs(r->s.q - s.s.q) != 1) continue;
        for (const auto& [id, t] : r->s.tiles)
            if (ghost(t)) return false;
    }
    for (const Tile& t : done_tiles_)
        if (std::llabs(t.q - s.s.q) == 1 && ghost(t)) return false;
    return true;
}

std::optional<LiveListener::RenderJob> LiveListener::update_tile(Tile& t, const PassResult& p, double heard,
                                                                const std::optional<store::Key>& key,
                                                                std::optional<double> progress) {
    const types::TrackReport& r = p.report;
    t.f_hz = p.f_hz;
    t.snr_db = std::isfinite(r.snr2500_db) ? std::optional<double>(r.snr2500_db) : std::nullopt;
    t.z_ref = std::isfinite(r.z_ref) ? std::optional<double>(r.z_ref) : std::nullopt;
    t.progress = progress ? *progress : progress_locked(t.q);
    t.heard = heard;
    std::size_t pos = 0;
    for (const float w : p.w) pos += static_cast<double>(w) > 0.0 ? 1 : 0;
    t.received = p.w.empty() ? 0.0 : static_cast<double>(pos) / static_cast<double>(p.w.size());
    t.mean_w_db = t.received > 0 ? std::optional<double>(receiver::mean_w_db(p)) : std::nullopt;
    // (elapsed in seconds: progress is a share of the slot's frame, the
    // longer one when both waveforms are heard, not of the pass's own)
    t.confidence = receiver::confidence_db(p, &spec_of(p), t.progress * dur_);
    if (const auto& h = p.header) {
        headers_[t.id] = *h;
        t.callsign = h->callsign;
        t.grid = h->grid;
        t.picture_id = hex08(h->picture_id);
        t.mode = abc(h->mode);
        t.segment = h->segment;
    }
    if (p.cw && p.cw->text && !p.cw->text->empty()) t.cw_text = *p.cw->text;
    t.updated = now_;
    if (!seams_.renderer) return std::nullopt;
    if (p.z.size() != static_cast<std::size_t>(picture::SENT)) return std::nullopt;   // nowhere to place the latents
    RenderJob job;
    job.tile_id = t.id;
    job.gen = ++next_gen_;
    render_gen_[t.id] = job.gen;
    auto lean = std::make_shared<PassResult>();
    lean->uid = p.uid;
    lean->q = p.q;
    lean->frame = p.frame;
    lean->waveform = p.waveform;
    lean->f_hz = p.f_hz;
    lean->z = p.z;
    lean->w = p.w;
    lean->header = p.header;
    job.p = std::move(lean);
    job.key = key;
    return job;
}

Tile* LiveListener::find_tile_locked(const std::string& id) {
    for (const auto& r : slots_)
        if (Tile* t = r->s.tile(id)) return t;
    for (Tile& t : done_tiles_)
        if (t.id == id) return &t;
    return nullptr;
}

// --- pictures ------------------------------------------------------------------------------------

void LiveListener::render(const RenderJob& job) {
    const std::shared_ptr<Renderer> renderer = seams_.renderer;
    if (!renderer) return;
    const PassResult& p = *job.p;
    const std::optional<header::HeaderFields>& h = p.header;
    // A headerless pass the store matched to a picture is drawn as that picture.
    std::optional<store::Accumulator> matched;
    if (!h && job.key && store_ && store_->has_accumulator(*job.key)) matched = store_->accumulator(*job.key);
    const std::optional<std::int64_t> codec_id =
        h ? std::optional<std::int64_t>(h->codec_id) : matched ? std::optional<std::int64_t>(matched->codec_id) : std::nullopt;
    {
        Lock lk(mu_);
        if (codec_error_) return;
    }
    std::optional<std::string> err;
    try {
        err = renderer->load(codec_id, [this](const std::string& m) { log(m); });
    } catch (const std::exception& e) {
        err = std::string(e.what()).empty() ? error_name(e) : std::string(e.what());
    }
    if (err) {
        {
            Lock lk(mu_);
            if (!codec_error_) {
                codec_error_ = *err;
                log_locked("no pictures: " + *err);
            }
        }
        deliver();
        return;
    }
    std::optional<std::int64_t> seg;
    std::int64_t mode = 0;
    const bool guess = !h && !matched;
    std::optional<std::string> mismatch;
    if (matched) mode = matched->mode;   // the pass is a member already
    if (guess) {
        seg = 0;
        mode = 0;
    } else {
        if (h) {
            seg = h->segment;
            mode = h->mode;
        }
        try {
            mismatch = renderer->check(codec_id);
        } catch (const std::exception& e) {
            mismatch = std::string(e.what());
        }
    }
    bool weight = false;
    std::int64_t n = 1;
    std::optional<images::Picture> pic;
    std::optional<std::string> failed;
    if (!mismatch) {
        const std::size_t plane = static_cast<std::size_t>(picture::N_GROUPS) * picture::GROUP_LATENTS;
        std::vector<double> S(plane, 0.0), W(plane, 0.0);
        const std::optional<store::Key> k =
            job.key ? job.key : (h ? std::optional<store::Key>(store::Key{h->callsign, h->picture_id}) : std::nullopt);
        bool member = false;
        if (k && store_ && store_->has_accumulator(*k)) {
            const store::Accumulator acc = store_->accumulator(*k);
            for (std::size_t i = 0; i < plane; ++i) {
                S[i] += static_cast<double>(acc.S[i]);
                W[i] += static_cast<double>(acc.W[i]);
            }
            // (a copy of a transmission the store holds already is that
            // member; on the joint path compared by band centre)
            const std::vector<std::string> uids = acc.uids();
            member = std::find(uids.begin(), uids.end(), p.uid) != uids.end();
            if (!member && seg) {
                const std::optional<std::int64_t> wf = joint_ ? std::optional<std::int64_t>(waveform_of(p)) : std::nullopt;
                member = store_->twin(acc, p.q, p.f_hz, *seg, std::nullopt, wf).has_value();
            }
            n = static_cast<std::int64_t>(acc.members.size()) + (member ? 0 : 1);
        }
        if (!member && seg) {
            const std::span<const std::uint16_t> idx = picture::air_to_canonical(*seg);
            if (p.w.size() != p.z.size())
                throw std::invalid_argument("operands could not be broadcast together");
            const std::size_t base = static_cast<std::size_t>(*seg) * picture::GROUP_LATENTS;
            for (std::size_t i = 0; i < idx.size(); ++i) {
                const double wi = p.w[i];
                S[base + idx[i]] += wi > 0 ? wi * static_cast<double>(p.z[i]) : 0.0;
                W[base + idx[i]] += wi;
            }
        }
        weight = std::any_of(W.begin(), W.end(), [](double v) { return v > 0; });
        if (weight) {
            try {
                pic = renderer->render(S, W, mode, codec_id);
            } catch (const std::exception& e) {
                failed = std::string(e.what());
            }
        }
    }
    const std::string rel = "tiles/" + job.tile_id + ".png";
    std::unique_lock<std::mutex> png(png_mu_, std::defer_lock);
    if (pic) {
        png.lock();   // one file write at a time, the newest picture last
        {
            Lock lk(mu_);
            const auto g = render_gen_.find(job.tile_id);
            if (g == render_gen_.end() || g->second != job.gen) return;   // a newer update draws its own
        }
        const std::string tmp = dir_ + "/tiles/." + job.tile_id + "." + std::to_string(process_id()) + ".png.tmp";
        images::save_png(*pic, tmp);
        replace_or_throw(tmp, dir_ + "/" + rel);
    }
    Lock lk(mu_);
    const auto g = render_gen_.find(job.tile_id);
    Tile* t = find_tile_locked(job.tile_id);
    if (!t || g == render_gen_.end() || g->second != job.gen) return;
    if (matched) {
        t->callsign = job.key->callsign;
        t->picture_id = hex08(job.key->picture_id);
        t->mode = abc(matched->mode);
    }
    const std::string guess_note(GUESS_NOTE);
    if (guess) {
        if (t->note.find(guess_note) == std::string::npos) t->note = t->note.empty() ? guess_note : t->note + "; " + guess_note;
    } else if (mismatch) {
        t->note = *mismatch;   // CodecMismatch: a silently wrong picture
        return;
    } else {
        t->note = replace_all(replace_all(t->note, "; " + guess_note, ""), guess_note, "");
    }
    if (!weight) return;
    if (failed) {
        t->note = "render failed: " + *failed;
        return;
    }
    t->image = rel;
    t->image_rev += 1;
    t->passes = n;
}

// --- the receives ----------------------------------------------------------------------------------

std::vector<Detection> LiveListener::candidates_for(const std::shared_ptr<Rec>& s, const receiver::Prepared& prep,
                                                    std::span<const std::uint8_t> ok, std::int64_t origin) {
    const std::int64_t q = s->s.q;
    std::vector<acquire::Hint> hints;
    bool need = false;
    std::vector<Detection> cands;
    {
        Lock lk(mu_);
        hints = hints_locked(q, dial_.dial_at(static_cast<double>(slot_t0(q))));
        need = !s->s.cands || !subset(hints, s->s.hints);
        if (need && !hints.empty() && !set_equal(hints, s->s.hints)) {
            std::string where;
            for (std::size_t k = 0; k < hints.size(); ++k) {
                if (k) where += ", ";
                where += f1(hints[k].f_hz);
                if (joint_) where += " Hz (" + wf_name(hints[k].waveform.value_or(qcfg::WAVEFORM_CE)) + ")";
            }
            if (!joint_) where += " Hz";
            log_locked("slot " + slot_utc(q) + ": a later pass is due at " + where);
        }
        if (!need) cands = *s->s.cands;
    }
    deliver();
    if (need) {
        cands = seams_.candidates(prep, sel_, hints);
        // Acquisition reads only the preamble and a few seconds around it:
        // once the audio read is past that, its answer is final.
        std::int64_t last = -1;
        for (std::int64_t k = static_cast<std::int64_t>(ok.size()) - 1; k >= 0; --k)
            if (ok[static_cast<std::size_t>(k)]) {
                last = k;
                break;
            }
        const bool final = last >= 0 && static_cast<double>(last + 1 - origin) / FE_FS_D > acquire::live_end_s() + 1.0;
        Lock lk(mu_);
        if (final)
            s->s.cands = cands;
        else
            s->s.cands.reset();
        s->s.hints = hints;
    }
    std::vector<Detection> out;
    for (const Detection& c : cands)
        if (c.method != "hint") out.push_back(c);
    return out;
}

void LiveListener::refresh(const std::shared_ptr<Rec>& s) {
    const std::int64_t q = s->s.q;
    const std::int64_t t0 = slot_t0(q);
    double t_start = 0.0;
    {
        Lock lk(mu_);
        if (s->s.done || s->applying) return;   // (a final refresh queued behind a finish that has run)
        t_start = clock_locked();
    }
    const SlotRead rd = capture_read(ring_, q, sel_);
    dial_.refresh();
    double elapsed = 0.0, progress = 0.0;
    std::optional<std::int64_t> dial;
    DialEpochs heard_by_dial;
    {
        Lock lk(mu_);
        elapsed = now_ - static_cast<double>(t0);
        // the tile's progress is when its audio was read, not when the
        // receive returned
        progress = progress_locked(q);
        dial = dial_.dial_at(static_cast<double>(t0));
        heard_by_dial = dial_.by_dial(static_cast<double>(t0) - static_cast<double>(rd.origin) / FE_FS_D,
                                      static_cast<double>(t0) + static_cast<double>(rd.x.size()) / FE_FS_D);
    }
    // live, a slot is received on the dial it started on: anything heard on
    // another counts as not heard
    std::vector<std::pair<double, double>> spans;
    for (const auto& [d, sp] : heard_by_dial)
        if (d == dial) spans = sp;
    const std::vector<std::uint8_t> ok = restrict(rd.ok, rd.origin, q, spans);
    {
        const auto [lo, hi] = py_slice(static_cast<std::int64_t>(ok.size()), rd.origin,
                                       rd.origin + static_cast<std::int64_t>(PREAMBLE_S) * FE_FS_I);
        const double m = mask_mean(std::span<const std::uint8_t>(ok.data() + lo, static_cast<std::size_t>(hi - lo)));
        if (m < 0.5) return;   // joined after its preamble: only the end-of-slot search can find it
    }
    PrepareSeams ps;
    ps.noise = seams_.noise;
    const SlotCapture cap = capture_prepare(rd.x, ok, rd.origin, q, Selection(sel_), std::nullopt, ps);
    bool first_branch = false, recheck = false;
    std::vector<double> known;
    {
        Lock lk(mu_);
        first_branch = elapsed < cfg_.detect_until_s || s->s.dets.empty();
        if (!first_branch && elapsed < cfg_.recheck_until_s) {
            recheck = true;
            for (const Detection& d : s->s.dets) known.push_back(centre(d));
        }
    }
    std::vector<Detection> found;
    if (first_branch) {
        receiver::DetectOptions o;
        o.live_only = true;
        o.erase_s = cap.erase;
        o.cands = candidates_for(s, cap.prep, ok, rd.origin);
        found = seams_.detect(cap.prep, sel_, o);
    } else if (recheck) {
        std::vector<Detection> cands = candidates_for(s, cap.prep, ok, rd.origin);
        const bool fresh = std::any_of(cands.begin(), cands.end(), [&](const Detection& c) {
            return std::all_of(known.begin(), known.end(), [&](double f) { return std::fabs(centre(c) - f) >= MERGE_HZ; });
        });
        if (fresh) {
            receiver::DetectOptions o;
            o.live_only = true;
            o.erase_s = cap.erase;
            o.cands = std::move(cands);
            o.skip_hz = known;
            found = seams_.detect(cap.prep, sel_, o);
        }
    }
    std::vector<Detection> dets;
    {
        Lock lk(mu_);
        for (const Detection& d : found) keep(*s, d);   // keep one detection per signal
        if (s->s.dets.empty()) return;
        dets = s->s.dets;
    }
    receiver::SlotOptions so;
    so.live_only = false;
    so.estimator = cfg_.estimator;
    so.erase_s = cap.erase;
    so.round_b = false;
    so.dets = dets;
    std::vector<PassResult> passes = seams_.receive_slot(cap.prep, sel_, so);
    const std::size_t n_passes = passes.size();
    for (PassResult& p : passes) {
        std::optional<RenderJob> job;
        {
            Lock lk(mu_);
            if (s->s.done || s->applying) break;
            known_header_in_place(*s, p, dial);
            if (!plausible(*s, p, dial)) continue;
            const auto [wf, fr] = kind(p);
            Tile& t = tile_for(*s, p.f_hz, dial, wf, fr);
            if (!takes(t, p)) continue;
            job = update_tile(t, p, cap.heard, std::nullopt, progress);
            t.status = s->s.finishing ? "finishing" : "receiving";
        }
        deliver();
        if (job) render(*job);
        {
            Lock lk(mu_);
            log_locked("slot " + slot_utc(q) + ": " + std::to_string(n_passes) + " signal(s) at " +
                       fmt(100 * progress_locked(q), "%.0f") + "% (" + fmt(clock_locked() - t_start, "%.0f") + " s)");
        }
        deliver();
    }
}

void LiveListener::finish(const std::shared_ptr<Rec>& s) {
    const std::int64_t q = s->s.q;
    const std::int64_t t0 = slot_t0(q);
    double t_start = 0.0;
    FeReader reader;
    std::vector<Detection> dets;
    {
        Lock lk(mu_);
        t_start = clock_locked();
        reader = s->s.reader;
        dets = s->s.dets;
    }
    const SlotRead rd = reader ? capture_read(reader, q, sel_) : capture_read(ring_, q, sel_);
    dial_.refresh();
    DialEpochs dials;
    std::optional<std::int64_t> first;
    std::vector<acquire::Hint> hints;
    {
        Lock lk(mu_);
        dials = dial_.by_dial(static_cast<double>(t0) - static_cast<double>(rd.origin) / FE_FS_D,
                              static_cast<double>(t0) + static_cast<double>(rd.x.size()) / FE_FS_D);
        first = dial_.dial_at(static_cast<double>(t0));
        hints = hints_locked(q, first);
    }
    deliver();
    struct Result {
        std::optional<std::int64_t> dial;
        std::vector<PassResult> passes;
        double heard = 0.0;
    };
    // one receive per dial the slot was heard on, each with the others
    // erased; the preamble's detections belong to the dial it started on
    std::vector<Result> results;
    for (const auto& [dial, spans] : dials) {
        const std::vector<std::uint8_t> ok = restrict(rd.ok, rd.origin, q, spans);
        const bool own = dial == first;
        FinishResult r = seams_.finish(rd.x, ok, rd.origin, q, finish_rx_,
                                       own ? std::span<const Detection>(dets) : std::span<const Detection>(),
                                       own ? std::span<const acquire::Hint>(hints) : std::span<const acquire::Hint>());
        results.push_back({dial, std::move(r.passes), r.heard});
    }

    // _finish_apply
    std::set<std::string> seen;
    {
        Lock lk(mu_);
        s->applying = true;
    }
    for (Result& res : results) {
        std::vector<PassResult> keep_passes;
        {
            Lock lk(mu_);
            for (PassResult& p : res.passes) known_header_in_place(*s, p, res.dial);
            for (PassResult& p : res.passes)
                if (plausible(*s, p, res.dial)) keep_passes.push_back(std::move(p));
        }
        for (const PassResult& p : keep_passes) {
            std::string tid;
            {
                Lock lk(mu_);
                const auto [wf, fr] = kind(p);
                Tile& t = tile_for(*s, p.f_hz, res.dial, wf, fr);
                if (!takes(t, p)) {
                    log_locked("slot " + t.slot_utc + ": the pass at " + f1(p.f_hz) + " Hz as waveform " + wf +
                               " is not kept: the signal read stronger as " + t.waveform);
                    continue;
                }
                seen.insert(t.id);
                tid = t.id;
            }
            deliver();
            std::optional<store::Key> key;
            std::optional<std::string> note;
            bool stored = false;
            if (store_) {
                try {
                    const associate::Result a = seams_.associate(p, *store_);
                    key = a.key;
                    note = key ? "stored (" + (a.rule.empty() ? std::string("already a member") : a.rule) + ")"
                               : std::string("stored (provisional)");
                    stored = true;
                } catch (const std::exception& e) {   // never lose the tile over the store
                    note = std::string("store failed: ") + e.what();
                }
            }
            std::optional<RenderJob> job;
            {
                Lock lk(mu_);
                Tile* t = s->s.tile(tid);
                if (!t) continue;
                if (note) t->note = *note;
                if (stored) t->uid = p.uid;
                job = update_tile(*t, p, res.heard, key, std::nullopt);
                t->status = "complete";
            }
            deliver();
            if (job) render(*job);
        }
    }
    {
        Lock lk(mu_);
        for (auto& [id, t] : s->s.tiles)
            if (!seen.count(id)) {
                t.status = "lost";
                t.updated = now_;
            }
    }
    if (!seen.empty()) identify_provisional();
    {
        Lock lk(mu_);
        s->applying = false;
        s->s.done = true;
        std::size_t n = 0;
        std::optional<double> heard;
        for (const Result& r : results) {
            n += r.passes.size();
            heard = heard ? py_max(*heard, r.heard) : r.heard;
        }
        const double h = heard.value_or(0.0);
        std::string what = h >= FINISH_SEARCH_MIN_HEARD ? "" : ", " + fmt(100 * h, "%.0f") + "% heard: no whole-slot search";
        if (results.size() > 1) what += ", on " + std::to_string(results.size()) + " dials";
        if (s->s.reader) what += ", from the passband store";
        log_locked("slot " + slot_utc(q) + ": frame over, " + std::to_string(n) + " pass(es)" + what + " (" +
                   fmt(clock_locked() - t_start, "%.0f") + " s)");
    }
    deliver();
}

void LiveListener::identify_provisional() {
    struct Todo {
        std::string id;
        std::string uid;
    };
    std::vector<Todo> todo;
    {
        Lock lk(mu_);
        for (const Tile& t : done_tiles_)
            if (t.uid && !t.uid->empty() && t.status == "complete" && t.note.rfind("stored (provisional)", 0) == 0)
                todo.push_back({t.id, *t.uid});
        if (todo.empty() || !store_) return;
    }
    for (const Todo& td : todo) {
        std::optional<store::Key> key;
        store::Accumulator acc;
        std::optional<Value> m;
        try {
            key = store_->find_member(td.uid);
            if (!key) continue;
            acc = store_->accumulator(*key);
            m = acc.member(td.uid);
        } catch (const std::exception&) {   // a store problem must not stop the finish
            continue;
        }
        Lock lk(mu_);
        Tile* t = nullptr;
        for (Tile& x : done_tiles_)
            if (x.id == td.id && x.uid == td.uid) {
                t = &x;
                break;
            }
        if (!t) continue;
        t->callsign = key->callsign;
        t->picture_id = hex08(key->picture_id);
        t->mode = abc(acc.mode);
        const Value* seg = (m && m->kind() == Value::Kind::Object) ? m->find("segment") : nullptr;
        if (seg && !seg->is_null()) t->segment = py_int(*seg);
        const Value* method = (m && m->kind() == Value::Kind::Object) ? m->find("method") : nullptr;
        t->note = "stored (" + (method ? py_str(*method) : std::string("corr")) + ")";
        t->updated = now_;
        log_locked("slot " + t->slot_utc + ": the pass at " + f1(t->f_hz) + " Hz is " + *t->callsign + " " +
                   *t->picture_id);
    }
    deliver();
}

// --- run ---------------------------------------------------------------------------------------------

void run(LiveListener& listener, Source& source, bool realtime, const std::atomic<bool>* stop) {
    listener.set_source(source.name());
    std::vector<std::string> names;
    for (const frame::FrameSpec& s : listener.specs())
        names.push_back(std::string(s.waveform() == frame::Waveform::L ? "L" : "CE") + " (" + s.name() + ")");
    std::string joined;
    for (std::size_t k = 0; k < names.size(); ++k) joined += (k ? " and " : "") + names[k];
    listener.log("listening: " + source.name() + ", frame " + listener.config().spec.name() + ", waveform" +
                 (names.size() > 1 ? "s" : "") + " " + joined);
    listener.write_state();
    double last_write = 0.0;
    for (;;) {
        std::optional<Chunk> c = source.next(1.0);
        if (!c && source.eof()) break;
        if (stop && stop->load()) break;
        if (!realtime) listener.wait_idle();   // a replay waits for its receives
        if (c) listener.feed(c->x, c->t_end);
        const double now = realtime ? listener.clock() : listener.audio_end().value_or(0.0);
        if (now != 0.0 && (listener.step(now) || listener.clock() - last_write > 2.0)) {
            listener.write_state();
            last_write = listener.clock();
        }
    }
    if (!realtime && listener.audio_end()) {
        listener.wait_idle();
        if (const std::optional<double> end = listener.replay_end()) listener.step(*end);
        listener.wait_idle();
        listener.step(listener.now());   // retire the slots the worker finished
    }
    listener.log("audio ended");
    listener.write_state(false);
}

}  // namespace sstvae::qrss::live
