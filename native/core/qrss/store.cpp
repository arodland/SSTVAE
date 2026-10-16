// Port of sstvae/qrss/store.py; see store.hpp.

#include "qrss/store.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <stdexcept>
#include <system_error>
#include <unordered_set>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <stdlib.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

#include "math/errors.hpp"
#include "math/npstats.hpp"
#include "qrss/associate.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/npsum.hpp"
#include "qrss/pynum.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
#include "util/fileio.hpp"
#include "util/sha256.hpp"

namespace sstvae::qrss::store {

using pyjson::Value;

namespace {

// --- Python's conversions of JSON values ---------------------------------------------

const char* type_name(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Null: return "NoneType";
        case Value::Kind::Bool: return "bool";
        case Value::Kind::Int: return "int";
        case Value::Kind::Float: return "float";
        case Value::Kind::String: return "str";
        case Value::Kind::Array: return "list";
        case Value::Kind::Object: return "dict";
    }
    return "object";
}

// str(v), as an f-string formats a JSON value.
std::string py_str(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Null: return "None";
        case Value::Kind::Bool: return v.as_bool() ? "True" : "False";
        case Value::Kind::String: return v.as_string();
        default: return pyjson::dumps(v);   // an int, or a float's repr (containers: approximately)
    }
}

bool py_equal(const Value& a, const Value& b);

bool numeric(const Value& v) {
    return v.kind() == Value::Kind::Bool || v.kind() == Value::Kind::Int || v.kind() == Value::Kind::Float;
}

double number_of(const Value& v) {
    if (v.kind() == Value::Kind::Bool) return v.as_bool() ? 1.0 : 0.0;
    return v.as_double();
}

// Python's ==: numbers by value (1 == 1.0 == True, NaN never), dicts
// regardless of order.
bool py_equal(const Value& a, const Value& b) {
    if (numeric(a) && numeric(b)) {
        if (a.kind() != Value::Kind::Float && b.kind() != Value::Kind::Float) {
            const auto ia = a.kind() == Value::Kind::Bool ? std::int64_t(a.as_bool()) : a.as_int();
            const auto ib = b.kind() == Value::Kind::Bool ? std::int64_t(b.as_bool()) : b.as_int();
            return ia == ib;
        }
        return number_of(a) == number_of(b);
    }
    if (a.kind() != b.kind()) return false;
    switch (a.kind()) {
        case Value::Kind::Null: return true;
        case Value::Kind::String: return a.as_string() == b.as_string();
        case Value::Kind::Array: {
            const auto& x = a.items();
            const auto& y = b.items();
            if (x.size() != y.size()) return false;
            for (std::size_t i = 0; i < x.size(); ++i)
                if (!py_equal(x[i], y[i])) return false;
            return true;
        }
        case Value::Kind::Object: {
            if (a.keys().size() != b.keys().size()) return false;
            for (std::size_t i = 0; i < a.keys().size(); ++i) {
                const Value* o = b.find(a.keys()[i]);
                if (!o || !py_equal(a.values()[i], *o)) return false;
            }
            return true;
        }
        default: return false;
    }
}

// A member's "uid" against an optional uid (nullopt: Python's None).
bool uid_matches(const Value& v, std::optional<std::string_view> uid) {
    return uid ? equals(v, *uid) : v.is_null();
}

// --- files ---------------------------------------------------------------------------

[[noreturn]] void file_error(const char* verb, const std::string& path, int err) {
    throw FileError(verb, path, err);
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    util::File f = util::open_file(path, "rb");
    if (!f) file_error("cannot open", path, errno);
    std::vector<std::uint8_t> out;
    std::uint8_t buf[1 << 16];
    for (;;) {
        const std::size_t n = std::fread(buf, 1, sizeof buf, f.get());
        out.insert(out.end(), buf, buf + n);
        if (n < sizeof buf) {
            if (std::ferror(f.get())) file_error("cannot read", path, errno ? errno : EIO);
            break;
        }
    }
    return out;
}

// bytes.decode() (strict UTF-8: no surrogates, no overlong forms).
void check_utf8(std::span<const std::uint8_t> s, const std::string& path) {
    std::size_t i = 0;
    const std::size_t n = s.size();
    auto bad = [&] {
        throw std::invalid_argument("'utf-8' codec can't decode byte 0x" +
                                    std::string(1, "0123456789abcdef"[s[i] >> 4]) +
                                    std::string(1, "0123456789abcdef"[s[i] & 15]) + " in position " +
                                    std::to_string(i) + " (" + path + ")");
    };
    while (i < n) {
        const std::uint8_t b = s[i];
        if (b < 0x80) {
            ++i;
            continue;
        }
        std::size_t len = 0;
        std::uint8_t lo = 0x80, hi = 0xBF;
        if (b >= 0xC2 && b <= 0xDF) {
            len = 2;
        } else if (b >= 0xE0 && b <= 0xEF) {
            len = 3;
            if (b == 0xE0) lo = 0xA0;
            if (b == 0xED) hi = 0x9F;
        } else if (b >= 0xF0 && b <= 0xF4) {
            len = 4;
            if (b == 0xF0) lo = 0x90;
            if (b == 0xF4) hi = 0x8F;
        } else {
            bad();
        }
        if (i + 1 >= n || s[i + 1] < lo || s[i + 1] > hi) bad();
        for (std::size_t k = 2; k < len; ++k)
            if (i + k >= n || (s[i + k] & 0xC0) != 0x80) bad();
        i += len;
    }
}

Value read_json(const std::string& path) {
    const std::vector<std::uint8_t> b = read_file(path);
    check_utf8(b, path);
    return pyjson::loads(std::string_view(reinterpret_cast<const char*>(b.data()), b.size()));
}

// np.load(path)[name]'s KeyError text.
const npz::Array& member_of(const npz::Npz& z, std::string_view name) {
    if (!z.contains(name)) {
        const std::string what = std::string(name) + " is not a file in the archive";
        throw KeyError(what, "KeyError: " + pytext::repr(what));
    }
    return z.at(name);
}

// a.astype(np.float32) of an accumulator plane, which must hold 3 x 52800.
std::vector<float> plane_of(const npz::Array& a, const char* what) {
    if (a.size() != PLANE)
        throw std::invalid_argument(std::string("accumulator ") + what + " holds " + std::to_string(a.size()) +
                                    " values, not 3 x 52800");
    if (a.dtype == npz::DType::f4) return a.values<float>();
    const std::vector<double> d = a.to_double();
    return std::vector<float>(d.begin(), d.end());
}

std::vector<double> plane_double(const npz::Array& a, const std::string& what) {
    if (a.size() != PLANE)
        throw std::invalid_argument("foreign part " + what + " holds " + std::to_string(a.size()) +
                                    " values, not 3 x 52800");
    return a.to_double();
}

std::string shape_repr(const std::vector<std::size_t>& shape) {
    std::string s = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(shape[i]);
    }
    if (shape.size() == 1) s += ",";
    return s + ")";
}

#ifdef _WIN32
std::string lower_ascii(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}
#endif

// The names in `dir` matching "*<suffix>" as pathlib's glob matches them
// (hidden names included; case-insensitively on Windows), or none when the
// directory cannot be read.
std::vector<std::string> glob_suffix(const std::string& dir, std::string_view suffix) {
    std::error_code ec;
    std::vector<std::string> out;
    for (std::string& name : util::list_dir(dir, ec)) {
        if (name.size() < suffix.size()) continue;
        std::string tail = name.substr(name.size() - suffix.size());
#ifdef _WIN32
        tail = lower_ascii(tail);
#endif
        if (tail == suffix) out.push_back(std::move(name));
    }
    return out;
}

// sorted(Path objects in one directory): by name, case-insensitively on
// Windows (a WindowsPath compares its lowercased text).
void sort_paths(std::vector<std::string>& names) {
#ifdef _WIN32
    std::stable_sort(names.begin(), names.end(),
                     [](const std::string& a, const std::string& b) { return lower_ascii(a) < lower_ascii(b); });
#else
    std::sort(names.begin(), names.end());
#endif
}

std::string join(const std::string& dir, const std::string& name) {
#ifdef _WIN32
    return dir + "\\" + name;
#else
    return dir + "/" + name;
#endif
}

void remove_or_throw(const std::string& path, bool missing_ok) {
    std::error_code ec;
    if (util::remove_file(path, ec)) return;
    if (missing_ok && ec == std::errc::no_such_file_or_directory) return;
    file_error("cannot remove", path, util::errno_of(ec));
}

void default_replace(const std::string& from, const std::string& to) {
    std::error_code ec;
    if (!util::replace_file(from, to, ec)) file_error("cannot replace", from, util::errno_of(ec));
}

std::string env(const char* name) {
#ifdef _WIN32
    std::wstring w;
    for (const char* c = name; *c; ++c) w += static_cast<wchar_t>(*c);
    const wchar_t* v = ::_wgetenv(w.c_str());
    return v ? util::utf8_of(std::wstring_view(v)) : std::string();
#else
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
#endif
}

#ifdef _WIN32
bool has_env(const char* name) {
    std::wstring w;
    for (const char* c = name; *c; ++c) w += static_cast<wchar_t>(*c);
    return ::_wgetenv(w.c_str()) != nullptr;
}
#endif

std::string pathlib(const std::string& p) {
#ifdef _WIN32
    std::string s = p;
    std::replace(s.begin(), s.end(), '/', '\\');
    return s;
#else
    return pypath::pathlib_str(p);
#endif
}

Value key_list(const Key& k) {
    return Value::array({Value::string(k.callsign), Value::integer(k.picture_id)});
}

// The mutex of the store at `root` (made already), shared by every Store open
// on it in this process: keyed by the canonical path, so "a/../s", "./s"
// and a symlink to s are one store. Leaked, as a function-local static that
// Stores destroyed during static destruction may still reach.
std::shared_ptr<std::recursive_mutex> root_mutex(const std::string& root) {
    static std::mutex* const reg_mu = new std::mutex;
    static auto* const reg = new std::map<std::filesystem::path, std::weak_ptr<std::recursive_mutex>>;
    std::filesystem::path key;
    if (const auto p = util::fs_path(root)) {
        std::error_code ec;
        key = std::filesystem::canonical(*p, ec);
        if (ec) key = std::filesystem::absolute(*p, ec).lexically_normal();
        if (ec) key = *p;
    }
    const std::lock_guard<std::mutex> lk(*reg_mu);
    for (auto it = reg->begin(); it != reg->end();)
        it = it->second.expired() ? reg->erase(it) : std::next(it);
    auto& slot = (*reg)[key];
    auto mu = slot.lock();
    if (!mu) {
        mu = std::make_shared<std::recursive_mutex>();
        slot = mu;
    }
    return mu;
}

}  // namespace

// --- conversions shared with associate ------------------------------------------------------

std::int64_t py_int(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Int: return v.as_int();
        case Value::Kind::Bool: return v.as_bool() ? 1 : 0;
        case Value::Kind::Float: {
            const double f = v.as_double();
            if (std::isnan(f)) throw std::invalid_argument("cannot convert float NaN to integer");
            if (std::isinf(f)) throw std::overflow_error("cannot convert float infinity to integer");
            const double t = std::trunc(f);
            if (t < -9223372036854775808.0 || t >= 9223372036854775808.0)
                throw std::overflow_error("int() of " + pyjson::dumps(v) + " does not fit int64");
            return static_cast<std::int64_t>(t);
        }
        case Value::Kind::String: {
            // int(str): pynum's rules (ASCII, whatever the C locale says)
            const auto parsed = pynum::py_int(v.as_string(), 10);
            if (!parsed) throw std::invalid_argument(pynum::invalid_int_literal(v.as_string(), 10));
            if (!parsed->value) throw std::overflow_error("int() of " + parsed->decimal + " does not fit int64");
            return *parsed->value;
        }
        default:
            throw math::TypeError(std::string("int() argument must be a string, a bytes-like object or a real "
                                              "number, not '") + type_name(v) + "'");
    }
}

double py_float(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Int:
        case Value::Kind::Float: return v.as_double();
        case Value::Kind::Bool: return v.as_bool() ? 1.0 : 0.0;
        case Value::Kind::String: {
            // float(str): pynum's rules ('.' the point whatever LC_NUMERIC
            // says; no hex; an underflow is 0, an overflow inf)
            if (const auto f = pynum::py_float(v.as_string())) return *f;
            throw std::invalid_argument("could not convert string to float: " + pytext::repr(v.as_string()));
        }
        default:
            throw math::TypeError(std::string("float() argument must be a string or a real number, not '") +
                                  type_name(v) + "'");
    }
}

bool truthy(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Null: return false;
        case Value::Kind::Bool: return v.as_bool();
        case Value::Kind::Int: return v.as_int() != 0;
        case Value::Kind::Float: return v.as_double() != 0.0;
        case Value::Kind::String: return !v.as_string().empty();
        case Value::Kind::Array: return !v.items().empty();
        case Value::Kind::Object: return !v.keys().empty();
    }
    return false;
}

bool equals(const Value& v, std::string_view s) {
    return v.kind() == Value::Kind::String && v.as_string() == s;
}

bool contains(const Value& c, std::string_view s) {
    switch (c.kind()) {
        case Value::Kind::Array:
            for (const Value& x : c.items())
                if (equals(x, s)) return true;
            return false;
        case Value::Kind::String: return c.as_string().find(s) != std::string::npos;
        case Value::Kind::Object: return c.find(s) != nullptr;
        default:
            throw math::TypeError(std::string("argument of type '") + type_name(c) + "' is not iterable");
    }
}

// --- module functions ---------------------------------------------------------------------

#ifndef _WIN32
namespace {

// pwd.getpwuid(os.getuid()).pw_dir, through the reentrant call (a Store may
// be opened on several threads at once); nullopt when there is no entry.
std::optional<std::string> passwd_home() {
    long hint = ::sysconf(_SC_GETPW_R_SIZE_MAX);
    std::vector<char> buf(hint > 0 ? static_cast<std::size_t>(hint) : 4096);
    for (;;) {
        passwd pw{};
        passwd* out = nullptr;
        const int rc = ::getpwuid_r(::getuid(), &pw, buf.data(), buf.size(), &out);
        if (rc == ERANGE && buf.size() < (std::size_t(1) << 20)) {
            buf.resize(buf.size() * 2);
            continue;
        }
        if (rc == 0 && out && out->pw_dir) return std::string(out->pw_dir);
        return std::nullopt;
    }
}

}  // namespace
#endif

std::string default_root() {
    if (const std::string e = env("QRSSTVAE_HOME"); !e.empty()) return pathlib(e);
    std::string home;
#ifdef _WIN32
    if (has_env("USERPROFILE")) {
        home = env("USERPROFILE");
    } else if (has_env("HOMEPATH")) {
        home = env("HOMEDRIVE") + env("HOMEPATH");
    } else {
        home = "~";
    }
    return pathlib(home + "\\.local\\share\\qrsstvae");
#else
    if (!env("HOME").empty()) {      // (an empty $HOME counts as unset: store.py's _home)
        home = env("HOME");
    } else if (const auto dir = passwd_home()) {
        home = *dir;
    } else {
        home = "~";
    }
    if (home != "~") {
        while (!home.empty() && home.back() == '/') home.pop_back();
        if (home.empty()) home = "/";
    }
    // (Path("/") / ".local" is "/.local": a POSIX "//" root would survive pathlib)
    return pathlib((home == "/" ? std::string() : home) + "/.local/share/qrsstvae");
#endif
}

std::string key_name(const Key& key) {
    std::string call = key.callsign;
    std::replace(call.begin(), call.end(), '/', '_');
    const bool neg = key.picture_id < 0;
    const std::uint64_t mag = neg ? 0 - static_cast<std::uint64_t>(key.picture_id)
                                  : static_cast<std::uint64_t>(key.picture_id);
    char buf[32];
    std::snprintf(buf, sizeof buf, "%llx", static_cast<unsigned long long>(mag));
    std::string hex = buf;
    const std::size_t width = neg ? 7 : 8;
    if (hex.size() < width) hex.insert(0, width - hex.size(), '0');
    return call + "_" + (neg ? "-" : "") + hex;
}

double slot_time(std::int64_t q) { return static_cast<double>(q) * static_cast<double>(SLOT_S); }

double mean_w_db(std::span<const double> w) {
    const double m = w.empty() ? 0.0 : npsum::mean(w);
    return m > 0 ? 10.0 * std::log10(m) : -99.0;
}

double mean_w_db(std::span<const float> w) {
    const std::vector<double> d(w.begin(), w.end());
    return mean_w_db(std::span<const double>(d));
}

std::span<const std::uint16_t> canonical_index(std::int64_t segment, std::int64_t n) {
    if (!(0 < n && n <= SENT))
        throw std::invalid_argument("a pass carries 1.." + std::to_string(SENT) + " latents, not " + std::to_string(n));
    return picture::air_to_canonical(segment).subspan(0, static_cast<std::size_t>(n));
}

// --- the accumulator -------------------------------------------------------------------------

Accumulator Accumulator::empty(Key key, std::int64_t mode, std::int64_t codec_id, double now) {
    Accumulator a;
    a.key = std::move(key);
    a.mode = mode;
    a.codec_id = codec_id;
    a.S.assign(PLANE, 0.0f);
    a.W.assign(PLANE, 0.0f);
    a.S_ext.assign(PLANE, 0.0f);
    a.W_ext.assign(PLANE, 0.0f);
    a.created = now;
    a.updated = now;
    return a;
}

std::string member_uid(const Value& uid) {
    if (uid.kind() != Value::Kind::String) throw std::invalid_argument("bad pass uid " + pyjson::repr(uid));
    return uid.as_string();
}

std::vector<std::string> Accumulator::uids() const {
    std::vector<std::string> out;
    out.reserve(members.size());
    for (const Value& m : members) out.push_back(member_uid(m.at("uid")));
    return out;
}

std::optional<std::size_t> Accumulator::member_index(std::string_view uid) const {
    for (std::size_t i = 0; i < members.size(); ++i)
        if (equals(members[i].at("uid"), uid)) return i;
    return std::nullopt;
}

std::optional<Value> Accumulator::member(std::string_view uid) const {
    const auto i = member_index(uid);
    if (!i) return std::nullopt;
    return members[*i];
}

const Part* Accumulator::part(std::string_view sha) const {
    for (const Part& p : parts)
        if (p.sha == sha) return &p;
    return nullptr;
}

Accumulator::Sums Accumulator::sums(Store& store, std::optional<std::string_view> skip, bool ext) const {
    Sums s;
    if (ext) {
        s.S.assign(S_ext.begin(), S_ext.end());
        s.W.assign(W_ext.begin(), W_ext.end());
    } else {
        s.S.assign(PLANE, 0.0);
        s.W.assign(PLANE, 0.0);
    }
    for (const Value& m : members) {
        if (uid_matches(m.at("uid"), skip)) continue;
        const ZWPtr zw = store.pass_zw(member_uid(m.at("uid")));
        const auto idx = canonical_index(py_int(m.at("segment")), static_cast<std::int64_t>(zw->z.size()));
        const std::size_t base = static_cast<std::size_t>(py_int(m.at("segment"))) * GROUP_LATENTS;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const double w = zw->w[i];
            const std::size_t k = base + idx[i];
            s.S[k] += w > 0 ? w * zw->z[i] : 0.0;
            s.W[k] += w;
        }
    }
    return s;
}

void Accumulator::rebuild(Store& store) {
    const Sums s = sums(store);
    S.resize(PLANE);   // (in place: a binding's views of S and W stay valid)
    W.resize(PLANE);
    for (std::size_t i = 0; i < PLANE; ++i) {   // float64 -> float32, rounded to nearest
        S[i] = static_cast<float>(s.S[i]);
        W[i] = static_cast<float>(s.W[i]);
    }
}

std::pair<std::vector<double>, std::vector<double>> Accumulator::loo(std::string_view uid, Store& store) const {
    const std::optional<Value> m = member(uid);
    if (!m) {
        const std::string what = std::string(uid) + " is not a member of " + key_name(key);
        throw KeyError(what, "KeyError: " + pytext::repr(what));
    }
    const Sums s = sums(store, uid);
    const ZWPtr zw = store.pass_zw(std::string(uid));
    const auto idx = canonical_index(py_int(m->at("segment")), static_cast<std::int64_t>(zw->z.size()));
    const std::size_t base = static_cast<std::size_t>(py_int(m->at("segment"))) * GROUP_LATENTS;
    std::pair<std::vector<double>, std::vector<double>> out;
    out.first.resize(idx.size());
    out.second.resize(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        out.first[i] = s.S[base + idx[i]];
        out.second[i] = s.W[base + idx[i]];
    }
    return out;
}

bool Accumulator::has_data(std::int64_t g) const {
    if (g < -N_GROUPS || g >= N_GROUPS)
        throw std::out_of_range("index " + std::to_string(g) + " is out of bounds for axis 0 with size 3");
    const std::size_t base = static_cast<std::size_t>((g + N_GROUPS) % N_GROUPS) * GROUP_LATENTS;
    for (std::size_t i = 0; i < static_cast<std::size_t>(GROUP_LATENTS); ++i)
        if (W[base + i] > 0) return true;
    return false;
}

double Accumulator::median_w(std::int64_t g, std::int64_t n) const {
    const auto idx = canonical_index(g, n);
    const std::size_t base = static_cast<std::size_t>(g) * GROUP_LATENTS;
    std::vector<float> v(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) v[i] = W[base + idx[i]];
    return static_cast<double>(math::median(std::span<const float>(v)));
}

std::vector<std::optional<double>> Accumulator::seg_snr_db() const {
    std::vector<std::optional<double>> out;
    for (std::int64_t g = 0; g < mode + 1; ++g) {
        const double med = has_data(g) ? median_w(g) : 0.0;
        if (med > 0)
            out.emplace_back(10.0 * std::log10(med));
        else
            out.emplace_back(std::nullopt);
    }
    return out;
}

Value Accumulator::meta() const {
    Value m = Value::object();
    m.set("callsign", Value::string(key.callsign));
    m.set("picture_id", Value::integer(key.picture_id));
    m.set("mode", Value::integer(mode));
    m.set("codec_id", Value::integer(codec_id));
    m.set("created", Value::number(created));
    m.set("updated", Value::number(updated));
    m.set("members", Value::array(members));
    m.set("foreign", Value::array(foreign));
    return m;
}

// --- the store ---------------------------------------------------------------------------------

Store::Store(std::optional<std::string> root, Seams seams)
    : root_(root ? pathlib(*root) : default_root()), seams_(std::move(seams)) {
    for (const char* d : {"passes", "acc", "prov"}) {
        // (self.root / d).mkdir(parents=True, exist_ok=True): the OSError
        // names the directory whose own mkdir failed, this one or a parent
        // (--store /proc/st is "No such file or directory: '/proc/st'", as
        // an unwritable parent is its PermissionError)
        const std::string path = dir(d);
        std::error_code ec;
        std::string failed;
        if (!rawfile::make_dirs(path, ec, &failed)) file_error("cannot create", failed, util::errno_of(ec));
    }
    mu_ = root_mutex(root_);
}

std::string Store::dir(const char* name) const { return join(root_, name); }

double Store::now() const {
    if (seams_.clock) return seams_.clock();
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void Store::write_atomic(const std::string& path, std::span<const std::uint8_t> data, bool sync) const {
    const auto slash = path.find_last_of(
#ifdef _WIN32
        "/\\"
#else
        "/"
#endif
    );
    const std::string parent = path.substr(0, slash + 1);
    const std::string name = path.substr(slash + 1);
    const std::string tmp = parent + "." + name + "." + std::to_string(rawfile::process_id()) + "." +
                            std::to_string(rawfile::thread_ident()) + ".tmp";
    try {
        {
            util::File f = util::open_file(tmp, "wb");
            if (!f) file_error("cannot open", tmp, errno);
            if (!data.empty() && std::fwrite(data.data(), 1, data.size(), f.get()) != data.size())
                file_error("cannot write", tmp, errno ? errno : EIO);
            if (std::fflush(f.get()) != 0) file_error("cannot write", tmp, errno ? errno : EIO);
            if (sync) {
#ifdef _WIN32
                if (::_commit(::_fileno(f.get())) != 0) file_error("cannot sync", tmp, errno);
#else
                if (::fsync(::fileno(f.get())) != 0) file_error("cannot sync", tmp, errno);
#endif
            }
            std::FILE* raw = f.release();
            if (std::fclose(raw) != 0) file_error("cannot write", tmp, errno ? errno : EIO);
        }
        if (seams_.replace)
            seams_.replace(tmp, path);
        else
            default_replace(tmp, path);
    } catch (...) {
        std::error_code ec;
        util::remove_file(tmp, ec);
        throw;
    }
}

void Store::write_json(const std::string& path, const Value& v) const {
    const std::string text = pyjson::dumps(v, 1, true);
    write_atomic(path, std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()),
                 true);
}

void Store::forget(const std::string& uid) {
    if (const auto it = zw_index_.find(uid); it != zw_index_.end()) {
        zw_.erase(it->second);
        zw_index_.erase(it);
    }
    light_.erase(uid);
}

rawfile::Signature Store::signature(const std::string& path) const {
    std::error_code ec;
    const auto sig = rawfile::signature(path, ec);
    if (!sig) file_error("cannot stat", path, util::errno_of(ec));
    return *sig;
}

// --- passes ---

std::string Store::pass_path(std::string_view uid) const {
    if (uid.find('/') != std::string_view::npos || (!uid.empty() && uid.front() == '.'))
        throw std::invalid_argument("bad pass uid " + pytext::repr(uid));
    return join(dir("passes"), std::string(uid) + ".npz");
}

void Store::add_pass(const types::PassResult& p) {
    const auto lk = hold();
    const std::string path = pass_path(p.uid);
    const std::vector<std::uint8_t> data = p.bytes();
    write_atomic(path, data, false);
    forget(p.uid);
}

bool Store::has_pass(std::string_view uid) const { return util::exists(pass_path(uid)); }

types::PassResult Store::load_pass(std::string_view uid) const {
    return types::PassResult::load(pass_path(uid));
}

std::vector<std::string> Store::pass_uids() const {
    const auto lk = hold();
    std::vector<std::string> out;
    for (const std::string& name : glob_suffix(dir("passes"), ".npz"))
        out.push_back(name.size() > 4 ? name.substr(0, name.size() - 4) : name);
    std::sort(out.begin(), out.end());
    return out;
}

ZWPtr Store::pass_zw(const std::string& uid) {
    const auto lk = hold();
    const std::size_t cap = seams_.zw_cache ? seams_.zw_cache() : ZW_CACHE;
    const std::string path = pass_path(uid);
    const rawfile::Signature sig = signature(path);   // (before the read: a rewrite during it shows next time)
    if (const auto it = zw_index_.find(uid); it != zw_index_.end()) {
        if (it->second->second.sig == sig) {
            zw_.splice(zw_.end(), zw_, it->second);
            return it->second->second.zw;
        }
        zw_.erase(it->second);       // the file changed: read it again
        zw_index_.erase(it);
    }
    const npz::Npz d = npz::load_npz(path);
    const npz::Array& za = member_of(d, "z");
    const npz::Array& wa = member_of(d, "w");
    auto zw = std::make_shared<ZW>();
    zw->z = za.to_double();
    zw->w = wa.to_double();
    if (za.shape != wa.shape || za.ndim() != 1)
        throw std::invalid_argument("pass " + uid + ": z" + shape_repr(za.shape) + " and w" + shape_repr(wa.shape) +
                                    " do not match");
    ZWPtr out = zw;
    zw_.emplace_back(uid, Cached{out, sig});
    zw_index_[uid] = std::prev(zw_.end());
    while (zw_.size() > cap) {
        zw_index_.erase(zw_.front().first);
        zw_.pop_front();
    }
    return out;
}

std::vector<std::string> Store::zw_cached() const {
    const auto lk = hold();
    std::vector<std::string> out;
    for (const auto& e : zw_) out.push_back(e.first);
    return out;
}

std::vector<std::pair<std::string, ZWPtr>> Store::zw_items() const {
    const auto lk = hold();
    std::vector<std::pair<std::string, ZWPtr>> out;
    for (const auto& e : zw_) out.emplace_back(e.first, e.second.zw);
    return out;
}

PassInfo Store::pass_info(const std::string& uid) {
    const auto lk = hold();
    const std::string path = pass_path(uid);
    const rawfile::Signature sig = signature(path);
    if (const auto it = light_.find(uid); it != light_.end() && it->second.sig == sig) return it->second.info;
    const npz::Npz d = npz::load_npz(path);
    const npz::Array& ma = member_of(d, "meta");
    check_utf8(ma.bytes, path);
    const Value meta =
        pyjson::loads(std::string_view(reinterpret_cast<const char*>(ma.bytes.data()), ma.bytes.size()));
    PassInfo info;
    info.hdr_llr = member_of(d, "hdr_llr").to_double();
    info.uid = uid;
    info.q = py_int(meta.at("q"));
    info.f_hz = py_float(meta.at("f_hz"));
    info.header = meta.at("header");
    info.z_ref = py_float(meta.at("report").at("z_ref"));
    info.em_round = py_int(meta.at("em_round"));
    const Value* wf = meta.find("waveform");
    info.waveform = wf ? py_int(*wf) : 0;
    const Value* fr = meta.find("frame");
    info.frame = fr ? *fr : Value::null();
    light_.insert_or_assign(uid, LightEntry{info, sig});
    return info;
}

std::vector<Key> Store::update_pass(const types::PassResult& p) {
    const auto lk = hold();
    add_pass(p);
    std::vector<Key> keys;
    for (const Key& key : open_keys()) {
        Accumulator acc = accumulator(key);
        const auto i = acc.member_index(p.uid);
        if (!i) continue;
        acc.members[*i].set("em_round", Value::integer(p.em_round));
        acc.members[*i].set("mean_w_db", Value::number(mean_w_db(std::span<const float>(p.w))));
        commit(acc);
        keys.push_back(key);
    }
    return keys;
}

// --- accumulators ---

std::pair<std::string, std::string> Store::acc_paths(const Key& key) const {
    const std::string stem = join(dir("acc"), key_name(key));
    return {stem + ".json", stem + ".npz"};
}

bool Store::has_accumulator(const Key& key) const { return util::exists(acc_paths(key).first); }

Accumulator Store::accumulator(const Key& key) const {
    const auto lk = hold();
    const auto [jpath, npath] = acc_paths(key);
    if (!util::exists(jpath)) {
        const std::string what = "no accumulator " + key_name(key);
        throw KeyError(what, "KeyError: " + pytext::repr(what));
    }
    const Value meta = read_json(jpath);
    const npz::Npz d = npz::load_npz(npath);
    Accumulator acc;
    acc.S = plane_of(member_of(d, "S"), "S");
    acc.W = plane_of(member_of(d, "W"), "W");
    acc.S_ext = plane_of(member_of(d, "S_ext"), "S_ext");
    acc.W_ext = plane_of(member_of(d, "W_ext"), "W_ext");
    for (const Value& f : meta.at("foreign").items()) {
        const std::string sha = py_str(f.at("sha"));
        if (!d.contains("Sx_" + sha)) continue;
        Part part{sha, member_of(d, "Sx_" + sha), member_of(d, "Wx_" + py_str(f.at("sha")))};
        bool replaced = false;
        for (Part& p : acc.parts)
            if (p.sha == sha) {
                p = part;
                replaced = true;
            }
        if (!replaced) acc.parts.push_back(std::move(part));
    }
    acc.key = Key{meta.at("callsign").as_string("callsign"), py_int(meta.at("picture_id"))};
    acc.mode = py_int(meta.at("mode"));
    acc.codec_id = py_int(meta.at("codec_id"));
    acc.members = meta.at("members").items();
    acc.foreign = meta.at("foreign").items();
    acc.created = py_float(meta.at("created"));
    acc.updated = py_float(meta.at("updated"));
    return acc;
}

Accumulator Store::create(const Key& key, std::int64_t mode, std::int64_t codec_id) {
    const auto lk = hold();
    if (has_accumulator(key)) return accumulator(key);
    Accumulator acc = Accumulator::empty(key, mode, codec_id, now());
    save_accumulator(acc);
    return acc;
}

void Store::save_accumulator(const Accumulator& acc) {
    const auto lk = hold();
    const auto [jpath, npath] = acc_paths(acc.key);
    const std::vector<std::size_t> shape{static_cast<std::size_t>(N_GROUPS), static_cast<std::size_t>(GROUP_LATENTS)};
    for (const std::vector<float>* v : {&acc.S, &acc.W, &acc.S_ext, &acc.W_ext})
        if (v->size() != PLANE) throw std::invalid_argument("an accumulator's planes hold 3 x 52800 values");
    npz::Npz z;
    z.add("S", npz::Array::of(acc.S, shape));
    z.add("W", npz::Array::of(acc.W, shape));
    z.add("S_ext", npz::Array::of(acc.S_ext, shape));
    z.add("W_ext", npz::Array::of(acc.W_ext, shape));
    for (const Part& p : acc.parts) {
        z.add("Sx_" + p.sha, p.S);
        z.add("Wx_" + p.sha, p.W);
    }
    write_atomic(npath, npz::npz_bytes(z), true);
    write_json(jpath, acc.meta());
}

void Store::commit(Accumulator& acc) {
    const auto lk = hold();
    acc.rebuild(*this);
    acc.updated = now();
    save_accumulator(acc);
}

std::vector<Key> Store::open_keys() const {
    const auto lk = hold();
    std::vector<std::string> names = glob_suffix(dir("acc"), ".json");
    sort_paths(names);
    std::vector<Key> keys;
    for (const std::string& n : names) {
        const Value meta = read_json(join(dir("acc"), n));
        keys.push_back(Key{meta.at("callsign").as_string("callsign"), py_int(meta.at("picture_id"))});
    }
    return keys;
}

std::vector<Due> Store::due(std::int64_t q) {
    const auto lk = hold();
    std::vector<std::string> names = glob_suffix(dir("acc"), ".json");
    sort_paths(names);
    std::vector<Due> out;
    for (const std::string& n : names) {
        std::vector<Due> rows;   // (an accumulator that fails part way gives none)
        try {
            const Value meta = read_json(join(dir("acc"), n));
            std::unordered_set<std::int64_t> have;
            for (const Value& m : meta.at("members").items()) have.insert(py_int(m.at("segment")));
            for (const Value& m : meta.at("members").items()) {
                const std::int64_t k = py_int(m.at("segment"));
                const PassInfo info = pass_info(member_uid(m.at("uid")));
                for (std::int64_t g = k + 1; g < py_int(meta.at("mode")) + 1; ++g) {
                    if (!have.count(g) && info.q + 2 * (g - k) == q)
                        rows.push_back(Due{Key{meta.at("callsign").as_string("callsign"), py_int(meta.at("picture_id"))},
                                           g, info.f_hz, info.q, info.waveform, info.frame});
                }
            }
        } catch (const FileError&) {          // OSError: an accumulator being written,
            continue;                         // or a pass expired
        } catch (const KeyError&) {
            continue;
        } catch (const std::invalid_argument&) {   // ValueError
            continue;
        } catch (const math::TypeError&) {        // TypeError: a hand-edited file whose
            continue;                             // values are not the types it holds
        } catch (const std::overflow_error&) {    // OverflowError: int() of an infinity, or
            continue;                             // (here) an integer past int64 in the file
        }
        out.insert(out.end(), std::make_move_iterator(rows.begin()), std::make_move_iterator(rows.end()));
    }
    return out;
}

std::optional<Key> Store::find_member(std::string_view uid) const {
    const auto lk = hold();
    for (const Key& key : open_keys()) {
        const Value meta = read_json(acc_paths(key).first);
        for (const Value& m : meta.at("members").items())
            if (equals(m.at("uid"), uid)) return key;
    }
    return std::nullopt;
}

Accumulator Store::attach(const Key& key, const std::string& uid, std::int64_t segment, const std::string& method,
                          std::optional<std::int64_t> mode, std::optional<std::int64_t> codec_id) {
    const auto lk = hold();
    if (!has_pass(uid)) {
        const std::string what = "no pass " + uid;
        throw KeyError(what, "KeyError: " + pytext::repr(what));
    }
    if (!has_accumulator(key)) {
        const Value hdr = pass_info(uid).header;
        if (!mode) mode = truthy(hdr) ? py_int(hdr.at("mode")) : segment;
        if (!codec_id) codec_id = truthy(hdr) ? py_int(hdr.at("codec_id")) : 0;
        create(key, *mode, *codec_id);
    }
    Accumulator acc = accumulator(key);
    if (!(0 <= segment && segment <= acc.mode))
        throw std::invalid_argument("segment " + std::to_string(segment) + " is not in mode " +
                                    std::to_string(acc.mode));
    const std::optional<Key> other = find_member(uid);
    if (other && !(*other == acc.key)) detach(*other, uid);
    prov_remove(uid);
    const PassInfo info = pass_info(uid);
    const ZWPtr zw = pass_zw(uid);
    Value rec = Value::object();
    rec.set("uid", Value::string(uid));
    rec.set("segment", Value::integer(segment));
    rec.set("method", Value::string(method));
    rec.set("z_ref", Value::number(info.z_ref));
    rec.set("mean_w_db", Value::number(mean_w_db(std::span<const double>(zw->w))));
    rec.set("em_round", Value::integer(info.em_round));
    // One transmission received twice is one member: whichever copy carries more.
    const std::optional<Value> tw = twin(acc, info.q, info.f_hz, segment, uid, info.waveform);
    if (tw) {
        const bool keep_twin = py_float(tw->at("mean_w_db")) >= py_float(rec.at("mean_w_db"));
        const Value& keep = keep_twin ? *tw : rec;
        const Value& drop = keep_twin ? rec : *tw;
        Value ev = Value::object();
        ev.set("event", Value::string("duplicate"));
        ev.set("key", key_list(acc.key));
        ev.set("uid", drop.at("uid"));
        ev.set("kept", keep.at("uid"));
        log(ev);
        const Value drop_uid = drop.at("uid");
        std::vector<Value> kept;
        for (Value& m : acc.members)
            if (!py_equal(m.at("uid"), drop_uid)) kept.push_back(std::move(m));
        acc.members = std::move(kept);
        if (keep_twin) {
            commit(acc);   // (uid may have been a member already)
            return acc;
        }
    }
    if (const auto i = acc.member_index(uid)) {
        for (std::size_t k = 0; k < rec.keys().size(); ++k) acc.members[*i].set(rec.keys()[k], rec.values()[k]);
    } else {
        acc.members.push_back(rec);
    }
    commit(acc);
    return acc;
}

std::optional<Value> Store::twin(const Accumulator& acc, std::int64_t q, double f_hz, std::int64_t segment,
                                 std::optional<std::string_view> uid, std::optional<std::int64_t> waveform) {
    const auto lk = hold();
    for (const Value& m : acc.members) {
        if (uid_matches(m.at("uid"), uid) || py_int(m.at("segment")) != segment) continue;
        const PassInfo o = pass_info(member_uid(m.at("uid")));
        double f_o = o.f_hz, f = f_hz;
        if (waveform && o.waveform != *waveform) {
            constexpr double off = rxconfig::receiver::L_EXCLUDE_OFFSET_HZ;
            f_o += o.waveform == config::WAVEFORM_L ? off : 0.0;
            f += *waveform == config::WAVEFORM_L ? off : 0.0;
        }
        if (o.q == q && std::abs(f_o - f) < SAME_PASS_HZ) return m;
    }
    return std::nullopt;
}

Accumulator Store::detach(const Key& key, std::string_view uid) {
    const auto lk = hold();
    Accumulator acc = accumulator(key);
    std::vector<Value> kept;
    for (Value& m : acc.members)
        if (!equals(m.at("uid"), uid)) kept.push_back(std::move(m));
    acc.members = std::move(kept);
    commit(acc);
    return acc;
}

void Store::delete_accumulator(const Key& key) {
    const auto lk = hold();
    const auto [jpath, npath] = acc_paths(key);
    remove_or_throw(jpath, true);
    remove_or_throw(npath, true);
}

bool Store::merge(const Key& key, std::span<const float> S, std::span<const float> W, const std::string& source) {
    const auto lk = hold();
    for (const auto* x : {&S, &W})
        if (x->size() != PLANE)
            throw std::invalid_argument("cannot reshape array of size " + std::to_string(x->size()) +
                                        " into shape (3,52800)");
    for (std::size_t i = 0; i < PLANE; ++i)
        if (!std::isfinite(S[i])) return false;
    for (std::size_t i = 0; i < PLANE; ++i)
        if (!std::isfinite(W[i])) return false;
    for (std::size_t i = 0; i < PLANE; ++i)
        if (W[i] < 0) return false;
    Accumulator acc = accumulator(key);
    util::Sha256 h;
    h.update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(S.data()), S.size_bytes()));
    h.update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(W.data()), W.size_bytes()));
    const util::Sha256Digest dg = h.digest();
    const std::string digest = util::to_hex(dg).substr(0, 16);
    for (const Value& f : acc.foreign) {
        const Value* s = f.find("sha");
        if (s && equals(*s, digest)) return false;
    }
    std::vector<std::size_t> stale;
    for (std::size_t i = 0; i < acc.foreign.size(); ++i) {
        const Value* s = acc.foreign[i].find("source");
        if (!source.empty() && s && equals(*s, source)) stale.push_back(i);
    }
    const Accumulator::Sums mine = acc.sums(*this, std::nullopt, false);
    std::vector<double> Wf(PLANE), mf(PLANE, 0.0), mm(PLANE, 0.0);
    for (std::size_t i = 0; i < PLANE; ++i) {
        Wf[i] = W[i];
        if (Wf[i] > 0) mf[i] = static_cast<double>(S[i]) / Wf[i];
        if (mine.W[i] > 0) mm[i] = mine.S[i] / mine.W[i];
    }
    const associate::Corr c = associate::corr_stat(mf, Wf, mm, mine.W);
    const double t_exp = associate::expected_t(Wf, mine.W);
    const bool ok = associate::corr_accept(c.t, t_exp);
    Value ev = Value::object();
    ev.set("event", Value::string("merge"));
    ev.set("key", key_list(acc.key));
    ev.set("source", Value::string(source));
    ev.set("t", Value::number(c.t));
    ev.set("M", Value::integer(c.M));
    ev.set("t_expected", Value::number(t_exp));
    // an np.bool_ (t an np.float64) goes through default=float
    ev.set("accepted", Value::boolean(ok));
    log(ev);
    if (!ok) return false;
    std::vector<Value> stale_entries;
    for (std::size_t i : stale) stale_entries.push_back(acc.foreign[i]);
    // list.remove(f): the first entry that is f or == f
    for (std::size_t k = 0; k < stale.size(); ++k) {
        const Value& f = stale_entries[k];
        for (std::size_t i = 0; i < acc.foreign.size(); ++i) {
            if (py_equal(acc.foreign[i], f)) {
                acc.foreign.erase(acc.foreign.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        const std::string sha = py_str(f.at("sha"));
        acc.parts.erase(std::remove_if(acc.parts.begin(), acc.parts.end(),
                                       [&](const Part& p) { return p.sha == sha; }),
                        acc.parts.end());
    }
    Value entry = Value::object();
    entry.set("source", Value::string(source));
    entry.set("sha", Value::string(digest));
    entry.set("t", Value::number(c.t));
    entry.set("M", Value::integer(c.M));
    entry.set("merged", Value::number(now()));
    Value replaced = Value::array();
    for (const Value& f : stale_entries) replaced.push_back(f.at("sha"));
    entry.set("replaced", std::move(replaced));
    acc.foreign.push_back(std::move(entry));
    const std::vector<std::size_t> shape{static_cast<std::size_t>(N_GROUPS), static_cast<std::size_t>(GROUP_LATENTS)};
    Part part{digest, npz::Array::of(S, shape), npz::Array::of(W, shape)};
    bool replaced_part = false;
    for (Part& p : acc.parts)
        if (p.sha == digest) {
            p = part;
            replaced_part = true;
        }
    if (!replaced_part) acc.parts.push_back(std::move(part));
    std::vector<double> S_ext(PLANE, 0.0), W_ext(PLANE, 0.0);
    for (const Value& f : acc.foreign) {
        const std::string sha = py_str(f.at("sha"));
        const Part* p = acc.part(sha);
        if (!p) throw KeyError(sha, "KeyError: " + pytext::repr(sha));
        const std::vector<double> Sx = plane_double(p->S, "Sx_" + sha);
        const std::vector<double> Wx = plane_double(p->W, "Wx_" + sha);
        for (std::size_t i = 0; i < PLANE; ++i) {
            S_ext[i] += Sx[i];
            W_ext[i] += Wx[i];
        }
    }
    for (std::size_t i = 0; i < PLANE; ++i) {
        acc.S_ext[i] = static_cast<float>(S_ext[i]);
        acc.W_ext[i] = static_cast<float>(W_ext[i]);
    }
    commit(acc);
    return true;
}

// --- provisional clusters ---

std::string Store::prov_path(std::string_view cid) const { return join(dir("prov"), std::string(cid) + ".json"); }

std::vector<Value> Store::prov_clusters() const {
    const auto lk = hold();
    std::vector<std::string> names = glob_suffix(dir("prov"), ".json");
    sort_paths(names);
    std::vector<Value> out;
    for (const std::string& n : names) out.push_back(read_json(join(dir("prov"), n)));
    return out;
}

std::optional<Value> Store::prov_cluster_of(std::string_view uid) const {
    const auto lk = hold();
    for (const Value& c : prov_clusters())
        if (contains(c.at("members"), uid)) return c;
    return std::nullopt;
}

std::string Store::prov_add(const std::string& uid, std::optional<std::string> cluster) {
    const auto lk = hold();
    if (cluster) {
        const Value c = read_json(prov_path(*cluster));
        if (contains(c.at("members"), uid)) return *cluster;
    }
    prov_remove(uid);
    const double t = now();
    Value c;
    if (!cluster) {
        cluster = "c_" + uid;
        for (std::int64_t n = 2; util::exists(prov_path(*cluster)); ++n)   // (never over another cluster)
            cluster = "c_" + uid + "_" + std::to_string(n);
        c = Value::object();
        c.set("cluster", Value::string(*cluster));
        c.set("created", Value::number(t));
        c.set("members", Value::array());
    } else {
        c = read_json(prov_path(*cluster));
    }
    Value* members = c.find("members");
    if (!members) throw KeyError("members", "KeyError: 'members'");
    if (members->kind() != Value::Kind::Array)
        throw math::TypeError(std::string("'") + type_name(*members) + "' object has no attribute 'append'");
    members->push_back(Value::string(uid));
    c.set("updated", Value::number(t));
    write_json(prov_path(*cluster), c);
    return *cluster;
}

void Store::prov_remove(std::string_view uid) {
    const auto lk = hold();
    std::optional<Value> c = prov_cluster_of(uid);
    if (!c) return;
    std::vector<Value> rest;
    for (const Value& u : c->at("members").items())
        if (!equals(u, uid)) rest.push_back(u);
    const bool any = !rest.empty();
    c->set("members", Value::array(std::move(rest)));
    if (any) {
        c->set("updated", Value::number(now()));
        write_json(prov_path(py_str(c->at("cluster"))), *c);
    } else {
        remove_or_throw(prov_path(py_str(c->at("cluster"))), false);
    }
}

// --- housekeeping ---

void Store::log(const Value& event) {
    const auto lk = hold();
    Value e = Value::object();
    e.set("time", Value::number(now()));
    for (std::size_t k = 0; k < event.keys().size(); ++k) e.set(event.keys()[k], event.values()[k]);
    const std::string path = join(root_, "log.jsonl");
    util::File f = util::open_file(path, "a");
    if (!f) file_error("cannot open", path, errno);
    const std::string line = pyjson::dumps(e) + "\n";
    if (std::fwrite(line.data(), 1, line.size(), f.get()) != line.size())
        file_error("cannot write", path, errno ? errno : EIO);
    std::FILE* raw = f.release();
    if (std::fclose(raw) != 0) file_error("cannot write", path, errno ? errno : EIO);
}

Expired Store::expire(std::optional<double> now_, double days) {
    const auto lk = hold();
    const double t = now_ ? *now_ : now();
    const double cutoff = t - days * DAY_S;
    Expired removed;
    std::unordered_set<std::string> held;
    for (const Key& key : open_keys()) {
        const Accumulator acc = accumulator(key);
        if (acc.updated < cutoff) {
            delete_accumulator(key);
            removed.acc.push_back(key);
        } else {
            for (const std::string& u : acc.uids()) held.insert(u);
        }
    }
    for (const Value& c : prov_clusters()) {
        const Value& upd = c.at("updated");
        if (!numeric(upd))
            throw math::TypeError(std::string("'<' not supported between instances of '") + type_name(upd) +
                                  "' and 'float'");
        if (number_of(upd) < cutoff) {
            remove_or_throw(prov_path(py_str(c.at("cluster"))), false);
            removed.prov.push_back(c.at("cluster"));
        } else {
            for (const Value& u : c.at("members").items())
                if (u.kind() == Value::Kind::String) held.insert(u.as_string());
        }
    }
    for (const std::string& uid : pass_uids()) {
        if (held.count(uid)) continue;
        if (slot_time(pass_info(uid).q) < cutoff) {
            remove_or_throw(pass_path(uid), false);
            forget(uid);
            removed.passes.push_back(uid);
        }
    }
    return removed;
}

}  // namespace sstvae::qrss::store
