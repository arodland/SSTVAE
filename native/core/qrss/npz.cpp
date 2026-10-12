#include "qrss/npz.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <type_traits>

#include "qrss/crc32.hpp"

namespace sstvae::qrss::npz {

namespace {

// --- dtypes ---------------------------------------------------------------------------

struct DTypeInfo {
    DType t;
    const char* descr;
    std::size_t size;
};

constexpr std::array<DTypeInfo, 14> DTYPES = {{
    {DType::f2, "<f2", 2},  {DType::f4, "<f4", 4},   {DType::f8, "<f8", 8},  {DType::c8, "<c8", 8},
    {DType::c16, "<c16", 16}, {DType::i1, "|i1", 1}, {DType::u1, "|u1", 1},  {DType::i2, "<i2", 2},
    {DType::u2, "<u2", 2},  {DType::i4, "<i4", 4},   {DType::u4, "<u4", 4},  {DType::i8, "<i8", 8},
    {DType::u8, "<u8", 8},  {DType::b1, "|b1", 1},
}};

const DTypeInfo& info(DType t) {
    for (const auto& d : DTYPES)
        if (d.t == t) return d;
    throw std::invalid_argument("npz: unknown dtype");
}

[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument(what); }

std::size_t checked_product(const std::vector<std::size_t>& shape, std::size_t item) {
    std::size_t n = item;
    for (const std::size_t d : shape) {
        if (d != 0 && n > std::numeric_limits<std::size_t>::max() / d) bad("npy: array size overflows");
        n *= d;
    }
    return n;
}

// --- little-endian bytes ---------------------------------------------------------------

void put16(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
}
void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    put16(out, v & 0xffffu);
    put16(out, v >> 16);
}
void put64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    put32(out, static_cast<std::uint32_t>(v));
    put32(out, static_cast<std::uint32_t>(v >> 32));
}
void put(std::vector<std::uint8_t>& out, std::string_view s) { out.insert(out.end(), s.begin(), s.end()); }

std::uint64_t get_le(std::span<const std::uint8_t> b, std::size_t at, int n, const char* what) {
    if (at > b.size() || b.size() - at < static_cast<std::size_t>(n)) bad(std::string("npz: truncated ") + what);
    std::uint64_t v = 0;
    for (int i = n - 1; i >= 0; --i) v = (v << 8) | b[at + static_cast<std::size_t>(i)];
    return v;
}
std::uint32_t get16(std::span<const std::uint8_t> b, std::size_t at, const char* what) {
    return static_cast<std::uint32_t>(get_le(b, at, 2, what));
}
std::uint32_t get32(std::span<const std::uint8_t> b, std::size_t at, const char* what) {
    return static_cast<std::uint32_t>(get_le(b, at, 4, what));
}
std::uint64_t get64(std::span<const std::uint8_t> b, std::size_t at, const char* what) {
    return get_le(b, at, 8, what);
}
bool sig_at(std::span<const std::uint8_t> b, std::size_t at, const char* sig) {
    return at <= b.size() && b.size() - at >= 4 && std::memcmp(b.data() + at, sig, 4) == 0;
}

// --- the .npy header: a Python literal dict ---------------------------------------------

struct Value {
    enum Kind { Str, Int, Bool, None, Tuple, List, Dict } kind = None;
    std::string text;   // Str: the string; Int: the digits, with a leading '-' if negative
    bool flag = false;  // Bool
    std::vector<Value> items;                       // Tuple, List
    std::vector<std::pair<Value, Value>> entries;   // Dict
};

class Literal {
public:
    explicit Literal(std::string_view s) : s_(s) {}

    Value parse() {
        Value v = value();
        space();
        if (i_ != s_.size()) fail("trailing text");
        return v;
    }

private:
    [[noreturn]] void fail(const char* why) const {
        bad(std::string("npy: cannot parse header (") + why + "): " + std::string(s_));
    }
    void space() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r' ||
                                  s_[i_] == '\f' || s_[i_] == '\v'))
            ++i_;
    }
    bool eat(char c) {
        space();
        if (i_ < s_.size() && s_[i_] == c) {
            ++i_;
            return true;
        }
        return false;
    }
    bool word(std::string_view w) {
        if (s_.substr(i_, w.size()) != w) return false;
        const std::size_t e = i_ + w.size();
        if (e < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[e])) || s_[e] == '_')) return false;
        i_ = e;
        return true;
    }
    Value value() {
        space();
        if (i_ >= s_.size()) fail("unexpected end");
        const char c = s_[i_];
        Value v;
        if (c == '\'' || c == '"') {
            v.kind = Value::Str;
            v.text = string(c);
        } else if (c == '(' || c == '[') {
            const char close = c == '(' ? ')' : ']';
            ++i_;
            v.kind = c == '(' ? Value::Tuple : Value::List;
            bool comma = false;
            while (!eat(close)) {
                if (!v.items.empty() && !comma) fail("expected ','");
                v.items.push_back(value());
                comma = eat(',');
            }
            // "(x)" is x itself, not a tuple.
            if (c == '(' && v.items.size() == 1 && !comma) return v.items[0];
        } else if (c == '{') {
            ++i_;
            v.kind = Value::Dict;
            bool comma = false;
            while (!eat('}')) {
                if (!v.entries.empty() && !comma) fail("expected ','");
                Value k = value();
                if (!eat(':')) fail("expected ':'");
                Value val = value();
                // Python keeps the last of repeated keys.
                bool replaced = false;
                for (auto& e : v.entries)
                    if (e.first.kind == k.kind && e.first.text == k.text && e.first.flag == k.flag &&
                        k.kind != Value::Tuple && k.kind != Value::List && k.kind != Value::Dict) {
                        e.second = val;
                        replaced = true;
                    }
                if (!replaced) v.entries.emplace_back(std::move(k), std::move(val));
                comma = eat(',');
            }
        } else if (word("True")) {
            v.kind = Value::Bool;
            v.flag = true;
        } else if (word("False")) {
            v.kind = Value::Bool;
        } else if (word("None")) {
            v.kind = Value::None;
        } else if (c == '-' || c == '+' || std::isdigit(static_cast<unsigned char>(c))) {
            bool neg = false;
            while (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) {
                neg ^= s_[i_] == '-';
                ++i_;
                space();
            }
            if (!(i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_])))) fail("bad number");
            std::string digits;
            while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '_')) {
                if (s_[i_] != '_') digits += s_[i_];
                ++i_;
            }
            if (i_ < s_.size() && (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == 'j' ||
                                   s_[i_] == 'J' || s_[i_] == 'x' || s_[i_] == 'X'))
                fail("only decimal integers are expected");
            v.kind = Value::Int;
            v.text = (neg ? "-" : "") + digits;
        } else {
            fail("unexpected character");
        }
        return v;
    }
    std::string string(char q) {
        ++i_;
        std::string out;
        while (i_ < s_.size() && s_[i_] != q) {
            char ch = s_[i_++];
            if (ch == '\n') fail("newline in string");
            if (ch == '\\') {
                if (i_ >= s_.size()) fail("unterminated string");
                const char e = s_[i_++];
                switch (e) {
                    case 'n': ch = '\n'; break;
                    case 't': ch = '\t'; break;
                    case 'r': ch = '\r'; break;
                    case '0': ch = '\0'; break;
                    case '\\': case '\'': case '"': ch = e; break;
                    case 'x': {
                        if (i_ + 2 > s_.size() || !std::isxdigit(static_cast<unsigned char>(s_[i_])) ||
                            !std::isxdigit(static_cast<unsigned char>(s_[i_ + 1])))
                            fail("bad \\x escape");
                        ch = static_cast<char>(std::stoi(std::string(s_.substr(i_, 2)), nullptr, 16));
                        i_ += 2;
                        break;
                    }
                    default: out += '\\'; ch = e; break;
                }
            }
            out += ch;
        }
        if (i_ >= s_.size()) fail("unterminated string");
        ++i_;
        return out;
    }

    std::string_view s_;
    std::size_t i_ = 0;
};

constexpr std::size_t MAX_HEADER_SIZE = 10000;   // numpy's _MAX_HEADER_SIZE
constexpr std::size_t ARRAY_ALIGN = 64;
constexpr std::size_t GROWTH_AXIS_MAX_DIGITS = 21;
const std::string_view MAGIC("\x93NUMPY", 6);

std::string shape_repr(const std::vector<std::size_t>& shape) {
    if (shape.empty()) return "()";
    std::string s = "(";
    for (std::size_t k = 0; k < shape.size(); ++k) {
        if (k) s += ", ";
        s += std::to_string(shape[k]);
    }
    if (shape.size() == 1) s += ",";
    return s + ")";
}

std::vector<std::uint8_t> fortran_to_c(const std::vector<std::uint8_t>& f, const std::vector<std::size_t>& shape,
                                       std::size_t item) {
    if (shape.size() < 2) return f;
    const std::size_t n = f.size() / item;
    std::vector<std::uint8_t> c(f.size());
    std::vector<std::size_t> idx(shape.size(), 0);
    for (std::size_t k = 0; k < n; ++k) {   // k walks C order
        std::size_t off = 0, stride = 1;
        for (std::size_t d = 0; d < shape.size(); ++d) {
            off += idx[d] * stride;
            stride *= shape[d];
        }
        std::memcpy(c.data() + k * item, f.data() + off * item, item);
        for (std::size_t d = shape.size(); d-- > 0;) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
    return c;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) throw std::runtime_error("cannot read " + path);
    return data;
}

void write_file(const std::string& path, const std::vector<std::uint8_t>& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    out.close();
    if (!out) throw std::runtime_error("cannot write " + path);
}

// --- zip -------------------------------------------------------------------------------

constexpr std::uint64_t ZIP64_LIMIT = (std::uint64_t(1) << 31) - 1;
constexpr std::size_t ZIP_FILECOUNT_LIMIT = 0xffff;
constexpr std::uint32_t EXTERNAL_ATTR = 0600u << 16;
constexpr std::uint32_t VERSION_ZIP64 = 45;
constexpr std::uint32_t CREATE_SYSTEM_UNIX = 3;
constexpr std::size_t SIZE_FILE_HEADER = 30, SIZE_CENTRAL_DIR = 46, SIZE_END = 22, SIZE_END64 = 56,
                      SIZE_END64_LOCATOR = 20, MAX_COMMENT = 0xffff;

}  // namespace

// --- Array ---------------------------------------------------------------------------------

std::string_view descr(DType t) { return info(t).descr; }
std::size_t itemsize(DType t) { return info(t).size; }

std::optional<DType> from_descr(std::string_view d) {
    for (const auto& x : DTYPES)
        if (d == x.descr) return x.t;
    // A one-byte type has no byte order; numpy writes '|', but any prefix means the same.
    if (d.size() == 3 && (d[0] == '<' || d[0] == '>' || d[0] == '=') &&
        (d.substr(1) == "i1" || d.substr(1) == "u1" || d.substr(1) == "b1"))
        return from_descr(std::string("|") + std::string(d.substr(1)));
    return std::nullopt;
}

std::size_t Array::size() const {
    std::size_t n = 1;
    for (const std::size_t d : shape) n *= d;
    return n;
}

void Array::set_shape(std::size_t n, std::optional<std::vector<std::size_t>> s) {
    if (!s) {
        shape = {n};
        return;
    }
    shape = std::move(*s);
    if (checked_product(shape, 1) != n)
        bad("npz: shape " + shape_repr(shape) + " does not hold " + std::to_string(n) + " values");
}

Array Array::of_bools(std::span<const std::uint8_t> v, std::optional<std::vector<std::size_t>> s) {
    Array a;
    a.dtype = DType::b1;
    a.set_shape(v.size(), std::move(s));
    a.bytes.resize(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) a.bytes[i] = v[i] ? 1 : 0;
    return a;
}

void Array::check_dtype(DType want) const {
    if (dtype != want)
        bad("npz: array is " + std::string(descr(dtype)) + ", not " + std::string(descr(want)));
}

std::vector<std::uint8_t> Array::bools() const {
    check_dtype(DType::b1);
    std::vector<std::uint8_t> out(bytes.size());
    for (std::size_t i = 0; i < bytes.size(); ++i) out[i] = bytes[i] ? 1 : 0;
    return out;
}

namespace {

template <typename T>
void widen(const Array& a, std::vector<double>& out) {
    const std::size_t n = a.size();
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        T v;
        std::memcpy(&v, a.bytes.data() + i * sizeof(T), sizeof(T));
        out[i] = static_cast<double>(v);
    }
}

template <typename T>
std::int64_t one_int(const Array& a) {
    T v;
    std::memcpy(&v, a.bytes.data(), sizeof(T));
    if constexpr (std::is_same_v<T, std::uint64_t>) {
        if (v > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            bad("npz: value does not fit int64");
    }
    return static_cast<std::int64_t>(v);
}

}  // namespace

std::vector<double> Array::to_double() const {
    std::vector<double> out;
    switch (dtype) {
        case DType::f2: {
            const std::size_t n = size();
            out.resize(n);
            for (std::size_t i = 0; i < n; ++i)
                out[i] = half::to_double(static_cast<std::uint16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8)));
            break;
        }
        case DType::f4: widen<float>(*this, out); break;
        case DType::f8: widen<double>(*this, out); break;
        case DType::i1: widen<std::int8_t>(*this, out); break;
        case DType::u1: widen<std::uint8_t>(*this, out); break;
        case DType::i2: widen<std::int16_t>(*this, out); break;
        case DType::u2: widen<std::uint16_t>(*this, out); break;
        case DType::i4: widen<std::int32_t>(*this, out); break;
        case DType::u4: widen<std::uint32_t>(*this, out); break;
        case DType::i8: widen<std::int64_t>(*this, out); break;
        case DType::u8: widen<std::uint64_t>(*this, out); break;
        case DType::b1: {
            out.resize(bytes.size());
            for (std::size_t i = 0; i < bytes.size(); ++i) out[i] = bytes[i] ? 1.0 : 0.0;
            break;
        }
        case DType::c8:
        case DType::c16: bad("npz: a complex array has no float64 value");
    }
    return out;
}

std::int64_t Array::to_int64() const {
    if (size() != 1) bad("npz: expected one value, the array has " + std::to_string(size()));
    switch (dtype) {
        case DType::i1: return one_int<std::int8_t>(*this);
        case DType::u1: return one_int<std::uint8_t>(*this);
        case DType::i2: return one_int<std::int16_t>(*this);
        case DType::u2: return one_int<std::uint16_t>(*this);
        case DType::i4: return one_int<std::int32_t>(*this);
        case DType::u4: return one_int<std::uint32_t>(*this);
        case DType::i8: return one_int<std::int64_t>(*this);
        case DType::u8: return one_int<std::uint64_t>(*this);
        case DType::b1: return bytes[0] ? 1 : 0;
        default: bad("npz: expected an integer array, not " + std::string(descr(dtype)));
    }
}

// --- Npz -----------------------------------------------------------------------------------

void Npz::add(std::string name, Array a) {
    if (name.empty()) bad("npz: empty member name");
    if (name.find('\0') != std::string::npos) bad("npz: member name contains NUL");
    if (contains(name)) bad("npz: duplicate member " + name);
    if (a.bytes.size() != checked_product(a.shape, itemsize(a.dtype)))
        bad("npz: member " + name + " has " + std::to_string(a.bytes.size()) + " bytes for shape " +
            shape_repr(a.shape) + " of " + std::string(descr(a.dtype)));
    members_.emplace_back(std::move(name), std::move(a));
}

bool Npz::contains(std::string_view name) const {
    return std::any_of(members_.begin(), members_.end(), [&](const auto& m) { return m.first == name; });
}

const Array& Npz::at(std::string_view name) const {
    for (const auto& m : members_)
        if (m.first == name) return m.second;
    throw std::out_of_range(std::string(name) + " is not a file in the archive");
}

std::vector<std::string> Npz::files() const {
    std::vector<std::string> out;
    out.reserve(members_.size());
    for (const auto& m : members_) out.push_back(m.first);
    return out;
}

// --- .npy ----------------------------------------------------------------------------------

std::vector<std::uint8_t> npy_bytes(const Array& a) {
    if (a.bytes.size() != checked_product(a.shape, itemsize(a.dtype)))
        bad("npy: " + std::to_string(a.bytes.size()) + " bytes do not fill shape " + shape_repr(a.shape));
    // numpy's _write_array_header: sorted keys, repr values, then room to
    // grow the first axis, then _wrap_header's padding to ARRAY_ALIGN.
    std::string header = "{'descr': '" + std::string(descr(a.dtype)) + "', 'fortran_order': False, 'shape': " +
                         shape_repr(a.shape) + ", }";
    if (!a.shape.empty()) header.append(GROWTH_AXIS_MAX_DIGITS - std::to_string(a.shape[0]).size(), ' ');
    const std::size_t hlen = header.size() + 1;
    std::uint8_t major = 1;
    std::size_t len_bytes = 2;
    std::size_t padlen = ARRAY_ALIGN - ((MAGIC.size() + 2 + len_bytes + hlen) % ARRAY_ALIGN);
    if (hlen + padlen > 0xffff) {
        major = 2;
        len_bytes = 4;
        padlen = ARRAY_ALIGN - ((MAGIC.size() + 2 + len_bytes + hlen) % ARRAY_ALIGN);
    }
    std::vector<std::uint8_t> out;
    out.reserve(MAGIC.size() + 2 + len_bytes + hlen + padlen + a.bytes.size());
    put(out, MAGIC);
    out.push_back(major);
    out.push_back(0);
    if (major == 1)
        put16(out, static_cast<std::uint32_t>(hlen + padlen));
    else
        put32(out, static_cast<std::uint32_t>(hlen + padlen));
    put(out, header);
    out.insert(out.end(), padlen, ' ');
    out.push_back('\n');
    out.insert(out.end(), a.bytes.begin(), a.bytes.end());
    return out;
}

Array parse_npy(std::span<const std::uint8_t> f) {
    if (f.size() < 8 || std::memcmp(f.data(), MAGIC.data(), MAGIC.size()) != 0)
        bad("npy: the magic string is not correct");
    const unsigned major = f[6], minor = f[7];
    std::size_t at = 8, hlen;
    if (major == 1 && minor == 0) {
        hlen = get16(f, at, "npy header length");
        at += 2;
    } else if ((major == 2 || major == 3) && minor == 0) {
        hlen = get32(f, at, "npy header length");
        at += 4;
    } else {
        bad("npy: format version (" + std::to_string(major) + ", " + std::to_string(minor) +
            ") is not supported; only (1, 0), (2, 0) and (3, 0)");
    }
    if (f.size() - at < hlen) bad("npy: truncated header");
    const std::string_view header(reinterpret_cast<const char*>(f.data() + at), hlen);
    at += hlen;
    // numpy measures the decoded text; latin1 is one char per byte, and
    // for utf8 (3.0) counting bytes only ever refuses sooner.
    if (header.size() > MAX_HEADER_SIZE)
        bad("npy: header info length (" + std::to_string(header.size()) + ") is large and may not be safe to load");

    const Value d = Literal(header).parse();
    if (d.kind != Value::Dict) bad("npy: header is not a dictionary");
    const Value *v_descr = nullptr, *v_fortran = nullptr, *v_shape = nullptr;
    for (const auto& [k, v] : d.entries) {
        if (k.kind != Value::Str) bad("npy: header does not contain the correct keys");
        if (k.text == "descr") v_descr = &v;
        else if (k.text == "fortran_order") v_fortran = &v;
        else if (k.text == "shape") v_shape = &v;
        else bad("npy: header does not contain the correct keys (" + k.text + ")");
    }
    if (!v_descr || !v_fortran || !v_shape || d.entries.size() != 3)
        bad("npy: header does not contain the correct keys");
    if (v_shape->kind != Value::Tuple) bad("npy: shape is not valid");
    if (v_fortran->kind != Value::Bool) bad("npy: fortran_order is not a valid bool");
    if (v_descr->kind != Value::Str)
        bad("npy: descr is not a supported dtype (structured or object arrays are not supported)");
    const std::optional<DType> t = from_descr(v_descr->text);
    if (!t) bad("npy: dtype '" + v_descr->text + "' is not supported");

    Array a;
    a.dtype = *t;
    for (const Value& x : v_shape->items) {
        if (x.kind == Value::Bool) {   // isinstance(True, int) in Python
            a.shape.push_back(x.flag ? 1 : 0);
            continue;
        }
        if (x.kind != Value::Int) bad("npy: shape is not valid");
        if (!x.text.empty() && x.text[0] == '-') bad("npy: negative dimensions are not allowed");
        if (x.text.size() > 18) bad("npy: dimension too large");
        a.shape.push_back(static_cast<std::size_t>(std::stoull(x.text)));
    }
    const std::size_t nbytes = checked_product(a.shape, itemsize(a.dtype));
    if (f.size() - at < nbytes)
        bad("npy: EOF reading array data, expected " + std::to_string(nbytes) + " bytes got " +
            std::to_string(f.size() - at));
    a.bytes.assign(f.begin() + static_cast<std::ptrdiff_t>(at),
                   f.begin() + static_cast<std::ptrdiff_t>(at + nbytes));
    if (v_fortran->flag) a.bytes = fortran_to_c(a.bytes, a.shape, itemsize(a.dtype));
    return a;
}

// --- .npz ----------------------------------------------------------------------------------

std::vector<std::uint8_t> npz_bytes(const Npz& z, DosTime stamp) {
    struct Entry {
        std::string name;
        std::uint32_t crc;
        std::uint64_t size;
        std::uint64_t offset;
    };
    std::vector<Entry> entries;
    std::vector<std::uint8_t> out;
    for (const auto& [key, arr] : z.members()) {
        const std::string name = key + ".npy";
        const std::vector<std::uint8_t> data = npy_bytes(arr);
        Entry e{name, crc32(std::span<const std::uint8_t>(data)), data.size(), out.size()};
        // Local header, as zipfile writes it with force_zip64=True: sizes
        // 0xFFFFFFFF here, the real ones in the zip64 extra.
        put(out, "PK\x03\x04");
        out.push_back(VERSION_ZIP64);
        out.push_back(0);
        const bool ascii = std::all_of(name.begin(), name.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; });
        put16(out, ascii ? 0 : 0x800);
        put16(out, 0);   // stored
        put16(out, stamp.time);
        put16(out, stamp.date);
        put32(out, e.crc);
        put32(out, 0xffffffffu);
        put32(out, 0xffffffffu);
        put16(out, static_cast<std::uint32_t>(name.size()));
        put16(out, 20);
        put(out, name);
        put16(out, 1);
        put16(out, 16);
        put64(out, e.size);   // file size
        put64(out, e.size);   // compressed size
        out.insert(out.end(), data.begin(), data.end());
        entries.push_back(std::move(e));
    }
    // Central directory, as zipfile._write_end_record.
    const std::uint64_t start_dir = out.size();
    for (const Entry& e : entries) {
        std::vector<std::uint64_t> extra;
        std::uint32_t size32 = static_cast<std::uint32_t>(e.size), offset32 = static_cast<std::uint32_t>(e.offset);
        if (e.size > ZIP64_LIMIT) {
            extra.push_back(e.size);
            extra.push_back(e.size);
            size32 = 0xffffffffu;
        }
        if (e.offset > ZIP64_LIMIT) {
            extra.push_back(e.offset);
            offset32 = 0xffffffffu;
        }
        const bool ascii =
            std::all_of(e.name.begin(), e.name.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; });
        put(out, "PK\x01\x02");
        out.push_back(VERSION_ZIP64);
        out.push_back(CREATE_SYSTEM_UNIX);
        out.push_back(VERSION_ZIP64);
        out.push_back(0);
        put16(out, ascii ? 0 : 0x800);
        put16(out, 0);
        put16(out, stamp.time);
        put16(out, stamp.date);
        put32(out, e.crc);
        put32(out, size32);   // compressed
        put32(out, size32);   // uncompressed
        put16(out, static_cast<std::uint32_t>(e.name.size()));
        put16(out, extra.empty() ? 0 : static_cast<std::uint32_t>(4 + 8 * extra.size()));
        put16(out, 0);   // comment
        put16(out, 0);   // disk
        put16(out, 0);   // internal attributes
        put32(out, EXTERNAL_ATTR);
        put32(out, offset32);
        put(out, e.name);
        if (!extra.empty()) {
            put16(out, 1);
            put16(out, static_cast<std::uint32_t>(8 * extra.size()));
            for (const std::uint64_t x : extra) put64(out, x);
        }
    }
    const std::uint64_t pos2 = out.size();
    std::uint64_t count = entries.size(), cd_size = pos2 - start_dir, cd_offset = start_dir;
    if (count > ZIP_FILECOUNT_LIMIT || cd_offset > ZIP64_LIMIT || cd_size > ZIP64_LIMIT) {
        put(out, "PK\x06\x06");
        put64(out, SIZE_END64 - 12);
        put16(out, VERSION_ZIP64);
        put16(out, VERSION_ZIP64);
        put32(out, 0);
        put32(out, 0);
        put64(out, count);
        put64(out, count);
        put64(out, cd_size);
        put64(out, cd_offset);
        put(out, "PK\x06\x07");
        put32(out, 0);
        put64(out, pos2);
        put32(out, 1);
        count = std::min<std::uint64_t>(count, 0xffff);
        cd_size = std::min<std::uint64_t>(cd_size, 0xffffffffu);
        cd_offset = std::min<std::uint64_t>(cd_offset, 0xffffffffu);
    }
    put(out, "PK\x05\x06");
    put16(out, 0);
    put16(out, 0);
    put16(out, static_cast<std::uint32_t>(count));
    put16(out, static_cast<std::uint32_t>(count));
    put32(out, static_cast<std::uint32_t>(cd_size));
    put32(out, static_cast<std::uint32_t>(cd_offset));
    put16(out, 0);
    return out;
}

Npz parse_npz(std::span<const std::uint8_t> f) {
    // zipfile._EndRecData: the end record is last, unless a comment follows it.
    std::size_t end_at = std::string::npos;
    if (f.size() >= SIZE_END && sig_at(f, f.size() - SIZE_END, "PK\x05\x06") && f[f.size() - 2] == 0 &&
        f[f.size() - 1] == 0) {
        end_at = f.size() - SIZE_END;
    } else {
        const std::size_t lo = f.size() > MAX_COMMENT + SIZE_END ? f.size() - MAX_COMMENT - SIZE_END : 0;
        for (std::size_t p = f.size() >= 4 ? f.size() - 4 + 1 : 0; p-- > lo;)
            if (sig_at(f, p, "PK\x05\x06")) {
                end_at = p;
                break;
            }
        if (end_at == std::string::npos || f.size() - end_at < SIZE_END) bad("npz: file is not a zip file");
    }
    if (get16(f, end_at + 4, "end record") != 0 || get16(f, end_at + 6, "end record") != 0)
        bad("npz: zipfiles that span multiple disks are not supported");
    std::uint64_t cd_size = get32(f, end_at + 12, "end record");
    std::uint64_t cd_offset = get32(f, end_at + 16, "end record");
    bool zip64 = false;
    // zipfile._EndRecData64: a locator right before the end record means a
    // zip64 end record right before that.
    if (end_at >= SIZE_END64_LOCATOR && sig_at(f, end_at - SIZE_END64_LOCATOR, "PK\x06\x07")) {
        const std::size_t loc = end_at - SIZE_END64_LOCATOR;
        if (get32(f, loc + 4, "zip64 locator") != 0 || get32(f, loc + 16, "zip64 locator") > 1)
            bad("npz: zipfiles that span multiple disks are not supported");
        if (loc < SIZE_END64 || !sig_at(f, loc - SIZE_END64, "PK\x06\x06"))
            bad("npz: zip64 end record missing");
        const std::size_t e64 = loc - SIZE_END64;
        cd_size = get64(f, e64 + 40, "zip64 end record");
        cd_offset = get64(f, e64 + 48, "zip64 end record");
        zip64 = true;
    }
    // Data prepended to the archive ("concat") shifts every offset.
    const std::int64_t concat = static_cast<std::int64_t>(end_at) - static_cast<std::int64_t>(cd_size) -
                                static_cast<std::int64_t>(cd_offset) -
                                (zip64 ? static_cast<std::int64_t>(SIZE_END64 + SIZE_END64_LOCATOR) : 0);
    const std::int64_t start_dir = static_cast<std::int64_t>(cd_offset) + concat;
    if (start_dir < 0 || static_cast<std::uint64_t>(start_dir) + cd_size > f.size())
        bad("npz: bad offset for central directory");

    Npz z;
    std::vector<std::pair<std::string, Array>> members;
    std::size_t p = static_cast<std::size_t>(start_dir);
    const std::size_t cd_end = p + static_cast<std::size_t>(cd_size);
    while (p < cd_end) {
        if (cd_end - p < SIZE_CENTRAL_DIR) bad("npz: truncated central directory");
        if (!sig_at(f, p, "PK\x01\x02")) bad("npz: bad magic number for central directory");
        const std::uint32_t flags = get16(f, p + 8, "central directory");
        const std::uint32_t method = get16(f, p + 10, "central directory");
        const std::uint32_t crc = get32(f, p + 16, "central directory");
        std::uint64_t csize = get32(f, p + 20, "central directory");
        std::uint64_t usize = get32(f, p + 24, "central directory");
        const std::size_t nlen = get16(f, p + 28, "central directory");
        const std::size_t elen = get16(f, p + 30, "central directory");
        const std::size_t clen = get16(f, p + 32, "central directory");
        std::uint64_t offset = get32(f, p + 42, "central directory");
        const std::size_t name_at = p + SIZE_CENTRAL_DIR;
        if (cd_end - name_at < nlen + elen + clen) bad("npz: truncated central directory");
        const std::string name(reinterpret_cast<const char*>(f.data() + name_at), nlen);
        // zipfile._decodeExtra: the zip64 field holds, in order, whichever
        // of the sizes and the offset are 0xFFFFFFFF here.
        for (std::size_t x = name_at + nlen; x + 4 <= name_at + nlen + elen;) {
            const std::uint32_t tp = get16(f, x, "extra field"), ln = get16(f, x + 2, "extra field");
            if (x + 4 + ln > name_at + nlen + elen) bad("npz: corrupt extra field");
            if (tp == 1) {
                std::size_t q = x + 4;
                const std::size_t qend = q + ln;
                auto next = [&](const char* what) {
                    if (qend - q < 8) bad(std::string("npz: corrupt zip64 extra field (") + what + ")");
                    const std::uint64_t v = get64(f, q, "zip64 extra");
                    q += 8;
                    return v;
                };
                if (usize == 0xffffffffu) usize = next("file size");
                if (csize == 0xffffffffu) csize = next("compress size");
                if (offset == 0xffffffffu) offset = next("header offset");
            }
            x += 4 + ln;
        }
        p = name_at + nlen + elen + clen;

        if (flags & 0x1u) bad("npz: member " + name + " is encrypted");
        if (method != 0)
            bad("npz: member " + name + " is compressed (method " + std::to_string(method) +
                "); only np.savez's stored members are supported, not np.savez_compressed");
        if (csize != usize) bad("npz: member " + name + " has different stored and file sizes");
        // The local header: its own name and extra lengths, never its sizes.
        const std::uint64_t lh = offset + static_cast<std::uint64_t>(concat);
        if (lh > f.size() || f.size() - lh < SIZE_FILE_HEADER) bad("npz: truncated file header");
        if (!sig_at(f, static_cast<std::size_t>(lh), "PK\x03\x04")) bad("npz: bad magic number for file header");
        const std::size_t lnlen = get16(f, static_cast<std::size_t>(lh) + 26, "file header");
        const std::size_t lelen = get16(f, static_cast<std::size_t>(lh) + 28, "file header");
        const std::size_t lname = static_cast<std::size_t>(lh) + SIZE_FILE_HEADER;
        if (f.size() - lname < lnlen + lelen) bad("npz: truncated file header");
        if (std::string_view(reinterpret_cast<const char*>(f.data() + lname), lnlen) != name)
            bad("npz: file name in directory and header differ: " + name);
        const std::size_t data_at = lname + lnlen + lelen;
        if (f.size() - data_at < csize) bad("npz: truncated member " + name);
        const std::span<const std::uint8_t> data = f.subspan(data_at, static_cast<std::size_t>(csize));
        if (crc32(data) != crc) bad("npz: bad CRC-32 for file " + name);
        // np.load's NpzFile names a member "x.npy" as "x".
        std::string key = name;
        if (key.size() >= 4 && key.compare(key.size() - 4, 4, ".npy") == 0) key.resize(key.size() - 4);
        if (data.size() < 6 || std::memcmp(data.data(), MAGIC.data(), MAGIC.size()) != 0)
            bad("npz: member " + name + " is not a .npy array");
        Array a = parse_npy(data);
        // zipfile keeps the last of repeated names.
        bool replaced = false;
        for (auto& m : members)
            if (m.first == key) {
                m.second = std::move(a);
                replaced = true;
                break;
            }
        if (!replaced) members.emplace_back(std::move(key), std::move(a));
    }
    for (auto& [k, a] : members) z.add(std::move(k), std::move(a));
    return z;
}

void save_npy(const std::string& path, const Array& a) { write_file(path, npy_bytes(a)); }
Array load_npy(const std::string& path) {
    const auto data = read_file(path);
    return parse_npy(data);
}
void save_npz(const std::string& path, const Npz& z, DosTime stamp) { write_file(path, npz_bytes(z, stamp)); }
Npz load_npz(const std::string& path) {
    const auto data = read_file(path);
    return parse_npz(data);
}

}  // namespace sstvae::qrss::npz
