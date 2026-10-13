#include "qrss/npz.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

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

// Header text in a message: np.load decodes the header as Latin-1, so do
// the same into UTF-8, which keeps every message valid UTF-8 (a Python
// caller reads it as str) whatever bytes the file held.
std::string latin1(std::string_view s) {
    std::string out;
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        if (c < 0x80) {
            out.push_back(ch);
        } else {
            out.push_back(static_cast<char>(0xC0 | (c >> 6)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    return out;
}

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

// --- zip member names, as zipfile decodes them -----------------------------------------

// bytes.decode("utf-8"): strict, so no overlong forms, surrogates or code
// points past U+10FFFF.
bool valid_utf8(std::string_view s) {
    const auto* b = reinterpret_cast<const unsigned char*>(s.data());
    const std::size_t n = s.size();
    std::size_t i = 0;
    auto cont = [&](std::size_t k, unsigned lo = 0x80, unsigned hi = 0xBF) {
        return k < n && b[k] >= lo && b[k] <= hi;
    };
    while (i < n) {
        const unsigned c = b[i];
        if (c < 0x80) {
            i += 1;
        } else if (c >= 0xC2 && c <= 0xDF) {
            if (!cont(i + 1)) return false;
            i += 2;
        } else if (c >= 0xE0 && c <= 0xEF) {
            const unsigned lo = c == 0xE0 ? 0xA0 : 0x80, hi = c == 0xED ? 0x9F : 0xBF;
            if (!cont(i + 1, lo, hi) || !cont(i + 2)) return false;
            i += 3;
        } else if (c >= 0xF0 && c <= 0xF4) {
            const unsigned lo = c == 0xF0 ? 0x90 : 0x80, hi = c == 0xF4 ? 0x8F : 0xBF;
            if (!cont(i + 1, lo, hi) || !cont(i + 2) || !cont(i + 3)) return false;
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

// Python's cp437 codec, bytes 0x80..0xFF (0x00..0x7F are ASCII).
constexpr std::array<std::uint16_t, 128> CP437_HIGH = {
    0x00C7, 0x00FC, 0x00E9, 0x00E2, 0x00E4, 0x00E0, 0x00E5, 0x00E7,
    0x00EA, 0x00EB, 0x00E8, 0x00EF, 0x00EE, 0x00EC, 0x00C4, 0x00C5,
    0x00C9, 0x00E6, 0x00C6, 0x00F4, 0x00F6, 0x00F2, 0x00FB, 0x00F9,
    0x00FF, 0x00D6, 0x00DC, 0x00A2, 0x00A3, 0x00A5, 0x20A7, 0x0192,
    0x00E1, 0x00ED, 0x00F3, 0x00FA, 0x00F1, 0x00D1, 0x00AA, 0x00BA,
    0x00BF, 0x2310, 0x00AC, 0x00BD, 0x00BC, 0x00A1, 0x00AB, 0x00BB,
    0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562, 0x2556,
    0x2555, 0x2563, 0x2551, 0x2557, 0x255D, 0x255C, 0x255B, 0x2510,
    0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x255E, 0x255F,
    0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x2567,
    0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256B,
    0x256A, 0x2518, 0x250C, 0x2588, 0x2584, 0x258C, 0x2590, 0x2580,
    0x03B1, 0x00DF, 0x0393, 0x03C0, 0x03A3, 0x03C3, 0x00B5, 0x03C4,
    0x03A6, 0x0398, 0x03A9, 0x03B4, 0x221E, 0x03C6, 0x03B5, 0x2229,
    0x2261, 0x00B1, 0x2265, 0x2264, 0x2320, 0x2321, 0x00F7, 0x2248,
    0x00B0, 0x2219, 0x00B7, 0x221A, 0x207F, 0x00B2, 0x25A0, 0x00A0,
};

// A zip name as zipfile reads it, in UTF-8: the bytes decoded as UTF-8
// when the entry's flag bit 11 says so, otherwise as cp437 (zipfile's
// default, np.load passing no metadata_encoding). Invalid UTF-8 is
// zipfile's UnicodeDecodeError.
std::string decode_zip_name(std::span<const std::uint8_t> raw, bool utf8) {
    if (utf8) {
        std::string s(raw.begin(), raw.end());
        if (!valid_utf8(s)) bad("npz: a file name flagged UTF-8 is not valid UTF-8");
        return s;
    }
    std::string out;
    for (const std::uint8_t c : raw) {
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
            continue;
        }
        const std::uint32_t cp = CP437_HIGH[c - 0x80u];   // all below U+10000
        if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        }
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// zipfile._sanitize_filename: a name ends at its first NUL, and on
// Windows (os.sep "\\") a backslash is read as "/".
std::string sanitize_zip_name(std::string s) {
    const std::size_t nul = s.find('\0');
    if (nul != std::string::npos) s.resize(nul);
#ifdef _WIN32
    std::replace(s.begin(), s.end(), '\\', '/');
#endif
    return s;
}

// A zip offset as zipfile holds it, a Python int: the central directory's
// offset (up to 2^64 - 1 in a zip64 field) plus the prepended-data shift
// (negative when bytes are missing), so it lies in (-2^64, 2^65). A low
// word and a carry hold that exactly; ordering and equality are what the
// "Overlapped entries" check needs.
struct Wide {
    int hi = 0;
    std::uint64_t lo = 0;
    friend bool operator==(const Wide&, const Wide&) = default;
    friend bool operator<(const Wide& a, const Wide& b) { return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo; }
};
Wide wide_add(std::uint64_t a, std::uint64_t b) { return {a + b < a ? 1 : 0, a + b}; }
Wide wide_add(std::uint64_t a, std::int64_t c) {
    if (c >= 0) return wide_add(a, static_cast<std::uint64_t>(c));
    const std::uint64_t m = std::uint64_t(0) - static_cast<std::uint64_t>(c);
    return {a < m ? -1 : 0, a - m};
}

// --- the .npy header: a Python literal dict ---------------------------------------------

struct Value {
    enum Kind { Str, Int, Bool, None, Tuple, List, Dict } kind = None;
    std::string text;   // Str: the string; Int: the digits, with a leading '-' if negative
    bool flag = false;  // Bool
    std::vector<Value> items;                       // Tuple, List
    std::vector<std::pair<Value, Value>> entries;   // Dict
};

// The header as np.load reads it, with ast.literal_eval -- and so with
// Python's tokenizer deciding what separates the tokens. Anything this
// accepts, np.load accepts with the same value; a few spellings np.load
// also accepts are refused here (string prefixes, adjacent strings, hex
// or octal integers, backslash continuations, "-(5)", and string escapes
// beyond \n \t \r \0 \\ \' \" and \xNN -- an octal \141 among them),
// none of which numpy writes.
class Literal {
public:
    explicit Literal(std::string_view s) : s_(s) {}

    Value parse() {
        // literal_eval strips leading spaces and tabs, and the line the
        // value starts on is then a line like any other.
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t')) ++i_;
        line_start();
        Value v = value();
        space();
        if (i_ != s_.size()) fail("trailing text");
        return v;
    }

private:
    [[noreturn]] void fail(const char* why) const {
        bad(std::string("npy: cannot parse header (") + why + "): " + latin1(s_));
    }
    // What Python's tokenizer skips between tokens: spaces, tabs and form
    // feeds, '#' comments, and line breaks ("\r\n", "\r" or "\n"). Never
    // '\v', which Python refuses. Outside brackets a line break ends a
    // logical line, and the next one goes through line_start().
    void space() {
        const std::size_t n = s_.size();
        for (;;) {
            while (i_ < n && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\f')) ++i_;
            if (i_ < n && s_[i_] == '#')
                while (i_ < n && s_[i_] != '\n' && s_[i_] != '\r') ++i_;
            if (!(i_ < n && (s_[i_] == '\n' || s_[i_] == '\r'))) return;
            if (s_[i_] == '\r' && i_ + 1 < n && s_[i_ + 1] == '\n') ++i_;
            ++i_;
            if (depth_ == 0) line_start();
        }
    }
    // A line outside brackets that is not blank (only whitespace or a
    // comment, then a line break) must not be indented: Python's
    // "unexpected indent", which a whitespace-only last line with no line
    // break also is. Columns as the tokenizer counts them: tabs to the next
    // multiple of 8, and a form feed back to 0.
    void line_start() {
        const std::size_t n = s_.size();
        std::size_t col = 0;
        for (; i_ < n; ++i_) {
            if (s_[i_] == ' ') ++col;
            else if (s_[i_] == '\t') col = (col / 8 + 1) * 8;
            else if (s_[i_] == '\f') col = 0;
            else break;
        }
        const bool blank = i_ < n && (s_[i_] == '#' || s_[i_] == '\n' || s_[i_] == '\r');
        if (!blank && col > 0) fail("unexpected indent");
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
            ++depth_;
            v.kind = c == '(' ? Value::Tuple : Value::List;
            bool comma = false;
            while (!eat(close)) {
                if (!v.items.empty() && !comma) fail("expected ','");
                v.items.push_back(value());
                comma = eat(',');
            }
            --depth_;
            // "(x)" is x itself, not a tuple.
            if (c == '(' && v.items.size() == 1 && !comma) return v.items[0];
        } else if (c == '{') {
            ++i_;
            ++depth_;
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
            --depth_;
        } else if (word("True")) {
            v.kind = Value::Bool;
            v.flag = true;
        } else if (word("False")) {
            v.kind = Value::Bool;
        } else if (word("None")) {
            v.kind = Value::None;
        } else if (c == '-' || c == '+' || std::isdigit(static_cast<unsigned char>(c))) {
            // literal_eval takes one sign on a number, never two ("--5").
            bool neg = false;
            if (c == '-' || c == '+') {
                neg = c == '-';
                ++i_;
                space();
            }
            const auto digit = [&](std::size_t k) {
                return k < s_.size() && std::isdigit(static_cast<unsigned char>(s_[k])) != 0;
            };
            if (!digit(i_)) fail("bad number");
            // Python's decinteger: nonzerodigit (["_"] digit)* | "0"+ (["_"] "0")*.
            std::string digits;
            for (;;) {
                digits += s_[i_++];
                if (i_ < s_.size() && s_[i_] == '_' && digit(i_ + 1)) ++i_;
                if (!digit(i_)) break;
            }
            if (digits[0] == '0' && digits.find_first_not_of('0') != std::string::npos)
                fail("leading zeros in decimal integer literals are not permitted");
            if (i_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[i_])) || s_[i_] == '_' || s_[i_] == '.'))
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
    int depth_ = 0;   // open brackets
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
constexpr std::uint32_t MAX_EXTRACT_VERSION = 63;   // zipfile's: newer archives are refused
constexpr std::uint32_t CREATE_SYSTEM_UNIX = 3;
constexpr std::size_t SIZE_FILE_HEADER = 30, SIZE_CENTRAL_DIR = 46, SIZE_END = 22, SIZE_END64 = 56,
                      SIZE_END64_LOCATOR = 20, MAX_COMMENT = 0xffff;

}  // namespace

// --- Array ---------------------------------------------------------------------------------

std::string_view descr(DType t) { return info(t).descr; }
std::size_t itemsize(DType t) { return info(t).size; }

std::optional<DType> from_descr(std::string_view d) {
    // np.dtype(d) for a kind and size: the byte order is '<', '=' (native,
    // which is little-endian here), '|' or absent; a one-byte type has none,
    // so any prefix -- '>' too -- means the same. '?' is bool. A
    // multi-byte '>' is big-endian and unsupported, and so is every other
    // spelling np.dtype takes ('d', 'float64', a subarray "2f8", ...).
    char order = '=';
    if (!d.empty() && (d[0] == '<' || d[0] == '>' || d[0] == '=' || d[0] == '|')) {
        order = d[0];
        d.remove_prefix(1);
    }
    if (d == "?") d = "b1";
    for (const auto& x : DTYPES)
        if (d == std::string_view(x.descr).substr(1)) {
            if (order == '>' && x.size > 1) return std::nullopt;
            return x.t;
        }
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
            throw std::overflow_error("npz: value does not fit int64");
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
    if (ndim() != 0) bad("npz: only 0-dimensional arrays can be converted to an integer, not shape " + shape_repr(shape));
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
        case DType::f2:
        case DType::f4:
        case DType::f8: {
            const double v = to_double()[0];   // exact for every float dtype
            if (std::isnan(v)) bad("cannot convert float NaN to integer");
            if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
            const double t = std::trunc(v);
            // [-2^63, 2^63): both ends are exact doubles.
            if (!(t >= -0x1p63 && t < 0x1p63)) throw std::overflow_error("npz: value does not fit int64");
            return static_cast<std::int64_t>(t);
        }
        case DType::c8:
        case DType::c16: bad("npz: a complex array has no integer value");
    }
    bad("npz: unknown dtype");
}

// --- Npz -----------------------------------------------------------------------------------

void Npz::add(std::string name, Array a) {
    // np.savez(f, **{"": a}) writes ".npy", which np.load reads back as "",
    // so an empty name is a name. A NUL (which zipfile would cut the name
    // at, ".npy" and all) and bytes that are not UTF-8 (a Python str always
    // is, and np.load could not read the name back) are refused.
    if (name.find('\0') != std::string::npos) bad("npz: member name contains NUL");
    if (!valid_utf8(name)) bad("npz: member name is not valid UTF-8");
    if (contains(name)) bad("npz: duplicate member " + name);
    if (a.bytes.size() != checked_product(a.shape, itemsize(a.dtype)))
        bad("npz: member " + name + " has " + std::to_string(a.bytes.size()) + " bytes for shape " +
            shape_repr(a.shape) + " of " + std::string(descr(a.dtype)));
    index_.emplace(name, members_.size());
    members_.emplace_back(std::move(name), std::move(a));
}

bool Npz::contains(std::string_view name) const { return index_.count(std::string(name)) != 0; }

const Array& Npz::at(std::string_view name) const {
    const auto it = index_.find(std::string(name));
    if (it == index_.end()) throw std::out_of_range(std::string(name) + " is not a file in the archive");
    return members_[it->second].second;
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
        else bad("npy: header does not contain the correct keys (" + latin1(k.text) + ")");
    }
    if (!v_descr || !v_fortran || !v_shape || d.entries.size() != 3)
        bad("npy: header does not contain the correct keys");
    if (v_shape->kind != Value::Tuple) bad("npy: shape is not valid");
    if (v_fortran->kind != Value::Bool) bad("npy: fortran_order is not a valid bool");
    if (v_descr->kind != Value::Str)
        bad("npy: descr is not a supported dtype (structured or object arrays are not supported)");
    const std::optional<DType> t = from_descr(v_descr->text);
    if (!t) bad("npy: dtype '" + latin1(v_descr->text) + "' is not supported");

    Array a;
    a.dtype = *t;
    // numpy's limits on a shape: at most 64 dimensions, each an int (a
    // bool passes the header check, isinstance(True, int), and then fails
    // the reshape), none negative (-0 is 0), none past npy_intp, and the
    // nonzero ones times the item size within npy_intp -- checked even
    // when another dimension is 0 and the array holds nothing.
    constexpr std::size_t NPY_MAXDIMS = 64;
    if (v_shape->items.size() > NPY_MAXDIMS)
        bad("npy: maximum supported dimension for an ndarray is currently 64, found " +
            std::to_string(v_shape->items.size()));
    constexpr std::uint64_t INTP_MAX = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    std::uint64_t nonzero_bytes = itemsize(a.dtype);
    bool empty = false;
    for (const Value& x : v_shape->items) {
        if (x.kind == Value::Bool) bad("npy: an integer is required (a bool dimension)");
        if (x.kind != Value::Int) bad("npy: shape is not valid");
        const bool minus = !x.text.empty() && x.text[0] == '-';
        const std::string_view digits = std::string_view(x.text).substr(minus ? 1 : 0);
        const std::size_t lead = digits.find_first_not_of('0');
        const std::string_view sig = lead == std::string_view::npos ? std::string_view() : digits.substr(lead);
        if (sig.empty()) {
            a.shape.push_back(0);
            empty = true;
            continue;
        }
        if (minus) bad("npy: negative dimensions are not allowed");
        std::uint64_t v = 0;
        for (const char c : sig) {
            const auto d = static_cast<std::uint64_t>(c - '0');
            if (v > (INTP_MAX - d) / 10) bad("npy: Maximum allowed dimension exceeded");
            v = v * 10 + d;
        }
        if (nonzero_bytes > INTP_MAX / v)
            bad("npy: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.");
        nonzero_bytes *= v;
        a.shape.push_back(static_cast<std::size_t>(v));
    }
    const std::size_t nbytes = empty ? 0 : static_cast<std::size_t>(nonzero_bytes);
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
    // np.load decides by the first bytes: a zip local header or an (empty
    // archive's) end record is an npz; anything else is a .npy file or
    // pickled data, which allow_pickle=False refuses. zipfile alone would
    // accept data prepended to an archive, and np.load never gets that far.
    if (f.empty()) bad("npz: file is empty (np.load: no data left in file)");
    if (!sig_at(f, 0, "PK\x03\x04") && !sig_at(f, 0, "PK\x05\x06"))
        bad("npz: file does not start like a zip archive (np.load would read it as .npy or pickled data)");
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
    // (zipfile reads no disk numbers from this record, only from a zip64 locator.)
    std::uint64_t cd_size = get32(f, end_at + 12, "end record");
    std::uint64_t cd_offset = get32(f, end_at + 16, "end record");
    std::uint64_t location = end_at;   // zipfile's endrec[_ECD_LOCATION]
    // zipfile._EndRecData64 (CPython 3.13): a zip64 locator right before the
    // end record points at a zip64 end record, which must sit right after
    // the central directory it describes. Every value comes from the file,
    // so each is checked before it is used as an offset: a record claiming
    // a directory of 2^64 - 1 bytes is corrupt, never an empty archive.
    if (end_at >= SIZE_END64_LOCATOR && sig_at(f, end_at - SIZE_END64_LOCATOR, "PK\x06\x07")) {
        const std::size_t loc = end_at - SIZE_END64_LOCATOR;
        if (get32(f, loc + 4, "zip64 locator") != 0 || get32(f, loc + 16, "zip64 locator") > 1)
            bad("npz: zipfiles that span multiple disks are not supported");
        const std::uint64_t reloff = get64(f, loc + 8, "zip64 locator");
        if (loc < SIZE_END64 || reloff > loc - SIZE_END64)
            bad("npz: corrupt zip64 end of central directory locator");
        const std::size_t offset = loc - SIZE_END64;
        std::size_t at = static_cast<std::size_t>(reloff);
        std::uint64_t extrasz = offset - reloff;   // zip64 extensible data
        if (!sig_at(f, at, "PK\x06\x06") && reloff != offset) {
            at = offset;   // data prepended to the archive: assume no extensible data
            extrasz = 0;
        }
        if (!sig_at(f, at, "PK\x06\x06")) bad("npz: zip64 end of central directory record not found");
        const std::uint64_t sz = get64(f, at + 4, "zip64 end record");
        const std::uint64_t dirsize = get64(f, at + 40, "zip64 end record");
        const std::uint64_t diroffset = get64(f, at + 48, "zip64 end record");
        // diroffset + dirsize == reloff and sz + 12 == 56 + extrasz, without overflow.
        if (dirsize > reloff || diroffset != reloff - dirsize || sz != SIZE_END64 - 12 + extrasz)
            bad("npz: corrupt zip64 end of central directory record");
        cd_size = dirsize;
        cd_offset = diroffset;
        location = offset - extrasz;
    }
    // Data prepended to the archive ("concat") shifts every offset. Both
    // sizes are now at most the file's (32-bit fields, or checked above), so
    // none of this can overflow.
    if (cd_size > location) bad("npz: bad offset for central directory");
    const std::uint64_t start_dir = location - cd_size;
    const std::int64_t concat = static_cast<std::int64_t>(start_dir) - static_cast<std::int64_t>(cd_offset);

    // The central directory first (zipfile._RealGetContents), then the
    // members np.load reads, each through its local header (ZipFile.open).
    struct Entry {
        std::string orig;              // the name as the directory spells it (orig_filename)
        std::string filename;          // as namelist() reports it: sanitized, or a 0x7075 field's
        std::uint32_t flags, method, crc;
        std::uint64_t csize, size;     // compress size; the bytes the member holds
        Wide header;                   // the local header's position in f
        Wide end;                      // where the next member (or the directory) starts
    };
    std::vector<Entry> entries;
    std::size_t p = static_cast<std::size_t>(start_dir);
    const std::size_t cd_end = static_cast<std::size_t>(location);
    while (p < cd_end) {
        if (cd_end - p < SIZE_CENTRAL_DIR) bad("npz: truncated central directory");
        if (!sig_at(f, p, "PK\x01\x02")) bad("npz: bad magic number for central directory");
        const std::uint32_t extract_version = f[p + 6];
        const std::uint32_t flags = get16(f, p + 8, "central directory");
        const std::uint32_t method = get16(f, p + 10, "central directory");
        const std::uint32_t crc = get32(f, p + 16, "central directory");
        std::uint64_t csize = get32(f, p + 20, "central directory");
        std::uint64_t usize = get32(f, p + 24, "central directory");
        const std::size_t nlen = get16(f, p + 28, "central directory");
        const std::size_t elen = get16(f, p + 30, "central directory");
        const std::size_t clen = get16(f, p + 32, "central directory");
        std::uint64_t offset = get32(f, p + 42, "central directory");
        // The name, extra field and comment are read as zipfile reads them:
        // whatever of their declared lengths the directory still holds.
        const std::size_t name_at = p + SIZE_CENTRAL_DIR;
        const std::size_t extra_at = std::min(name_at + nlen, cd_end);
        const std::size_t extra_end = std::min(name_at + nlen + elen, cd_end);
        const std::span<const std::uint8_t> raw = f.subspan(name_at, extra_at - name_at);
        const std::string orig = decode_zip_name(raw, (flags & 0x800u) != 0);
        std::string filename = sanitize_zip_name(orig);
        if (extract_version > MAX_EXTRACT_VERSION)
            bad("npz: zip file version " + std::to_string(extract_version / 10) + "." +
                std::to_string(extract_version % 10) + " is not supported");
        // zipfile._decodeExtra.
        for (std::size_t x = extra_at; x + 4 <= extra_end;) {
            const std::uint32_t tp = get16(f, x, "extra field"), ln = get16(f, x + 2, "extra field");
            if (x + 4 + ln > extra_end) bad("npz: corrupt extra field");
            const std::size_t q0 = x + 4, qend = q0 + ln;
            if (tp == 1) {
                // zip64: in order, whichever of the sizes and the offset
                // are 0xFFFFFFFF in the fixed fields.
                std::size_t q = q0;
                auto next = [&](const char* what) {
                    if (qend - q < 8) bad(std::string("npz: corrupt zip64 extra field (") + what + " not found)");
                    const std::uint64_t v = get64(f, q, "zip64 extra");
                    q += 8;
                    return v;
                };
                if (usize == 0xffffffffu) usize = next("file size");
                if (csize == 0xffffffffu) csize = next("compress size");
                if (offset == 0xffffffffu) offset = next("header offset");
            } else if (tp == 0x7075) {
                // Info-ZIP Unicode Path: version 1 and the CRC-32 of the
                // raw name it replaces, then that name in UTF-8.
                if (ln < 5) bad("npz: corrupt unicode path extra field (0x7075)");
                if (f[q0] == 1 && get32(f, q0 + 1, "extra field") == crc32(raw)) {
                    const std::string up(reinterpret_cast<const char*>(f.data() + q0 + 5), ln - 5);
                    if (!valid_utf8(up)) bad("npz: corrupt unicode path extra field (0x7075): invalid utf-8 bytes");
                    if (!up.empty()) filename = sanitize_zip_name(up);   // empty: zipfile only warns
                }
            }
            x = qend;
        }
        p = name_at + nlen + elen + clen;
        // A stored member is min(compress size, file size) bytes: ZipExtFile
        // stops at whichever runs out first (and checks the CRC of those).
        entries.push_back({orig, std::move(filename), flags, method, crc, csize, std::min(csize, usize),
                           wide_add(offset, concat), {}});
    }
    // zipfile: each member must end before the next one's local header (or
    // the central directory) starts -- its "Overlapped entries" check, with
    // its order: sorted by offset, walked from the end, so of two entries
    // at one offset the later one is checked against the next header and
    // the earlier one only warned about (CPython 3.13).
    {
        std::vector<std::size_t> order(entries.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return entries[a].header < entries[b].header; });
        Wide end{0, start_dir};
        for (std::size_t k = order.size(); k-- > 0;) {
            entries[order[k]].end = end;
            end = entries[order[k]].header;
        }
    }

    // ZipFile.open and the read, for one entry.
    auto read_member = [&](const Entry& e) -> Array {
        const std::string& name = e.orig;
        if (e.header.hi != 0 || e.header.lo > f.size() || f.size() - e.header.lo < SIZE_FILE_HEADER)
            bad("npz: truncated file header for " + name);
        const auto lh = static_cast<std::size_t>(e.header.lo);
        if (!sig_at(f, lh, "PK\x03\x04")) bad("npz: bad magic number for file header");
        const std::uint32_t lflags = get16(f, lh + 6, "file header");
        const std::size_t lnlen = get16(f, lh + 26, "file header");
        const std::size_t lelen = get16(f, lh + 28, "file header");
        const std::size_t lname = lh + SIZE_FILE_HEADER;
        // (Past the end, zipfile's name or data comes up short: a mismatch
        // or an EOF, so refused either way.)
        if (f.size() - lname < lnlen + lelen) bad("npz: truncated file header");
        if (e.flags & 0x20u) bad("npz: member " + name + " is compressed patched data (flag bit 5)");
        if (e.flags & 0x40u) bad("npz: member " + name + " uses strong encryption (flag bit 6)");
        if (decode_zip_name(f.subspan(lname, lnlen), (lflags & 0x800u) != 0) != name)
            bad("npz: file name in directory and header differ: " + name);
        const std::size_t data_at = lname + lnlen + lelen;
        // An entry whose end is its own start shares its offset with
        // another; zipfile only warns about those.
        if (e.end < wide_add(static_cast<std::uint64_t>(data_at), e.csize) && !(e.end == e.header))
            bad("npz: overlapped entries: " + name + " (possible zip bomb)");
        if (e.flags & 0x1u) bad("npz: member " + name + " is encrypted");
        if (e.method != 0)
            bad("npz: member " + name + " is compressed (method " + std::to_string(e.method) +
                "); only np.savez's stored members are supported, not np.savez_compressed");
        if (f.size() - data_at < e.size) bad("npz: truncated member " + name);
        const std::span<const std::uint8_t> data = f.subspan(data_at, static_cast<std::size_t>(e.size));
        if (crc32(data) != e.crc) bad("npz: bad CRC-32 for file " + name);
        if (data.size() < 6 || std::memcmp(data.data(), MAGIC.data(), MAGIC.size()) != 0)
            bad("npz: member " + name + " is not a .npy array");
        return parse_npy(data);
    };

    // np.load's NpzFile: its files are the names without ".npy", in
    // archive order; a key reads the member named exactly that if there is
    // one, else the one named key + ".npy" (and zipfile's NameToInfo keeps
    // the last entry of a repeated name). Only those members are read,
    // which is what reading every key of np.load's result does.
    std::unordered_map<std::string, std::size_t> by_name;
    for (std::size_t i = 0; i < entries.size(); ++i) by_name[entries[i].filename] = i;
    std::vector<std::string> keys;
    std::unordered_set<std::string> seen;
    for (const Entry& e : entries) {
        std::string key = e.filename;
        if (key.size() >= 4 && key.compare(key.size() - 4, 4, ".npy") == 0) key.resize(key.size() - 4);
        if (seen.insert(key).second) keys.push_back(std::move(key));
    }
    Npz z;
    for (const std::string& key : keys) {
        const auto exact = by_name.find(key);
        const std::size_t i = exact != by_name.end() ? exact->second : by_name.at(key + ".npy");
        z.add(key, read_member(entries[i]));
    }
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
