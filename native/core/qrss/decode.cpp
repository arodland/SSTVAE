#include "qrss/decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/npsum.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
#include "qrss/pyunicode_tables.hpp"

namespace sstvae::qrss::decode {

namespace {

using pyjson::Value;

// One code point of strict UTF-8 at s[i] (RFC 3629: shortest form, no
// surrogates), its length in `len`; nullopt where Python's UTF-8 decoder
// would start an error (each byte of which surrogateescape keeps as itself).
std::optional<char32_t> code_point(std::string_view s, std::size_t i, std::size_t& len) {
    const auto b = static_cast<unsigned char>(s[i]);
    if (b < 0x80) {
        len = 1;
        return b;
    }
    unsigned char lo = 0x80, hi = 0xBF;
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
        return std::nullopt;
    }
    if (i + len > s.size()) return std::nullopt;
    const auto b1 = static_cast<unsigned char>(s[i + 1]);
    if (b1 < lo || b1 > hi) return std::nullopt;
    char32_t cp = b & (len == 2 ? 0x1Fu : len == 3 ? 0x0Fu : 0x07u);
    for (std::size_t k = 1; k < len; ++k) {
        const auto c = static_cast<unsigned char>(s[i + k]);
        if ((c & 0xC0) != 0x80) return std::nullopt;
        cp = (cp << 6) | (c & 0x3Fu);
    }
    return cp;
}

const char* upper_of(char32_t cp) {
    const auto* first = std::begin(pyunicode::UPPER);
    const auto* last = std::end(pyunicode::UPPER);
    const auto* it = std::lower_bound(first, last, cp,
                                      [](const pyunicode::Upper& u, char32_t c) { return u.cp < c; });
    return it != last && it->cp == cp ? it->utf8 : nullptr;
}

std::string fmt(const char* spec, double x) { return frame::py_format(x, spec); }

// The type name Python's messages give the object json.loads made.
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

// format(v, spec) for a float spec ("%.1f", "%+.2f") on what json.loads made.
std::string fmt_value(const char* spec, const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Bool: return fmt(spec, v.as_bool() ? 1.0 : 0.0);
        case Value::Kind::Int: return fmt(spec, static_cast<double>(v.as_int()));
        case Value::Kind::Float: return fmt(spec, v.as_double());
        case Value::Kind::String:
            throw PyException("ValueError", "Unknown format code 'f' for object of type 'str'");
        default:
            throw PyException("TypeError",
                              std::string("unsupported format string passed to ") + type_name(v) + ".__format__");
    }
}

// Python's truth value of what json.loads made.
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
    return true;
}

// WAVEFORM_NAMES.get(wf, wf)
std::string waveform_name(std::int64_t wf) {
    if (wf == config::WAVEFORM_CE) return "CE";
    if (wf == config::WAVEFORM_L) return "L";
    return std::to_string(wf);
}

// Decimal text (no sign) divided by 16 in place; the remainder.
int divmod16(std::string& digits) {
    std::string q;
    int rem = 0;
    for (const char c : digits) {
        const int cur = rem * 10 + (c - '0');
        const int d = cur / 16;
        rem = cur % 16;
        if (!q.empty() || d != 0) q += static_cast<char>('0' + d);
    }
    digits = q.empty() ? std::string("0") : q;
    return rem;
}

std::string zero_pad(const std::string& sign, std::string digits, int width) {
    const std::size_t have = sign.size() + digits.size();
    if (static_cast<std::size_t>(std::max(width, 0)) > have)
        digits.insert(0, static_cast<std::size_t>(width) - have, '0');
    return sign + digits;
}

}  // namespace

std::string py_upper(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t len = 1;
        const std::optional<char32_t> cp = code_point(text, i, len);
        if (!cp) {
            out += text[i++];   // an escaped byte: upper() leaves it
            continue;
        }
        if (const char* up = upper_of(*cp)) out += up;
        else out.append(text.substr(i, len));
        i += len;
    }
    return out;
}

std::string py_hex(std::int64_t v, int width) {
    // |v| as unsigned: -INT64_MIN does not fit an int64
    std::uint64_t u = v < 0 ? 0 - static_cast<std::uint64_t>(v) : static_cast<std::uint64_t>(v);
    std::string digits;
    do {
        digits.insert(digits.begin(), "0123456789abcdef"[u % 16]);
        u /= 16;
    } while (u);
    return zero_pad(v < 0 ? "-" : "", digits, width);
}

std::string py_hex(std::string_view decimal, int width) {
    std::string sign;
    if (!decimal.empty() && decimal.front() == '-') {
        sign = "-";
        decimal.remove_prefix(1);
    }
    std::string rest(decimal);
    if (rest.empty() || rest.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("py_hex: not a decimal integer: " + std::string(decimal));
    std::string hex;
    do {
        hex.insert(hex.begin(), "0123456789abcdef"[divmod16(rest)]);
    } while (rest != "0");
    if (hex == "0") sign.clear();
    return zero_pad(sign, hex, width);
}

std::string py_repr(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Null: return "None";
        case Value::Kind::Bool: return v.as_bool() ? "True" : "False";
        case Value::Kind::Int: return std::to_string(v.as_int());
        case Value::Kind::Float: return frame::py_repr(v.as_double());
        case Value::Kind::String: return pytext::repr(std::string_view(v.as_string()));
        case Value::Kind::Array: {
            std::string out = "[";
            for (std::size_t k = 0; k < v.items().size(); ++k) out += (k ? ", " : "") + py_repr(v.items()[k]);
            return out + "]";
        }
        case Value::Kind::Object: {
            std::string out = "{";
            for (std::size_t k = 0; k < v.keys().size(); ++k)
                out += (k ? ", " : "") + pytext::repr(std::string_view(v.keys()[k])) + ": " + py_repr(v.values()[k]);
            return out + "}";
        }
    }
    return "?";
}

std::string py_str(const Value& v) { return v.kind() == Value::Kind::String ? v.as_string() : py_repr(v); }

PictureKey parse_key(std::string_view s) {
    const std::size_t sep = s.rfind(':');
    const std::string shown = pytext::repr(pypath::fs_decode(s));
    if (sep == std::string_view::npos || sep == 0)
        throw Refusal("--key " + shown + ": expected CALL:PICTUREID (hexadecimal)");
    const std::string_view call = s.substr(0, sep), pid = s.substr(sep + 1);
    std::optional<pynum::PyInt> v = pynum::py_int(pid, 16);
    if (!v)
        throw Refusal("--key " + shown + ": picture ID " + pytext::repr(pypath::fs_decode(pid)) +
                      " is not hexadecimal");
    return PictureKey{py_upper(call), std::move(*v)};
}

std::string member_wave(store::Store& st, const Value& uid) {
    // store.pass_info(uid): _pass_path refuses a uid that is no string
    // (ValueError), which member_wave shows as "?".
    if (uid.kind() != Value::Kind::String) return "?";
    store::PassInfo info;
    try {
        info = st.pass_info(uid.as_string());
    } catch (const store::FileError&) {   // OSError
        return "?";
    } catch (const store::KeyError&) {
        return "?";
    } catch (const std::invalid_argument&) {   // ValueError
        return "?";
    }
    return waveform_name(info.waveform) + " " + (truthy(info.frame) ? py_str(info.frame) : std::string("?"));
}

std::vector<std::string> acc_lines(const store::Accumulator& acc, store::Store* st) {
    // picture.MODE_NAMES[acc.mode]: a tuple of three, negative indices from the end.
    if (acc.mode < -3 || acc.mode > 2) throw PyException("IndexError", "tuple index out of range");
    const char mode_name = "ABC"[(acc.mode + 3) % 3];
    std::vector<std::string> out;
    std::string first = "picture " + acc.key.callsign + " " + py_hex(acc.key.picture_id, 8) + "  mode " +
                        std::string(1, mode_name) + "  codec " + py_hex(acc.codec_id, 4) + "  " +
                        std::to_string(acc.members.size()) + " passes";
    if (!acc.foreign.empty()) first += ", " + std::to_string(acc.foreign.size()) + " merged";
    first += "  mean W " + fmt("%+.2f", em::acc_mean_w_db(acc)) + " dB";
    out.push_back(std::move(first));
    const std::vector<std::optional<double>> snr = acc.seg_snr_db();
    for (std::size_t g = 0; g < snr.size(); ++g)
        out.push_back("  segment " + std::to_string(g) + ": " +
                      (snr[g] ? "effective SNR " + fmt("%+.2f", *snr[g]) + " dB (median W)" : std::string("no data")));
    for (const Value& m : acc.members) {
        const std::string wave = st ? "  " + member_wave(*st, m.at("uid")) : std::string();
        // the f-string's fields, left to right
        std::string line = "    " + py_str(m.at("uid")) + wave;
        line += "  segment " + py_str(m.at("segment"));
        line += "  by " + py_str(m.at("method"));
        line += "  Z_ref " + fmt_value("%.1f", m.at("z_ref"));
        line += "  mean W " + fmt_value("%+.2f", m.at("mean_w_db")) + " dB";
        line += "  EM round " + py_str(m.at("em_round"));
        out.push_back(std::move(line));
    }
    return out;
}

Planes planes_from_pass(const types::PassResult& p) {
    if (!p.header)
        throw Refusal("the pass has no decoded header: its latents cannot be placed "
                      "(store it and decode the accumulator instead)");
    const header::HeaderFields& h = *p.header;
    Planes out;
    out.S.assign(store::PLANE, 0.0);
    out.W.assign(store::PLANE, 0.0);
    const auto idx = store::canonical_index(h.segment, static_cast<std::int64_t>(p.z.size()));
    const std::size_t nz = p.z.size(), nw = p.w.size();
    // w * z broadcast as numpy does (one value against many), then assigned
    // to the len(z) places idx names.
    if (nw != nz && nw != 1 && nz != 1)
        throw std::invalid_argument("operands could not be broadcast together with shapes (" + std::to_string(nw) +
                                    ",) (" + std::to_string(nz) + ",) ");
    if (nw != nz && nz == 1)
        throw std::invalid_argument("shape mismatch: value array of shape (" + std::to_string(nw) +
                                    ",) could not be broadcast to indexing result of shape (1,)");
    const std::size_t base = static_cast<std::size_t>(h.segment) * store::GROUP_LATENTS;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        const double w = static_cast<double>(p.w[nw == 1 ? 0 : i]);
        out.S[base + idx[i]] = w * static_cast<double>(p.z[i]);
        out.W[base + idx[i]] = w;
    }
    out.mode = h.mode;
    return out;
}

Planes planes_from_accumulator(const store::Accumulator& acc) {
    Planes out;
    out.S.assign(acc.S.begin(), acc.S.end());
    out.W.assign(acc.W.begin(), acc.W.end());
    out.mode = acc.mode;
    return out;
}

std::string pass_line(const types::PassResult& p) {
    // float(np.mean(p.w)): float32 arithmetic (nan for no values), then
    // max(m, 1e-30), which keeps a NaN.
    const double m = p.w.empty() ? std::numeric_limits<double>::quiet_NaN()
                                 : static_cast<double>(npsum::mean(std::span<const float>(p.w)));
    const double v = 1e-30 > m ? 1e-30 : m;
    return "pass " + p.uid + ": " + waveform_name(p.waveform) + " " + p.frame + " frame, mean W " +
           fmt("%+.2f", 10.0 * std::log10(v)) + " dB";
}

}  // namespace sstvae::qrss::decode
