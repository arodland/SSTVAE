#include "qrss/pyjson.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "qrss/frame.hpp"
#include "qrss/pynum.hpp"

namespace sstvae::qrss::pyjson {

namespace {

const char* kind_name(Value::Kind k) {
    switch (k) {
        case Value::Kind::Null: return "null";
        case Value::Kind::Bool: return "bool";
        case Value::Kind::Int: return "int";
        case Value::Kind::Float: return "float";
        case Value::Kind::String: return "str";
        case Value::Kind::Array: return "list";
        case Value::Kind::Object: return "dict";
    }
    return "?";
}

[[noreturn]] void wrong(std::string_view what, const char* want, Value::Kind got) {
    throw std::invalid_argument(std::string(what) + ": expected " + want + ", got " + kind_name(got));
}

// --- dumps ------------------------------------------------------------------

void append_utf16_escape(std::string& out, std::uint32_t u) {
    static const char* hex = "0123456789abcdef";
    out += "\\u";
    out += hex[(u >> 12) & 0xF];
    out += hex[(u >> 8) & 0xF];
    out += hex[(u >> 4) & 0xF];
    out += hex[u & 0xF];
}

// Decode one UTF-8 sequence (surrogates in their 3-byte form accepted).
std::uint32_t next_code_point(std::string_view s, std::size_t& i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    auto cont = [&](std::size_t k) -> std::uint32_t {
        if (i + k >= s.size()) throw std::invalid_argument("json.dumps: truncated UTF-8 in a string");
        const auto b = static_cast<unsigned char>(s[i + k]);
        if ((b & 0xC0) != 0x80) throw std::invalid_argument("json.dumps: invalid UTF-8 in a string");
        return b & 0x3Fu;
    };
    if (b0 < 0x80) {
        i += 1;
        return b0;
    }
    if ((b0 & 0xE0) == 0xC0) {
        const std::uint32_t u = ((b0 & 0x1Fu) << 6) | cont(1);
        i += 2;
        return u;
    }
    if ((b0 & 0xF0) == 0xE0) {
        const std::uint32_t u = ((b0 & 0x0Fu) << 12) | (cont(1) << 6) | cont(2);
        i += 3;
        return u;
    }
    if ((b0 & 0xF8) == 0xF0) {
        const std::uint32_t u = ((b0 & 0x07u) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3);
        i += 4;
        return u;
    }
    throw std::invalid_argument("json.dumps: invalid UTF-8 in a string");
}

// json's py_encode_basestring_ascii.
void dump_string(std::string& out, std::string_view s) {
    out += '"';
    for (std::size_t i = 0; i < s.size();) {
        const std::uint32_t u = next_code_point(s, i);
        switch (u) {
            case '\\': out += "\\\\"; continue;
            case '"': out += "\\\""; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            default: break;
        }
        if (u >= 0x20 && u <= 0x7E) {
            out += static_cast<char>(u);
        } else if (u > 0xFFFF) {
            const std::uint32_t v = u - 0x10000;
            append_utf16_escape(out, 0xD800 | ((v >> 10) & 0x3FF));
            append_utf16_escape(out, 0xDC00 | (v & 0x3FF));
        } else {
            append_utf16_escape(out, u);
        }
    }
    out += '"';
}

void dump(std::string& out, const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Null: out += "null"; return;
        case Value::Kind::Bool: out += v.as_bool() ? "true" : "false"; return;
        case Value::Kind::Int: out += std::to_string(v.as_int()); return;
        case Value::Kind::Float: {
            const double f = v.as_double();
            if (std::isnan(f))
                out += "NaN";
            else if (std::isinf(f))
                out += f > 0 ? "Infinity" : "-Infinity";
            else
                out += frame::py_repr(f);
            return;
        }
        case Value::Kind::String: dump_string(out, v.as_string()); return;
        case Value::Kind::Array: {
            out += '[';
            bool first = true;
            for (const Value& e : v.items()) {
                if (!first) out += ", ";
                first = false;
                dump(out, e);
            }
            out += ']';
            return;
        }
        case Value::Kind::Object: {
            out += '{';
            for (std::size_t k = 0; k < v.keys().size(); ++k) {
                if (k) out += ", ";
                dump_string(out, v.keys()[k]);
                out += ": ";
                dump(out, v.values()[k]);
            }
            out += '}';
            return;
        }
    }
}

// --- loads ------------------------------------------------------------------

void put_utf8(std::string& out, std::uint32_t u) {
    if (u < 0x80) {
        out += static_cast<char>(u);
    } else if (u < 0x800) {
        out += static_cast<char>(0xC0 | (u >> 6));
        out += static_cast<char>(0x80 | (u & 0x3F));
    } else if (u < 0x10000) {   // surrogates too, as surrogatepass writes them
        out += static_cast<char>(0xE0 | (u >> 12));
        out += static_cast<char>(0x80 | ((u >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (u & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (u >> 18));
        out += static_cast<char>(0x80 | ((u >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((u >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (u & 0x3F));
    }
}

class Parser {
public:
    explicit Parser(std::string_view s) : s_(s) {}

    Value document() {
        skip_ws();
        Value v = value();
        skip_ws();
        if (i_ != s_.size()) fail("Extra data", i_);
        return v;
    }

private:
    std::string_view s_;
    std::size_t i_ = 0;

    [[noreturn]] void fail(const char* msg, std::size_t pos) const {
        std::size_t line = 1, col_start = 0;
        for (std::size_t k = 0; k < pos && k < s_.size(); ++k)
            if (s_[k] == '\n') {
                ++line;
                col_start = k + 1;
            }
        throw std::invalid_argument(std::string(msg) + ": line " + std::to_string(line) + " column " +
                                    std::to_string(pos - col_start + 1) + " (char " + std::to_string(pos) + ")");
    }

    void skip_ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
    }

    bool literal(std::string_view word) {
        if (s_.substr(i_, word.size()) == word) {
            i_ += word.size();
            return true;
        }
        return false;
    }

    Value value() {
        if (i_ >= s_.size()) fail("Expecting value", i_);
        const char c = s_[i_];
        if (c == '"') return Value::string(string());
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == 'n' && literal("null")) return Value::null();
        if (c == 't' && literal("true")) return Value::boolean(true);
        if (c == 'f' && literal("false")) return Value::boolean(false);
        if (c == 'N' && literal("NaN")) return Value::number(std::numeric_limits<double>::quiet_NaN());
        if (c == 'I' && literal("Infinity")) return Value::number(std::numeric_limits<double>::infinity());
        if (c == '-' && literal("-Infinity")) return Value::number(-std::numeric_limits<double>::infinity());
        return number();
    }

    Value number() {
        // json's NUMBER_RE: (-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?
        const std::size_t start = i_;
        std::size_t k = i_;
        auto digit = [&](std::size_t p) { return p < s_.size() && s_[p] >= '0' && s_[p] <= '9'; };
        if (k < s_.size() && s_[k] == '-') ++k;
        if (!digit(k)) fail("Expecting value", start);
        if (s_[k] == '0')
            ++k;
        else
            while (digit(k)) ++k;
        const std::size_t int_end = k;
        bool is_float = false;
        if (k < s_.size() && s_[k] == '.' && digit(k + 1)) {
            k += 1;
            while (digit(k)) ++k;
            is_float = true;
        }
        if (k < s_.size() && (s_[k] == 'e' || s_[k] == 'E')) {
            std::size_t e = k + 1;
            if (e < s_.size() && (s_[e] == '+' || s_[e] == '-')) ++e;
            if (digit(e)) {
                while (digit(e)) ++e;
                k = e;
                is_float = true;
            }
        }
        i_ = k;
        const std::string_view text = s_.substr(start, k - start);
        if (is_float) {
            const auto f = pynum::py_float(text);
            if (!f) fail("Expecting value", start);
            return Value::number(*f);
        }
        const auto parsed = pynum::py_int(s_.substr(start, int_end - start), 10);
        if (!parsed) fail("Expecting value", start);
        if (!parsed->value)
            throw std::overflow_error("json integer " + parsed->decimal + " does not fit in int64");
        return Value::integer(*parsed->value);
    }

    std::uint32_t hex4(std::size_t p) const {
        if (p + 4 > s_.size()) fail("Invalid \\uXXXX escape", p - 1);
        std::uint32_t u = 0;
        for (std::size_t k = 0; k < 4; ++k) {
            const char c = s_[p + k];
            u <<= 4;
            if (c >= '0' && c <= '9')
                u |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f')
                u |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                u |= static_cast<std::uint32_t>(c - 'A' + 10);
            else
                fail("Invalid \\uXXXX escape", p - 1);
        }
        return u;
    }

    std::string string() {
        const std::size_t begin = i_;   // at the opening quote
        ++i_;
        std::string out;
        while (true) {
            if (i_ >= s_.size()) fail("Unterminated string starting at", begin);
            const auto c = static_cast<unsigned char>(s_[i_]);
            if (c == '"') {
                ++i_;
                return out;
            }
            if (c < 0x20) fail("Invalid control character at", i_);
            if (c != '\\') {
                out += static_cast<char>(c);
                ++i_;
                continue;
            }
            if (i_ + 1 >= s_.size()) fail("Unterminated string starting at", begin);
            const char e = s_[i_ + 1];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    std::uint32_t u = hex4(i_ + 2);
                    i_ += 6;
                    if (u >= 0xD800 && u <= 0xDBFF && i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                        const std::uint32_t u2 = hex4(i_ + 2);
                        if (u2 >= 0xDC00 && u2 <= 0xDFFF) {
                            u = 0x10000 + (((u - 0xD800) << 10) | (u2 - 0xDC00));
                            i_ += 6;
                        }
                    }
                    put_utf8(out, u);
                    continue;
                }
                default: fail("Invalid \\escape", i_);
            }
            i_ += 2;
        }
    }

    Value object() {
        const std::size_t open = i_;
        (void)open;
        ++i_;
        Value obj = Value::object();
        skip_ws();
        if (i_ < s_.size() && s_[i_] == '}') {
            ++i_;
            return obj;
        }
        while (true) {
            skip_ws();
            if (i_ >= s_.size() || s_[i_] != '"') fail("Expecting property name enclosed in double quotes", i_);
            std::string key = string();
            skip_ws();
            if (i_ >= s_.size() || s_[i_] != ':') fail("Expecting ':' delimiter", i_);
            ++i_;
            skip_ws();
            Value v = value();
            obj.set(std::move(key), std::move(v));
            skip_ws();
            if (i_ < s_.size() && s_[i_] == '}') {
                ++i_;
                return obj;
            }
            if (i_ >= s_.size() || s_[i_] != ',') fail("Expecting ',' delimiter", i_);
            ++i_;
        }
    }

    Value array() {
        ++i_;
        Value arr = Value::array();
        skip_ws();
        if (i_ < s_.size() && s_[i_] == ']') {
            ++i_;
            return arr;
        }
        while (true) {
            skip_ws();
            arr.push_back(value());
            skip_ws();
            if (i_ < s_.size() && s_[i_] == ']') {
                ++i_;
                return arr;
            }
            if (i_ >= s_.size() || s_[i_] != ',') fail("Expecting ',' delimiter", i_);
            ++i_;
        }
    }
};

}  // namespace

Value Value::boolean(bool b) {
    Value v;
    v.kind_ = Kind::Bool;
    v.b_ = b;
    return v;
}

Value Value::integer(std::int64_t i) {
    Value v;
    v.kind_ = Kind::Int;
    v.i_ = i;
    return v;
}

Value Value::number(double f) {
    Value v;
    v.kind_ = Kind::Float;
    v.f_ = f;
    return v;
}

Value Value::string(std::string s) {
    Value v;
    v.kind_ = Kind::String;
    v.s_ = std::move(s);
    return v;
}

Value Value::array(std::vector<Value> items) {
    Value v;
    v.kind_ = Kind::Array;
    v.items_ = std::move(items);
    return v;
}

Value Value::object() {
    Value v;
    v.kind_ = Kind::Object;
    return v;
}

bool Value::as_bool(std::string_view what) const {
    if (kind_ != Kind::Bool) wrong(what, "bool", kind_);
    return b_;
}

std::int64_t Value::as_int(std::string_view what) const {
    if (kind_ != Kind::Int) wrong(what, "int", kind_);
    return i_;
}

double Value::as_double(std::string_view what) const {
    if (kind_ == Kind::Float) return f_;
    if (kind_ == Kind::Int) return static_cast<double>(i_);
    wrong(what, "float", kind_);
}

const std::string& Value::as_string(std::string_view what) const {
    if (kind_ != Kind::String) wrong(what, "str", kind_);
    return s_;
}

const std::vector<Value>& Value::items() const {
    if (kind_ != Kind::Array) wrong("value", "list", kind_);
    return items_;
}

const std::vector<std::string>& Value::keys() const {
    if (kind_ != Kind::Object) wrong("value", "dict", kind_);
    return keys_;
}

const std::vector<Value>& Value::values() const {
    if (kind_ != Kind::Object) wrong("value", "dict", kind_);
    return items_;
}

const Value* Value::find(std::string_view key) const {
    if (kind_ != Kind::Object) wrong("value", "dict", kind_);
    for (std::size_t k = 0; k < keys_.size(); ++k)
        if (keys_[k] == key) return &items_[k];
    return nullptr;
}

const Value& Value::at(std::string_view key) const {
    const Value* v = find(key);
    if (!v) throw std::out_of_range("KeyError: " + frame::py_repr(key));
    return *v;
}

void Value::set(std::string key, Value v) {
    if (kind_ != Kind::Object) wrong("value", "dict", kind_);
    for (std::size_t k = 0; k < keys_.size(); ++k)
        if (keys_[k] == key) {
            items_[k] = std::move(v);
            return;
        }
    keys_.push_back(std::move(key));
    items_.push_back(std::move(v));
}

void Value::push_back(Value v) {
    if (kind_ != Kind::Array) wrong("value", "list", kind_);
    items_.push_back(std::move(v));
}

bool Value::operator==(const Value& o) const {
    if (kind_ != o.kind_) return false;
    switch (kind_) {
        case Kind::Null: return true;
        case Kind::Bool: return b_ == o.b_;
        case Kind::Int: return i_ == o.i_;
        case Kind::Float: return f_ == o.f_ || (std::isnan(f_) && std::isnan(o.f_));
        case Kind::String: return s_ == o.s_;
        case Kind::Array: return items_ == o.items_;
        case Kind::Object: return keys_ == o.keys_ && items_ == o.items_;
    }
    return false;
}

std::string dumps(const Value& v) {
    std::string out;
    dump(out, v);
    return out;
}

Value loads(std::string_view text) { return Parser(text).document(); }

}  // namespace sstvae::qrss::pyjson
