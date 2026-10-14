// Python's json module, as far as PassResult's `meta` needs it: json.dumps
// with its defaults and json.loads, on JSON values with Python's types
// (None, bool, int, float, str, list, dict in insertion order).
//
// dumps is byte for byte json.dumps(value): separators ", " and ": ",
// ensure_ascii (every code point outside space..'~' written \uXXXX, a
// code point past U+FFFF as its surrogate pair; '"', '\\', \b \f \n \r \t
// by their short escapes), floats as float.__repr__ except NaN,
// Infinity and -Infinity (allow_nan), ints in decimal, dict keys in
// insertion order. Strings are UTF-8 and may carry lone surrogates in the
// surrogatepass form (ED A0 80 ...), which come out as \udXXX, as Python
// writes them.
//
// loads accepts what json.loads accepts: RFC 8259 JSON plus NaN, Infinity
// and -Infinity, a repeated key keeping its last value (in the position the
// key first had, as a dict does), \u escapes with surrogate pairs combined
// and lone surrogates kept, control characters refused inside strings
// (strict=True). Numbers with a fraction or exponent are floats (correctly
// rounded, as Python's float()); the rest are ints, which must fit int64
// (std::overflow_error otherwise -- Python's are unbounded). Malformed text
// is std::invalid_argument naming the position, in json's words
// ("Expecting value: line 1 column 5 (char 4)") where it can.

#ifndef SSTVAE_QRSS_PYJSON_HPP
#define SSTVAE_QRSS_PYJSON_HPP

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sstvae::qrss::pyjson {

// A dict lookup of an absent key (Python's KeyError): what() is
// "KeyError: '<key>'", key() the key itself.
class KeyError : public std::out_of_range {
public:
    KeyError(std::string key, const std::string& what) : std::out_of_range(what), key_(std::move(key)) {}
    const std::string& key() const { return key_; }

private:
    std::string key_;
};

class Value {
public:
    enum class Kind : std::uint8_t { Null, Bool, Int, Float, String, Array, Object };

    Value() = default;   // None
    static Value null() { return Value(); }
    static Value boolean(bool b);
    static Value integer(std::int64_t i);
    static Value number(double f);
    static Value string(std::string s);
    static Value array(std::vector<Value> items = {});
    static Value object();

    Kind kind() const { return kind_; }
    bool is_null() const { return kind_ == Kind::Null; }

    // The value as its kind; std::invalid_argument naming `what` otherwise.
    bool as_bool(std::string_view what = "value") const;
    std::int64_t as_int(std::string_view what = "value") const;
    // A float, or an int converted (Python's float() of it).
    double as_double(std::string_view what = "value") const;
    const std::string& as_string(std::string_view what = "value") const;
    const std::vector<Value>& items() const;
    // A list's items to change in place (the store edits members and clusters).
    std::vector<Value>& items();

    // dict access, in insertion order.
    const std::vector<std::string>& keys() const;
    const std::vector<Value>& values() const;
    const Value* find(std::string_view key) const;
    // The value of `key` to change in place, or null when absent.
    Value* find(std::string_view key);
    // KeyError (a std::out_of_range) when absent.
    const Value& at(std::string_view key) const;
    // d[key] = v: replaces in place, or appends.
    void set(std::string key, Value v);
    // list.append
    void push_back(Value v);

    bool operator==(const Value& o) const;

private:
    Kind kind_ = Kind::Null;
    bool b_ = false;
    std::int64_t i_ = 0;
    double f_ = 0.0;
    std::string s_;
    std::vector<Value> items_;        // Array items, or Object values
    std::vector<std::string> keys_;   // Object keys
};

// json.dumps(v)
std::string dumps(const Value& v);
// json.dumps(v, indent=indent, sort_keys=sort_keys), byte for byte, for an
// int indent (the store writes indent=1, sort_keys=True). With an indent
// (0 included: newlines and no spaces; a negative one is 0, as ' ' * n is
// '') every non-empty list and dict puts each item on a line of its own,
// indented by `indent` spaces per level, the item separator becomes ","
// and the key separator stays ": " (json's defaults once indent is set);
// an empty one is still "[]" / "{}". Without one the separators are
// json.dumps' defaults, ", " and ": ". sort_keys orders each dict's items
// by key as Python's str comparison does (code points, which UTF-8 bytes
// compared unsigned preserve, surrogates included); otherwise insertion
// order. Every key is a str here, so sorting never meets the TypeError
// that mixed key types raise in Python.
std::string dumps(const Value& v, std::optional<int> indent, bool sort_keys);
// json.loads(text); text is UTF-8.
Value loads(std::string_view text);

}  // namespace sstvae::qrss::pyjson

#endif  // SSTVAE_QRSS_PYJSON_HPP
