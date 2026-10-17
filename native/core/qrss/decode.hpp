// qrss_decode.py's helpers, the part of `sstvae-qrss decode` that needs no
// codec: parse_key, member_wave, acc_lines (with em.acc_mean_w_db) and
// planes_from_pass, and the line the script prints for a single pass.
// The codec half (render.load, render.render, the PNG) is
// qrss/codec/decode.hpp; the command line is apps/sstvae_qrss/decode.cpp.
//
// The texts are the script's, character for character, on a store either
// implementation wrote (tests/test_qrss_native_decode.py runs the two side
// by side). Numbers are formatted as Python's f-strings format them
// (`:+.2f`, `:.1f`, `:08x` with the sign inside the width, str() of a JSON
// value); 10 log10 is the C library's, where the script's comes from numpy
// (which may differ in the last place: only a value on a rounding boundary
// of its two decimals could print differently).
//
// Errors are the script's, as the exceptions it raises or lets escape:
// Refusal is its SystemExit (the message), PyException an uncaught
// exception the store layer does not already name (IndexError,
// TypeError, ValueError, with Python's class and text); and from the store
// (qrss/store.hpp) KeyError, FileError (OSError), std::invalid_argument
// (ValueError) and math::TypeError pass through. member_wave catches what
// the script catches there (OSError, KeyError, ValueError): "?".
//
// Plain functions of their arguments; thread-safe as the Store they are
// handed is (store.hpp).

#ifndef SSTVAE_QRSS_DECODE_HPP
#define SSTVAE_QRSS_DECODE_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/em.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/pynum.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::decode {

// raise SystemExit(message): what() is the message.
class Refusal : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// An exception the script would let escape, by the class its traceback's
// last line names ("IndexError") and its message.
class PyException : public std::runtime_error {
public:
    PyException(std::string cls, const std::string& message) : std::runtime_error(message), cls_(std::move(cls)) {}
    const std::string& cls() const noexcept { return cls_; }

private:
    std::string cls_;
};

// str.upper() of `text`: every UTF-8 code point by Python's full case
// mapping (U+00DF is "SS"; pyunicode_tables.hpp's UPPER), every byte that
// is not part of a UTF-8 sequence kept as it is -- what upper() does to the
// surrogate escape a str holds for it (and to an encoded lone surrogate).
std::string py_upper(std::string_view text);

// f"{v:0{width}x}", as Python formats a negative one: the sign inside the
// width (format(-5, "08x") is "-0000005").
std::string py_hex(std::int64_t v, int width);
// The same of an int of any size, given by its decimal text ("-12", as
// pynum::PyInt::decimal holds it).
std::string py_hex(std::string_view decimal, int width);

// str() and repr() of the object json.loads gives for a JSON value
// (None, True, 12, 0.1, 'text', [..], {..}; a float as repr, "nan" and
// "inf" for the non-finite ones json reads as NaN and Infinity).
std::string py_str(const pyjson::Value& v);
std::string py_repr(const pyjson::Value& v);

// parse_key(s): (call.upper(), int(pid, 16)) of "CALL:PICTUREID", split at
// its last ':'. `s` is the argument as sstvae-qrss holds it (UTF-8, or on
// POSIX a file name's bytes). Refusal "--key '<s>': expected CALL:PICTUREID
// (hexadecimal)" or "--key '<s>': picture ID '<pid>' is not hexadecimal".
// The picture ID is any int (int() has no limit): picture_id.value is
// nullopt beyond int64, where no accumulator can be keyed.
struct PictureKey {
    std::string callsign;
    pynum::PyInt picture_id;
};
PictureKey parse_key(std::string_view s);

// member_wave(store, uid): a stored pass's waveform and frame ("CE full",
// "L l-short", "CE ?" for a pass without a frame), "?" when it cannot be
// read (OSError, KeyError, ValueError). The uid as the member's JSON holds
// it: anything but a string is the script's TypeError. A pass file np.load
// cannot open is "?" here too (the C++ store's ValueError), where the script
// lets np.load's BadZipFile / EOFError escape (apps/sstvae_qrss/decode.cpp).
std::string member_wave(store::Store& store, const pyjson::Value& uid);

// em.acc_mean_w_db(acc): 10 log10 of the mean W over the latents heard
// (W > 0), -inf when none is. It lives in qrss/em.hpp (em.py's own) and is
// re-exported here for the callers that knew it as decode's.
using em::acc_mean_w_db;

// acc_lines(acc, store): the summary line, one line per segment, one per
// member (with its member_wave when `store` is given). PyException
// IndexError for a mode outside MODE_NAMES, KeyError for a member missing a
// field, TypeError / ValueError for a field that will not format.
std::vector<std::string> acc_lines(const store::Accumulator& acc, store::Store* store = nullptr);

// (S, W, mode): float64 [3 * 52800] each, canonical row-major.
struct Planes {
    std::vector<double> S, W;
    std::int64_t mode = 0;
};

// planes_from_pass(p): the pass at its segment's canonical offsets,
// S = w * z and W = w in float64. Refusal "the pass has no decoded
// header: ..." without a header; store::canonical_index's ValueError for
// a pass of no latents; std::invalid_argument (numpy's broadcast text)
// when z and w differ in length.
Planes planes_from_pass(const types::PassResult& p);

// acc.S, acc.W, acc.mode as render.render_accumulator hands them over
// (float32 to float64).
Planes planes_from_accumulator(const store::Accumulator& acc);

// The script's line for --pass: "pass <uid>: CE full frame, mean W +x.xx dB"
// (10 log10 of max(mean w, 1e-30), the mean in float32 as numpy's).
std::string pass_line(const types::PassResult& p);

}  // namespace sstvae::qrss::decode

#endif  // SSTVAE_QRSS_DECODE_HPP
