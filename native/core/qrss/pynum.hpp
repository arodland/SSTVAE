// Python's int() and float() on text, for the QRSS command line (plan stage
// 1 item 17): argparse's `type=int` / `type=float` and qrss_encode.py's
// int(args.codec_id, 16) accept and refuse by these rules, and the CLIs'
// refusals quote them.
//
// Python's int has no size limit, so a parsed integer comes back as its
// canonical decimal text (str(int(text))) and, when it fits, as an int64:
// a refusal that quotes the value ("segment must be 0..1, not
// 99999999999999999999") then reads as the reference's even where the C++
// cannot hold the number.
//
// ASCII only. Python also accepts Unicode whitespace around the number and
// Unicode decimal digits inside it (" 12", "١"); those are
// refused here, which a command line typed by an operator never meets.

#ifndef SSTVAE_QRSS_PYNUM_HPP
#define SSTVAE_QRSS_PYNUM_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace sstvae::qrss::pynum {

struct PyInt {
    std::string decimal;                // str() of the value: "-418", "0"
    std::optional<std::int64_t> value;  // when it fits in int64
};

// int(text, base) for base 10 or 16 (std::invalid_argument for another
// base): surrounding ASCII whitespace (\t \n \v \f \r and space), one
// sign, base 16's optional 0x/0X prefix after it, single underscores
// between digits (and one straight after the prefix). nullopt where Python
// raises ValueError.
std::optional<PyInt> py_int(std::string_view text, int base);

// The ValueError int(text, base) raises: "invalid literal for int() with
// base 16: 'xyz'" (the repr cut to 200 characters, as "%.200R").
std::string invalid_int_literal(std::string_view text, int base);

// float(text): surrounding whitespace, one sign, "inf", "infinity" or "nan"
// in any case, or decimal digits with an optional point and exponent and
// single underscores between digits. Correctly rounded (strtod), and '.'
// is the point whatever LC_NUMERIC says.
// nullopt where Python raises ValueError.
std::optional<double> py_float(std::string_view text);

}  // namespace sstvae::qrss::pynum

#endif  // SSTVAE_QRSS_PYNUM_HPP
