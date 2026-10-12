// Python's text conventions, for refusal messages that must read exactly
// as sstvae/qrss's do (the CLI and the app show them, and tests grep them).
//
// Python strings are sequences of code points; the C++ port carries text
// as UTF-8. `decode` is the bridge: well-formed UTF-8 decodes as usual,
// a surrogate encoded in three bytes (what Python's "surrogatepass"
// error handler writes) decodes to the surrogate, and any other byte that
// does not start a well-formed sequence decodes to U+DC00 + byte --
// Python's "surrogateescape", which is how Python itself decodes an argv
// that is not valid UTF-8. So a C++ CLI handed the same bytes as a Python
// one counts the same characters and prints the same repr.

#ifndef SSTVAE_QRSS_PYTEXT_HPP
#define SSTVAE_QRSS_PYTEXT_HPP

#include <string>
#include <string_view>

namespace sstvae::qrss::pytext {

// The code points of `utf8`, decoded as described above.
std::u32string decode(std::string_view utf8);

// str.isprintable() of one code point. Exact for every assigned code
// point of Unicode 15.1 (Python 3.13's database); an unassigned one
// (category Cn) reads as printable here, where Python says it is not.
bool isprintable(char32_t cp);

// repr() of a str: the quote Python picks, its escapes (\\ \' \t \n \r,
// \xhh, \uhhhh, \Uhhhhhhhh, lowercase hex), non-printable code points
// escaped and printable ones kept (as UTF-8).
std::string repr(std::u32string_view text);
inline std::string repr(std::string_view utf8) { return repr(decode(utf8)); }

}  // namespace sstvae::qrss::pytext

#endif  // SSTVAE_QRSS_PYTEXT_HPP
