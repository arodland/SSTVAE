// How the qrss_*.py scripts spell a path: pathlib's PurePosixPath, on the
// text sstvae-qrss and the codec glue were given (UTF-8, or on POSIX the
// file name's own bytes: '/' and '.' are ASCII, so the bytes in between
// are carried through whatever they are).
//
// One copy for the CLI (apps/sstvae_qrss) and the codec glue (core/qrss/
// codec), which used to keep one each because the glue is not linked in a
// build without the codec.
//
// POSIX spelling on every platform: on Windows Python's Path is a
// WindowsPath and prints "a\\b" for "a/b"; sstvae-qrss keeps the slashes it
// was given. A message naming a path there differs in its separators only.

#ifndef SSTVAE_QRSS_PYPATH_HPP
#define SSTVAE_QRSS_PYPATH_HPP

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sstvae::qrss::pypath {

// pathlib's split of an already normalised path string: (root, parts),
// root "", "/" or "//"; "" and "." parts dropped.
std::pair<std::string, std::vector<std::string>> split(std::string_view text);

// The inverse: root and parts joined by '/', "." for nothing.
std::string join(const std::string& root, const std::vector<std::string>& parts);

// str(PurePosixPath(text)): "." for "", '.' components and repeated or
// trailing slashes dropped, exactly two leading slashes kept.
std::string pathlib_str(std::string_view text);

// PurePosixPath(text).name: the last part, "" for none.
std::string pathlib_name(std::string_view text);

// PurePosixPath(text).suffix: the name's last ".xyz", "" for none (".onnx"
// and "a." have none).
std::string pathlib_suffix(std::string_view text);

// str(PurePosixPath(dir) / name), for a name with no slash.
std::string child(std::string_view dir, const std::string& name);

// str(PurePosixPath(p).with_name(name)); std::invalid_argument "<p> has an
// empty name" for a path with no name (pathlib's ValueError).
std::string with_name(std::string_view p, const std::string& name);

// `text` (a message, which may name a file by its bytes) as the scripts'
// sys.stderr writes the str they hold for it, os.fsdecode(text): UTF-8 as
// it is, and a byte that is not part of a UTF-8 sequence as "\\udcXX"
// (surrogateescape's U+DCXX through stderr's "backslashreplace", which
// Python uses for stderr whatever the locale or PYTHONIOENCODING say). On
// Windows the file-system codec is "surrogatepass" (PEP 529), so a lone
// surrogate's three bytes (WTF-8, util/fileio.hpp) are one "\\udXXX". Every
// other platform reads them as three bytes that are not UTF-8.
//
// For a stream: appends to `out` what `text` gives, up to a sequence that
// the end of `text` cuts short, which is left for the next call (unless
// `final`), and returns how many bytes of `text` it used.
std::size_t stderr_text(std::string_view text, bool final, std::string& out);

}  // namespace sstvae::qrss::pypath

#endif  // SSTVAE_QRSS_PYPATH_HPP
