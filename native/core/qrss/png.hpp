// A picture written as a PNG file by its UTF-8 name, on every platform.
//
// images::save_png (SSTVAE's, which this port may not edit) hands the name
// to stbi_write_png, which opens it with a narrow fopen: on Windows that
// reads the name through the ANSI code page, so a picture under a name
// outside it (a Cyrillic user name in the state directory's path, say) is
// not written at all. This encodes with the same stb encoder, through
// stbi_write_png_to_func (stb's implementation is compiled once, in
// images.cpp), and writes the bytes through util/fileio.hpp -- the port's
// one way to open a file -- to a temporary sibling that is then moved into
// place, as every other file the QRSS code writes. The bytes are exactly
// images::save_png's for the same picture (native/tests/qrss/test_png.cpp
// compares them): stb's PNG writer builds the whole file in memory and
// stbi_write_png only fwrites that buffer.
//
// Threads: no state of its own (stb's encoder reads two globals, its
// compression level and forced filter, which nothing here sets), so any
// number of threads may write at once -- to different paths; two writers of
// one path each write their own temporary, and the last move wins. Blocks
// for the encoding (~10 ms for 640x480 RGB) and the write (a page-cache
// copy of ~0.5 MB).

#ifndef SSTVAE_QRSS_PNG_HPP
#define SSTVAE_QRSS_PNG_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "images/types.hpp"

namespace sstvae::qrss::png {

// The PNG file images::save_png writes for `img` (RGB, 8 bits a channel).
// std::runtime_error "cannot encode a <w>x<h> picture as PNG" when stb
// cannot (no pixels, or a buffer that does not hold width x height x 3).
std::vector<std::uint8_t> png_bytes(const images::Picture& img);

// Writes png_bytes(img) to `path` (UTF-8): to the temporary sibling
// ".<name>.<pid>.<thread id>.tmp" (store.py's _tmp_name), flushed, closed
// and moved over `path` with util::replace_file (os.replace). On failure
// the temporary is removed and the error thrown: png_bytes' refusal, or
// rawfile::FileError (an errno and the file it concerns) when the
// temporary cannot be opened or written, or cannot replace `path`.
void write_png(const images::Picture& img, const std::string& path);

}  // namespace sstvae::qrss::png

#endif  // SSTVAE_QRSS_PNG_HPP
