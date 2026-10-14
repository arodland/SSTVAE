// A picture written as a PNG file by its UTF-8 name (png.hpp).

#include "qrss/png.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "qrss/rawfile.hpp"
#include "util/fileio.hpp"

// The declarations only: stb's implementation is compiled once, in
// native/core/images/images.cpp (not STB_IMAGE_WRITE_STATIC, so its symbols
// are the library's).
#include "stb_image_write.h"

namespace sstvae::qrss::png {

namespace {

void append(void* context, void* data, int size) {
    auto* out = static_cast<std::vector<std::uint8_t>*>(context);
    const auto* p = static_cast<const std::uint8_t*>(data);
    if (size > 0) out->insert(out->end(), p, p + size);
}

[[noreturn]] void fail(int err, const std::string& path) {
    throw rawfile::FileError(std::error_code(err ? err : EIO, std::generic_category()), path);
}

}  // namespace

std::vector<std::uint8_t> png_bytes(const images::Picture& img) {
    const std::string what = std::to_string(img.width) + "x" + std::to_string(img.height);
    if (img.width < 0 || img.height < 0 ||
        img.rgb.size() != static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height) * 3)
        throw std::runtime_error("cannot encode a " + what + " picture as PNG: its pixels are not " + what + " RGB");
    std::vector<std::uint8_t> out;
    if (stbi_write_png_to_func(&append, &out, img.width, img.height, 3, img.rgb.data(), img.width * 3) == 0)
        throw std::runtime_error("cannot encode a " + what + " picture as PNG");
    return out;
}

void write_png(const images::Picture& img, const std::string& path) {
    const std::vector<std::uint8_t> data = png_bytes(img);
#ifdef _WIN32
    const std::size_t slash = path.find_last_of("/\\");
#else
    const std::size_t slash = path.find_last_of('/');
#endif
    const std::string parent = slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const std::string tmp = parent + "." + name + "." + std::to_string(rawfile::process_id()) + "." +
                            std::to_string(rawfile::thread_ident()) + ".tmp";
    try {
        {
            util::File f = util::open_file(tmp, "wb");
            if (!f) fail(errno, tmp);
            errno = 0;   // (stdio need not set it: fail() says EIO then)
            if (!data.empty() && std::fwrite(data.data(), 1, data.size(), f.get()) != data.size()) fail(errno, tmp);
            if (std::fflush(f.get()) != 0) fail(errno, tmp);
            std::FILE* raw = f.release();
            if (std::fclose(raw) != 0) fail(errno, tmp);
        }
        std::error_code ec;
        if (!util::replace_file(tmp, path, ec)) throw rawfile::FileError(ec, tmp);
    } catch (...) {
        std::error_code ignore;
        util::remove_file(tmp, ignore);
        throw;
    }
}

}  // namespace sstvae::qrss::png
