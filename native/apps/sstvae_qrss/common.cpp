#include "common.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "qrss/npz.hpp"
#include "qrss/picture.hpp"
#include "qrss/pytext.hpp"
#include "util/fileio.hpp"

namespace sstvae::qrss::cli {

namespace {

// PEP 3151's OSError subclasses, by errno.
std::string os_error_class(int err) {
    switch (err) {
        case ENOENT: return "FileNotFoundError";
        case EISDIR: return "IsADirectoryError";
        case ENOTDIR: return "NotADirectoryError";
        case EACCES:
        case EPERM: return "PermissionError";
        case EEXIST: return "FileExistsError";
        case EINTR: return "InterruptedError";
        default: return "OSError";
    }
}

}  // namespace

int system_exit(std::ostream& err, const std::string& message) {
    err << message << "\n";
    return 1;
}

int uncaught(std::ostream& err, const std::string& cls, const std::string& message) {
    err << cls << ": " << message << "\n";
    return 1;
}

OsError::OsError(int err, const std::string& path)
    : PyError(os_error_class(err),
              "[Errno " + std::to_string(err) + "] " + std::strerror(err) + ": " + pytext::repr(path)),
      err_(err) {}

std::vector<std::uint8_t> read_file(const std::string& path) {
    errno = 0;
    const util::File f = util::open_file(path, "rb");
    if (!f) throw OsError(errno ? errno : EIO, path);
    std::vector<std::uint8_t> out;
    std::uint8_t buf[1 << 16];
    for (;;) {
        errno = 0;
        const std::size_t n = std::fread(buf, 1, sizeof buf, f.get());
        out.insert(out.end(), buf, buf + n);
        if (n < sizeof buf) {
            if (std::ferror(f.get())) throw OsError(errno ? errno : EIO, path);
            break;
        }
    }
    return out;
}

void write_file(const std::string& path, std::span<const std::uint8_t> data) {
    errno = 0;
    util::File f = util::open_file(path, "wb");
    if (!f) throw OsError(errno ? errno : EIO, path);
    errno = 0;
    if (!data.empty() && std::fwrite(data.data(), 1, data.size(), f.get()) != data.size())
        throw OsError(errno ? errno : EIO, path);
    errno = 0;
    if (std::fclose(f.release()) != 0) throw OsError(errno ? errno : EIO, path);
}

bool is_file(const std::string& path) { return util::is_file(path); }

picture::StoredPicture load_qrsp(const std::string& path) {
    const std::vector<std::uint8_t> data = read_file(path);
    // np.load reads six bytes and decides: a zip, an .npy, or a pickle.
    auto starts = [&](std::string_view magic) {
        return data.size() >= magic.size() && std::equal(magic.begin(), magic.end(), data.begin(),
                                                         [](char a, std::uint8_t b) { return static_cast<std::uint8_t>(a) == b; });
    };
    if (data.empty()) throw PyError("EOFError", "No data left in file");
    if (!starts(std::string_view("PK\x03\x04", 4)) && !starts(std::string_view("PK\x05\x06", 4))) {
        if (data.size() >= 6 && starts(std::string_view("\x93NUMPY", 6)))
            throw PyError("TypeError", "'numpy.ndarray' object does not support the context manager protocol");
        throw PyError("ValueError",
                      "This file contains pickled (object) data. If you trust the file you can load it unsafely "
                      "using the `allow_pickle=` keyword argument or `pickle.load()`.");
    }
    npz::Npz z;
    try {
        z = npz::parse_npz(data);
    } catch (const std::exception& e) {
        // The class, not the text: zipfile's for the archive, numpy's
        // ValueError for a member's .npy header ("npy: ..."). The reference
        // reads members lazily, so where a corrupt member and a missing
        // one meet, which is reported first can differ too.
        const std::string what = e.what();
        throw PyError(what.rfind("npy: ", 0) == 0 ? "ValueError" : "zipfile.BadZipFile", what);
    }
    try {
        return picture::from_npz(z, path);
    } catch (const std::overflow_error& e) {
        // int() of an infinite float member.
        throw PyError("OverflowError", e.what());
    } catch (const std::exception& e) {
        // int() of a member that is no number: numpy's and Python's words.
        const std::string what = e.what();
        if (what.rfind("npz: only 0-dimensional arrays", 0) == 0)
            throw PyError("TypeError", "only 0-dimensional arrays can be converted to Python scalars");
        if (what.rfind("npz: a complex array", 0) == 0)
            throw PyError("TypeError",
                          "int() argument must be a string, a bytes-like object or a real number, not 'complex'");
        throw PyError("ValueError", what);
    }
}

}  // namespace sstvae::qrss::cli
