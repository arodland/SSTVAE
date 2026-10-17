// How the qrss_*.py scripts end, for their sstvae-qrss subcommands.
//
// - A refusal is `raise SystemExit(message)`: Python prints the message on
//   stderr and exits 1. system_exit() does the same.
// - An exception the script does not catch (a malformed .qrsp read by
//   qrss_beacon.py, int("xyz", 16) in qrss_encode.py, an unwritable output)
//   ends in a traceback whose last line is "ValueError: message" and exit
//   1. uncaught() prints that last line only: the native has no Python
//   stack to show, and a made-up one would mislead.
// - Files are read and written whole, with Python's OSError text on
//   failure ("FileNotFoundError: [Errno 2] No such file or directory:
//   'x.qrsp'"), and an output file is only created once its bytes exist,
//   so a refusal leaves none behind (as the scripts, which serialise
//   before or while opening the file, never fail between the two).

#ifndef SSTVAE_APPS_SSTVAE_QRSS_COMMON_HPP
#define SSTVAE_APPS_SSTVAE_QRSS_COMMON_HPP

#include <cstdint>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/picture.hpp"

namespace sstvae::qrss::cli {

// raise SystemExit(message): `message` and a newline on `err`; returns 1.
int system_exit(std::ostream& err, const std::string& message);

// An exception a script lets escape: the class as the traceback's last
// line names it ("ValueError", "zipfile.BadZipFile") and the message.
class PyError : public std::runtime_error {
public:
    PyError(std::string cls, const std::string& message) : std::runtime_error(message), cls_(std::move(cls)) {}
    const std::string& cls() const noexcept { return cls_; }

private:
    std::string cls_;
};

// An uncaught Python exception: "<cls>: <message>" on `err`; returns 1.
int uncaught(std::ostream& err, const std::string& cls, const std::string& message);
inline int uncaught(std::ostream& err, const PyError& e) { return uncaught(err, e.cls(), e.what()); }

// An OSError from open(): errno and the path, as Python formats it
// ("[Errno 2] No such file or directory: 'x.qrsp'"), with PEP 3151's
// class (FileNotFoundError, IsADirectoryError, PermissionError, ...).
class OsError : public PyError {
public:
    OsError(int err, const std::string& path);
    int err() const noexcept { return err_; }

private:
    int err_;
};

// open(path, "rb").read() / open(path, "wb").write(data); OsError on failure.
std::vector<std::uint8_t> read_file(const std::string& path);
void write_file(const std::string& path, std::span<const std::uint8_t> data);

// os.path.isfile(path): an existing regular file, symlinks followed.
bool is_file(const std::string& path);

// picture.load_qrsp(path) as a script that does not catch its exceptions
// sees it: the file read whole (OsError), np.load's view of what is not an
// .npz (EOFError for an empty file, the pickle refusal for anything else,
// a TypeError for an .npy), int() of a member that is no scalar number
// (numpy's TypeError for an array, OverflowError for infinity, ValueError
// for NaN, TypeError for a complex), and load_qrsp's own ValueErrors with
// their texts (`path` quoted as given). A damaged archive is zipfile.BadZipFile
// (a damaged member header numpy's ValueError) with the native parser's
// words, not zipfile's. PyError for all of them.
picture::StoredPicture load_qrsp(const std::string& path);

}  // namespace sstvae::qrss::cli

#endif  // SSTVAE_APPS_SSTVAE_QRSS_COMMON_HPP
