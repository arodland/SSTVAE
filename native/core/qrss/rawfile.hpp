// Positioned file I/O for the stage-3 stores (passband hours, the FLAC
// packer, the directory lock's neighbours): open a file by its UTF-8 name,
// read and write at an offset, size it, make it sparse, ask its mtime.
//
// Names go through util/fileio.hpp's conversions (bytes on POSIX; UTF-16
// through fs_path/wide_of on Windows), so this is the same "one way to
// name a file" as everything else. What it adds is what a FILE* cannot
// do: pread/pwrite (no shared cursor, so two threads may read one handle
// at once), ftruncate to a size with no data (a sparse 57.6 MB hour), and
// on Windows FSCTL_SET_SPARSE, without which that size is allocated. The
// Windows half lives in rawfile.cpp only, so no header here pulls in
// <windows.h> (CLAUDE.md).
//
// On Windows a file is opened with FILE_SHARE_READ | FILE_SHARE_WRITE |
// FILE_SHARE_DELETE, so a reader holding an hour open never stops the
// packer's rename or expiry's delete (os.replace / unlink of an open file
// otherwise fail there with a sharing violation).
//
// Thread safety: a RawFile may be read (read_at) from several threads at
// once; writes and set_size need the caller's own exclusion against each
// other, as with any file. All calls block for the duration of the I/O
// they do (a page-cache copy normally; a disk read on a cold cache).

#ifndef SSTVAE_QRSS_RAWFILE_HPP
#define SSTVAE_QRSS_RAWFILE_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace sstvae::qrss::rawfile {

// An OSError: errno (or a Windows error), and the file it concerns (UTF-8),
// for the stores' callers to raise as Python's OSError naming that file.
class FileError : public std::system_error {
public:
    FileError(std::error_code ec, std::string path) : std::system_error(ec, path), path_(std::move(path)) {}
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

enum class Mode {
    Read,        // existing file, read only
    ReadWrite,   // existing file, read and write
    Create,      // read and write, created if absent (never truncated)
    Truncate,    // read and write, created or truncated to 0
};

class RawFile {
public:
    RawFile() = default;
    ~RawFile();
    RawFile(RawFile&& o) noexcept;
    RawFile& operator=(RawFile&& o) noexcept;
    RawFile(const RawFile&) = delete;
    RawFile& operator=(const RawFile&) = delete;

    // Opens `path` (UTF-8); on failure is_open() is false and `ec` says why
    // (the system's errno; EINVAL for a name holding a NUL byte).
    static RawFile open(const std::string& path, Mode mode, std::error_code& ec);

    bool is_open() const;
    void close();

    // Reads up to n bytes at `offset`: the count read, short only at the
    // end of the file. 0 with `ec` set on an error.
    std::size_t read_at(std::uint64_t offset, void* buf, std::size_t n, std::error_code& ec) const;
    // Writes all n bytes at `offset` (extending the file as needed).
    bool write_at(std::uint64_t offset, const void* buf, std::size_t n, std::error_code& ec);
    // The file's size in bytes.
    std::optional<std::uint64_t> size(std::error_code& ec) const;
    // ftruncate: grows with a hole (no data written) or shrinks.
    bool set_size(std::uint64_t n, std::error_code& ec);
    // Marks the file sparse (Windows: FSCTL_SET_SPARSE; elsewhere every
    // file system that can hold holes does so without being asked, and
    // this is a no-op returning true). Call before set_size grows it.
    bool make_sparse(std::error_code& ec);
    // Data and metadata to the device (fsync / FlushFileBuffers).
    bool sync(std::error_code& ec);

private:
#ifdef _WIN32
    void* h_ = nullptr;   // HANDLE; null when closed (INVALID_HANDLE_VALUE never stored)
#else
    int fd_ = -1;
#endif
};

// os.stat(path).st_mtime_ns; nullopt with `ec` set when it cannot be
// stat'ed. On Windows the last-write FILETIME as ns since 1970 (Python's
// conversion), so values compare with Python's own for the same file.
std::optional<std::int64_t> mtime_ns(const std::string& path, std::error_code& ec);

// The whole file as bytes; nullopt with `ec` set when it cannot be read
// (ENOENT for a missing file).
std::optional<std::string> read_all(const std::string& path, std::error_code& ec);

// Writes `data` to `tmp`, then moves it over `path` (os.replace). The
// temporary is removed again when the move fails. False with `ec` set.
bool write_replace(const std::string& tmp, const std::string& path, std::string_view data, std::error_code& ec);

// mkdir -p, as Path.mkdir(parents=True, exist_ok=True) does it: false
// with `ec` set when `dir` cannot be made or exists as something other
// than a directory (EEXIST), and `failed` (when given) naming the path
// whose mkdir failed -- `dir` or one of its parents, as the reference's
// OSError does.
bool make_dirs(const std::string& dir, std::error_code& ec, std::string* failed = nullptr);

// The calling process's id and thread's id, as os.getpid() and
// threading.get_ident() give them (pthread_self / GetCurrentThreadId),
// for temporary names a concurrent Python process could meet.
std::int64_t process_id();
std::uint64_t thread_ident();

}  // namespace sstvae::qrss::rawfile

#endif  // SSTVAE_QRSS_RAWFILE_HPP
