// Positioned file I/O (rawfile.hpp). POSIX: open/pread/pwrite/ftruncate.
// Windows: CreateFileW with every share mode, ReadFile/WriteFile at an
// OVERLAPPED offset, SetFileInformationByHandle and FSCTL_SET_SPARSE --
// the only translation unit of the port that includes <windows.h>.

#include "qrss/rawfile.hpp"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <utility>

#include "util/fileio.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#include <process.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace sstvae::qrss::rawfile {

namespace {

std::error_code errno_code(int e) { return {e, std::generic_category()}; }

#ifdef _WIN32
std::error_code win_code(DWORD e) { return {static_cast<int>(e), std::system_category()}; }
#endif

}  // namespace

RawFile::~RawFile() { close(); }

#ifdef _WIN32
RawFile::RawFile(RawFile&& o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
RawFile& RawFile::operator=(RawFile&& o) noexcept {
    if (this != &o) {
        close();
        h_ = std::exchange(o.h_, nullptr);
    }
    return *this;
}
bool RawFile::is_open() const { return h_ != nullptr; }
void RawFile::close() {
    if (h_) ::CloseHandle(static_cast<HANDLE>(h_));
    h_ = nullptr;
}

RawFile RawFile::open(const std::string& path, Mode mode, std::error_code& ec) {
    ec.clear();
    RawFile f;
    if (path.find('\0') != std::string::npos) {
        ec = errno_code(EINVAL);
        return f;
    }
    const std::optional<std::wstring> w = util::wide_of(path);
    if (!w) {
        ec = errno_code(EILSEQ);
        return f;
    }
    const DWORD access = mode == Mode::Read ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
    const DWORD disp = mode == Mode::Read || mode == Mode::ReadWrite ? OPEN_EXISTING
                       : mode == Mode::Create                       ? OPEN_ALWAYS
                                                                    : CREATE_ALWAYS;
    HANDLE h = ::CreateFileW(w->c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                             disp, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        ec = win_code(::GetLastError());
        return f;
    }
    f.h_ = h;
    return f;
}

std::size_t RawFile::read_at(std::uint64_t offset, void* buf, std::size_t n, std::error_code& ec) const {
    ec.clear();
    auto* p = static_cast<char*>(buf);
    std::size_t got = 0;
    while (got < n) {
        OVERLAPPED ov{};
        const std::uint64_t at = offset + got;
        ov.Offset = static_cast<DWORD>(at & 0xFFFFFFFFu);
        ov.OffsetHigh = static_cast<DWORD>(at >> 32);
        const std::size_t left = n - got;
        const DWORD want = left > (1u << 30) ? (1u << 30) : static_cast<DWORD>(left);
        DWORD done = 0;
        if (!::ReadFile(static_cast<HANDLE>(h_), p + got, want, &done, &ov)) {
            const DWORD e = ::GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            ec = win_code(e);
            return 0;
        }
        if (done == 0) break;
        got += done;
    }
    return got;
}

bool RawFile::write_at(std::uint64_t offset, const void* buf, std::size_t n, std::error_code& ec) {
    ec.clear();
    const auto* p = static_cast<const char*>(buf);
    std::size_t put = 0;
    while (put < n) {
        OVERLAPPED ov{};
        const std::uint64_t at = offset + put;
        ov.Offset = static_cast<DWORD>(at & 0xFFFFFFFFu);
        ov.OffsetHigh = static_cast<DWORD>(at >> 32);
        const std::size_t left = n - put;
        const DWORD want = left > (1u << 30) ? (1u << 30) : static_cast<DWORD>(left);
        DWORD done = 0;
        if (!::WriteFile(static_cast<HANDLE>(h_), p + put, want, &done, &ov)) {
            ec = win_code(::GetLastError());
            return false;
        }
        put += done;
    }
    return true;
}

std::optional<std::uint64_t> RawFile::size(std::error_code& ec) const {
    ec.clear();
    LARGE_INTEGER s;
    if (!::GetFileSizeEx(static_cast<HANDLE>(h_), &s)) {
        ec = win_code(::GetLastError());
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(s.QuadPart);
}

bool RawFile::set_size(std::uint64_t n, std::error_code& ec) {
    ec.clear();
    FILE_END_OF_FILE_INFO info{};
    info.EndOfFile.QuadPart = static_cast<LONGLONG>(n);
    if (!::SetFileInformationByHandle(static_cast<HANDLE>(h_), FileEndOfFileInfo, &info, sizeof info)) {
        ec = win_code(::GetLastError());
        return false;
    }
    return true;
}

bool RawFile::make_sparse(std::error_code& ec) {
    ec.clear();
    DWORD ret = 0;
    if (!::DeviceIoControl(static_cast<HANDLE>(h_), FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &ret, nullptr)) {
        const DWORD e = ::GetLastError();
        // A file system without sparse files (FAT, some network shares):
        // the file is simply allocated, as Python's memmap is there.
        if (e == ERROR_INVALID_FUNCTION || e == ERROR_NOT_SUPPORTED) return true;
        ec = win_code(e);
        return false;
    }
    return true;
}

bool RawFile::sync(std::error_code& ec) {
    ec.clear();
    if (!::FlushFileBuffers(static_cast<HANDLE>(h_))) {
        ec = win_code(::GetLastError());
        return false;
    }
    return true;
}

std::optional<std::int64_t> mtime_ns(const std::string& path, std::error_code& ec) {
    ec.clear();
    const std::optional<std::wstring> w = util::wide_of(path);
    if (!w || path.find('\0') != std::string::npos) {
        ec = errno_code(EINVAL);
        return std::nullopt;
    }
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!::GetFileAttributesExW(w->c_str(), GetFileExInfoStandard, &a)) {
        ec = win_code(::GetLastError());
        return std::nullopt;
    }
    const std::uint64_t t = (static_cast<std::uint64_t>(a.ftLastWriteTime.dwHighDateTime) << 32) |
                            a.ftLastWriteTime.dwLowDateTime;
    return (static_cast<std::int64_t>(t) - 116444736000000000LL) * 100;
}

std::int64_t process_id() { return static_cast<std::int64_t>(::GetCurrentProcessId()); }
std::uint64_t thread_ident() { return static_cast<std::uint64_t>(::GetCurrentThreadId()); }

#else  // POSIX

RawFile::RawFile(RawFile&& o) noexcept : fd_(std::exchange(o.fd_, -1)) {}
RawFile& RawFile::operator=(RawFile&& o) noexcept {
    if (this != &o) {
        close();
        fd_ = std::exchange(o.fd_, -1);
    }
    return *this;
}
bool RawFile::is_open() const { return fd_ >= 0; }
void RawFile::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

RawFile RawFile::open(const std::string& path, Mode mode, std::error_code& ec) {
    ec.clear();
    RawFile f;
    if (path.find('\0') != std::string::npos) {
        ec = errno_code(EINVAL);
        return f;
    }
    int flags = O_CLOEXEC;
    switch (mode) {
        case Mode::Read: flags |= O_RDONLY; break;
        case Mode::ReadWrite: flags |= O_RDWR; break;
        case Mode::Create: flags |= O_RDWR | O_CREAT; break;
        case Mode::Truncate: flags |= O_RDWR | O_CREAT | O_TRUNC; break;
    }
    int fd;
    do {
        fd = ::open(path.c_str(), flags, 0666);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        ec = errno_code(errno);
        return f;
    }
    f.fd_ = fd;
    return f;
}

std::size_t RawFile::read_at(std::uint64_t offset, void* buf, std::size_t n, std::error_code& ec) const {
    ec.clear();
    auto* p = static_cast<char*>(buf);
    std::size_t got = 0;
    while (got < n) {
        const ssize_t r = ::pread(fd_, p + got, n - got, static_cast<off_t>(offset + got));
        if (r < 0) {
            if (errno == EINTR) continue;
            ec = errno_code(errno);
            return 0;
        }
        if (r == 0) break;
        got += static_cast<std::size_t>(r);
    }
    return got;
}

bool RawFile::write_at(std::uint64_t offset, const void* buf, std::size_t n, std::error_code& ec) {
    ec.clear();
    const auto* p = static_cast<const char*>(buf);
    std::size_t put = 0;
    while (put < n) {
        const ssize_t r = ::pwrite(fd_, p + put, n - put, static_cast<off_t>(offset + put));
        if (r < 0) {
            if (errno == EINTR) continue;
            ec = errno_code(errno);
            return false;
        }
        put += static_cast<std::size_t>(r);
    }
    return true;
}

std::optional<std::uint64_t> RawFile::size(std::error_code& ec) const {
    ec.clear();
    struct stat st;
    if (::fstat(fd_, &st) != 0) {
        ec = errno_code(errno);
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(st.st_size);
}

bool RawFile::set_size(std::uint64_t n, std::error_code& ec) {
    ec.clear();
    int r;
    do {
        r = ::ftruncate(fd_, static_cast<off_t>(n));
    } while (r != 0 && errno == EINTR);
    if (r != 0) {
        ec = errno_code(errno);
        return false;
    }
    return true;
}

bool RawFile::make_sparse(std::error_code& ec) {
    ec.clear();
    return true;
}

bool RawFile::sync(std::error_code& ec) {
    ec.clear();
    if (::fsync(fd_) != 0) {
        ec = errno_code(errno);
        return false;
    }
    return true;
}

std::optional<std::int64_t> mtime_ns(const std::string& path, std::error_code& ec) {
    ec.clear();
    if (path.find('\0') != std::string::npos) {
        ec = errno_code(EINVAL);
        return std::nullopt;
    }
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        ec = errno_code(errno);
        return std::nullopt;
    }
#if defined(__APPLE__)
    return static_cast<std::int64_t>(st.st_mtimespec.tv_sec) * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
    return static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
#endif
}

std::int64_t process_id() { return static_cast<std::int64_t>(::getpid()); }
std::uint64_t thread_ident() {
    // CPython's PyThread_get_thread_ident: (unsigned long)pthread_self().
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(reinterpret_cast<void*>(::pthread_self())));
}

#endif

std::optional<std::string> read_all(const std::string& path, std::error_code& ec) {
    ec.clear();
    RawFile f = RawFile::open(path, Mode::Read, ec);
    if (!f.is_open()) return std::nullopt;
    std::string out;
    char buf[1 << 14];
    std::uint64_t at = 0;
    for (;;) {
        const std::size_t r = f.read_at(at, buf, sizeof buf, ec);
        if (ec) return std::nullopt;
        if (r == 0) break;
        out.append(buf, r);
        at += r;
    }
    return out;
}

bool write_replace(const std::string& tmp, const std::string& path, std::string_view data, std::error_code& ec) {
    ec.clear();
    {
        RawFile f = RawFile::open(tmp, Mode::Truncate, ec);
        if (!f.is_open()) return false;
        if (!f.write_at(0, data.data(), data.size(), ec)) {
            f.close();
            std::error_code ignore;
            util::remove_file(tmp, ignore);
            return false;
        }
    }
    if (!util::replace_file(tmp, path, ec)) {
        std::error_code ignore;
        util::remove_file(tmp, ignore);
        return false;
    }
    return true;
}

bool make_dirs(const std::string& dir, std::error_code& ec) {
    ec.clear();
    const auto p = util::fs_path(dir);
    if (!p) {
        ec = errno_code(EINVAL);
        return false;
    }
    std::filesystem::create_directories(*p, ec);
    if (ec) {
        // exist_ok: an existing directory is fine (create_directories
        // reports EEXIST for some races); anything else is not.
        std::error_code e2;
        if (std::filesystem::is_directory(*p, e2)) {
            ec.clear();
            return true;
        }
        return false;
    }
    std::error_code e2;
    if (!std::filesystem::is_directory(*p, e2)) {
        ec = errno_code(EEXIST);
        return false;
    }
    return true;
}

}  // namespace sstvae::qrss::rawfile
