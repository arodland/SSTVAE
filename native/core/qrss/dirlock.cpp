// live.dir_lock: see dirlock.hpp. The Windows half is here only, so no
// header pulls in <windows.h> (CLAUDE.md). Every FileError thrown here
// carries an errno code (util::errno_code_of), on Windows too.

#include "qrss/dirlock.hpp"

#include <cerrno>
#include <string>
#include <system_error>
#include <utility>

#include "qrss/rawfile.hpp"
#include "util/fileio.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace sstvae::qrss::dirlock {

namespace {

std::string lock_path(const std::string& dir) {
    if (dir.empty()) return ".lock";
    const char last = dir.back();
#ifdef _WIN32
    if (last == '/' || last == '\\') return dir + ".lock";
#else
    if (last == '/') return dir + ".lock";
#endif
    return dir + "/.lock";
}

// "<pid>\n" as the reference's text-mode file writes it.
std::string pid_line() {
#ifdef _WIN32
    return std::to_string(rawfile::process_id()) + "\r\n";
#else
    return std::to_string(rawfile::process_id()) + "\n";
#endif
}

}  // namespace

DirLock::DirLock(DirLock&& o) noexcept : path_(std::move(o.path_)) {
#ifdef _WIN32
    h_ = std::exchange(o.h_, nullptr);
#else
    fd_ = std::exchange(o.fd_, -1);
#endif
}

DirLock& DirLock::operator=(DirLock&& o) noexcept {
    if (this != &o) {
        close();
        path_ = std::move(o.path_);
#ifdef _WIN32
        h_ = std::exchange(o.h_, nullptr);
#else
        fd_ = std::exchange(o.fd_, -1);
#endif
    }
    return *this;
}

DirLock::~DirLock() { close(); }

#ifdef _WIN32

namespace {

// The last Win32 error as an errno code (util::errno_of: ERROR_ACCESS_DENIED
// -> EACCES, EIO for one with no errno), which is what the bindings and the
// CLI read a FileError's code as.
std::error_code win32_error() {
    return util::errno_code_of(std::error_code(static_cast<int>(::GetLastError()), std::system_category()));
}

}  // namespace

std::optional<DirLock> DirLock::try_acquire(const std::string& dir) {
    std::error_code ec;
    std::string failed;
    if (!rawfile::make_dirs(dir, ec, &failed)) throw rawfile::FileError(util::errno_code_of(ec), failed);
    DirLock l;
    l.path_ = lock_path(dir);
    const std::optional<std::wstring> w = util::wide_of(l.path_);
    if (!w || l.path_.find('\0') != std::string::npos)
        throw rawfile::FileError(std::make_error_code(std::errc::invalid_argument), l.path_);
    // "a+": read and append, created if absent; shared as the C runtime shares a file
    HANDLE h = ::CreateFileW(w->c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                             OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        throw rawfile::FileError(win32_error(), l.path_);
    l.h_ = h;
    OVERLAPPED ov{};   // byte 0
    if (!::LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &ov)) return std::nullopt;
    LARGE_INTEGER zero{};
    if (!::SetFilePointerEx(h, zero, nullptr, FILE_BEGIN) || !::SetEndOfFile(h))
        throw rawfile::FileError(win32_error(), l.path_);
    const std::string line = pid_line();
    DWORD done = 0;
    if (!::WriteFile(h, line.data(), static_cast<DWORD>(line.size()), &done, nullptr) || done != line.size())
        throw rawfile::FileError(win32_error(), l.path_);
    return l;
}

void DirLock::close() {
    if (h_) {
        OVERLAPPED ov{};
        ::UnlockFileEx(static_cast<HANDLE>(h_), 0, 1, 0, &ov);
        ::CloseHandle(static_cast<HANDLE>(h_));
    }
    h_ = nullptr;
}

bool DirLock::is_open() const { return h_ != nullptr; }

long long DirLock::native_handle() const { return h_ ? reinterpret_cast<long long>(h_) : -1; }

#else

std::optional<DirLock> DirLock::try_acquire(const std::string& dir) {
    std::error_code ec;
    std::string failed;
    if (!rawfile::make_dirs(dir, ec, &failed)) throw rawfile::FileError(util::errno_code_of(ec), failed);
    DirLock l;
    l.path_ = lock_path(dir);
    if (l.path_.find('\0') != std::string::npos)
        throw rawfile::FileError(std::make_error_code(std::errc::invalid_argument), l.path_);
    int fd = -1;
    do {
        fd = ::open(l.path_.c_str(), O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC, 0666);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) throw rawfile::FileError(std::error_code(errno, std::generic_category()), l.path_);
    l.fd_ = fd;
    int r = 0;
    do {
        r = ::flock(fd, LOCK_EX | LOCK_NB);
    } while (r != 0 && errno == EINTR);
    if (r != 0) return std::nullopt;   // (l closes the file)
    if (::ftruncate(fd, 0) != 0) throw rawfile::FileError(std::error_code(errno, std::generic_category()), l.path_);
    const std::string line = pid_line();
    std::size_t put = 0;
    while (put < line.size()) {
        const ssize_t n = ::write(fd, line.data() + put, line.size() - put);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw rawfile::FileError(std::error_code(errno, std::generic_category()), l.path_);
        }
        put += static_cast<std::size_t>(n);
    }
    return l;
}

void DirLock::close() {
    if (fd_ >= 0) ::close(fd_);   // (closing the last descriptor of the open file releases the flock)
    fd_ = -1;
}

bool DirLock::is_open() const { return fd_ >= 0; }

long long DirLock::native_handle() const { return fd_; }

#endif

}  // namespace sstvae::qrss::dirlock
