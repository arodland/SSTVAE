// live.dir_lock (sstvae/qrss/live.py, the reference): one writer per
// directory. A listener holds an exclusive lock on <dir>/.lock for its
// whole life, so two listeners on one store, tile directory or passband
// store are refused rather than overwriting each other's files.
//
// The lock is the reference's own, so a Python listener and a C++ one
// exclude each other:
//   - POSIX: flock(LOCK_EX | LOCK_NB) on the file opened for append
//     (O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC, as open(path, "a+") opens
//     it). flock locks belong to the open file, so a second try_acquire in
//     the same process is refused too, as the reference's second open is.
//   - Windows: LockFileEx(LOCKFILE_EXCLUSIVE_LOCK |
//     LOCKFILE_FAIL_IMMEDIATELY) on byte 0, the byte msvcrt.locking(fd,
//     LK_NBLCK, 1) locks after the reference's seek(0); the two conflict.
// Not QLockFile: that is a different protocol (a PID file with stale-lock
// detection) which a Python listener would not see.
//
// Once locked, the file is truncated and holds "<pid>\n" (Windows: "\r\n",
// as the reference's text-mode write puts it there). The lock goes with
// the DirLock (close() or its destructor) and, whatever happens, with the
// process.
//
// Thread safety: a DirLock is one handle; close() and the accessors are
// for its owner's thread. try_acquire blocks only for a mkdir, an open and
// a 10-byte write -- never for the lock, which is tried, not waited for.

#ifndef SSTVAE_QRSS_DIRLOCK_HPP
#define SSTVAE_QRSS_DIRLOCK_HPP

#include <optional>
#include <string>

namespace sstvae::qrss::dirlock {

class DirLock {
public:
    // Locks `dir` (made first, mkdir -p): the lock, or nullopt when another
    // holder has it (any error of the lock call itself counts as that, as
    // the reference's `except OSError` does). std::system_error (an
    // OSError) when the directory cannot be made or the file opened or
    // written.
    static std::optional<DirLock> try_acquire(const std::string& dir);

    DirLock(DirLock&& o) noexcept;
    DirLock& operator=(DirLock&& o) noexcept;
    DirLock(const DirLock&) = delete;
    DirLock& operator=(const DirLock&) = delete;
    ~DirLock();

    // Releases the lock and closes the file; idempotent.
    void close();
    bool is_open() const;
    // <dir>/.lock (UTF-8).
    const std::string& path() const { return path_; }
    // The file descriptor (POSIX) or C runtime descriptor-free handle value
    // (Windows: the HANDLE as an integer); -1 once closed.
    long long native_handle() const;

private:
    DirLock() = default;
    std::string path_;
#ifdef _WIN32
    void* h_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace sstvae::qrss::dirlock

#endif  // SSTVAE_QRSS_DIRLOCK_HPP
