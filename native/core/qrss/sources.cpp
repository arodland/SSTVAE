// The live listener's audio sources (sources.hpp): live.py's Stamper,
// StdinSource and WavSource.

#include "qrss/sources.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/live.hpp"
#include "util/fileio.hpp"

#ifdef _WIN32
// MSVC's <windows.h> defines min and max as macros in C++ too (MinGW's does
// not, so the cross compile cannot see it): without NOMINMAX every
// std::min( below is a syntax error (C2589).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <stdlib.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <sys/select.h>
#include <unistd.h>
#endif

namespace sstvae::qrss::live {

namespace {

constexpr int FS = config::FS;

// int(round(v)): Python's round() of a float, half to even, as an int64.
std::int64_t round_index(double v) {
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    const double r = std::nearbyint(v);     // the default rounding mode: half to even
    if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0))
        throw std::overflow_error("time past an int64 sample index");
    return static_cast<std::int64_t>(r);
}

// Python's exact `a > b` of an int and a float.
bool int_gt_float(std::int64_t a, double b) {
    if (std::isnan(b)) return false;
    if (b >= 9223372036854775808.0) return false;
    if (b < -9223372036854775808.0) return true;
    return a > static_cast<std::int64_t>(std::floor(b));
}

}  // namespace

double system_time() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    // _PyTime_AsSecondsDouble: whole seconds by integer division, else ns / 1e9.
    const std::int64_t sec = 1000000000;
    if (ns % sec == 0) return static_cast<double>(ns / sec);
    volatile double d = static_cast<double>(ns);
    return d / 1e9;
}

// --- Stamper -----------------------------------------------------------------------------

Stamper::Stamper(double hold_s) : hold_s_(hold_s) {}

std::int64_t Stamper::start_locked() const {
    std::int64_t n = 0;
    std::optional<std::int64_t> best;
    for (const auto& [c, t] : held_) {
        n += static_cast<std::int64_t>(c.size());
        const std::int64_t a = round_index(t * static_cast<double>(FS)) - n + 1;
        best = best ? std::min(*best, a) : a;
    }
    return *best;
}

std::vector<Chunk> Stamper::push(std::vector<double> x, double t) {
    std::lock_guard lk(mu_);
    held_.emplace_back(std::move(x), t);      // held even when start() refuses it, as there
    const std::int64_t start = start_locked();
    if (t - held_.front().second < hold_s_ &&
        (!next_ || int_gt_float(start - *next_, RESYNC_S * static_cast<double>(FS))))
        return {};
    return flush_locked();
}

std::vector<Chunk> Stamper::flush() {
    std::lock_guard lk(mu_);
    return flush_locked();
}

std::vector<Chunk> Stamper::flush_locked() {
    if (held_.empty()) return {};
    const std::int64_t start = start_locked();
    std::size_t total = 0;
    for (const auto& h : held_) total += h.first.size();
    Chunk out;
    out.x.reserve(total);
    for (auto& h : held_) out.x.insert(out.x.end(), h.first.begin(), h.first.end());
    held_.clear();
    const auto len = static_cast<std::int64_t>(total);
    next_ = start + len;
    out.t_end = static_cast<double>(start + len - 1) / static_cast<double>(FS);
    std::vector<Chunk> r;
    r.push_back(std::move(out));
    return r;
}

// --- InputHandle -------------------------------------------------------------------------

InputHandle InputHandle::standard_input() {
#ifdef _WIN32
    return InputHandle(reinterpret_cast<std::intptr_t>(GetStdHandle(STD_INPUT_HANDLE)));
#else
    return InputHandle(0);
#endif
}

#ifdef _MSC_VER
namespace {
// _get_osfhandle of a descriptor that is not open runs the CRT's
// invalid-parameter handler, whose default ends the process on the spot
// (Python's os.read raises OSError EBADF there instead): silenced on this
// thread for the one call, as CPython's _Py_BEGIN_SUPPRESS_IPH does, so the
// invalid handle reaches the first read and is reported there.
void __cdecl ignore_invalid_parameter(const wchar_t*, const wchar_t*, const wchar_t*, unsigned int,
                                      std::uintptr_t) {}
}  // namespace
#endif

InputHandle InputHandle::from_fd(int fd) {
#ifdef _WIN32
#ifdef _MSC_VER
    const _invalid_parameter_handler old = _set_thread_local_invalid_parameter_handler(ignore_invalid_parameter);
#endif
    const std::intptr_t h = _get_osfhandle(fd);
#ifdef _MSC_VER
    _set_thread_local_invalid_parameter_handler(old);
#endif
    return InputHandle(h);
#else
    return InputHandle(fd);
#endif
}

InputHandle InputHandle::from_native(std::intptr_t handle) { return InputHandle(handle); }

// --- StreamSource ------------------------------------------------------------------------

struct StreamSource::State {
    std::intptr_t handle = 0;
    bool f32 = true;
    std::size_t itemsize = 4;
    std::size_t nbytes = 0;
    Clock clock;
    Stamper stamper;

    explicit State(double hold_s) : stamper(hold_s) {}

    std::mutex mu;
    std::condition_variable cv;
    std::deque<Chunk> queue;
    bool ended = false;          // the reader has queued its last chunk
    bool eof = false;            // next() has seen the end
    bool stop = false;           // close() asked the reader to stop
    bool reader_done = false;
    std::exception_ptr error;
#ifdef _WIN32
    HANDLE thread = nullptr;     // the reader's own handle, for CancelSynchronousIo
#else
    // A self-pipe close() writes a byte to, which the reader waits on beside
    // the input: set before the reader starts, closed after it has gone
    // (-1, -1: no pipe could be made, and the reader looks at the flag
    // every 0.1 s instead).
    int wake[2] = {-1, -1};
    ~State() {
        for (const int fd : wake)
            if (fd >= 0) ::close(fd);
    }
#endif

    void put(std::vector<Chunk> out) {
        if (out.empty()) return;
        std::lock_guard lk(mu);
        for (auto& c : out) queue.push_back(std::move(c));
        cv.notify_all();
    }
    bool stopping() {
        std::lock_guard lk(mu);
        return stop;
    }
};

namespace {

enum class ReadResult { data, end, stopped };

#ifndef _WIN32
// Wait until `fd` has something for read() -- data, the end, an error, or
// it is not open -- or `wake` (close()'s self-pipe) is readable: true when
// read() should be called, false to look at the stop flag again. select()
// where both fit an fd_set, poll() past FD_SETSIZE: macOS's poll() answers
// POLLNVAL at once for a terminal (a character device), and the read()
// behind it then blocked until a key was pressed, close() waiting on it;
// select() waits on a terminal on every POSIX system. Without a wake pipe
// (`wake` < 0) it returns every 0.1 s.
bool wait_readable(int fd, int wake) {
    if (fd < 0) return true;                       // not a descriptor: read() reports EBADF
    if (fd < FD_SETSIZE && wake < FD_SETSIZE) {
        fd_set rd;
        FD_ZERO(&rd);
        FD_SET(fd, &rd);
        if (wake >= 0) FD_SET(wake, &rd);
        timeval tick{0, 100000};
        const int r = ::select(std::max(fd, wake) + 1, &rd, nullptr, nullptr, wake >= 0 ? nullptr : &tick);
        if (r < 0) return errno != EINTR;          // (EBADF: the input is not open; read() says so)
        if (wake >= 0 && FD_ISSET(wake, &rd)) return false;
        return r > 0 && FD_ISSET(fd, &rd);
    }
    pollfd p[2] = {{fd, POLLIN, 0}, {wake, POLLIN, 0}};
    const int r = ::poll(p, wake >= 0 ? 2 : 1, wake >= 0 ? -1 : 100);
    if (r < 0) {
        if (errno == EINTR) return false;
        throw std::system_error(errno, std::generic_category(), "poll");
    }
    if (wake >= 0 && p[1].revents != 0) return false;
    return r > 0 && p[0].revents != 0;
}
#endif

// One raw read of up to buf.size() bytes (BufferedReader.read1 with nothing
// buffered): data (got > 0), the end, or a stop. Throws std::system_error
// on a read error.
template <typename State>
ReadResult read_once(State& st, unsigned char* buf, std::size_t n, std::size_t& got) {
#ifdef _WIN32
    const HANDLE h = reinterpret_cast<HANDLE>(st.handle);
    for (;;) {
        if (st.stopping()) return ReadResult::stopped;
        DWORD k = 0;
        const DWORD want = static_cast<DWORD>(std::min<std::size_t>(n, 0x7fffffff));
        if (ReadFile(h, buf, want, &k, nullptr)) {
            if (k == 0) return ReadResult::end;
            got = k;
            return ReadResult::data;
        }
        const DWORD e = GetLastError();
        // The CRT's _read: a broken pipe is the end of the input.
        if (e == ERROR_BROKEN_PIPE || e == ERROR_HANDLE_EOF) return ReadResult::end;
        if (e == ERROR_OPERATION_ABORTED) {
            if (st.stopping()) return ReadResult::stopped;
            continue;               // a cancel that raced a read that finished
        }
        throw std::system_error(util::errno_code_of(std::error_code(static_cast<int>(e), std::system_category())),
                                "read");
    }
#else
    const int fd = static_cast<int>(st.handle);
    const int wake = st.wake[0];
    for (;;) {
        if (st.stopping()) return ReadResult::stopped;
        if (!wait_readable(fd, wake)) continue;
        // Readable, at the end, hung up, in error or not open: read() says which.
        const ssize_t k = ::read(fd, buf, n);
        if (k > 0) {
            got = static_cast<std::size_t>(k);
            return ReadResult::data;
        }
        if (k == 0) return ReadResult::end;
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
        throw std::system_error(errno, std::generic_category(), "read");
    }
#endif
}

}  // namespace

void StreamSource::read_loop(const std::shared_ptr<State>& sp) {
    auto& st = *sp;
#ifdef _WIN32
    {
        HANDLE self = nullptr;
        DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self, 0,
                        FALSE, DUPLICATE_SAME_ACCESS);
        std::lock_guard lk(st.mu);
        st.thread = self;
    }
#endif
    std::exception_ptr failure;
    try {
        std::vector<unsigned char> buf(st.nbytes);
        std::vector<unsigned char> b;      // rest + this read
        for (;;) {
            std::size_t got = 0;
            const ReadResult r = read_once(st, buf.data(), buf.size(), got);
            if (r != ReadResult::data) break;
            const double t = st.clock ? st.clock() : system_time();
            b.insert(b.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(got));
            const std::size_t k = b.size() / st.itemsize * st.itemsize;
            if (k == 0) continue;
            std::vector<double> x(k / st.itemsize);
            if (st.f32) {
                for (std::size_t i = 0; i < x.size(); ++i) {
                    const unsigned char* q = b.data() + 4 * i;
                    const std::uint32_t u = std::uint32_t(q[0]) | (std::uint32_t(q[1]) << 8) |
                                            (std::uint32_t(q[2]) << 16) |
                                            (std::uint32_t(q[3]) << 24);
                    float f;
                    std::memcpy(&f, &u, 4);
                    x[i] = static_cast<double>(f) * 1.0;
                }
            } else {
                for (std::size_t i = 0; i < x.size(); ++i) {
                    const unsigned char* q = b.data() + 2 * i;
                    const auto v = static_cast<std::int16_t>(
                        static_cast<std::uint16_t>(q[0] | (static_cast<unsigned>(q[1]) << 8)));
                    x[i] = static_cast<double>(v) * (1.0 / 32768.0);
                }
            }
            b.erase(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(k));
            st.put(st.stamper.push(std::move(x), t));
        }
    } catch (...) {
        failure = std::current_exception();
    }
    // The end of the input, a stop or a failure: what is held goes out, then the end.
    try {
        st.put(st.stamper.flush());
    } catch (...) {
        if (!failure) failure = std::current_exception();
    }
    std::lock_guard lk(st.mu);
    st.error = failure;
    st.ended = true;
    st.reader_done = true;
    st.cv.notify_all();
}

StreamSource::StreamSource(InputHandle in, Options opt) {
    const double n = opt.chunk_s * static_cast<double>(FS);
    if (!(n >= 1.0) || !(n < 1e15))
        throw std::invalid_argument("chunk_s must make a read of at least one sample");
    st_ = std::make_shared<State>(opt.hold_s);
    st_->handle = in.native();
    st_->f32 = opt.fmt == "f32";
    st_->itemsize = st_->f32 ? 4 : 2;
    st_->nbytes = static_cast<std::size_t>(static_cast<std::int64_t>(n)) * st_->itemsize;
    st_->clock = std::move(opt.clock);
    name_ = "stdin (" + opt.fmt + ", 8 kHz)";
#ifndef _WIN32
    int p[2];
    if (::pipe(p) == 0) {
        for (const int fd : p) {
            ::fcntl(fd, F_SETFD, FD_CLOEXEC);
            ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) | O_NONBLOCK);   // (close()'s write never blocks)
        }
        st_->wake[0] = p[0];
        st_->wake[1] = p[1];
    }
#endif
    auto sp = st_;
    reader_ = std::thread([sp] { read_loop(sp); });
}

StreamSource::~StreamSource() { close(); }

void StreamSource::close() {
    std::lock_guard jl(join_mu_);
    if (!reader_.joinable()) return;
    {
        std::lock_guard lk(st_->mu);
        st_->stop = true;
    }
#ifndef _WIN32
    // Wake the reader from its wait (the byte stays: every wait after it
    // returns at once, and the flag ends the loop).
    if (st_->wake[1] >= 0) {
        const char b = 1;
        [[maybe_unused]] const ssize_t w = ::write(st_->wake[1], &b, 1);
    }
#endif
#ifdef _WIN32
    // A blocking ReadFile does not look at the flag: cancel it until the reader has gone.
    for (;;) {
        HANDLE h = nullptr;
        {
            std::lock_guard lk(st_->mu);
            if (st_->reader_done) break;
            h = st_->thread;
        }
        if (h) ::CancelSynchronousIo(h);
        ::Sleep(10);   // (the Win32 one, not live::Sleep)
    }
#endif
    reader_.join();
#ifdef _WIN32
    std::lock_guard lk(st_->mu);
    if (st_->thread) {
        CloseHandle(st_->thread);
        st_->thread = nullptr;
    }
#endif
}

std::optional<Chunk> StreamSource::next(double timeout_s) {
    if (std::isnan(timeout_s) || timeout_s < 0)
        throw std::invalid_argument("'timeout' must be a non-negative number");
    std::unique_lock lk(st_->mu);
    auto ready = [&] { return !st_->queue.empty() || st_->ended; };
    // Past ~31 years the deadline would overflow steady_clock's int64
    // nanoseconds (UB; in practice a deadline in the past, so next() would
    // return at once and a caller asking for "a very long time" would spin).
    if (timeout_s >= 1e9) {
        st_->cv.wait(lk, ready);
    } else {
        const auto d = std::chrono::duration<double>(timeout_s);
        st_->cv.wait_until(lk,
                           std::chrono::steady_clock::now() +
                               std::chrono::duration_cast<std::chrono::steady_clock::duration>(d),
                           ready);
    }
    if (!st_->queue.empty()) {
        Chunk c = std::move(st_->queue.front());
        st_->queue.pop_front();
        return c;
    }
    if (st_->ended) st_->eof = true;
    return std::nullopt;
}

bool StreamSource::eof() const {
    std::lock_guard lk(st_->mu);
    return st_->eof;
}

std::string StreamSource::name() const { return name_; }

std::exception_ptr StreamSource::error() const {
    std::lock_guard lk(st_->mu);
    return st_->error;
}

// --- WavSource ---------------------------------------------------------------------------

WavSource::WavSource(const std::string& path, double start_unix, Options opt)
    : opt_(std::move(opt)) {
    x_ = wav::read_wav(path, &warnings_);
    init(path, start_unix);
}

WavSource::WavSource(std::vector<double> x, const std::string& path, double start_unix,
                     Options opt)
    : x_(std::move(x)), opt_(std::move(opt)) {
    init(path, start_unix);
}

void WavSource::init(const std::string& path, double start_unix) {
    start_ = start_unix;
    const double n = opt_.chunk_s * static_cast<double>(FS);
    if (std::isnan(n)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (!(n >= 1.0) || !(n < 1e15))
        throw std::invalid_argument("chunk_s must make a chunk of at least one sample");
    chunk_ = static_cast<std::size_t>(static_cast<std::int64_t>(n));
    name_ = (opt_.shown_as ? *opt_.shown_as : path) + " (replay from " + utc_iso(start_unix) + ")";
}

std::optional<Chunk> WavSource::next(double /*timeout_s*/) {
    std::optional<double> w0;
    {
        std::lock_guard lk(mu_);
        w0 = wall0_;
    }
    if (!w0) {
        const double w = opt_.clock ? opt_.clock() : system_time();
        std::lock_guard lk(mu_);
        if (!wall0_) wall0_ = w;
        w0 = wall0_;
    }
    std::size_t a = 0;
    {
        std::lock_guard lk(mu_);
        if (a_ >= x_.size()) {
            eof_ = true;
            return std::nullopt;
        }
        a = a_;
        a_ += chunk_;
    }
    Chunk c;
    const std::size_t len = std::min(chunk_, x_.size() - a);
    c.x.assign(x_.begin() + static_cast<std::ptrdiff_t>(a),
               x_.begin() + static_cast<std::ptrdiff_t>(a + len));
    c.t_end = start_ + static_cast<double>(static_cast<std::int64_t>(a + len) - 1) /
                           static_cast<double>(FS);
    if (opt_.speed > 0) {
        const double now = opt_.clock ? opt_.clock() : system_time();
        const double lag = (c.t_end - start_) / opt_.speed - (now - *w0);
        if (lag > 0) {
            if (opt_.sleep)
                opt_.sleep(lag);
            else
                std::this_thread::sleep_for(std::chrono::duration<double>(lag));
        }
    }
    return c;
}

bool WavSource::eof() const {
    std::lock_guard lk(mu_);
    return eof_;
}

}  // namespace sstvae::qrss::live
