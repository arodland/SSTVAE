// The live listener's audio sources: sstvae/qrss/live.py's Stamper,
// StdinSource and WavSource (the reference), behind source.hpp's Source.
//
//   live.py        here
//   Stamper        Stamper: the same chunks out, at the same times, bit for
//                  bit (the reference's integer and float arithmetic in its
//                  order: round(t * FS) half to even, Python's exact
//                  int-against-float comparison, (start + n - 1) / FS)
//   StdinSource    StreamSource, over any readable descriptor (InputHandle:
//                  a POSIX file descriptor or a Windows HANDLE; stdin by
//                  default)
//   WavSource      WavSource (wavread.hpp reads the file, as wavio.read_wav)
//
// Not a format module: nothing here goes on the air. Constants are
// rx_config.hpp's (live.py's, generated).
//
// StreamSource is the reference's StdinSource: raw mono 8 kHz little-endian
// float32 ("f32") or int16 (anything else, scaled by 1/32768); a reader
// thread reads whatever the descriptor has (up to chunk_s of samples a read:
// BufferedReader.read1's one raw read), carries a sample split across two
// reads over to the next, stamps each read with the clock as it returns
// (through a Stamper), and queues what the Stamper lets out; at the end of
// the stream it flushes the Stamper, drops a trailing partial sample, and
// ends. It never waits on the listener: the queue is unbounded, so a receive
// that takes a minute loses no audio.
//
// Fixed rather than copied: a read error (EBADF, EIO, a broken console),
// or a clock that raises, ends the reference's reader thread with a
// traceback and nothing queued after it, so its chunks() times out forever
// and never reports the end; here it ends the stream as the end of the
// input does (the Stamper flushed, eof() true) and error() says why.
//
// Not as the reference (realistic inputs never meet these): a chunk_s whose
// read size int(chunk_s * 8000) is not positive, or NaN, is refused
// (std::invalid_argument) where the reference reads nothing and ends, or
// raises later; a NaN or negative timeout is refused as queue.get refuses a
// negative one, and an infinite one (or any from 1e9 s, about 31 years, on:
// past that the deadline would overflow steady_clock) waits for the next
// chunk or the end (queue.get raises OverflowError past PY_TIMEOUT_MAX).
//
// Threads. Every class here is safe to call from several threads: Stamper
// and WavSource hold a mutex for each call, StreamSource keeps its queue
// under one. Results are values (chunks are copies). What blocks, and for
// how long:
//   Stamper::push/flush      the arithmetic and a concatenation (microseconds)
//   StreamSource::next       up to timeout_s, waiting for the reader thread
//   StreamSource::close and  until the reader thread has stopped: on POSIX
//   its destructor           at once (the reader waits on the input and on a
//                            self-pipe close() writes to, with select(), which
//                            waits on a terminal where macOS's poll() does
//                            not; past FD_SETSIZE poll(), and a terminal
//                            there on macOS still waits for its next read);
//                            one cancelled read on Windows; a clock seam that
//                            is itself waiting (a Python callable wanting the
//                            GIL) delays it -- a binding releases the GIL first
//   WavSource(path, ...)     reading the whole file (a 30-minute recording:
//                            about a second)
//   WavSource::next          the replay's pacing sleep when speed > 0 (up
//                            to chunk_s / speed), else nothing
// The clock and sleep seams are called with no lock of the caller's held,
// on the reader thread for StreamSource's clock.

#ifndef SSTVAE_QRSS_SOURCES_HPP
#define SSTVAE_QRSS_SOURCES_HPP

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "qrss/rx_config.hpp"
#include "qrss/source.hpp"
#include "qrss/wavread.hpp"

namespace sstvae::qrss::live {

inline constexpr double RESYNC_S = rxconfig::live::RESYNC_S;           // 1.0
inline constexpr double STAMP_HOLD_S = rxconfig::live::STAMP_HOLD_S;   // 5.0

// Unix seconds now (the reference's time.time()); a seam, for tests and
// for a replay's pacing.
using Clock = std::function<double()>;
// Sleep this many seconds (time.sleep).
using Sleep = std::function<void(double)>;

// time.time(): the system clock as unix seconds (nanoseconds since 1970
// divided by 1e9, as CPython converts them).
double system_time();

// live.Stamper: unix times for audio read from a pipe, which says when it
// was read, not when it was heard. A chunk that reads as late is held, with
// the chunks after it, until one of them reads as following on (a backlog,
// now drained) or hold_s has passed (audio really was missing); held chunks
// go out as one, timed by the chunk that arrived least late. The stream's
// first audio is held hold_s too. See live.py for the whole argument.
//
// Errors are the reference's round(): std::invalid_argument "cannot convert
// float NaN to integer" for a NaN time, std::overflow_error "cannot convert
// float infinity to integer" for an infinite one; std::overflow_error "time
// past an int64 sample index" for a time past 2^63 / 8000 s (year 36
// million), which Python's ints would hold. Past 2^53 samples (year 37,000)
// a chunk's time is the double quotient (start + n - 1) / 8000, which may be
// an ulp from Python's exactly rounded int / int.
class Stamper {
public:
    explicit Stamper(double hold_s = STAMP_HOLD_S);

    // Chunk x, read at unix time t: the chunks ready to go out (none, or
    // everything held as one).
    std::vector<Chunk> push(std::vector<double> x, double t);
    // Everything held, as one chunk (none when nothing is held).
    std::vector<Chunk> flush();

    double hold_s() const { return hold_s_; }

private:
    std::vector<Chunk> flush_locked();
    std::int64_t start_locked() const;

    mutable std::mutex mu_;
    double hold_s_;
    std::vector<std::pair<std::vector<double>, double>> held_;
    std::optional<std::int64_t> next_;   // 8 kHz index after the last sample put out
};

// A readable file the source does not own (never closes): a POSIX file
// descriptor, or a Windows HANDLE.
class InputHandle {
public:
    // fd 0 / GetStdHandle(STD_INPUT_HANDLE).
    static InputHandle standard_input();
    // A C runtime file descriptor (on Windows, the HANDLE behind it:
    // _get_osfhandle; an invalid descriptor gives an invalid handle, which
    // the first read reports).
    static InputHandle from_fd(int fd);
    // A Windows HANDLE as its integer value (POSIX: a file descriptor).
    static InputHandle from_native(std::intptr_t handle);

    std::intptr_t native() const { return h_; }

private:
    explicit InputHandle(std::intptr_t h) : h_(h) {}
    std::intptr_t h_;
};

// StreamSource's options (namespace scope: a nested struct with member
// initializers cannot be a default argument inside its own class).
struct StreamOptions {
    std::string fmt = "f32";      // "f32": float32; anything else int16 (live.py's test)
    double chunk_s = 0.25;        // a read takes up to int(chunk_s * 8000) samples
    double hold_s = STAMP_HOLD_S;
    Clock clock;                  // empty: system_time
};

// live.StdinSource over any readable descriptor (see the header comment).
class StreamSource final : public Source {
public:
    using Options = StreamOptions;

    // Starts the reader thread. std::invalid_argument for a chunk_s whose
    // read size is not a positive number of bytes.
    explicit StreamSource(InputHandle in = InputHandle::standard_input(), Options opt = {});
    ~StreamSource() override;
    StreamSource(const StreamSource&) = delete;
    StreamSource& operator=(const StreamSource&) = delete;

    // The next chunk the Stamper let out; nullopt after timeout_s with none,
    // or at the end (eof() then true). std::invalid_argument "'timeout'
    // must be a non-negative number" for a negative or NaN timeout.
    std::optional<Chunk> next(double timeout_s) override;
    bool eof() const override;
    // "stdin (<fmt>, 8 kHz)", as the reference names it.
    std::string name() const override;

    // Stops the reader (no more reads; what was read stays queued, and the
    // end follows it). Idempotent; the destructor calls it.
    void close();
    // What ended the stream, if not its end or close(): a read error (a
    // std::system_error: an errno, or on Windows a system error), or what
    // the clock or the Stamper threw. Null otherwise.
    std::exception_ptr error() const;

private:
    struct State;
    static void read_loop(const std::shared_ptr<State>& st);

    std::shared_ptr<State> st_;
    std::mutex join_mu_;          // one close() joins the reader
    std::thread reader_;
    std::string name_;
};

// WavSource's options.
struct WavOptions {
    double speed = 0.0;
    double chunk_s = 1.0;                     // chunk = int(chunk_s * 8000) samples
    Clock clock;                              // empty: system_time
    Sleep sleep;                              // empty: this thread sleeps
    std::optional<std::string> shown_as;      // the path as name() spells it (default: path)
};

// live.WavSource: a recording replayed from start_unix, stamped from its own
// sample count: chunk k (chunk_s of samples, the last one shorter) ends at
// start_unix + (a + len - 1) / 8000. speed 0 (or any speed not > 0) replays
// as fast as next() is called; otherwise next() first sleeps until the
// chunk's place at that many times real time, counted from the first
// next() call's clock reading. The timeout next() is given is not used (a
// replay never waits for audio).
class WavSource final : public Source {
public:
    using Options = WavOptions;

    // Reads `path` (UTF-8, util/fileio.hpp) as wavio.read_wav: wav::ReadError
    // or wav::FormatError (wavread.hpp's refusals, the reference's), then
    // utc_iso(start_unix)'s refusals (live.hpp), in the reference's order.
    // std::invalid_argument for a chunk_s whose chunk is not a positive
    // number of samples.
    WavSource(const std::string& path, double start_unix, Options opt = {});
    // The samples given (8 kHz mono, as read_wav returns them); `path` only
    // names the source.
    WavSource(std::vector<double> x, const std::string& path, double start_unix, Options opt = {});

    std::optional<Chunk> next(double timeout_s) override;
    bool eof() const override;
    // "<path> (replay from <utc_iso(start_unix)>)".
    std::string name() const override { return name_; }

    // The samples (immutable), and the warnings reading them gave (scipy's
    // and numpy's, wavread.hpp), for a front end to print.
    const std::vector<double>& samples() const { return x_; }
    const std::vector<wav::Warning>& warnings() const { return warnings_; }
    double start() const { return start_; }

private:
    void init(const std::string& path, double start_unix);

    std::vector<double> x_;           // never changed after construction
    std::vector<wav::Warning> warnings_;
    double start_ = 0.0;
    std::string name_;
    Options opt_;
    std::size_t chunk_ = 0;

    mutable std::mutex mu_;
    std::size_t a_ = 0;               // the next chunk's first sample
    std::optional<double> wall0_;     // the clock at the first next()
    bool eof_ = false;
};

}  // namespace sstvae::qrss::live

#endif  // SSTVAE_QRSS_SOURCES_HPP
