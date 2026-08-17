// A test harness small enough not to be a dependency.
//
// Catch2 or doctest would be nicer, but every dependency added here is
// one vcpkg has to resolve on three platforms before the parity harness
// -- the thing that makes the whole port checkable -- can run at all.
// The harness is ~60 lines; the corpus it checks is the valuable part.

#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// SetErrorMode is declared by hand rather than by including <windows.h>.
//
// This header is in *every* test, and <windows.h> is not a polite
// guest: it defines `min` and `max` as macros, so every later
// `std::max(a, b)` becomes a syntax error (C2589), which is what
// including it here actually did. NOMINMAX and WIN32_LEAN_AND_MEAN
// would suppress that, but only for as long as nothing pulls in a
// Windows header ahead of us, and the tests link Qt. Three lines beat
// a constraint on include order that nothing checks. The signature is
// winbase.h's exactly -- UINT WINAPI SetErrorMode(UINT) -- so a
// translation unit that sees both still sees one declaration.
extern "C" __declspec(dllimport) unsigned int __stdcall SetErrorMode(unsigned int);
#ifdef _DEBUG
#include <crtdbg.h>
#endif
#endif

namespace sstvae::check {

inline int failures = 0;
inline int checks = 0;

inline void fail(const std::string& what, const std::string& detail) {
    ++failures;
    std::fprintf(stderr, "FAIL %s: %s\n", what.c_str(), detail.c_str());
}

inline void is_true(bool ok, const std::string& what) {
    ++checks;
    if (!ok) fail(what, "expected true");
}

// Values are rendered through this rather than std::to_string directly,
// so a check can compare strings (dtype names, mostly) as readily as
// numbers.
inline std::string to_text(const std::string& s) { return "\"" + s + "\""; }
inline std::string to_text(const char* s) { return to_text(std::string(s)); }
template <typename T>
std::string to_text(const T& v) {
    return std::to_string(v);
}

template <typename A, typename B>
void equal(const A& got, const B& want, const std::string& what) {
    ++checks;
    if (!(got == want))
        fail(what, "got " + to_text(got) + ", want " + to_text(want));
}

// Largest absolute difference between two sequences, reported with the
// index where it occurs -- a bare "1.7e-09 > 1e-12" says nothing about
// whether one element is wrong or all of them are.
template <typename T>
void close(const std::vector<T>& got, const std::vector<T>& want, double tol,
           const std::string& what) {
    ++checks;
    if (got.size() != want.size()) {
        fail(what, "size " + std::to_string(got.size()) + " != " +
                       std::to_string(want.size()));
        return;
    }
    double worst = 0.0;
    std::size_t where = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::abs(got[i] - want[i]);
        if (d > worst) {
            worst = d;
            where = i;
        }
    }
    if (!(worst <= tol)) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "max |diff| = %.6g at index %zu (tolerance %.6g)", worst,
                      where, tol);
        fail(what, buf);
    }
}

// ---------------------------------------------------------------------
// Staying visible when a test does not return
//
// Everything above reports a wrong answer. These two report the other
// failure mode, which is the one an external library brings with it:
// the test that never finishes. A hang produces no output at all, so
// it is indistinguishable between platforms and between causes, and
// that is exactly when a bug is only reproducible on the runner.

// The step a test is currently inside, for the watchdog to name.
// A string literal, so publishing it is a pointer store and safe to
// read from another thread while this one is wedged inside a library.
inline std::atomic<const char*> current_step{"(not started)"};

// Publish the step *and* leave a trail on stderr.
//
// The trail is for the third failure mode, which neither the watchdog
// nor a FAIL line covers: the test that **crashes**. A segfault prints
// nothing of its own, so the log shows a test that produced no output at
// all and died, which says only "somewhere in a few hundred lines".
//
// CLAUDE.md records that a printf is not a diagnostic for a *hang*, and
// that is still true -- ctest holds a test's output until the test
// finishes, and a wedged process never does. A crash is the opposite
// case: the process dies, its pipe closes, and ctest reports everything
// written before that point. So per-step output is worth nothing for a
// hang and is the whole answer for a crash. Both mechanisms are kept
// because they cover different things.
//
// Unbuffered on purpose (`fflush`): a crash gives no chance to drain a
// buffer, and stderr's buffering is not the same on the three platforms.
// Costs nothing in the normal case, because ctest discards a passing
// test's output entirely -- the same argument SSTVAE_HAMLIB_DEBUG makes.
inline void step(const char* name) {
    current_step.store(name);
    std::fprintf(stderr, "-- %s\n", name);
    std::fflush(stderr);
}

// Windows: fail loudly rather than waiting on a dialog nobody can see.
//
// An unhandled exception on a headless runner raises Windows Error
// Reporting, and a CRT assert opens a message box. Both block forever
// with an empty stderr, which is *identical in the log* to a deadlock
// -- so a crash gets diagnosed as a hang and the search goes to the
// wrong place. Route the diagnostics to stderr and let the process die.
// No-op everywhere else, because no other platform does this.
inline void report_crashes_instead_of_prompting() {
#ifdef _WIN32
    // Spelled out rather than named, for the same reason the function
    // above is declared here: the SEM_ names are <windows.h> macros, and
    // redefining them would break any translation unit that does include
    // it. The values are ABI, not implementation detail.
    constexpr unsigned int fail_critical_errors = 0x0001;
    constexpr unsigned int no_gp_fault_error_box = 0x0002;
    constexpr unsigned int no_open_file_error_box = 0x8000;
    SetErrorMode(fail_critical_errors | no_gp_fault_error_box |
                 no_open_file_error_box);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#ifdef _DEBUG
    // Only a debug CRT has these; in a release build they are no-op
    // macros that do not even use their argument, which is a warning of
    // its own (C4189) for a call that was never going to do anything.
    for (const int report_type : {_CRT_WARN, _CRT_ERROR, _CRT_ASSERT}) {
        _CrtSetReportMode(report_type, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(report_type, _CRTDBG_FILE_STDERR);
    }
#endif
#endif
}

// A deadline for the whole process, generously sized.
//
// This is not a latency assertion and must never be read as one: it
// belongs at many times the measured worst case, so that expiring means
// "wedged", never "slower than I guessed". Its value over a ctest
// TIMEOUT is that it runs *inside* the process, so it can say which
// step was in progress -- and it uses _Exit deliberately, because
// unwinding through static destructors is itself a place a wedged
// library can hang, and a watchdog that can hang is not one.
class Watchdog {
public:
    Watchdog(double seconds, const char* suite) {
        thread_ = std::thread([this, seconds, suite] {
            std::unique_lock<std::mutex> lock(mutex_);
            const auto limit = std::chrono::duration<double>(seconds);
            if (cv_.wait_for(lock, limit, [this] { return done_; })) return;
            std::fprintf(stderr,
                         "\nTIMEOUT: %s made no progress for %.0f s, stuck in "
                         "%s\n",
                         suite, seconds, current_step.load());
            std::fflush(stderr);
            std::_Exit(1);
        });
    }

    ~Watchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
    std::thread thread_;
};

inline int report(const char* suite) {
    if (failures == 0) {
        std::printf("ok: %s (%d checks)\n", suite, checks);
        return 0;
    }
    std::fprintf(stderr, "%d of %d checks FAILED in %s\n", failures, checks, suite);
    return 1;
}

}  // namespace sstvae::check
