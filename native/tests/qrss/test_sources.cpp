// native/core/qrss/sources against live.py's Stamper, StdinSource and
// WavSource (the reference).
//
// The Stamper's expected chunks are the reference's, pinned by
// gen_sources_pins.py (sources_pins.hpp). tests/test_qrss_native_sources.py
// compares all three with the Python directly (fuzzing the Stamper, the
// same scripted writes and clock into both readers, the same WAV files);
// this file keeps the C++ honest without Python and covers what only the
// C++ has: close(), a read error ending the stream, the seams.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "qrss/sources.hpp"
#include "sources_pins.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace sstvae;
namespace lv = sstvae::qrss::live;

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

template <typename E, typename F>
void raises(F&& f, const std::string& text, const std::string& what) {
    ++check::checks;
    try {
        f();
    } catch (const E& e) {
        if (std::string(e.what()) != text) check::fail(what, std::string("text: ") + e.what());
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("wrong exception: ") + e.what());
        return;
    }
    check::fail(what, "no exception");
}

// A chunk holding the run first, first + 1, ..., first + len - 1.
bool is_run(const std::vector<double>& x, long long first) {
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i] != static_cast<double>(first + static_cast<long long>(i))) return false;
    return true;
}

void test_stamper_pins() {
    check::current_step = "stamper pins";
    for (const auto& c : sources_pins::cases()) {
        lv::Stamper st(c.hold_s);
        std::vector<sources_pins::Out> got;
        long long at = 0;
        auto take = [&](int after, const std::vector<lv::Chunk>& out) {
            for (const lv::Chunk& ch : out) {
                const long long first = ch.x.empty() ? -1 : static_cast<long long>(ch.x[0]);
                check::is_true(ch.x.empty() || is_run(ch.x, first), std::string(c.name) + ": a run");
                got.push_back({after, first, static_cast<long long>(ch.x.size()), ch.t_end});
            }
        };
        for (std::size_t k = 0; k < c.pushes.size(); ++k) {
            std::vector<double> x(static_cast<std::size_t>(c.pushes[k].n));
            for (auto& v : x) v = static_cast<double>(at++);
            take(static_cast<int>(k), st.push(std::move(x), c.pushes[k].t));
        }
        take(-1, st.flush());
        check::equal(got.size(), c.outs.size(), std::string(c.name) + ": chunks");
        for (std::size_t i = 0; i < std::min(got.size(), c.outs.size()); ++i) {
            const auto& g = got[i];
            const auto& w = c.outs[i];
            const std::string what = std::string(c.name) + " chunk " + std::to_string(i);
            check::equal(g.after, w.after, what + ": after push");
            check::equal(g.first, w.first, what + ": first");
            check::equal(g.len, w.len, what + ": len");
            ++check::checks;
            if (std::memcmp(&g.t_end, &w.t_end, sizeof(double)) != 0)
                check::fail(what + ": t_end", std::to_string(g.t_end) + " != " + std::to_string(w.t_end));
        }
        check::is_true(st.flush().empty(), std::string(c.name) + ": nothing held after the flush");
    }
}

void test_stamper_refusals() {
    check::current_step = "stamper refusals";
    for (double bad : {NaN, INF, -INF}) {
        lv::Stamper st;
        st.push({1.0, 2.0}, 100.0);
        const bool nan = std::isnan(bad);
        const std::string text = nan ? "cannot convert float NaN to integer"
                                     : "cannot convert float infinity to integer";
        auto push_bad = [&] { (void)st.push({3.0}, bad); };
        auto push_ok = [&] { (void)st.push({4.0}, 101.0); };
        auto flush = [&] { (void)st.flush(); };
        if (nan) {
            raises<std::invalid_argument>(push_bad, text, "push NaN");
            raises<std::invalid_argument>(push_ok, text, "push after NaN: still held");
            raises<std::invalid_argument>(flush, text, "flush after NaN");
        } else {
            raises<std::overflow_error>(push_bad, text, "push inf");
            raises<std::overflow_error>(push_ok, text, "push after inf: still held");
            raises<std::overflow_error>(flush, text, "flush after inf");
        }
    }
    lv::Stamper st;
    raises<std::overflow_error>([&] { (void)st.push({1.0}, 2e15); }, "time past an int64 sample index",
                                "a time past int64 samples");
}

void test_stamper_threads() {
    check::current_step = "stamper threads";
    // Pushes from four threads: every sample goes out exactly once.
    lv::Stamper st(0.0);
    std::atomic<long long> total{0};
    std::vector<std::thread> ts;
    for (int k = 0; k < 4; ++k)
        ts.emplace_back([&, k] {
            for (int i = 0; i < 500; ++i)
                for (const lv::Chunk& c : st.push(std::vector<double>(7, k), 1000.0 + i * 0.01))
                    total += static_cast<long long>(c.x.size());
        });
    for (auto& t : ts) t.join();
    for (const lv::Chunk& c : st.flush()) total += static_cast<long long>(c.x.size());
    check::equal(total.load(), 4LL * 500 * 7, "every sample out once");
}

std::vector<unsigned char> f32_bytes(const std::vector<float>& v) {
    std::vector<unsigned char> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

#ifndef _WIN32

// Writes each blob once the clock has stamped the read before it.
struct Script {
    int r = -1, w = -1;
    std::vector<std::vector<unsigned char>> blobs;
    std::vector<double> times;
    std::atomic<int> calls{0};
    std::thread writer;

    Script() {
        int fds[2];
        if (::pipe(fds) != 0) throw std::runtime_error("pipe");
        r = fds[0];
        w = fds[1];
    }
    ~Script() {
        if (writer.joinable()) writer.join();
        if (r >= 0) ::close(r);
    }
    lv::Clock clock() {
        return [this] {
            const int k = calls.fetch_add(1);
            return times.at(static_cast<std::size_t>(k));
        };
    }
    void start() {
        writer = std::thread([this] {
            for (std::size_t i = 0; i < blobs.size(); ++i) {
                if (::write(w, blobs[i].data(), blobs[i].size()) != static_cast<ssize_t>(blobs[i].size())) break;
                const auto t0 = std::chrono::steady_clock::now();
                while (calls.load() < static_cast<int>(i + 1) &&
                       std::chrono::steady_clock::now() - t0 < std::chrono::seconds(20))
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ::close(w);
        });
    }
};

std::vector<lv::Chunk> drain(lv::StreamSource& s) {
    std::vector<lv::Chunk> out;
    const auto t0 = std::chrono::steady_clock::now();
    while (!s.eof()) {
        if (auto c = s.next(0.2)) out.push_back(std::move(*c));
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
            check::fail("drain", "the stream never ended");
            break;
        }
    }
    return out;
}

void test_stream_f32_split_samples() {
    check::current_step = "stream f32";
    // 10 samples written 3, 5, 1, 30, 1 bytes at a time (samples split across
    // reads), then one byte more that never completes a sample; hold 0 lets
    // every whole sample out at its read.
    std::vector<float> v;
    for (int i = 0; i < 10; ++i) v.push_back(0.5f * static_cast<float>(i) - 1.0f);
    const auto bytes = f32_bytes(v);
    Script sc;
    std::size_t at = 0;
    for (std::size_t n : {3, 5, 1, 30, 1}) {
        sc.blobs.emplace_back(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                              bytes.begin() + static_cast<std::ptrdiff_t>(at + n));
        at += n;
    }
    sc.blobs.push_back({0x7f});                  // the odd byte at the end
    sc.times = {100.0, 100.25, 100.5, 100.75, 101.0, 101.25};
    lv::StreamOptions o;
    o.hold_s = 0.0;
    o.clock = sc.clock();
    lv::StreamSource src(lv::InputHandle::from_fd(sc.r), o);
    check::equal(src.name(), std::string("stdin (f32, 8 kHz)"), "name");
    check::is_true(!src.eof(), "not at the end before reading");
    sc.start();
    const auto out = drain(src);
    check::equal(sc.calls.load(), 6, "one clock reading a read, whole sample or not");
    std::vector<double> all;
    for (const auto& c : out) all.insert(all.end(), c.x.begin(), c.x.end());
    const std::vector<double> want(v.begin(), v.end());
    check::is_true(all == want, "every whole sample, widened; the odd byte dropped");
    // reads of 3 B (no sample), 5 B (2), 1 B (none), 30 B (7), 1 B (1), 1 B (none)
    check::equal(out.size(), std::size_t{3}, "a chunk per read that completed a sample");
    if (out.size() == 3) {
        check::equal(out[0].x.size(), std::size_t{2}, "chunk 0");
        check::equal(out[1].x.size(), std::size_t{7}, "chunk 1");
        check::equal(out[2].x.size(), std::size_t{1}, "chunk 2");
        check::equal(out[0].t_end, 100.25, "t_end 0: round(t * 8000) / 8000");
        check::equal(out[1].t_end, 100.75, "t_end 1");
        check::equal(out[2].t_end, 101.0, "t_end 2");
    }
    check::is_true(!src.error(), "no error");
}

void test_stream_s16_scale() {
    check::current_step = "stream s16";
    Script sc;
    const std::vector<std::int16_t> v{0, 1, -1, 32767, -32768, 12345};
    std::vector<unsigned char> b(v.size() * 2);
    std::memcpy(b.data(), v.data(), b.size());
    sc.blobs = {b};
    sc.times = {5000.0};
    lv::StreamOptions o;
    o.fmt = "s16";
    o.clock = sc.clock();
    lv::StreamSource src(lv::InputHandle::from_fd(sc.r), o);
    check::equal(src.name(), std::string("stdin (s16, 8 kHz)"), "name s16");
    sc.start();
    const auto out = drain(src);       // held (the first audio), out at the end
    check::equal(out.size(), std::size_t{1}, "one chunk at the end");
    if (!out.empty()) {
        std::vector<double> want;
        for (auto s : v) want.push_back(static_cast<double>(s) * (1.0 / 32768.0));
        check::is_true(out[0].x == want, "int16 / 32768");
        check::equal(out[0].t_end, 5000.0, "t_end: the read time, the stream contiguous");
    }
}

void test_stream_close_and_timeout() {
    check::current_step = "stream close";
    int fds[2];
    if (::pipe(fds) != 0) throw std::runtime_error("pipe");
    {
        lv::StreamSource src(lv::InputHandle::from_fd(fds[0]));
        const auto t0 = std::chrono::steady_clock::now();
        check::is_true(!src.next(0.05).has_value(), "nothing: a timeout");
        check::is_true(std::chrono::steady_clock::now() - t0 >= std::chrono::milliseconds(45),
                       "next waited its timeout");
        check::is_true(!src.eof(), "a timeout is not the end");
        raises<std::invalid_argument>([&] { (void)src.next(-1.0); }, "'timeout' must be a non-negative number",
                                      "a negative timeout");
        raises<std::invalid_argument>([&] { (void)src.next(NaN); }, "'timeout' must be a non-negative number",
                                      "a NaN timeout");
        float one = 0.25f;
        if (::write(fds[1], &one, 4) != 4) throw std::runtime_error("write");
        std::this_thread::sleep_for(std::chrono::milliseconds(150));    // read and held
        src.close();
        src.close();                                                     // idempotent
        auto c = src.next(INF);
        check::is_true(c.has_value() && c->x == std::vector<double>{0.25}, "what was read goes out at the close");
        check::is_true(!src.next(INF).has_value() && src.eof(), "then the end");
        check::is_true(!src.error(), "a close is not an error");
    }   // the destructor after close
    {
        lv::StreamSource src(lv::InputHandle::from_fd(fds[0]));
        // destroyed while the reader waits on a pipe nobody writes
    }
    ::close(fds[0]);
    ::close(fds[1]);
    {
        // A timeout too long for steady_clock's nanoseconds (1e12 s) waits as
        // an infinite one does: here for the end, 0.1 s away. Without the
        // clamp the deadline overflows into the past and next() returns at
        // once, the stream not ended.
        int p[2];
        if (::pipe(p) != 0) throw std::runtime_error("pipe");
        lv::StreamSource src(lv::InputHandle::from_fd(p[0]));
        std::thread closer([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            ::close(p[1]);
        });
        const bool got = src.next(1e12).has_value();
        check::is_true(!got && src.eof(), "a 1e12 s timeout waits for the end");
        closer.join();
        ::close(p[0]);
    }
    raises<std::invalid_argument>([] { lv::StreamOptions o; o.chunk_s = 0.0; lv::StreamSource s(lv::InputHandle::from_fd(0), o); },
                                  "chunk_s must make a read of at least one sample", "chunk_s 0");
    raises<std::invalid_argument>([] { lv::StreamOptions o; o.chunk_s = NaN; lv::StreamSource s(lv::InputHandle::from_fd(0), o); },
                                  "chunk_s must make a read of at least one sample", "chunk_s NaN");
}

void test_stream_read_error() {
    check::current_step = "stream read error";
    const int fd = ::open(".", O_RDONLY);
    if (fd < 0) throw std::runtime_error("open .");
    {
        lv::StreamSource src(lv::InputHandle::from_fd(fd));
        const auto t0 = std::chrono::steady_clock::now();
        while (!src.eof() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) (void)src.next(0.5);
        check::is_true(src.eof(), "a read error ends the stream");
        int err = 0;
        try {
            if (auto e = src.error()) std::rethrow_exception(e);
        } catch (const std::system_error& e) {
            err = e.code().value();
        }
        check::equal(err, EISDIR, "the error kept: EISDIR");
    }
    ::close(fd);
}

void test_stream_clock_throws() {
    check::current_step = "stream clock throws";
    Script sc;
    sc.blobs = {f32_bytes({1.0f, 2.0f})};
    lv::StreamOptions o;
    o.clock = [] () -> double { throw std::runtime_error("clock broke"); };
    lv::StreamSource src(lv::InputHandle::from_fd(sc.r), o);
    sc.times = {0.0};
    // the writer waits for a clock reading that never succeeds; it gives up after 20 s,
    // so close the write end from here instead
    if (::write(sc.w, sc.blobs[0].data(), 8) != 8) throw std::runtime_error("write");
    const auto t0 = std::chrono::steady_clock::now();
    while (!src.eof() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) (void)src.next(0.5);
    check::is_true(src.eof(), "a clock that throws ends the stream");
    std::string what;
    try {
        if (auto e = src.error()) std::rethrow_exception(e);
    } catch (const std::exception& e) {
        what = e.what();
    }
    check::equal(what, std::string("clock broke"), "the clock's exception kept");
    ::close(sc.w);
}

#endif  // !_WIN32

void test_wav_source() {
    check::current_step = "wav source";
    std::vector<double> x(8000 * 2 + 300);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(i);
    std::vector<double> clock_times{10.0, 10.1, 10.2, 10.3, 10.4};
    std::size_t calls = 0;
    std::vector<double> sleeps;
    lv::WavOptions o;
    o.speed = 2.0;
    o.clock = [&] { return clock_times.at(calls++); };
    o.sleep = [&](double s) { sleeps.push_back(s); };
    lv::WavSource src(x, "rec.wav", 1760000000.0, o);
    check::equal(src.name(), std::string("rec.wav (replay from 2025-10-09T08:53:20Z)"), "name");
    std::vector<lv::Chunk> out;
    while (auto c = src.next(0.0)) out.push_back(std::move(*c));
    check::is_true(src.eof(), "eof after the last chunk");
    check::equal(out.size(), std::size_t{3}, "three chunks");
    check::equal(calls, std::size_t{4}, "the clock read at the start and once a chunk");
    if (out.size() == 3) {
        check::equal(out[0].x.size(), std::size_t{8000}, "chunk 0");
        check::equal(out[2].x.size(), std::size_t{300}, "the last chunk shorter");
        check::is_true(is_run(out[1].x, 8000), "chunk 1's samples");
        check::equal(out[0].t_end, 1760000000.0 + 7999.0 / 8000.0, "t_end 0");
        check::equal(out[2].t_end, 1760000000.0 + 16299.0 / 8000.0, "t_end 2");
    }
    // lag = (t_end - start) / speed - (clock - wall0)
    std::vector<double> want;
    for (std::size_t k = 0; k < 3; ++k) {
        const double te = out.size() == 3 ? out[k].t_end : 0.0;
        const double lag = (te - 1760000000.0) / 2.0 - (clock_times[k + 1] - 10.0);
        if (lag > 0) want.push_back(lag);
    }
    check::is_true(sleeps == want, "the pacing sleeps");

    // speed 0: the clock read once, no sleep; an empty recording ends at once
    calls = 0;
    sleeps.clear();
    lv::WavOptions f;
    f.clock = [&] { return clock_times.at(calls++); };
    f.sleep = [&](double s) { sleeps.push_back(s); };
    lv::WavSource fast(std::vector<double>{}, "e.wav", 0.0, f);
    check::is_true(!fast.next(1.0).has_value() && fast.eof(), "empty: the end at once");
    check::equal(calls, std::size_t{1}, "the clock read once, as the generator starts");
    check::is_true(sleeps.empty(), "no sleep at speed 0");
    raises<std::invalid_argument>([] { lv::WavOptions w; w.chunk_s = 0.0; lv::WavSource s(std::vector<double>{1.0}, "a", 0.0, w); },
                                  "chunk_s must make a chunk of at least one sample", "chunk_s 0");
    raises<std::invalid_argument>([] { lv::WavOptions w; w.chunk_s = NaN; lv::WavSource s(std::vector<double>{1.0}, "a", 0.0, w); },
                                  "cannot convert float NaN to integer", "chunk_s NaN");
    raises<std::invalid_argument>([] { lv::WavSource s(std::vector<double>{1.0}, "a", NaN); },
                                  "Invalid value NaN (not a number)", "start NaN: utc_iso's refusal");
}

void test_system_time() {
    check::current_step = "system time";
    const double a = lv::system_time();
    check::is_true(a > 1.7e9 && a < 4e9, "unix seconds");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog dog(240.0, "qrss_core_sources");
    try {
        test_stamper_pins();
        test_stamper_refusals();
        test_stamper_threads();
#ifndef _WIN32
        test_stream_f32_split_samples();
        test_stream_s16_scale();
        test_stream_close_and_timeout();
        test_stream_read_error();
        test_stream_clock_throws();
#endif
        test_wav_source();
        test_system_time();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_core_sources");
}
