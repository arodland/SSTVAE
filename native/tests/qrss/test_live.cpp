// native/core/qrss/live against sstvae/qrss/live.py (the reference) up to
// LiveListener: the small helpers, the FE ring, a slot's capture, the
// end-of-slot receive's decisions, the dial log, restrict, tiles and the
// listener's configuration.
//
// The expected values are the reference's, pinned: each was printed by the
// Python (utc_iso, frame_seconds, the capture span, FeRing reads, _spans,
// DialLog's points and spans, restrict, json.dumps(asdict(Tile))).
// tests/test_qrss_native_live.py compares the two directly (fuzzing the
// ring, the masks, the dial files and the end-of-slot receive) where Python
// is available; this file keeps the C++ honest without it, and covers what
// only the C++ has: the noise generator and the seams.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "qrss/live.hpp"
#include "util/fileio.hpp"

using namespace sstvae;
namespace lv = sstvae::qrss::live;
namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;
using cf = std::complex<float>;

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

std::vector<int> as_ints(const std::vector<std::uint8_t>& v) { return {v.begin(), v.end()}; }
std::vector<float> re(const std::vector<cf>& v) {
    std::vector<float> o;
    for (const cf& z : v) o.push_back(z.real());
    return o;
}
std::vector<float> im(const std::vector<cf>& v) {
    std::vector<float> o;
    for (const cf& z : v) o.push_back(z.imag());
    return o;
}

std::string vec_text(const std::vector<int>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "]";
}

void equal_vec(const std::vector<int>& got, const std::vector<int>& want, const std::string& what) {
    ++check::checks;
    if (got != want) check::fail(what, "got " + vec_text(got) + ", want " + vec_text(want));
}

void equal_floats(const std::vector<float>& got, const std::vector<float>& want, const std::string& what) {
    std::vector<int> g, w;
    for (float f : got) g.push_back(static_cast<int>(f));
    for (float f : want) w.push_back(static_cast<int>(f));
    ++check::checks;
    bool same = got.size() == want.size();
    for (std::size_t i = 0; same && i < got.size(); ++i) same = got[i] == want[i];
    if (!same) check::fail(what, "got " + vec_text(g) + ", want " + vec_text(w));
}

template <typename E, typename F>
void raises(F&& fn, const std::string& text, const std::string& what) {
    std::string got = "(no exception)";
    try {
        fn();
    } catch (const E& e) {
        got = e.what();
    } catch (const std::exception& e) {
        got = std::string("(another exception) ") + e.what();
    }
    check::equal(got, text, what);
}

// --- small things --------------------------------------------------------------------

void test_utc_iso() {
    const std::pair<double, const char*> rows[] = {
        {0.0, "1970-01-01T00:00:00Z"},
        {0.9999995, "1970-01-01T00:00:01Z"},   // 999999.5 us rounds half to even: the next second
        {0.9999985, "1970-01-01T00:00:00Z"},
        {-0.5, "1969-12-31T23:59:59Z"},
        {253402300799.0, "9999-12-31T23:59:59Z"},
        {-62135596800.0, "1-01-01T00:00:00Z"},   // glibc's %Y does not pad
        {1000000000.25, "2001-09-09T01:46:40Z"},
        {1760000000.5, "2025-10-09T08:53:20Z"},
    };
    for (const auto& [t, want] : rows) check::equal(lv::utc_iso(t), std::string(want), "utc_iso " + std::to_string(t));
    check::equal(lv::utc_iso_int(0), std::string("1970-01-01T00:00:00Z"), "utc_iso_int 0");
    check::equal(lv::utc_iso_int(253402300799), std::string("9999-12-31T23:59:59Z"), "utc_iso_int max");
    raises<std::invalid_argument>([] { lv::utc_iso(253402300800.0); }, "year 10000 is out of range", "utc_iso 10000");
    raises<std::invalid_argument>([] { lv::utc_iso(-62135596801.0); }, "year 0 is out of range", "utc_iso 0");
    raises<std::invalid_argument>([] { lv::utc_iso(NaN); }, "Invalid value NaN (not a number)", "utc_iso NaN");
    raises<std::overflow_error>([] { lv::utc_iso(INF); }, "timestamp out of range for platform time_t", "utc_iso inf");
    raises<std::overflow_error>([] { lv::utc_iso(1e19); }, "timestamp out of range for platform time_t",
                                "utc_iso 1e19");
    bool eoverflow = false;
    try {
        lv::utc_iso_int(std::numeric_limits<std::int64_t>::max());
    } catch (const std::system_error& e) {
        eoverflow = e.code() == std::errc::value_too_large;
    }
    check::is_true(eoverflow, "utc_iso_int(INT64_MAX): gmtime's EOVERFLOW");
}

void test_small() {
    check::equal(lv::slot_t0(0), std::int64_t{1}, "slot_t0(0)");
    check::equal(lv::slot_t0(1990584), std::int64_t{1990584} * 900 + 1, "slot_t0(1990584)");
    check::equal(lv::frame_seconds(fr::get("tiny")), 53.35, "frame_seconds tiny");
    check::equal(lv::frame_seconds(fr::get("short")), 255.76171875, "frame_seconds short");
    check::equal(lv::frame_seconds(fr::get("full")), 1782.66296875, "frame_seconds full");
    check::equal(lv::frame_seconds(fr::get_any("l-tiny")), 53.5925, "frame_seconds l-tiny");
    check::equal(lv::frame_seconds(fr::get_any("l-short")), 255.6859375, "frame_seconds l-short");
    const std::vector<fr::FrameSpec> both = {fr::get("tiny"), fr::get_any("l-tiny")};
    check::equal(lv::frame_seconds(lv::Selection(both)), 53.5925, "frame_seconds (tiny, l-tiny): the longest");
    raises<std::invalid_argument>([] { lv::frame_seconds(lv::Selection{}); }, "max() iterable argument is empty",
                                  "frame_seconds of none");

    check::is_true(lv::in_band(300.0), "in_band 300");
    check::is_true(!lv::in_band(299.999999999), "in_band under 300");
    check::is_true(lv::in_band(2700.0), "in_band 2700");
    check::is_true(!lv::in_band(2700.000000001), "in_band over 2700");
    check::is_true(lv::in_band(322.06907216494847, cfg::WAVEFORM_L), "in_band L: its data at 300");
    check::is_true(!lv::in_band(322.0690721639485, cfg::WAVEFORM_L), "in_band L: its data under 300");
    check::is_true(!lv::in_band(NaN), "in_band NaN");
    check::equal(lv::centre_of(1000.0, cfg::WAVEFORM_L), 990.5, "centre_of L");
    check::equal(lv::centre_of(1000.0, cfg::WAVEFORM_CE), 1000.0, "centre_of CE");

    check::equal(lv::waveform_id("CE"), 0, "waveform_id CE");
    check::equal(lv::waveform_id("L"), 1, "waveform_id L");
    check::equal(lv::waveform_id("l"), 0, "waveform_id l (unknown: CE)");
    using V = qrss::pyjson::Value;
    check::equal(lv::waveform_id(V::string("L")), std::int64_t{1}, "waveform_id json L");
    check::equal(lv::waveform_id(V::number(2.7)), std::int64_t{2}, "waveform_id 2.7 (int() truncates)");
    check::equal(lv::waveform_id(V::number(-1.5)), std::int64_t{-1}, "waveform_id -1.5");
    check::equal(lv::waveform_id(V::boolean(true)), std::int64_t{1}, "waveform_id true");
    check::equal(lv::waveform_id(V::integer(7)), std::int64_t{7}, "waveform_id 7");
    raises<std::invalid_argument>([] { lv::waveform_id(V::number(NaN)); }, "cannot convert float NaN to integer",
                                  "waveform_id NaN");
    raises<std::overflow_error>([] { lv::waveform_id(V::number(INF)); }, "cannot convert float infinity to integer",
                                "waveform_id inf");
    raises<math::TypeError>([] { lv::waveform_id(V::null()); },
                            "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'",
                            "waveform_id null");

    auto due = [](const char* text) { return lv::due_waveform(qrss::pyjson::loads(text)); };
    check::equal(due(R"({"frame": "l-tiny"})"), std::int64_t{1}, "due: an L frame");
    check::equal(due(R"({"frame": "tiny", "waveform": "L"})"), std::int64_t{0}, "due: the frame wins");
    check::equal(due(R"({"frame": "nope", "waveform": "L"})"), std::int64_t{1}, "due: an unknown frame, the waveform");
    check::equal(due(R"({"frame": 5, "waveform": 1})"), std::int64_t{1}, "due: a frame that is not a name");
    check::equal(due(R"({})"), std::int64_t{0}, "due: neither (stored before L)");
    check::is_true(lv::frame_waveform("l-short") == std::optional<int>(1), "frame_waveform l-short");
    check::is_true(!lv::frame_waveform("x"), "frame_waveform x");
}

// --- the FE ring ---------------------------------------------------------------------------

std::vector<cf> ramp(int a, int b, bool imag = false) {
    std::vector<cf> v;
    for (int i = a; i <= b; ++i) v.push_back(imag ? cf(0.0f, float(i)) : cf(float(i), 0.0f));
    return v;
}

void test_fe_ring() {
    lv::FeRing r(0.01);
    check::equal(r.n(), std::int64_t{40}, "ring n");
    check::is_true(!r.end(), "a new ring: no end");
    const lv::FeSpan none = r.read(3, 7);
    equal_vec(as_ints(none.ok), {0, 0, 0, 0}, "a new ring reads unwritten");

    r.write(0, ramp(1, 10));
    lv::FeSpan s = r.read(-5, 15);
    equal_floats(re(s.x), {0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0}, "r1 x");
    equal_vec(as_ints(s.ok), {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}, "r1 ok");

    // longer than the ring: its last 40 samples, 11..50 at 110..150
    r.write(100, ramp(1, 50));
    check::is_true(r.end() == std::optional<std::int64_t>(150), "r2 end");
    s = r.read(105, 152);
    std::vector<float> want_x(5, 0.0f);
    std::vector<int> want_ok(5, 0);
    for (int i = 11; i <= 50; ++i) {
        want_x.push_back(float(i));
        want_ok.push_back(1);
    }
    want_x.insert(want_x.end(), {0, 0});
    want_ok.insert(want_ok.end(), {0, 0});
    equal_floats(re(s.x), want_x, "r2 x");
    equal_vec(as_ints(s.ok), want_ok, "r2 ok");

    // past the end: the hole between is cleared
    r.write(300, ramp(1, 5, true));
    check::is_true(r.end() == std::optional<std::int64_t>(305), "r3 end");
    s = r.read(258, 306);
    std::vector<float> want_im(42, 0.0f);
    std::vector<int> want_ok3(42, 0);
    for (int i = 1; i <= 5; ++i) {
        want_im.push_back(float(i));
        want_ok3.push_back(1);
    }
    want_im.push_back(0.0f);
    want_ok3.push_back(0);
    equal_floats(im(s.x), want_im, "r3 x");
    equal_vec(as_ints(s.ok), want_ok3, "r3 ok");

    // behind the window: lands on 0 mod 40, the slot of index 280 (the reference's rule)
    r.write(0, std::vector<cf>{cf(9.0f, 0.0f)});
    check::is_true(r.end() == std::optional<std::int64_t>(305), "r4 end (unchanged)");
    s = r.read(278, 284);
    equal_floats(re(s.x), {0, 0, 9, 0, 0, 0}, "r4 x");
    equal_vec(as_ints(s.ok), {0, 0, 1, 0, 0, 0}, "r4 ok");

    // read_into agrees with read; clear
    std::vector<cf> x(6);
    std::vector<std::uint8_t> ok(6);
    r.read_into(278, x, ok);
    check::is_true(x == s.x && ok == s.ok, "read_into == read");
    r.clear(300, 302);
    equal_vec(as_ints(r.read(299, 304).ok), {0, 0, 0, 1, 1}, "clear [300, 302)");
    raises<std::invalid_argument>([&] { (void)r.read(5, 4); }, "negative dimensions are not allowed", "read b < a");
    raises<std::invalid_argument>([&] { r.read_into(0, x, std::span<std::uint8_t>(ok).first(2)); },
                                  "read_into: 6 samples and 2 flags", "read_into sizes");
    check::equal(r.samples().size(), std::size_t{40}, "samples()");
    check::equal(r.written().size(), std::size_t{40}, "written()");

    raises<std::invalid_argument>([] { lv::FeRing r(-1.0); }, "negative dimensions are not allowed", "FeRing(-1)");
    raises<std::invalid_argument>([] { lv::FeRing r(NaN); }, "cannot convert float NaN to integer", "FeRing(NaN)");
    raises<std::overflow_error>([] { lv::FeRing r(INF); }, "cannot convert float infinity to integer", "FeRing(inf)");
    lv::FeRing empty(0.0);
    empty.write(5, {});
    check::is_true(empty.end() == std::optional<std::int64_t>(5), "a zero ring: an empty write moves end");
    raises<std::out_of_range>([&] { empty.write(5, ramp(1, 2)); }, "index 0 is out of bounds for axis 0 with size 0",
                              "a zero ring: a write");
}

void test_fe_ring_threads() {
    // Writers append whole chunks; readers must only ever see a chunk whole
    // (every sample of a chunk carries the chunk's number).
    lv::FeRing r(0.25);   // 1000 samples
    constexpr int chunk = 50, chunks = 2000;
    std::atomic<bool> done{false};
    std::atomic<int> torn{0};
    auto reader = [&] {
        while (!done.load()) {
            const std::optional<std::int64_t> end = r.end();
            if (!end) continue;
            const std::int64_t a = (*end / chunk - 4) * chunk;
            const lv::FeSpan s = r.read(a, a + 4 * chunk);
            for (int c = 0; c < 4; ++c) {
                const std::size_t k = static_cast<std::size_t>(c * chunk);
                for (int i = 1; i < chunk; ++i)
                    if (s.ok[k + i] != s.ok[k] || (s.ok[k] && s.x[k + i] != s.x[k])) ++torn;
            }
        }
    };
    std::thread t1(reader), t2(reader);
    for (int c = 0; c < chunks; ++c)
        r.write(std::int64_t{c} * chunk, std::vector<cf>(chunk, cf(float(c), -float(c))));
    done = true;
    t1.join();
    t2.join();
    check::equal(torn.load(), 0, "no read sees a chunk part-written");
    check::is_true(r.end() == std::optional<std::int64_t>(std::int64_t{chunks} * chunk), "threads: end");
}

void test_spans() {
    const std::vector<std::uint8_t> m = {1, 0, 0, 1, 0};
    const lv::Erase e = lv::spans(m, 2.0, 1.0);
    check::is_true(e == lv::Erase{{0.0, 1.0}, {1.5, 2.0}}, "spans of 10010");
    check::is_true(lv::spans(std::vector<std::uint8_t>{0, 0}, 4000.0, 0.0) == lv::Erase{{0.0, 0.0005}}, "spans 00");
    check::is_true(lv::spans({}, 4000.0, 0.0).empty(), "spans of nothing");
    check::is_true(lv::spans(std::vector<std::uint8_t>{1, 1, 1}, 4000.0, 0.0).empty(), "spans of all heard");
}

// --- a slot's capture -------------------------------------------------------------------------

void test_capture_span() {
    struct Row {
        std::int64_t q;
        const char* frame;
        std::int64_t a, b, origin;
    };
    const Row rows[] = {
        {0, "tiny", -56000, 237400, 60000},
        {0, "short", -56000, 1047047, 60000},
        {1990584, "tiny", 7166102344000, 7166102637400, 60000},
        {1990584, "short", 7166102344000, 7166103447047, 60000},
    };
    for (const Row& r : rows) {
        const lv::CaptureSpan s = lv::capture_span(r.q, lv::frame_seconds(fr::get(r.frame)));
        const std::string w = std::string("capture_span ") + r.frame + " " + std::to_string(r.q);
        check::equal(s.a, r.a, w + " a");
        check::equal(s.b, r.b, w + " b");
        check::equal(s.origin, r.origin, w + " origin");
    }
    raises<std::invalid_argument>([] { lv::capture_span(0, NaN); }, "cannot convert float NaN to integer",
                                  "capture_span NaN");
    raises<std::overflow_error>([] { lv::capture_span(0, INF); }, "cannot convert float infinity to integer",
                                "capture_span inf");

    // capture_read through a ring and through a reader: the same span
    const std::vector<fr::FrameSpec> tiny = {fr::get("tiny")};
    lv::FeRing ring(lv::RING_S);
    const lv::CaptureSpan s = lv::capture_span(0, lv::frame_seconds(tiny[0]));
    ring.write(s.a + 10, std::vector<cf>(100, cf(1.0f, 2.0f)));
    const lv::SlotRead a = lv::capture_read(ring, 0, tiny);
    check::equal(a.x.size(), static_cast<std::size_t>(s.b - s.a), "capture_read length");
    check::equal(a.origin, s.origin, "capture_read origin");
    check::equal(int(a.ok[9]) * 1000 + int(a.ok[10]) * 100 + int(a.ok[109]) * 10 + int(a.ok[110]), 110,
                 "capture_read: what was written");
    const lv::FeReader reader = [&](std::int64_t lo, std::int64_t hi) { return ring.read(lo, hi); };
    const lv::SlotRead b = lv::capture_read(reader, 0, tiny);
    check::is_true(a.x == b.x && a.ok == b.ok && a.origin == b.origin, "capture_read: a reader reads the same");
    const lv::FeReader short_reader = [](std::int64_t, std::int64_t) { return lv::FeSpan{{cf{}}, {1}}; };
    raises<std::invalid_argument>([&] { lv::capture_read(short_reader, 0, tiny); },
                                  "a reader returned 1 samples and 1 flags for " + std::to_string(s.b - s.a),
                                  "capture_read: a reader's wrong length");
}

void test_default_noise() {
    const std::vector<cf> a = lv::default_noise(5, 200000);
    const std::vector<cf> b = lv::default_noise(5, 200000);
    const std::vector<cf> c = lv::default_noise(6, 1000);
    const std::vector<cf> head = lv::default_noise(5, 1000);
    check::is_true(a == b, "default_noise: deterministic");
    check::is_true(std::equal(head.begin(), head.end(), a.begin()), "default_noise: a prefix of a longer draw");
    check::is_true(!std::equal(c.begin(), c.end(), a.begin()), "default_noise: another seed, other samples");
    check::is_true(lv::default_noise(-1, 3).size() == 3 && lv::default_noise(0, 0).empty(), "default_noise sizes");
    double sre = 0, sim = 0, p = 0, cross = 0;
    std::vector<double> mag;
    for (const cf& z : a) {
        sre += z.real();
        sim += z.imag();
        p += std::norm(z);
        cross += double(z.real()) * z.imag();
        mag.push_back(std::norm(z));
    }
    const double n = double(a.size());
    std::nth_element(mag.begin(), mag.begin() + mag.size() / 2, mag.end());
    // E|n|^2 = 1/ln 2 and the median of |n|^2 is 1 (the level normalise leaves heard audio at)
    check::is_true(std::fabs(p / n - 1.0 / std::log(2.0)) < 0.01, "default_noise: E|n|^2 = 1/ln 2");
    check::is_true(std::fabs(mag[mag.size() / 2] - 1.0) < 0.01, "default_noise: median |n|^2 = 1");
    check::is_true(std::fabs(sre / n) < 0.006 && std::fabs(sim / n) < 0.006, "default_noise: zero mean");
    check::is_true(std::fabs(cross / n) < 0.006, "default_noise: re and im uncorrelated");
}

void test_capture_prepare_seams() {
    // identity front end and a constant noise: what capture_prepare itself does
    const std::int64_t margin = static_cast<std::int64_t>(lv::CAPTURE_MARGIN_S * 4000);
    const std::size_t n = static_cast<std::size_t>(3 * margin);
    std::vector<cf> x(n);
    std::vector<std::uint8_t> ok(n, 0);
    for (std::size_t i = 0; i < n; ++i) x[i] = cf(float(i % 97), -float(i % 89));
    for (std::size_t i = 100; i < 2000; ++i) ok[i] = 1;
    for (std::size_t i = 3000; i < 3010; ++i) ok[i] = 1;
    std::size_t blank_len = 0, norm_len = 0;
    std::optional<std::int64_t> seed_seen;
    std::size_t noise_len = 0;
    lv::PrepareSeams seams;
    seams.blank = [&](std::span<const cf> fe) {
        blank_len = fe.size();
        return qrss::frontend::Blanked{{fe.begin(), fe.end()}, std::vector<float>(fe.size(), 0.5f)};
    };
    seams.normalise = [&](std::span<const cf> fe, std::span<const float> keep) {
        norm_len = fe.size();
        bool keep_ok = true;
        for (std::size_t i = 0; i < keep.size(); ++i) keep_ok = keep_ok && keep[i] == (ok[i] ? 0.5f : 0.0f);
        check::is_true(keep_ok, "normalise sees keep * ok");
        std::vector<cf> o(fe.begin(), fe.end());
        for (cf& z : o) z *= 2.0f;
        return qrss::frontend::Normalised{o, std::vector<float>(fe.size(), 1.0f)};
    };
    seams.noise = [&](std::int64_t s, std::size_t m) {
        seed_seen = s;
        noise_len = m;
        return std::vector<cf>(m, cf(7.0f, -7.0f));
    };
    const std::int64_t origin = 500;
    const lv::SlotCapture c = lv::capture_prepare(x, ok, origin, 42, 0.25, std::nullopt, seams);
    check::equal(blank_len, static_cast<std::size_t>(3010 + margin), "blank sees up to the margin past the newest heard");
    check::equal(norm_len, blank_len, "normalise: the same samples");
    check::is_true(seed_seen == std::optional<std::int64_t>(42), "the noise's seed: q");
    std::size_t missing = 0;
    bool fe_ok = true, keep_ok = true;
    for (std::size_t i = 0; i < n; ++i) {
        missing += ok[i] ? 0 : 1;
        const cf want = ok[i] ? x[i] * 2.0f : cf(7.0f, -7.0f);
        fe_ok = fe_ok && c.prep.fe[i] == want;
        keep_ok = keep_ok && (*c.prep.keep)[i] == (ok[i] ? 0.5f : 1.0f);
    }
    check::equal(noise_len, missing, "noise for every sample not heard");
    check::is_true(fe_ok, "fe: normalised where heard, noise where not");
    check::is_true(keep_ok, "keep: the blanker's where heard, 1 where filled");
    check::equal(c.prep.q, std::int64_t{42}, "prep.q");
    check::equal(c.prep.t0_index, 500.0, "prep.t0_index");
    check::equal(c.prep.fs, std::int64_t{4000}, "prep.fs");
    check::is_true(c.erase == lv::spans(ok, 4000.0, 500.0), "erase: spans(ok)");
    check::equal(c.heard, 1.0, "heard: ok[500:1500]");
    const lv::SlotCapture c2 = lv::capture_prepare(x, ok, origin, 42, 1.0, std::optional<std::int64_t>(3), seams);
    check::is_true(seed_seen == std::optional<std::int64_t>(3), "the noise's seed: given");
    check::equal(c2.heard, 1510.0 / 4000.0, "heard: ok[500:4500]");
    const lv::SlotCapture c3 = lv::capture_prepare(x, ok, std::int64_t(n) + 5, 42, 1.0, std::nullopt, seams);
    check::equal(c3.heard, 0.0, "heard: origin past the capture");

    lv::PrepareSeams bad = seams;
    bad.noise = [](std::int64_t, std::size_t m) { return std::vector<cf>(m + 1); };
    raises<std::invalid_argument>([&] { lv::capture_prepare(x, ok, origin, 42, 1.0, std::nullopt, bad); },
                                  "the noise gave " + std::to_string(missing + 1) + " samples for " +
                                      std::to_string(missing),
                                  "a noise of the wrong length");
    raises<std::invalid_argument>(
        [&] { lv::capture_prepare(x, std::span<const std::uint8_t>(ok).first(5), origin, 42, 1.0); },
        "x has " + std::to_string(n) + " samples and ok 5", "x and ok differ");

    // the real front end: heard audio at unit level, the noise the default's
    std::vector<cf> hiss = lv::default_noise(99, n);
    std::vector<cf> loud(n);
    for (std::size_t i = 0; i < n; ++i) loud[i] = hiss[i] * 30.0f;
    const lv::SlotCapture r = lv::capture_prepare(loud, ok, origin, 42, 1.0);
    const std::vector<cf> fill = lv::default_noise(42, missing);
    std::size_t j = 0;
    bool filled = true, finite = true;
    for (std::size_t i = 0; i < n; ++i) {
        finite = finite && std::isfinite(r.prep.fe[i].real()) && std::isfinite(r.prep.fe[i].imag());
        if (!ok[i]) filled = filled && r.prep.fe[i] == fill[j++] && (*r.prep.keep)[i] == 1.0f;
    }
    check::is_true(filled, "the default noise, seeded by q, where not heard");
    check::is_true(finite, "the real front end: finite");
    double p = 0;
    for (std::size_t i = 100; i < 2000; ++i) p += std::norm(r.prep.fe[i]);
    check::is_true(std::fabs(p / 1900.0 - 1.0 / std::log(2.0)) < 0.25, "the real front end: heard at noise level");
}

// --- the end-of-slot receive's decisions -------------------------------------------------------

void test_finish_decisions() {
    check::is_true(lv::py_slice(10, 3, 7) == std::pair<std::int64_t, std::int64_t>(3, 7), "py_slice 3:7");
    check::is_true(lv::py_slice(10, -3, 20) == std::pair<std::int64_t, std::int64_t>(7, 10), "py_slice -3:20");
    check::is_true(lv::py_slice(10, 12, 20) == std::pair<std::int64_t, std::int64_t>(10, 10), "py_slice 12:20");
    check::is_true(lv::py_slice(10, 5, 2) == std::pair<std::int64_t, std::int64_t>(5, 5), "py_slice 5:2");
    check::is_true(lv::py_slice(10, -20, -15) == std::pair<std::int64_t, std::int64_t>(0, 0), "py_slice -20:-15");

    const double fs1 = 1000.0 / 4000.0;   // a frame of 1000 FE samples
    std::vector<std::uint8_t> ok(3000, 0);
    lv::FinishGate g = lv::finish_gate(ok, 100, fs1, true, true);
    check::is_true(!g.receive && g.k1 == 1100 && g.heard == 0.0, "gate: nothing heard");
    for (std::size_t i = 100; i < 500; ++i) ok[i] = 1;   // 40% of the frame
    g = lv::finish_gate(ok, 100, fs1, false, false);
    check::is_true(!g.receive && g.heard == 0.4, "gate: too little and nothing known");
    check::is_true(lv::finish_gate(ok, 100, fs1, true, false).receive, "gate: too little, with dets");
    check::is_true(lv::finish_gate(ok, 100, fs1, false, true).receive, "gate: too little, with hints");
    check::is_true(lv::finish_gate(ok, 100, fs1, false, false, 0.4).receive, "gate: min_heard 0.4");
    g = lv::finish_gate(ok, 3500, fs1, true, true);
    check::is_true(!g.receive && g.k1 == 3000 && g.heard == 0.0, "gate: origin past the capture");
    g = lv::finish_gate(ok, -2000, fs1, false, false);   // ok[-2000:-1000]: Python's negative slice
    check::is_true(!g.receive && g.k1 == -1000 && g.heard == 0.0, "gate: a negative origin");
    g = lv::finish_gate(ok, -2600, 0.25, false, false);   // ok[-2600:-1600] = ok[400:1400]
    check::is_true(!g.receive && g.k1 == -1600 && g.heard == 0.1, "gate: a negative origin, a tenth heard");

    using B = lv::FinishBranch;
    check::is_true(lv::finish_branch(0.5, false, false) == B::Search, "branch: half heard, a search");
    check::is_true(lv::finish_branch(0.49, true, false) == B::Known, "branch: dets");
    check::is_true(lv::finish_branch(0.49, false, true) == B::Known, "branch: hints");
    check::is_true(lv::finish_branch(0.0, true, true) == B::Nothing, "branch: nothing heard");
    check::is_true(lv::finish_branch(0.49, false, false) == B::Nothing, "branch: nothing known");
    check::is_true(lv::finish_branch(NaN, true, true) == B::Nothing, "branch: NaN heard");

    const qrss::api::ReceiveConfig c = lv::finish_config({fr::get("tiny")}, "plain");
    check::equal(c.estimator, std::string("plain"), "finish_config estimator");
    check::is_true(c.frames.size() == 1 && c.frames[0].name() == "tiny", "finish_config frames");
    // the budgets are finish_receive's own (the reference's constants), not the config's
    const lv::FinishOptions fo;
    check::is_true(fo.max_candidates == std::optional<std::int64_t>(lv::FINISH_MAX_CANDIDATES),
                   "FinishOptions candidates");
    check::is_true(fo.budget_s == std::optional<double>(lv::FINISH_BUDGET_S), "FinishOptions budget");
    check::equal(fo.min_heard, lv::FINISH_SEARCH_MIN_HEARD, "FinishOptions min_heard");

    qrss::types::Detection d;
    d.f_hz = 1000.0;
    d.waveform = cfg::WAVEFORM_L;
    qrss::types::Detection e;
    e.f_hz = 1500.0;
    const std::vector<qrss::types::Detection> dets = {d, e};
    check::is_true(lv::known_hz(dets, false) == std::vector<double>{1000.0, 1500.0}, "known_hz: their f");
    check::is_true(lv::known_hz(dets, true) == std::vector<double>{990.5, 1500.0}, "known_hz joint: band centres");

    // finish_receive that receives nothing: no receiver, no noise
    lv::FinishOptions opt;
    bool drew = false;
    opt.seams.noise = [&](std::int64_t, std::size_t m) {
        drew = true;
        return std::vector<cf>(m);
    };
    const qrss::api::ReceiveConfig tc = lv::finish_config({fr::get("tiny")});
    const std::vector<cf> x(1000);
    const std::vector<std::uint8_t> none(1000, 0);
    const lv::FinishResult fres = lv::finish_receive(x, none, 100, 7, tc, {}, {}, opt);
    check::is_true(fres.passes.empty() && fres.heard == 0.0 && !drew, "finish_receive: nothing heard");
}

// --- the dial ---------------------------------------------------------------------------------

const char DIAL_BYTES[] =
    "{\"t\": 10, \"dial_hz\": 7038600}\n{\"t\": 5, \"dial_hz\": 7038600}\ngarbage\n"
    "{\"t\": 20, \"dial_hz\": 14095600.5}\n{\"t\": 21, \"dial_hz\": 14095601.5}\n"
    "{\"t\": \"30\", \"dial_hz\": \"10138700\"}\n{\"t\": 40, \"dial_hz\": 0}\n{\"t\": NaN, \"dial_hz\": 5}\n"
    "{\"t\": 50, \"dial_hz\": true}\n{\"t\": 60, \"dial_hz\": 10138700}\n[1, 2]\n{\"dial_hz\": 1}\n{\"t\": 70}\n"
    "{\"t\": 123456789012345678901234567890, \"dial_hz\": 3}\n"
    "{\"t\": 80, \"dial_hz\": 4, \"x\": 123456789012345678901234567890}\n{\"t\": 90, \"dial_hz\": 5\r\n"
    "{\"t\": 95, \"dial_hz\": 6}\xc2\x85{\"t\": 96, \"dial_hz\": 7}\xe2\x80\xa8{\"t\": 97, \"dial_hz\": 8}\n"
    "{\"t\": 98, \"dial_hz\": 9}\xff\n";

// DialLog(...).refresh(); .points of DIAL_BYTES, the reference's
const std::vector<lv::DialPoint> DIAL_POINTS = {
    {5.0, 7038600},  {20.0, 14095600}, {21.0, 14095602}, {30.0, 10138700},
    {50.0, 1},       {60.0, 10138700}, {80.0, 4},        {95.0, 6},
    {96.0, 7},       {97.0, 8},        {1.2345678901234568e+29, 3},
};

std::string pts_text(const std::vector<lv::DialPoint>& p) {
    std::string s = "[";
    for (const lv::DialPoint& d : p) s += "(" + std::to_string(d.t) + ", " + std::to_string(d.dial_hz) + ") ";
    return s + "]";
}

void equal_points(const std::vector<lv::DialPoint>& got, const std::vector<lv::DialPoint>& want,
                  const std::string& what) {
    ++check::checks;
    if (got != want) check::fail(what, "got " + pts_text(got) + ", want " + pts_text(want));
}

void test_parse_dial() {
    equal_points(lv::parse_dial_text(std::string_view(DIAL_BYTES, sizeof DIAL_BYTES - 1)), DIAL_POINTS,
                 "parse_dial_text");
    equal_points(lv::parse_dial_text(""), {}, "parse_dial_text of nothing");
    equal_points(lv::parse_dial_text("{\"t\": 1, \"dial_hz\": 2}\x0b{\"t\": 2, \"dial_hz\": 3}\x1d"
                                     "{\"t\": 3, \"dial_hz\": 4}\x0c{\"t\": 4, \"dial_hz\": 5}"),
                 {{1.0, 2}, {2.0, 3}, {3.0, 4}, {4.0, 5}}, "parse_dial_text: splitlines' separators");
    // the port's skips (the reference raises OverflowError out of refresh, or keeps a Python int)
    equal_points(lv::parse_dial_text("{\"t\": 1, \"dial_hz\": 3}\n{\"t\": 5, \"dial_hz\": 1e400}\n"
                                     "{\"t\": 6, \"dial_hz\": \"-Infinity\"}\n"
                                     "{\"t\": 7, \"dial_hz\": 123456789012345678901234567890}\n"
                                     "{\"t\": 8, \"dial_hz\": 1e30}\n{\"t\": 9, \"dial_hz\": 9223372036854775807}\n"
                                     "{\"t\": 10, \"dial_hz\": 9223372036854774784}"),
                 {{1.0, 3}, {10.0, 9223372036854774784}}, "parse_dial_text: dials past int64 skipped");
    // a stable sort: equal times keep the file's order
    equal_points(lv::parse_dial_text("{\"t\": 5, \"dial_hz\": 1}\n{\"t\": 5, \"dial_hz\": 2}\n{\"t\": 1, \"dial_hz\": 3}"),
                 {{1.0, 3}, {5.0, 1}, {5.0, 2}}, "parse_dial_text: stable");
    // big integers inside strings are left alone, and an escaped quote does not end a string
    equal_points(lv::parse_dial_text("{\"s\": \"\\\"12345678901234567890123\", \"t\": 1e1, \"dial_hz\": 2, "
                                     "\"n\": -98765432109876543210}"),
                 {{10.0, 2}}, "parse_dial_text: a big int beside a string");
}

std::string dial_spans_text(const std::vector<lv::DialSpan>& v) {
    std::string s = "[";
    for (const lv::DialSpan& d : v)
        s += "(" + std::to_string(d.lo) + ", " + std::to_string(d.hi) + ", " +
             (d.dial ? std::to_string(*d.dial) : std::string("None")) + ") ";
    return s + "]";
}

void equal_spans(const std::vector<lv::DialSpan>& got, const std::vector<lv::DialSpan>& want, const std::string& what) {
    ++check::checks;
    bool same = got.size() == want.size();
    for (std::size_t i = 0; same && i < got.size(); ++i)
        same = got[i].lo == want[i].lo && got[i].hi == want[i].hi && got[i].dial == want[i].dial;
    if (!same) check::fail(what, "got " + dial_spans_text(got) + ", want " + dial_spans_text(want));
}

void test_dial_queries() {
    const std::vector<lv::DialPoint> pts = {{100.0, 7}, {200.0, 14}, {300.0, 7}, {400.0, 10}};
    const lv::DialGuard g = {2.0, 5.0};
    equal_spans(lv::dial_spans(pts, 150, 350, g), {{-INF, 198.0, 7}, {205.0, 298.0, 14}, {305.0, 398.0, 7}},
                "spans(150, 350)");
    equal_spans(lv::dial_spans(pts, -INF, INF, g),
                {{-INF, 198.0, 7}, {205.0, 298.0, 14}, {305.0, 398.0, 7}, {405.0, INF, 10}}, "spans(-inf, inf)");
    equal_spans(lv::dial_spans(pts, 210, 210, g), {{205.0, 298.0, 14}}, "spans(210, 210)");
    equal_spans(lv::dial_spans({}, 0, 1, g), {{-INF, INF, std::nullopt}}, "spans: no dial");
    const std::vector<lv::DialPoint> one = {{100.0, 7}};
    equal_spans(lv::dial_spans(one, 0, 1, g), {{-INF, INF, 7}}, "spans: one dial");
    const lv::DialEpochs by = lv::dial_by_dial(pts, 0, 1000, g);
    using SP = std::vector<std::pair<double, double>>;
    check::is_true(by.size() == 3 && by[0].first == std::optional<std::int64_t>(7) &&
                       by[0].second == SP{{-INF, 198.0}, {305.0, 398.0}} &&
                       by[1].first == std::optional<std::int64_t>(14) && by[1].second == SP{{205.0, 298.0}} &&
                       by[2].first == std::optional<std::int64_t>(10) && by[2].second == SP{{405.0, INF}},
                   "by_dial(0, 1000)");
    const std::pair<double, lv::DialSpan> at[] = {
        {50, {-INF, 198.0, 7}},    {100, {-INF, 198.0, 7}},  {199.9, {-INF, 198.0, 7}},
        {200, {205.0, 298.0, 14}}, {1e9, {405.0, INF, 10}},  {-INF, {-INF, 198.0, 7}},
        {NaN, {405.0, INF, 10}},   // numpy sorts NaN last
    };
    for (const auto& [t, want] : at) equal_spans({lv::dial_span_at(pts, t, g)}, {want}, "span_at " + std::to_string(t));
    equal_spans({lv::dial_span_at(one, 5, g)}, {{-INF, INF, 7}}, "span_at: one dial");
    check::is_true(lv::same_dial(1, 1) && lv::same_dial(std::nullopt, 2) && lv::same_dial(3, std::nullopt) &&
                       !lv::same_dial(1, 2),
                   "same_dial");
}

void test_dial_log() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "sstvae_qrss_test_live";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    const std::string path = util::path_text(dir / "dial.jsonl");
    auto put = [&](std::string_view bytes) {
        util::File f = util::open_file(path, "wb");
        if (!f) throw std::runtime_error("cannot write " + path);
        std::fwrite(bytes.data(), 1, bytes.size(), f.get());
    };
    lv::DialLog log(path);
    log.refresh();
    check::is_true(log.points().empty(), "DialLog: no file, no points");
    check::equal(log.dial_at(5.0).has_value(), false, "DialLog: no file, the dial unknown");
    put(std::string_view(DIAL_BYTES, sizeof DIAL_BYTES - 1));
    log.refresh();
    equal_points(log.points(), DIAL_POINTS, "DialLog.refresh");
    check::is_true(log.dial_at(25.0) == std::optional<std::int64_t>(14095602), "DialLog.dial_at");
    const lv::DialSpan s = log.span_at(25.0);
    check::is_true(s.lo == 21.0 + lv::DIAL_GUARD_S[1] && s.hi == 30.0 - lv::DIAL_GUARD_S[0], "DialLog.span_at");
    // DIAL_GUARD_S (2, 50) leaves three epochs: [(-inf, 18, 7038600), (147, 1.2e29, 8), (1.2e29, inf, 3)]
    check::equal(log.spans(0, 1e30).size(), std::size_t{3}, "DialLog.spans");
    check::equal(log.by_dial(0, 1e30).size(), std::size_t{3}, "DialLog.by_dial");
    check::is_true(log.spans(0, 1e30)[1].lo == 147.0 && log.spans(0, 1e30)[1].dial == std::optional<std::int64_t>(8),
                   "DialLog.spans: the guard after a change");

    // the same signature is not read again: points set by hand survive a refresh
    log.set_points({{1.0, 2}});
    log.refresh();
    equal_points(log.points(), {{1.0, 2}}, "DialLog: an unchanged file is not re-read");
    // a change of size is
    put("{\"t\": 3, \"dial_hz\": 4}\n");
    log.refresh();
    equal_points(log.points(), {{3.0, 4}}, "DialLog: a changed file is re-read");
    // removed: no dial
    fs::remove(fs::path(dir / "dial.jsonl"), ec);
    log.refresh();
    check::is_true(log.points().empty(), "DialLog: a removed file, no points");
    // a directory: stat works and the read fails, so the points are kept
    fs::create_directories(dir / "dial.jsonl");
    log.set_points({{1.0, 5}});
    log.refresh();
    equal_points(log.points(), {{1.0, 5}}, "DialLog: unreadable, the points kept");
    fs::remove_all(dir, ec);
}

// --- restrict --------------------------------------------------------------------------------

void test_restrict() {
    const std::int64_t q = 3;   // t0 = 2701
    std::vector<std::uint8_t> ok(40, 1);
    ok[5] = 0;
    auto kept = [&](std::vector<std::pair<double, double>> sp) {
        const std::vector<std::uint8_t> out = lv::restrict(ok, 10, q, sp);
        std::vector<int> idx;
        for (std::size_t i = 0; i < out.size(); ++i)
            if (out[i]) idx.push_back(int(i));
        return idx;
    };
    equal_vec(kept({{2701.001, 2701.004}}), {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}, "restrict 1");
    equal_vec(kept({{-INF, 2701.002}, {2701.006, INF}}),
              {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 34, 35, 36, 37, 38, 39}, "restrict 2");
    equal_vec(kept({{2700.0, 2700.5}}), {}, "restrict 3");
    equal_vec(kept({{2701.0011, 2701.00151}}), {15, 16}, "restrict 4");
    equal_vec(kept({}), {}, "restrict: no spans");
    std::vector<int> all;
    for (int i = 0; i < 40; ++i)
        if (i != 5) all.push_back(i);
    equal_vec(kept({{2701.001, 2701.004}, {-INF, INF}}), all, "restrict: an unbounded span, ok itself");
    check::is_true(!lv::restrict_ranges(40, 10, q, std::vector<std::pair<double, double>>{{-INF, INF}}),
                   "restrict_ranges: unbounded");
    const auto r = lv::restrict_ranges(40, 10, q, std::vector<std::pair<double, double>>{{2701.0, 2701.0}, {-1e300, 1e300}});
    check::is_true(r && *r == std::vector<std::pair<std::int64_t, std::int64_t>>{{0, 40}},
                   "restrict_ranges: an empty span dropped, a huge one clamped");
    raises<std::overflow_error>([&] { kept({{INF, INF}}); }, "cannot convert float infinity to integer",
                                "restrict: +inf as lo");
    raises<std::invalid_argument>([&] { kept({{NaN, 1.0}}); }, "cannot convert float NaN to integer", "restrict: NaN");
}

// --- tiles and the configuration ----------------------------------------------------------------

void test_tile() {
    lv::Tile t;
    t.id = "a";
    t.q = 7;
    t.slot_utc = "2026-10-09T06:00Z";
    t.frame = "tiny";
    t.f_hz = 1234.5;
    t.dial_hz = 14095600;
    t.snr_db = -3.25;
    t.z_ref = 8.0;
    t.callsign = "N0CALL";
    t.mode = "B";
    t.segment = 2;
    t.progress = 0.5;
    t.heard = 1.0;
    t.received = 0.25;
    t.mean_w_db = -1.5;
    t.confidence = {std::nullopt, 1.5, -2.0};
    t.passes = 2;
    t.image = "x.png";
    t.image_rev = 3;
    t.updated = 1760000000.125;
    t.note = "n";
    t.uid = "u";
    t.waveform = "L";
    const std::string want =
        R"({"id": "a", "q": 7, "slot_utc": "2026-10-09T06:00Z", "frame": "tiny", "f_hz": 1234.5, )"
        R"("dial_hz": 14095600, "status": "receiving", "snr_db": -3.25, "z_ref": 8.0, "callsign": "N0CALL", )"
        R"("grid": null, "picture_id": null, "mode": "B", "segment": 2, "cw_text": null, "progress": 0.5, )"
        R"("heard": 1.0, "received": 0.25, "mean_w_db": -1.5, "confidence": [null, 1.5, -2.0], "passes": 2, )"
        R"("image": "x.png", "image_rev": 3, "updated": 1760000000.125, "note": "n", "uid": "u", "waveform": "L"})";
    const qrss::pyjson::Value j = t.to_json();
    check::equal(qrss::pyjson::dumps(j), want, "Tile.to_json: json.dumps(asdict(tile))");
    const std::optional<lv::Tile> back = lv::Tile::from_json(j);
    check::is_true(back && qrss::pyjson::dumps(back->to_json()) == want, "Tile.from_json: a round trip");

    lv::Tile nf = t;
    nf.snr_db = NaN;
    nf.progress = INF;
    nf.confidence = {-INF};
    const std::string got = qrss::pyjson::dumps(nf.to_json());
    check::is_true(got.find("NaN") == std::string::npos && got.find("Infinity") == std::string::npos &&
                       got.find(R"("snr_db": null)") != std::string::npos &&
                       got.find(R"("progress": null)") != std::string::npos &&
                       got.find(R"("confidence": [null])") != std::string::npos,
                   "Tile.to_json: never a NaN token");

    auto from = [](const char* text) { return lv::Tile::from_json(qrss::pyjson::loads(text)); };
    const std::optional<lv::Tile> m = from(R"({"id": "z", "q": 1, "slot_utc": "s", "frame": "tiny", "f_hz": 900, "other": 1})");
    check::is_true(m && m->f_hz == 900.0 && m->status == "receiving" && m->passes == 1 && m->waveform == "CE",
                   "Tile.from_json: an int f_hz, defaults, an unknown key ignored");
    check::is_true(!from(R"({"id": "z", "q": 1, "slot_utc": "s", "frame": "tiny"})"), "Tile.from_json: f_hz missing");
    check::is_true(!from(R"({"id": "z", "q": "1", "slot_utc": "s", "frame": "tiny", "f_hz": 1.0})"),
                   "Tile.from_json: a str q");
    check::is_true(!from(R"({"id": "z", "q": 1.5, "slot_utc": "s", "frame": "tiny", "f_hz": 1.0})"),
                   "Tile.from_json: a float q");
    check::is_true(!from(R"([1])"), "Tile.from_json: not an object");

    lv::SlotState st;
    st.tiles.emplace_back("a", t);
    st.tiles.emplace_back("b", lv::Tile{});
    check::is_true(st.tile("a") && st.tile("a")->q == 7 && !st.tile("c"), "SlotState.tile");
    st.tile("b")->note = "x";
    check::equal(st.tiles[1].second.note, std::string("x"), "SlotState.tile: a pointer into the slot");
}

void test_live_config() {
    lv::LiveConfig c;
    check::is_true(c.specs().size() == 1 && c.specs()[0].name() == "full", "LiveConfig: full, CE alone");
    check::is_true(!c.joint() && c.selection().size() == 1, "LiveConfig: not joint");
    c.spec = fr::get("tiny");
    c.waveforms = std::vector<std::int64_t>{1, 0, 1};
    const std::vector<fr::FrameSpec> sp = c.specs();
    check::is_true(sp.size() == 2 && sp[0].name() == "tiny" && sp[1].name() == "l-tiny", "LiveConfig: CE's first");
    check::is_true(c.joint() && c.selection().size() == 2, "LiveConfig: joint");
    c.waveforms = std::vector<std::int64_t>{1};
    check::is_true(c.specs().size() == 1 && c.specs()[0].name() == "l-tiny" && c.joint() && c.selection().size() == 1,
                   "LiveConfig: L alone");
    c.waveforms = std::vector<std::int64_t>{2};
    raises<std::invalid_argument>([&] { (void)c.specs(); }, "waveforms (2,): one or more of 0, 1", "waveforms (2,)");
    c.waveforms = std::vector<std::int64_t>{};
    raises<std::invalid_argument>([&] { (void)c.specs(); }, "waveforms (): one or more of 0, 1", "waveforms ()");
    c.waveforms = std::vector<std::int64_t>{0, 5};
    raises<std::invalid_argument>([&] { (void)c.specs(); }, "waveforms (0, 5): one or more of 0, 1", "waveforms (0, 5)");
}

}  // namespace

int main() {
    try {
        test_utc_iso();
        test_small();
        test_fe_ring();
        test_fe_ring_threads();
        test_spans();
        test_capture_span();
        test_default_noise();
        test_capture_prepare_seams();
        test_finish_decisions();
        test_parse_dial();
        test_dial_queries();
        test_dial_log();
        test_restrict();
        test_tile();
        test_live_config();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_live");
}
