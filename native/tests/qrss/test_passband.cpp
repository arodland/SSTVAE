// The passband store (native/core/qrss/passband.hpp) and the directory lock
// (dirlock.hpp), with no Python. Byte-for-byte agreement with the reference's
// files, Python-written hours read here and the cross-process lock are
// tests/test_qrss_native_passband.py's; here the store's own contract: the
// files and their exact JSON, quantisation (clipping, NaN), gaps and masks
// across an hour boundary, meta.json winning on reopen, expiry by an
// injected clock, write_capture's power-of-2 gain and capture's trimming
// and coverage, packing to FLAC and back (a write into a packed hour
// unpacks it), the packer thread, writers and readers on several threads,
// and the lock refusing a second holder until the first lets go.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "qrss/dirlock.hpp"
#include "qrss/frontend.hpp"
#include "qrss/passband.hpp"
#include "qrss/rawfile.hpp"
#include "qrss_check.hpp"
#include "util/fileio.hpp"

#define CHECK(c) sstvae::check::is_true((c), std::string(#c) + " (line " + std::to_string(__LINE__) + ")")

namespace pb = sstvae::qrss::passband;
namespace rawfile = sstvae::qrss::rawfile;
using cf = std::complex<float>;
using sstvae::util::path_text;

namespace {

constexpr std::int64_t HOUR0 = 490'000;   // an hour count (2025-11-24)

std::string slurp(const std::string& path) {
    std::error_code ec;
    const auto s = rawfile::read_all(path, ec);
    return s ? *s : std::string("<missing>");
}

bool exists(const std::string& p) { return sstvae::util::exists(p); }

template <typename E, typename F>
bool throws(F&& f) {
    try {
        f();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// Values exactly representable at scale 8192.
std::vector<cf> ramp(std::size_t n) {
    std::vector<cf> x(n);
    for (std::size_t k = 0; k < n; ++k)
        x[k] = cf(static_cast<float>(static_cast<int>(k % 7) - 3) * 0.125f,
                  static_cast<float>(static_cast<int>(k % 5) - 2) * 0.25f);
    return x;
}

std::vector<cf> noise(std::size_t n, double sd, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<double> d(0.0, sd);
    std::vector<cf> x(n);
    for (auto& v : x) v = cf(static_cast<float>(d(g)), static_cast<float>(d(g)));
    return x;
}

bool same(const std::vector<cf>& a, const std::vector<cf>& b) { return a == b; }

void test_free() {
    CHECK(pb::slot_t0(0) == 1);
    CHECK(pb::slot_t0(-1) == -899);
    CHECK(pb::slot_t0(2'000'000) == 1'800'000'001);
    CHECK(throws<std::overflow_error>([] { pb::slot_t0(std::numeric_limits<std::int64_t>::max()); }));

    using R = std::vector<pb::Range>;
    CHECK((pb::merge_ranges(R{{5, 9}, {0, 3}, {3, 4}, {8, 12}}) == R{{0, 4}, {5, 12}}));
    CHECK((pb::merge_ranges(R{}) == R{}));
    CHECK((pb::merge_ranges(R{{0, 10}, {2, 3}}) == R{{0, 10}}));
    CHECK((pb::merge_ranges(R{{0, 3}, {4, 5}}) == R{{0, 3}, {4, 5}}));

    CHECK(pb::hour_of_name("fe_12.i16") == 12);
    CHECK(pb::hour_of_name("fe_007.flac") == 7);
    CHECK(pb::hour_of_name("fe_12.i16\n") == 12);   // `$` before a final newline
    CHECK(!pb::hour_of_name("fe_12.i16\n\n"));
    CHECK(!pb::hour_of_name("fe_12.json"));
    CHECK(!pb::hour_of_name("fe_.i16"));
    CHECK(!pb::hour_of_name("xfe_1.i16"));
    CHECK(!pb::hour_of_name("fe_1.i16x"));
    CHECK(!pb::hour_of_name("fe_-1.i16"));
    CHECK(!pb::hour_of_name("fe_1.flac.123.tmp"));
    CHECK(!pb::hour_of_name("fe_99999999999999999999.i16"));

    CHECK(pb::py_round(0.5) == 0 && pb::py_round(1.5) == 2 && pb::py_round(2.5) == 2);
    CHECK(pb::py_round(-0.5) == 0 && pb::py_round(-1.5) == -2 && pb::py_round(2.4999) == 2);
    CHECK(pb::py_round(1e15 + 0.5) == 1'000'000'000'000'000);
    CHECK(throws<std::invalid_argument>([] { pb::py_round(std::nan("")); }));
    CHECK(throws<std::overflow_error>([] { pb::py_round(INFINITY); }));
}

void test_store(const std::filesystem::path& dir) {
    const std::string root = path_text(dir / "store" / "nested");
    pb::StoreOptions opt;
    opt.scale = 8192.0;
    pb::PassbandStore s(root, opt);
    CHECK(slurp(root + "/meta.json") == "{\"fs\": 4000, \"scale\": 8192.0}");
    CHECK(s.per_hour() == 14'400'000);

    // across an hour boundary: the last second of HOUR0 and the first of the next
    const double t_a = HOUR0 * 3600.0 + 3599.0;
    const std::vector<cf> x = ramp(8000);
    s.write(t_a, x, false);
    CHECK((s.hours() == std::vector<std::int64_t>{HOUR0, HOUR0 + 1}));
    CHECK(slurp(s.index_path(HOUR0)) == "[[14396000, 14400000]]");
    CHECK(slurp(s.index_path(HOUR0 + 1)) == "[[0, 4000]]");
    std::error_code ec;
    rawfile::RawFile f = rawfile::RawFile::open(s.raw_path(HOUR0), rawfile::Mode::Read, ec);
    CHECK(f.is_open() && f.size(ec).value_or(0) == 57'600'000);
    f.close();

    pb::Masked r = s.read(t_a, t_a + 2.0);
    CHECK(same(r.x, x));
    CHECK(std::all_of(r.mask.begin(), r.mask.end(), [](std::uint8_t m) { return m == 1; }));
    r = s.read(t_a - 1.0, t_a + 3.0);   // a second unwritten either side: zeros, mask 0
    CHECK(r.x.size() == 16000);
    CHECK(std::count(r.mask.begin(), r.mask.end(), std::uint8_t{1}) == 8000);
    CHECK(r.mask[3999] == 0 && r.mask[4000] == 1 && r.mask[11999] == 1 && r.mask[12000] == 0);
    CHECK(r.x[0] == cf(0, 0) && r.x[15999] == cf(0, 0));
    CHECK(s.read(t_a, t_a).x.empty() && s.read(t_a + 1, t_a).x.empty());

    // a gap, then touching ranges merged
    s.write(t_a + 5.0, std::vector<cf>(x.begin(), x.begin() + 100), false);
    CHECK(slurp(s.index_path(HOUR0 + 1)) == "[[0, 4000], [16000, 16100]]");
    s.write(t_a + 2.0, std::vector<cf>(x.begin(), x.begin() + 100), false);
    CHECK(slurp(s.index_path(HOUR0 + 1)) == "[[0, 4100], [16000, 16100]]");

    // clipping to +-32767, NaN to 0, half-even rounding
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<cf> odd = {cf(10.0f, 0.0f), cf(0.0f, -10.0f), cf(nan, 1.0f), cf(1.0f, 1.0f),
                                 cf(2.5f / 8192.0f, -2.5f / 8192.0f)};
    s.write(1000.0, odd, false);
    r = s.read(1000.0, 1000.0 + 5.0 / 4000.0);
    CHECK(r.x.size() == 5);
    CHECK(r.x[0] == cf(32767.0f / 8192.0f, 0.0f) && r.x[1] == cf(0.0f, -32767.0f / 8192.0f));
    CHECK(r.x[2] == cf(0.0f, 1.0f) && r.x[3] == cf(1.0f, 1.0f));
    CHECK(r.x[4] == cf(2.0f / 8192.0f, -2.0f / 8192.0f));

    // reopened: meta.json's fs and scale, not the arguments
    pb::StoreOptions other;
    other.scale = 1.0;
    other.fs = 8000;
    pb::PassbandStore again(root, other);
    CHECK(again.scale() == 8192.0 && again.fs() == 4000);
    CHECK(same(again.read(t_a, t_a + 2.0).x, x));

    // expiry: 48 h after an hour ends, its files go (index and all)
    const double now = (HOUR0 + 1) * 3600.0 + 48 * 3600.0 + 1.0;
    CHECK((again.expire(now) == std::vector<std::int64_t>{0, HOUR0}));
    CHECK(!exists(again.raw_path(HOUR0)) && !exists(again.index_path(HOUR0)));
    CHECK((again.hours() == std::vector<std::int64_t>{HOUR0 + 1}));
    // a write expires by its own time
    again.write((HOUR0 + 60) * 3600.0, x);
    CHECK((again.hours() == std::vector<std::int64_t>{HOUR0 + 60}));

    // the injected clock is expire()'s "now"
    pb::StoreOptions clocked;
    clocked.clock = [] { return (HOUR0 + 200) * 3600.0; };
    pb::PassbandStore c(root, clocked);
    CHECK((c.expire() == std::vector<std::int64_t>{HOUR0 + 60}) && c.hours().empty());

    // a time that is no time
    CHECK(throws<std::invalid_argument>([&] { s.write(std::nan(""), x, false); }));
    CHECK(throws<std::overflow_error>([&] { (void)s.read(0.0, INFINITY); }));
}

void test_capture(const std::filesystem::path& dir) {
    pb::StoreOptions opt;
    opt.scale = 4096.0;
    pb::PassbandStore s(path_text(dir / "cap"), opt);
    sstvae::qrss::frontend::Capture cap;
    cap.q = 2'000'000;
    cap.t0_index = 15 * 4000 + 0.4;   // rounds to 60000
    cap.fe = noise(100 * 4000, 0.5, 7);
    cap.fe[1234] = cf(10.0f, 0.0f);   // peak 10 * 4096 = 40960: stored at 2^-ceil(log2(40960 / 16383)) = 1/4
    const pb::Written w = s.write_capture(cap, false);
    CHECK(w.gain == 0.25);
    CHECK(w.t_first == static_cast<double>(pb::slot_t0(cap.q) * 4000 - 60000) / 4000.0);
    const pb::SlotCapture got = s.capture(cap.q, 60.0);
    CHECK(got.capture.has_value() && got.coverage == 1.0);
    CHECK(got.capture->t0_index == 60000.0 && got.capture->q == cap.q && got.capture->fs == 4000);
    CHECK(got.capture->fe.size() == 80 * 4000);   // 15 s lead + 60 s + 5 s tail, all written
    const cf want = cap.fe[1234] * 0.25f;
    CHECK(std::abs(got.capture->fe[1234] - want) < 1e-6f);
    // half the frame written: coverage 0.5, the unwritten end trimmed
    pb::PassbandStore h(path_text(dir / "cap2"), opt);
    cap.fe.resize(45 * 4000);   // t0 - 15 s .. t0 + 30 s
    h.write_capture(cap, false);
    const pb::SlotCapture half = h.capture(cap.q, 60.0);
    CHECK(half.coverage == 0.5 && half.capture->fe.size() == 45 * 4000 && half.capture->t0_index == 60000.0);
    // nothing stored an hour later
    const pb::SlotCapture none = s.capture(cap.q + 4, 60.0);
    CHECK(!none.capture && none.coverage == 0.0);
    const auto q = s.slots(60.0);
    CHECK(std::find(q.begin(), q.end(), cap.q) != q.end());
    CHECK(std::is_sorted(q.begin(), q.end()));
    // a capture at another rate is refused
    cap.fs = 8000;
    CHECK(throws<std::invalid_argument>([&] { s.write_capture(cap, false); }));
}

void test_pack(const std::filesystem::path& dir) {
    pb::PassbandStore s(path_text(dir / "pack"));
    const double t_a = HOUR0 * 3600.0 + 3000.0;
    const std::vector<cf> x = noise(60 * 4000, 0.1, 3);
    s.write(t_a, x, false);
    s.write(t_a + 90.0, std::vector<cf>(x.begin(), x.begin() + 4000), false);
    const pb::Masked before = s.read(t_a - 10.0, t_a + 100.0);
    CHECK(!s.pack_hour(HOUR0 + 5));   // no such hour
    CHECK((s.raw_hours() == std::vector<std::int64_t>{HOUR0}) && s.pack_hour(HOUR0));
    CHECK(!exists(s.raw_path(HOUR0)) && exists(s.flac_path(HOUR0)));
    CHECK((s.hours() == std::vector<std::int64_t>{HOUR0}) && s.raw_hours().empty());
    const pb::Masked after = s.read(t_a - 10.0, t_a + 100.0);
    CHECK(same(after.x, before.x) && after.mask == before.mask);
    std::error_code ec;
    rawfile::RawFile f = rawfile::RawFile::open(s.flac_path(HOUR0), rawfile::Mode::Read, ec);
    CHECK(f.is_open() && f.size(ec).value_or(1 << 30) < 0.1 * 3600 * 4000 * 4);
    f.close();
    CHECK(!exists(s.flac_path(HOUR0) + "." + std::to_string(rawfile::process_id()) + ".tmp"));
    // a write into a packed hour unpacks it and keeps what was there
    s.write(t_a + 95.0, std::vector<cf>(x.begin(), x.begin() + 4000), false);
    CHECK(exists(s.raw_path(HOUR0)) && !exists(s.flac_path(HOUR0)));
    const pb::Masked again = s.read(t_a - 10.0, t_a + 100.0);
    CHECK(std::equal(again.x.begin(), again.x.begin() + 100 * 4000, before.x.begin()));
    CHECK(std::all_of(again.mask.begin() + 105 * 4000, again.mask.begin() + 106 * 4000,
                      [](std::uint8_t m) { return m == 1; }));
    CHECK(s.pack_hour(HOUR0));
    CHECK((s.expire((HOUR0 + 50) * 3600.0) == std::vector<std::int64_t>{HOUR0}) && s.hours().empty());
}

void test_packer_thread(const std::filesystem::path& dir) {
    pb::StoreOptions opt;
    opt.pack = true;
    pb::PassbandStore s(path_text(dir / "packer"), opt);
    const std::vector<cf> x(10 * 4000, cf(0.1f, 0.2f));
    s.write(HOUR0 * 3600.0 + 3590.0, x, false);
    s.write((HOUR0 + 1) * 3600.0 + 60.0, x);   // not yet due
    CHECK(!s.packer_started());
    s.write((HOUR0 + 1) * 3600.0 + pb::PB_PACK_AFTER_S + 70.0, x);
    CHECK(s.packer_started());
    CHECK(s.join_packer(60.0) && !s.packer_alive());
    CHECK((s.raw_hours() == std::vector<std::int64_t>{HOUR0 + 1}));
    CHECK((s.hours() == std::vector<std::int64_t>{HOUR0, HOUR0 + 1}));
    const pb::Masked r = s.read(HOUR0 * 3600.0 + 3590.0, HOUR0 * 3600.0 + 3600.0);
    CHECK(std::all_of(r.mask.begin(), r.mask.end(), [](std::uint8_t m) { return m == 1; }));
    // a store destroyed while its packer runs stops it and leaves no temporary behind
    {
        pb::PassbandStore t(path_text(dir / "packer2"), opt);
        t.write(HOUR0 * 3600.0, x, false);
        t.write((HOUR0 + 1) * 3600.0 + pb::PB_PACK_AFTER_S + 1.0, x);
    }
    pb::PassbandStore t(path_text(dir / "packer2"));
    CHECK(!exists(t.flac_path(HOUR0) + "." + std::to_string(rawfile::process_id()) + ".tmp"));
    CHECK(t.hours().size() == 2);
}

void test_threads(const std::filesystem::path& dir) {
    pb::PassbandStore s(path_text(dir / "threads"));
    const std::int64_t n0 = HOUR0 * 14'400'000;
    constexpr int W = 4, CHUNKS = 25, N = 1000;
    std::atomic<bool> done{false};
    std::atomic<int> reads{0}, bad{0};
    std::thread reader([&] {
        do {
            const pb::Masked r = s.read(HOUR0 * 3600.0, HOUR0 * 3600.0 + 30.0);
            reads.fetch_add(1);
            if (r.x.size() != 120'000) bad.fetch_add(1);
        } while (!done.load());
    });
    std::vector<std::thread> ws;
    for (int w = 0; w < W; ++w)
        ws.emplace_back([&, w] {
            for (int c = 0; c < CHUNKS; ++c) {
                const std::int64_t at = n0 + static_cast<std::int64_t>((c * W + w) * N);
                std::vector<cf> x(N, cf(static_cast<float>(w + 1) / 8.0f, static_cast<float>(c) / 64.0f));
                s.write_at(at, x, false);
            }
        });
    for (auto& t : ws) t.join();
    done.store(true);
    reader.join();
    CHECK(reads.load() > 0 && bad.load() == 0);
    CHECK(slurp(s.index_path(HOUR0)) == "[[0, " + std::to_string(W * CHUNKS * N) + "]]");
    const pb::Masked r = s.read(HOUR0 * 3600.0, HOUR0 * 3600.0 + W * CHUNKS * N / 4000.0);
    bool ok = std::all_of(r.mask.begin(), r.mask.end(), [](std::uint8_t m) { return m == 1; });
    for (int c = 0; c < CHUNKS && ok; ++c)
        for (int w = 0; w < W && ok; ++w)
            ok = r.x[static_cast<std::size_t>((c * W + w) * N)] ==
                 cf(static_cast<float>(w + 1) / 8.0f, static_cast<float>(c) / 64.0f);
    CHECK(ok);
}

// Packs racing writes into the hour being packed: no write may be lost,
// whichever way each pack goes (a write the encoder or the check has
// already passed must stop the swap, which is what pack_hour's mtime and
// write-count checks are for). Bursts of writes with pauses between them,
// so some packs land in a pause and swap, and some overlap a burst.
void test_pack_race(const std::filesystem::path& dir) {
    pb::StoreOptions opt;
    opt.fs = 40;   // a 576 kB hour: a pack takes milliseconds
    pb::PassbandStore s(path_text(dir / "race"), opt);
    const std::int64_t per = s.per_hour();
    const cf a(0.5f, 0.25f);
    std::vector<cf> want(static_cast<std::size_t>(per), a);
    s.write_at(HOUR0 * per, want, false);
    std::atomic<bool> stop{false};
    std::atomic<int> packed{0}, refused{0};
    std::thread packer([&] {
        while (!stop.load()) {
            if (s.pack_hour(HOUR0))
                packed.fetch_add(1);
            else
                refused.fetch_add(1);
        }
    });
    std::mt19937 g(7);
    int writes = 0;
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    while (std::chrono::steady_clock::now() < t_end || packed.load() == 0) {
        const int burst = 1 + static_cast<int>(g() % 300);   // (up to ~10 ms: some overlap a whole pack)
        for (int b = 0; b < burst; ++b, ++writes) {
            const std::int64_t k = static_cast<std::int64_t>(g() % static_cast<std::uint32_t>(per));
            // (exact at scale 8192, and never the background value)
            const cf v(static_cast<float>(writes % 1000 + 1) / 1024.0f, -static_cast<float>(writes % 777 + 1) / 1024.0f);
            s.write_at(HOUR0 * per + k, std::vector<cf>{v}, false);
            want[static_cast<std::size_t>(k)] = v;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(g() % 30));
        if (std::chrono::steady_clock::now() > t_end + std::chrono::seconds(20)) break;   // (never: a pause swaps)
    }
    stop.store(true);
    packer.join();
    const pb::Masked r = s.read(static_cast<double>(HOUR0 * 3600), static_cast<double>((HOUR0 + 1) * 3600));
    std::size_t lost = 0;
    for (std::size_t k = 0; k < want.size(); ++k)
        if (r.mask[k] != 1 || r.x[k] != want[k]) ++lost;
    if (lost || !packed.load() || !refused.load())
        std::fprintf(stderr, "pack race: %zu of %d writes lost; %d packs, %d refused\n", lost, writes, packed.load(),
                     refused.load());
    CHECK(lost == 0);
    CHECK(packed.load() > 0 && refused.load() > 0);   // (both outcomes were exercised)
}

// The same, placed deterministically through the before_swap seam: a write
// after the packed file checked out and before the swap stops the swap,
// whether it is this store's (counted) or another process's (seen by the
// raw file's mtime alone). Each check is load-bearing: without the count
// or without the mtime one of the two cases loses its write (mutation-checked).
void test_pack_write_meanwhile(const std::filesystem::path& dir) {
    int mode = 0;   // 0: no write, 1: a write through the store, 2: a write behind its back
    pb::PassbandStore* self = nullptr;
    pb::StoreOptions opt;
    opt.fs = 40;
    opt.before_swap = [&](std::int64_t hour) {
        const std::int64_t at = hour * self->per_hour() + 1000;
        if (mode == 1) {
            // with the mtime put back afterwards, as a file system with a
            // coarse clock (FAT's 2 s) leaves it: only the write count sees it
            const auto p = sstvae::util::fs_path(self->raw_path(hour)).value();
            const auto t = std::filesystem::last_write_time(p);
            self->write_at(at, std::vector<cf>{cf(0.75f, -0.75f)}, false);
            std::filesystem::last_write_time(p, t);
        } else if (mode == 2) {
            // (past the file system's coarse timestamp tick, so the mtime moves)
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            std::error_code ec;
            rawfile::RawFile f = rawfile::RawFile::open(self->raw_path(hour), rawfile::Mode::ReadWrite, ec);
            const std::int16_t v[2] = {4096, -4096};
            CHECK(f.is_open() && f.write_at(static_cast<std::uint64_t>(at - hour * self->per_hour()) * 4, v, 4, ec));
        }
    };
    pb::PassbandStore s(path_text(dir / "meanwhile"), opt);
    self = &s;
    s.write_at(HOUR0 * s.per_hour(), std::vector<cf>(5000, cf(0.5f, 0.25f)), false);
    for (int m : {1, 2}) {
        mode = m;
        CHECK(!s.pack_hour(HOUR0));
        CHECK(exists(s.raw_path(HOUR0)) && !exists(s.flac_path(HOUR0)));
        CHECK(!exists(s.flac_path(HOUR0) + "." + std::to_string(rawfile::process_id()) + ".tmp"));
        const pb::Masked r = s.read(HOUR0 * 3600.0 + 1000 / 40.0, HOUR0 * 3600.0 + 1001 / 40.0);
        CHECK(r.x.size() == 1 && r.mask[0] == 1 && r.x[0] == (m == 1 ? cf(0.75f, -0.75f) : cf(0.5f, -0.5f)));
    }
    mode = 0;
    CHECK(s.pack_hour(HOUR0) && !exists(s.raw_path(HOUR0)) && exists(s.flac_path(HOUR0)));
    const pb::Masked r = s.read(HOUR0 * 3600.0 + 1000 / 40.0, HOUR0 * 3600.0 + 1001 / 40.0);
    CHECK(r.x.size() == 1 && r.x[0] == cf(0.5f, -0.5f));
}

void test_short_raw_hour(const std::filesystem::path& dir) {
    // A raw hour left short (a crash between its creation and its sizing):
    // reads refuse it as numpy's mmap does, a write grows it back to size.
    pb::StoreOptions opt;
    opt.fs = 40;
    pb::PassbandStore s(path_text(dir / "short"), opt);
    std::error_code ec;
    CHECK(rawfile::write_replace(s.raw_path(HOUR0) + ".x", s.raw_path(HOUR0), std::string(100, '\1'), ec));
    CHECK(rawfile::write_replace(s.index_path(HOUR0) + ".x", s.index_path(HOUR0), "[[0, 10]]", ec));
    std::string what;
    try {
        s.read(HOUR0 * 3600.0, HOUR0 * 3600.0 + 1.0);
    } catch (const std::invalid_argument& e) {
        what = e.what();
    }
    CHECK(what == "mmap length is greater than file size");
    s.write(HOUR0 * 3600.0 + 5.0, std::vector<cf>(4, cf(1.0f, 1.0f)), false);
    rawfile::RawFile f = rawfile::RawFile::open(s.raw_path(HOUR0), rawfile::Mode::Read, ec);
    CHECK(f.is_open() && f.size(ec).value_or(0) == static_cast<std::uint64_t>(s.per_hour()) * 4);
    f.close();
    const pb::Masked r = s.read(HOUR0 * 3600.0, HOUR0 * 3600.0 + 6.0);
    CHECK(r.x[0] == cf(257.0f / 8192.0f, 257.0f / 8192.0f) && r.x[30] == cf(0.0f, 0.0f));   // (0x0101 kept)
    CHECK(r.x[200] == cf(1.0f, 1.0f) && r.mask[200] == 1 && r.mask[100] == 0);
    CHECK(s.pack_hour(HOUR0));
}

void test_dirlock(const std::filesystem::path& dir) {
    namespace dl = sstvae::qrss::dirlock;
    const std::string d = path_text(dir / "lock" / "deeper");
    std::optional<dl::DirLock> a = dl::DirLock::try_acquire(d);
    CHECK(a.has_value() && a->is_open());
    CHECK(slurp(d + "/.lock") == std::to_string(rawfile::process_id()) + "\n");
    CHECK(!dl::DirLock::try_acquire(d).has_value());   // a second holder is refused, in-process too
    dl::DirLock moved = std::move(*a);
    CHECK(!a->is_open() && moved.is_open());
    CHECK(!dl::DirLock::try_acquire(d).has_value());
    moved.close();
    moved.close();   // idempotent
    std::optional<dl::DirLock> b = dl::DirLock::try_acquire(d);
    CHECK(b.has_value() && b->path() == d + "/.lock");
    b.reset();
    CHECK(dl::DirLock::try_acquire(d).has_value());
    // a directory that cannot be made: an OSError, not "held"
    const std::string file = path_text(dir / "lock" / "plainfile");
    std::error_code ec;
    CHECK(rawfile::write_replace(file + ".tmp", file, "x", ec));
    CHECK(throws<rawfile::FileError>([&] { (void)dl::DirLock::try_acquire(file); }));
}

}  // namespace

int main() {
    std::random_device rd;
    const auto dir = std::filesystem::temp_directory_path() / ("qrss_passband_test_" + std::to_string(rd()));
    std::filesystem::create_directories(dir);
    try {
        test_free();
        test_store(dir);
        test_capture(dir);
        test_pack(dir);
        test_packer_thread(dir);
        test_threads(dir);
        test_pack_race(dir);
        test_pack_write_meanwhile(dir);
        test_short_raw_hour(dir);
        test_dirlock(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        std::filesystem::remove_all(dir);
        return 1;
    }
    std::filesystem::remove_all(dir);
    return sstvae::check::report("qrss_passband");
}
