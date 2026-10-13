// native/core/qrss/cwid against sstvae/qrss/cwid.py: the reader on a
// pinned LLR stack (read_ml's call, Z and margin to the bit, read, match),
// the windows of a stub track (unit sums, window_soft and window_llr in
// both keyings, through one digest, bitwise), keying_type, decode_units,
// the alphabet, and the reference's errors.
// tests/test_qrss_native_cwid.py compares against Python directly. The
// pins, with splitmix/unit of infra_testgen.hpp:
//
//   CALL = "K1ABC"
//   k = keying_units(CALL)[cwid.CALL_UNITS].astype(np.float64) * 2 - 1
//   mu = 2.0
//   noise = ((unit(splitmix(1200, 2 * 176)) - 0.5) * 3.0).reshape(2, 176)
//   llr = 2 * mu * (mu * k[None, :] + noise)
//   [v.hex() for v in cwid.read_ml(llr, margin=True)[1:]]; cwid.read(llr)
//   cwid.match(llr / (2 * mu), CALL).hex(), cwid.match(llr / (2 * mu), "VE3XYZ7").hex()
//
// and the stub track, for spec in (frame.SHORT, lframe.L_SHORT): OOK keying
// of CALL (the key-down units and the plain units 0-4, 184-191, which carry
// the carrier in OOK) at 0 Hz, amplitude 0.6, in uniform complex noise of
// variance 1/6 per sample:
//
//   tm = Timing(tau0=37.3, ppm=60.0, gamma=0.0, z=0.0, cov=np.zeros((3, 3)))
//   t0 = 50.5
//   n = int(t0 + T.rx_index(tm, spec.n_pos + 20, spec.n_pos, t_sym=spec.t_sym)) + 64
//   zz = splitmix(1210, 2 * n)
//   z = (unit(zz[:n]) - 0.5) + 1j * (unit(zz[n:]) - 0.5)
//   unitw, key = spec.cw_window // 192, keying_units(CALL)
//   for pw in spec.win_start_pos:
//       for u in range(192):
//           if not (key[u] or u < 5 or u >= 184):
//               continue
//           ta = pw - 0.5 + unitw * u
//           na = t0 + T.rx_index(tm, ta, spec.n_pos, t_sym=spec.t_sym)
//           nb = t0 + T.rx_index(tm, ta + unitw, spec.n_pos, t_sym=spec.t_sym)
//           m = int(math.ceil(na))
//           while m < nb:
//               z[m] += 0.6
//               m += 1
//   psd = (1.0 / 6.0) / CH_FS
//   chan = SimpleNamespace(t0_index=t0, fs=CH_FS, t_s=lambda k: (np.asarray(k, dtype=np.float64) - t0) / CH_FS,
//                          noise_psd_at=lambda t: np.full(np.shape(t), psd))
//   tr = SimpleNamespace(z_d=z, chan=chan, timing=tm)
//   out = io.BytesIO()
//   X0, X1, s2 = cwid._unit_sums(tr, spec)
//   out.write(X0.tobytes()); out.write(X1.tobytes()); out.write(s2.tobytes())
//   for ook in ((False, True) if spec.waveform == 0 else (False,)):
//       out.write(cwid.window_soft(None, CH_FS, tr, spec, ook=ook).tobytes())
//       out.write(cwid.window_llr(None, CH_FS, tr, spec, ook=ook).tobytes())
//   sha256(out.getvalue())

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "infra_testgen.hpp"
#include "qrss/cwid.hpp"
#include "qrss/frame.hpp"
#include "qrss/morse.hpp"
#include "qrss/trackbase.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
namespace cw = sstvae::qrss::cwid;
namespace fr = sstvae::qrss::frame;
namespace qcfg = sstvae::qrss::config;
namespace qtypes = sstvae::qrss::types;
using sstvae::qrss_testgen::splitmix;
using sstvae::qrss_testgen::unit;

static const char* const CALL = "K1ABC";

template <typename T>
static void append_all(std::vector<std::uint8_t>& out, const std::vector<T>& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    out.insert(out.end(), p, p + v.size() * sizeof(T));
}

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("threw another exception: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

static std::vector<double> keying_pm1(const char* call) {
    const auto key = qrss::morse::keying_units(call);
    std::vector<double> k;
    for (int i = cw::CALL_LO; i < cw::CALL_LO + cw::CALL_N; ++i) k.push_back(key[static_cast<std::size_t>(i)] * 2.0 - 1);
    return k;
}

static void test_reader() {
    const double mu = 2.0;
    const auto k = keying_pm1(CALL);
    const auto z = splitmix(1200, 2 * 176);
    std::vector<double> llr(2 * 176), soft(2 * 176);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t i = 0; i < 176; ++i) {
            const double noise = (unit(z[r * 176 + i]) - 0.5) * 3.0;
            llr[r * 176 + i] = 2 * mu * (mu * k[i] + noise);
            soft[r * 176 + i] = llr[r * 176 + i] / (2 * mu);
        }
    const cw::Combined L = cw::combine(llr, 2, 176);
    const cw::ReadMl r = cw::read_ml(L, 8, true);
    check::is_true(r.text && *r.text == CALL, "read_ml reads the call");
    check::equal(r.z, 0x1.30f0c92974ef2p+7, "read_ml Z");
    check::equal(r.margin, 0x1.586aa1f8ecb00p+3, "read_ml margin");
    const cw::ReadMl r2 = cw::read_ml(L);
    check::is_true(r2.text == r.text && r2.z == r.z && r2.margin == 0.0, "read_ml without the margin");
    const auto rd = cw::read(L);
    check::is_true(rd && *rd == CALL, "read accepts it");
    const cw::Combined S = cw::combine(soft, 2, 176);
    check::equal(cw::match(S, CALL), 0x1.30f0c92974ef2p+5, "match, the call");
    check::equal(cw::match(S, "VE3XYZ7"), 0x1.8fba3c9a31777p+4, "match, another call");
    // a 1-D row is itself, one row
    const cw::Combined one = cw::combine(std::span<const double>(llr).subspan(0, 176), 1, 176, true);
    check::is_true(one.nrow == 1 && one.s == std::vector<double>(llr.begin(), llr.begin() + 176), "combine, 1-D");
    // noise alone reads nothing acceptable
    std::vector<double> noise(176);
    for (std::size_t i = 0; i < 176; ++i) noise[i] = (unit(z[i]) - 0.5) * 3.0;
    check::is_true(!cw::read(cw::combine(noise, 1, 176, true)), "read rejects noise");
    // every unit key-up: the best call is the shortest, and its margin is
    // capped by the read against no call at all (its one unit: -5)
    const cw::ReadMl none = cw::read_ml(cw::combine(std::vector<double>(176, -5.0), 1, 176, true), 8, true);
    check::is_true(none.text && *none.text == "E" && none.margin == -5.0, "all key-up: 'E', margin -5");
}

static void test_alphabet_and_decode() {
    const auto& pats = cw::patterns();
    check::equal(pats.size(), std::size_t(37), "37 characters");
    check::equal(pats.front().ch, '/', "sorted: '/' first");
    check::equal(pats.back().ch, 'Z', "sorted: 'Z' last");
    // '/' is -..-. : 1110101011101
    check::is_true(pats.front().units == std::vector<std::int64_t>{1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1},
                   "the pattern of '/'");
    for (const char* call : {"K1ABC", "VE3XYZ7", "G4/W1AW", "M0A", "E", "5"}) {
        const auto key = qrss::morse::keying_units(call);
        const std::vector<std::uint8_t> region(key.begin() + cw::CALL_LO, key.begin() + cw::CALL_LO + cw::CALL_N);
        const auto d = cw::decode_units(region);
        check::is_true(d && *d == call, std::string("decode_units round trip ") + call);
    }
    check::is_true(!cw::decode_units(std::vector<std::uint8_t>{}), "nothing keyed");
    check::is_true(!cw::decode_units(std::vector<std::uint8_t>(6, 1)), "a 6-unit element");
    check::is_true(!cw::decode_units(std::vector<std::uint8_t>{1, 0, 0, 0, 0, 0, 0, 0, 1}), "a 7-unit gap");
    const auto late = cw::decode_units(std::vector<std::uint8_t>{0, 0, 0, 1, 1, 1});
    check::is_true(late && *late == "T", "a late first element");
}

struct Stub {
    std::vector<std::complex<double>> z;
    cw::UnitSource src;
};

static Stub stub(const fr::FrameSpec& spec) {
    Stub s;
    qtypes::Timing tm;
    tm.tau0 = 37.3;
    tm.ppm = 60.0;
    const double t0 = 50.5;
    const auto rx = [&](double tau) {
        return qrss::track::rx_index(tm, tau, spec.n_pos(), qcfg::CH_FS, spec.t_sym());
    };
    const auto n = static_cast<std::size_t>(static_cast<std::int64_t>(t0 + rx(static_cast<double>(spec.n_pos() + 20))) + 64);
    const auto zz = splitmix(1210, 2 * n);
    s.z.resize(n);
    for (std::size_t i = 0; i < n; ++i) s.z[i] = {unit(zz[i]) - 0.5, unit(zz[n + i]) - 0.5};
    const auto key = qrss::morse::keying_units(CALL);
    const std::int64_t unitw = spec.cw_window() / cw::UNITS;
    for (const std::int64_t pw : spec.win_start_pos())
        for (int u = 0; u < cw::UNITS; ++u) {
            if (!(key[static_cast<std::size_t>(u)] || u < 5 || u >= 184)) continue;
            const double ta = (static_cast<double>(pw) - 0.5) + static_cast<double>(unitw * u);
            const double na = t0 + rx(ta);
            const double nb = t0 + rx(ta + static_cast<double>(unitw));
            for (auto m = static_cast<std::int64_t>(std::ceil(na)); static_cast<double>(m) < nb; ++m)
                s.z[static_cast<std::size_t>(m)] += 0.6;
        }
    const double psd = (1.0 / 6.0) / qcfg::CH_FS;
    s.src.z = s.z;
    s.src.t0_index = t0;
    s.src.fs = qcfg::CH_FS;
    s.src.timing = tm;
    s.src.noise_at = [psd](std::span<const double>) { return std::vector<double>{psd}; };
    return s;
}

static std::string window_digest(const cw::UnitSource& src, const fr::FrameSpec& spec) {
    std::vector<std::uint8_t> out;
    const cw::UnitSums us = cw::unit_sums(src, spec);
    append_all(out, us.x0);
    append_all(out, us.x1);
    append_all(out, us.s2);
    const bool ce = spec.waveform() == fr::Waveform::CE;
    for (const bool ook : {false, true}) {
        if (ook && !ce) continue;
        append_all(out, cw::window_soft(src, spec, ook).data);
        append_all(out, cw::window_llr(src, spec, ook).data);
    }
    return util::to_hex(util::sha256(std::span<const std::uint8_t>(out)));
}

static void test_windows() {
    const fr::FrameSpec& sh = fr::get_any("short");
    const fr::FrameSpec& ls = fr::get_any("l-short");
    const Stub a = stub(sh);
    check::equal(a.z.size(), std::size_t(64251), "SHORT stub length");
    check::equal(window_digest(a.src, sh), std::string("71c891383d9e016b4392e8e5cc04b2eb07b443b86b46daa831088a479cbaf4ed"),
                 "SHORT unit sums and windows digest");
    const Stub b = stub(ls);
    check::equal(b.z.size(), std::size_t(64387), "L_SHORT stub length");
    check::equal(window_digest(b.src, ls), std::string("b3996c3cdfe4be7dd60de0e1835519f9c04ef1ee02700232958324c481cfd72f"),
                 "L_SHORT unit sums and windows digest");

    auto combined = [](const cw::Units32& u) {
        const std::vector<double> v(u.data.begin(), u.data.end());
        return cw::combine(v, u.rows, cw::CALL_N);
    };
    // OOK reads the call; its match, and FSK's, pinned
    const auto ook = cw::read(combined(cw::window_llr(a.src, sh, true)));
    check::is_true(ook && *ook == CALL, "SHORT OOK windows read the call");
    check::is_true(!cw::read(combined(cw::window_llr(a.src, sh, false))), "SHORT read as FSK: nothing");
    check::equal(cw::match(combined(cw::window_soft(a.src, sh, true)), CALL), 0x1.2da6a714a48fap+8, "SHORT OOK match");
    check::equal(cw::match(combined(cw::window_soft(a.src, sh, false)), CALL), -0x1.6dfe1632089fep+6,
                 "SHORT FSK match");
    check::equal(cw::match(combined(cw::window_soft(b.src, ls, false)), CALL), -0x1.74955d0c6c4a7p+6,
                 "L_SHORT FSK match");
    check::equal(cw::keying_type(a.src, sh), std::string("ook"), "keying_type, SHORT");
    check::equal(cw::keying_type(a.src, sh, CALL), std::string("ook"), "keying_type, SHORT, with the call");
    check::equal(cw::keying_type(b.src, ls), std::string("fsk"), "keying_type, L");
    // a frame without windows: no rows, nothing read
    cw::UnitSource empty;
    check::equal(cw::window_soft(empty, fr::get_any("tiny")).rows, std::size_t(0), "TINY has no windows");
    check::equal(cw::keying_type(empty, fr::get_any("tiny"), CALL), std::string("fsk"), "keying_type, TINY");

    // the windows from several threads at once: the serial answer
    const std::string want = window_digest(a.src, sh);
    std::vector<int> bad(4, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < 4; ++t)
        pool.emplace_back([&, t] {
            for (int rep = 0; rep < 3; ++rep)
                if (window_digest(a.src, sh) != want) ++bad[static_cast<std::size_t>(t)];
        });
    for (auto& th : pool) th.join();
    check::equal(bad[0] + bad[1] + bad[2] + bad[3], 0, "windows from 4 threads");
}

static void test_errors() {
    const fr::FrameSpec& ls = fr::get_any("l-short");
    throws<std::invalid_argument>([&] { cw::fsk_only(ls, true); }, "l-short: waveform L keys its callsign by FSK only",
                                  "OOK of an L frame");
    const std::vector<double> v175(175, 1.0);
    throws<std::invalid_argument>([&] { (void)cw::match(cw::combine(v175, 1, 175, true), CALL); },
                                  "shapes (176,) and (175,) not aligned: 176 (dim 0) != 175 (dim 0)", "match, 175 units");
    const std::vector<double> v176(176, 1.0);
    throws<std::invalid_argument>([&] { (void)cw::match(cw::combine(v176, 1, 176, true), "k1abc"); },
                                  "callsign 'k1abc' has characters outside [A-Z0-9/]: ['a', 'b', 'c', 'k']", "match, lowercase");
    throws<std::invalid_argument>([&] { (void)cw::read_ml(cw::combine(v176, 1, 176, true), -2); },
                                  "negative dimensions are not allowed", "read_ml, max_chars -2");
    throws<std::out_of_range>([&] { (void)cw::read_ml(cw::combine(v176, 1, 176, true), -1); },
                              "index 1 is out of bounds for axis 0 with size 0", "read_ml, max_chars -1");
    throws<std::out_of_range>([&] { (void)cw::read_ml(cw::combine(v176, 1, 176, true), 0); },
                              "index 1 is out of bounds for axis 0 with size 1", "read_ml, max_chars 0");
}

int main() {
    try {
        test_reader();
        test_alphabet_and_decode();
        test_windows();
        test_errors();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_cwid");
}
