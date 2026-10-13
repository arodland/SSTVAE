// native/core/qrss/types against sstvae/qrss/types.py: pass_uid, and a
// PassResult whose .npz is the same file Python's save writes (pinned by
// its SHA-256 and its meta text), loaded back unchanged; the load's
// refusals. tests/test_qrss_native_types.py checks both directions against
// Python directly. The pins:
//
//   z = np.arange(8, dtype=np.float32) - 3.5
//   qt.pass_uid(7, 1500.25, z), qt.pass_uid(0, 0.0025, z), qt.pass_uid(-3, -1234.5675, z)
//   p = qt.PassResult(uid=qt.pass_uid(7, 1500.25, z), q=7, frame="short", waveform=0, f_hz=1500.25,
//       timing=qt.Timing(tau0=3000.5, ppm=12.0, gamma=-0.25, z=40.0,
//                        cov=np.array([[1e-3, 2e-4], [2e-4, 5e-2]])),
//       report=qt.TrackReport(offset_hz=0.25, drift_hz_per_min=nan, wander_hz_rms=0.02,
//                             doppler_hz=inf, ppm=12.0, snr2500_db=-12.0, z_ref=40.0,
//                             kappa=1.02, slip_free=True, suspect=False),
//       z=z, w=np.full(8, 3.0, np.float32), hdr_llr=np.linspace(-1, 1, 5).astype(np.float32),
//       header=HeaderFields(callsign="K1ABC", grid=None, picture_id=0xD349C671, mode=1,
//                           segment=0, codec_id=0xD1D8),
//       cw=qt.CwIdResult(text="K1ABC", z_match=9.5, keying="fsk", agrees=None),
//       ch=(np.arange(4) + 1j * np.arange(4)[::-1]).astype(np.complex64), ch_fs=250,
//       ch_t0_index=3000.5, f_mix_hz=Fraction(96017, 64), psi=np.ones(3, np.float32),
//       estimator="jöint", em_round=2)
//   p.save(path); sha256(open(path, "rb").read())

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/types.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
namespace types = sstvae::qrss::types;

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    }
    check::fail(what, "did not throw");
}

static std::vector<float> zvec() {
    std::vector<float> z(8);
    for (int i = 0; i < 8; ++i) z[static_cast<std::size_t>(i)] = static_cast<float>(i) - 3.5f;
    return z;
}

static types::PassResult sample() {
    types::PassResult p;
    p.z = zvec();
    p.uid = types::pass_uid(7, 1500.25, p.z);
    p.q = 7;
    p.frame = "short";
    p.waveform = 0;
    p.f_hz = 1500.25;
    p.timing = {3000.5, 12.0, -0.25, 40.0, math::Matrix(2, 2, {1e-3, 2e-4, 2e-4, 5e-2})};
    p.report = {0.25, std::nan(""), 0.02, HUGE_VAL, 12.0, -12.0, 40.0, 1.02, true, false};
    p.w.assign(8, 3.0f);
    p.hdr_llr = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    qrss::header::HeaderFields h;
    h.callsign = "K1ABC";
    h.grid = std::nullopt;
    h.picture_id = 0xD349C671;
    h.mode = 1;
    h.segment = 0;
    h.codec_id = 0xD1D8;
    p.header = h;
    p.cw = types::CwIdResult{std::string("K1ABC"), 9.5, "fsk", std::nullopt};
    for (int i = 0; i < 4; ++i) p.ch.emplace_back(static_cast<float>(i), static_cast<float>(3 - i));
    p.ch_fs = 250;
    p.ch_t0_index = 3000.5;
    p.f_mix_hz = qrss::Rational(96017, 64);
    p.psi.assign(3, 1.0f);
    p.estimator = "j\xc3\xb6int";
    p.em_round = 2;
    return p;
}

static const char* META =
    "{\"uid\": \"7_1500250_c537cb1d\", \"q\": 7, \"frame\": \"short\", \"waveform\": 0, \"f_hz\": 1500.25, "
    "\"timing\": {\"tau0\": 3000.5, \"ppm\": 12.0, \"gamma\": -0.25, \"z\": 40.0}, \"report\": {\"offset_hz\": "
    "0.25, \"drift_hz_per_min\": NaN, \"wander_hz_rms\": 0.02, \"doppler_hz\": Infinity, \"ppm\": 12.0, "
    "\"snr2500_db\": -12.0, \"z_ref\": 40.0, \"kappa\": 1.02, \"slip_free\": true, \"suspect\": false}, "
    "\"header\": {\"callsign\": \"K1ABC\", \"grid\": null, \"picture_id\": 3544827505, \"mode\": 1, \"segment\": "
    "0, \"codec_id\": 53720, \"version\": 2, \"reserved\": 0}, \"cw\": {\"text\": \"K1ABC\", \"z_match\": 9.5, "
    "\"keying\": \"fsk\", \"agrees\": null}, \"ch_fs\": 250, \"ch_t0_index\": 3000.5, \"f_mix_hz\": "
    "\"96017/64\", \"estimator\": \"j\\u00f6int\", \"em_round\": 2}";

static void test_pass_uid() {
    const auto z = zvec();
    check::equal(types::pass_uid(7, 1500.25, z), std::string("7_1500250_c537cb1d"), "pass_uid");
    check::equal(types::pass_uid(0, 0.0025, z), std::string("0_2_c537cb1d"), "pass_uid rounds half to even");
    check::equal(types::pass_uid(-3, -1234.5675, z), std::string("-3_-1234568_c537cb1d"), "negative");
    throws<std::invalid_argument>([&] { types::pass_uid(1, std::nan(""), z); },
                                  "cannot convert float NaN to integer", "NaN f_hz");
    throws<std::overflow_error>([&] { types::pass_uid(1, HUGE_VAL, z); },
                                "cannot convert float infinity to integer", "infinite f_hz");
}

static void test_file() {
    const types::PassResult p = sample();
    check::equal(p.meta_json(), std::string(META), "meta is json.dumps(meta)");
    const auto bytes = p.bytes();
    check::equal(bytes.size(), std::size_t{2544}, "file size");
    check::equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(bytes))),
                 std::string("575b825505c9e9786802c97c48619330bd0ec31842302eecb1d4ecdd71b8f313"),
                 "the file Python's save writes");
    const types::PassResult back = types::PassResult::parse(bytes);
    check::equal(back.bytes() == bytes, true, "parse then write: the same file");
    check::equal(back.uid, p.uid, "uid");
    check::equal(back.header == p.header, true, "header");
    check::equal(back.cw == p.cw, true, "cw");
    check::equal(back.timing == p.timing, true, "timing (cov 2x2)");
    check::is_true(std::isnan(back.report.drift_hz_per_min) && std::isinf(back.report.doppler_hz), "NaN, inf");
    check::equal(back.f_mix_hz == p.f_mix_hz, true, "f_mix_hz");
    check::equal(back.ch == p.ch && back.z == p.z && back.psi == p.psi, true, "arrays");
    check::equal(back.estimator, p.estimator, "estimator (non-ASCII)");

    const auto dir = std::filesystem::temp_directory_path() / "sstvae_qrss_test_types";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "p.npz").string();
    p.save(path);
    check::equal(types::PassResult::load(path).bytes() == bytes, true, "save then load");
    std::filesystem::remove_all(dir);

    // no header, no cw, an empty cov
    types::PassResult q = sample();
    q.header.reset();
    q.cw.reset();
    q.timing.cov = math::Matrix();
    const auto qb = q.bytes();
    const auto qback = types::PassResult::parse(qb);
    check::is_true(!qback.header && !qback.cw && qback.timing.cov.empty(), "None header, None cw, 0x0 cov");
    check::equal(qback.bytes() == qb, true, "round trip without header");
}

static void test_refusals() {
    types::PassResult p = sample();
    auto z = p.to_npz();
    qrss::npz::Npz wrong;
    for (const auto& name : {"meta", "z", "w", "hdr_llr", "ch", "psi", "timing_cov"}) {
        if (std::string(name) == "z")
            wrong.add("z", qrss::npz::Array::of(std::vector<double>(8, 1.0)));
        else
            wrong.add(name, z.at(name));
    }
    throws<std::invalid_argument>([&] { types::PassResult::from_npz(wrong); },
                                  "PassResult member z is <f8, not float32", "z as float64");
    qrss::npz::Npz bad_meta;
    const std::string text = "\xff{}";
    bad_meta.add("meta", qrss::npz::Array::of(std::vector<std::uint8_t>(text.begin(), text.end())));
    for (const auto& name : {"z", "w", "hdr_llr", "ch", "psi", "timing_cov"}) bad_meta.add(name, z.at(name));
    throws<std::invalid_argument>([&] { types::PassResult::from_npz(bad_meta); },
                                  "PassResult meta is not valid UTF-8 (byte 0)", "meta not UTF-8");
}

int main() {
    try {
        test_pass_uid();
        test_file();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_types");
}
