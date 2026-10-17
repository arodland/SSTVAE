// native/core/qrss/acquire_tbd against A3 of sstvae/qrss/acquire.py
// (track-before-detect): the Viterbi, the frame starts and channels, the
// shape test, the path refinement and the L pilot re-anchoring through
// pinned digests (bitwise), A3 over a whole TINY capture (every hit field
// but p_value bitwise; p_value is gumbel_p's, to tolerance), the line fit
// (np.polyfit's LAPACK lstsq here a QR: tolerance), and the reference's
// errors. tests/test_qrss_native_acquire_tbd.py compares against Python
// directly. The pins, with splitmix/unit of infra_testgen.hpp:
//
//   out = io.BytesIO()
//   E = (unit(splitmix(1100, 60 * 81)) * 4.0 - 1.0).reshape(60, 81)
//   p, Z = A.tbd_viterbi(E); out.write(p.tobytes()); out.write(np.float64(Z).tobytes())
//   p, Z = A.tbd_viterbi(E.astype(np.float32), 3, 0.25); (the same)
//   sha256(out.getvalue())                                         # viterbi digest
//
//   out.write(A.tbd_frame_starts(frame.SHORT, 400000, 12345.5).tobytes())
//   out.write(A.tbd_frame_starts(lframe.L_FULL).tobytes())
//   out.write(A.tbd_channels([CarrierCandidate(f, 0.0, 0.0) for f in (1234.5, 320.0, 2690.0)]).tobytes())
//   sha256(...)                                                    # frames/channels digest
//
//   n, nb = 120, 400
//   G = (unit(splitmix(1101, n * nb)) * 3.0 - 1.0).reshape(n, nb).astype(np.float32)
//   G[:, 200] += np.float32(2.0)
//   bins = np.clip(200 + (unit(splitmix(1102, n)) * 7).astype(np.int64) - 3, 0, nb - 1)
//   keep = np.arange(n) % 5 != 2
//   z1, za = A.shape_z(G, bins, keep)                              # pinned .hex()
//   f, t = 1000.0 + 0.25 * np.arange(nb), -10.0 + 2.0 * np.arange(n)
//   p = A._bins_path(G, f, t, bins); out.write(p.t_s, p.f_hz, p.weight)
//   out.write(A.l_pilot_bins(G, f, t, np.array([0.0, 1045.0]), keep).tobytes())
//   tt = A.tbd_frame_starts(frame.SHORT)
//   out.write(A.shape_frames(tt, (frame.SHORT, lframe.L_SHORT)).astype(np.uint8).tobytes())
//   sha256(...)                                                    # grid digest; A._path_line(p) pinned
//
// and A3 over a TINY capture: a quarter-rate tone (2500 Hz audio) from
// sample 48000 in uniform complex noise,
//
//   t0 = 12 * FE_FS + 0.5
//   ts = A.tbd_frame_starts(frame.TINY, None, t0, FE_FS)
//   N = int(t0 + (ts[-1] + A.STFT_FRAME_S) * FE_FS) + 10
//   zz = splitmix(1103, 2 * N)
//   x = ((unit(zz[:N]) - 0.5).astype(np.float32) + 1j * (unit(zz[N:]) - 0.5).astype(np.float32)).astype(np.complex64)
//   k = np.arange(N)
//   tone = np.array([1, 1j, -1, -1j], dtype=np.complex64)[k % 4] * np.float32(0.08)
//   x = (x + np.where(k >= 48000, tone, np.complex64(0))).astype(np.complex64)
//   hits = A.track_before_detect(x, t0, frame.TINY, [CarrierCandidate(2500.3, 0.0, 0.0)], all_hits=True,
//                                shape_specs=(frame.TINY, lframe.L_TINY))
//   for h in hits: path arrays, z, n_frames (int64), centre_hz, shape_z, above_z,
//       waveform (int64, -1 for None), l_path present (int64), its arrays if so, l_shape_z, l_above_z
//   sha256(...)                                                    # A3 digest

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "infra_testgen.hpp"
#include "qrss/acquire_tbd.hpp"
#include "qrss/frame.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
namespace acq = sstvae::qrss::acquire;
namespace fr = sstvae::qrss::frame;
namespace qtypes = sstvae::qrss::types;
namespace qcfg = sstvae::qrss::config;
using sstvae::qrss_testgen::splitmix;
using sstvae::qrss_testgen::unit;

template <typename T>
static void append(std::vector<std::uint8_t>& out, const T& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
static void append_all(std::vector<std::uint8_t>& out, const std::vector<T>& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    out.insert(out.end(), p, p + v.size() * sizeof(T));
}

static std::string hex_of(const std::vector<std::uint8_t>& out) {
    return util::to_hex(util::sha256(std::span<const std::uint8_t>(out)));
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

static void test_viterbi() {
    acq::Grid<double> E(60, 81);
    const auto z = splitmix(1100, 60 * 81);
    for (std::size_t i = 0; i < E.data.size(); ++i) E.data[i] = unit(z[i]) * 4.0 - 1.0;
    acq::Grid<float> E32(60, 81);
    for (std::size_t i = 0; i < E.data.size(); ++i) E32.data[i] = static_cast<float>(E.data[i]);
    std::vector<std::uint8_t> out;
    const acq::TbdPath a = acq::tbd_viterbi(E);
    append_all(out, a.path);
    append(out, a.z);
    const acq::TbdPath b = acq::tbd_viterbi(E32, 3, 0.25);
    append_all(out, b.path);
    append(out, b.z);
    check::equal(hex_of(out), std::string("01b70880b1516fea000f6191ba394136b175b35df717b09eea3623bfbb474588"),
                 "tbd_viterbi digest");
    check::equal(a.z, 0x1.2886a98a2df1cp+4, "tbd_viterbi Z");
    // a channel read in place out of a wider grid is the same as the copy
    acq::Grid<double> wide(60, 100);
    for (std::size_t r = 0; r < 60; ++r)
        for (std::size_t c = 0; c < 81; ++c) wide(r, c + 7) = E(r, c);
    const acq::TbdPath w = acq::tbd_viterbi(wide.data.data(), 60, 100, 7, 81);
    check::is_true(w.path == a.path && w.z == a.z, "tbd_viterbi on a strided window");
}

static void test_frames_and_channels() {
    std::vector<std::uint8_t> out;
    append_all(out, acq::tbd_frame_starts(fr::get_any("short"), 400000, 12345.5));
    append_all(out, acq::tbd_frame_starts(fr::get_any("l-full")));
    const std::vector<double> cands = {1234.5, 320.0, 2690.0};
    append_all(out, acq::tbd_channels(cands));
    check::equal(hex_of(out), std::string("dc9ceb58715f90d408fb1720e9593a55ec10ddaa36e86c787baaeeb14fbf91ad"),
                 "frame starts / channels digest");
    check::equal(acq::tbd_n_frames(fr::get_any("short")), std::int64_t(132), "tbd_n_frames(SHORT)");
    check::equal(acq::tbd_n_frames(fr::get_any("l-full")), std::int64_t(896), "tbd_n_frames(L_FULL)");
}

static void test_grid() {
    const std::size_t n = 120, nb = 400;
    acq::Grid<float> G(n, nb);
    const auto z = splitmix(1101, n * nb);
    for (std::size_t i = 0; i < G.data.size(); ++i) G.data[i] = static_cast<float>(unit(z[i]) * 3.0 - 1.0);
    for (std::size_t r = 0; r < n; ++r) G(r, 200) += 2.0f;
    const auto zb = splitmix(1102, n);
    std::vector<std::int64_t> bins(n);
    for (std::size_t i = 0; i < n; ++i)
        bins[i] = std::clamp<std::int64_t>(200 + static_cast<std::int64_t>(unit(zb[i]) * 7) - 3, 0,
                                           static_cast<std::int64_t>(nb) - 1);
    std::vector<std::uint8_t> keep(n);
    for (std::size_t i = 0; i < n; ++i) keep[i] = i % 5 != 2;
    const std::span<const std::uint8_t> kp(keep);
    const auto [z1, za] = acq::shape_z(G, bins, kp);
    check::equal(z1, -0x1.441e366fc5968p-5, "shape_z z1");
    check::equal(za, 0x1.dc09e456cd9aap+4, "shape_z z_above");
    std::vector<double> f(nb), t(n);
    for (std::size_t i = 0; i < nb; ++i) f[i] = 1000.0 + 0.25 * static_cast<double>(i);
    for (std::size_t i = 0; i < n; ++i) t[i] = -10.0 + 2.0 * static_cast<double>(i);
    std::vector<std::uint8_t> out;
    const qtypes::FreqPath p = acq::bins_path(G, f, t, bins);
    append_all(out, p.t_s);
    append_all(out, p.f_hz);
    append_all(out, p.weight);
    const std::vector<double> line = {0.0, 1045.0};
    append_all(out, acq::l_pilot_bins(G, f, t, line, kp));
    const auto tt = acq::tbd_frame_starts(fr::get_any("short"));
    const std::vector<fr::FrameSpec> specs = {fr::get_any("short"), fr::get_any("l-short")};
    append_all(out, acq::shape_frames(tt, specs));
    check::equal(hex_of(out), std::string("840c78721d494176b941d5d9edf0d528cf36211a2deedf300a1bf2d8ef94b938"),
                 "shape_z / _bins_path / l_pilot_bins / shape_frames digest");
    // np.polyfit is LAPACK's SVD least squares in the reference: tolerance
    const auto c = acq::path_line(p);
    check::close(c, std::vector<double>{-0x1.2968c48807f5ap-11, 0x1.06829830e86d0p+10}, 1e-9, "_path_line");
}

struct Capture {
    std::vector<std::complex<float>> x;
    double t0 = 0.0;
};

static Capture tiny_capture() {
    Capture c;
    const double fs = qcfg::FE_FS;
    c.t0 = 12 * fs + 0.5;
    const auto ts = acq::tbd_frame_starts(fr::get_any("tiny"), std::nullopt, c.t0, fs);
    const auto N = static_cast<std::size_t>(
        static_cast<std::int64_t>(c.t0 + (ts.back() + qrss::rxconfig::acquire::STFT_FRAME_S) * fs) + 10);
    const auto zz = splitmix(1103, 2 * N);
    c.x.resize(N);
    const std::complex<float> q[4] = {{1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}};
    for (std::size_t k = 0; k < N; ++k) {
        std::complex<float> v(static_cast<float>(unit(zz[k]) - 0.5), static_cast<float>(unit(zz[N + k]) - 0.5));
        if (k >= 48000) {
            const std::complex<float> tone(q[k % 4].real() * 0.08f, q[k % 4].imag() * 0.08f);
            v = {v.real() + tone.real(), v.imag() + tone.imag()};
        }
        c.x[k] = v;
    }
    return c;
}

static std::string hits_digest(const std::vector<acq::TbdHit>& hits) {
    std::vector<std::uint8_t> out;
    for (const acq::TbdHit& h : hits) {
        append_all(out, h.path.t_s);
        append_all(out, h.path.f_hz);
        append_all(out, h.path.weight);
        append(out, h.z);
        append(out, h.n_frames);
        append(out, h.centre_hz);
        append(out, h.shape_z);
        append(out, h.above_z);
        append(out, h.waveform ? *h.waveform : std::int64_t(-1));
        append(out, std::int64_t(h.l_path.has_value()));
        if (h.l_path) {
            append_all(out, h.l_path->t_s);
            append_all(out, h.l_path->f_hz);
            append_all(out, h.l_path->weight);
        }
        append(out, h.l_shape_z);
        append(out, h.l_above_z);
    }
    return hex_of(out);
}

static void test_track_before_detect() {
    const Capture c = tiny_capture();
    check::equal(c.x.size(), std::size_t(264010), "TINY capture length");
    acq::TbdOptions opt;
    opt.all_hits = true;
    opt.shape_specs = std::vector<fr::FrameSpec>{fr::get_any("tiny"), fr::get_any("l-tiny")};
    const std::vector<double> cands = {2500.3};
    const auto hits = acq::track_before_detect(c.x, c.t0, fr::get_any("tiny"), cands, qcfg::FE_FS, opt);
    check::equal(hits.size(), std::size_t(38), "A3 hits");
    check::equal(hits_digest(hits), std::string("42323bb9571ce0beede5f12fcc068d664d78721bd876701dc3757182bc44373e"),
                 "A3 hits digest");
    // gumbel_p: numpy's SIMD exp/expm1
    check::close(std::vector<double>{hits[0].p_value, hits[1].p_value, hits[2].p_value},
                 std::vector<double>{0.0, 0.06763632296857537, 0.1861747431597127}, 4e-16, "A3 p-values");
    check::is_true(hits[0].passed() && !hits[1].passed() && !hits[2].passed(), "passed");
    check::is_true(!hits[0].waveform.has_value(), "the tone's shape is ambiguous (neither L nor CE)");
    check::close(std::vector<double>{hits[0].f_hz()}, std::vector<double>{2500.000891639298}, 1e-9, "f_hz (Hz)");
    const qtypes::Detection d = hits[0].to_detection(qcfg::WAVEFORM_L);
    check::close(std::vector<double>{d.f_hz}, std::vector<double>{2499.9992792869957}, 1e-9,
                 "to_detection(L): the pilot-anchored path's f_hz");
    check::is_true(d.method == "tbd" && d.waveform == qcfg::WAVEFORM_L && d.a3_z == hits[0].z, "to_detection fields");

    // the default (hits at p < TBD_PFA only): the tone
    const auto best = acq::track_before_detect(c.x, c.t0, fr::get_any("tiny"), cands);
    check::is_true(best.size() == 1 && best[0].z == hits[0].z && std::isnan(best[0].shape_z),
                   "the default keeps the passing hit, without the shape test");

    // the same capture through four threads at once: the serial answer
    const std::string want = hits_digest(hits);
    std::vector<int> bad(4, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < 4; ++t)
        pool.emplace_back([&, t] {
            if (hits_digest(acq::track_before_detect(c.x, c.t0, fr::get_any("tiny"), cands, qcfg::FE_FS, opt)) != want)
                ++bad[static_cast<std::size_t>(t)];
        });
    for (auto& th : pool) th.join();
    check::equal(bad[0] + bad[1] + bad[2] + bad[3], 0, "A3 from 4 threads");
}

static void test_small_pieces() {
    const qtypes::FreqPath p = acq::path_of(1234.5, -3.0, 300.0);
    check::is_true(p.t_s == std::vector<double>{-10.0, 0.0, 300.0} &&
                       p.f_hz == std::vector<double>{1234.5 + -3.0 * -10.0 / 60.0, 1234.5, 1234.5 + -3.0 * 300.0 / 60.0} &&
                       p.weight == std::vector<double>{1.0, 1.0, 1.0},
                   "_path_of");
    check::is_true(acq::shape_waveform(4.5, -3.0) == qcfg::WAVEFORM_L, "shape_waveform L");
    check::is_true(acq::shape_waveform(1.0, 3.5) == qcfg::WAVEFORM_CE, "shape_waveform CE");
    check::is_true(!acq::shape_waveform(5.0, 95.0) && !acq::shape_waveform(NAN, NAN), "shape_waveform ambiguous");
    check::is_true(acq::shape_verdict(qcfg::WAVEFORM_L, std::nullopt) == qcfg::WAVEFORM_L &&
                       acq::shape_verdict(std::nullopt, qcfg::WAVEFORM_CE) == qcfg::WAVEFORM_CE &&
                       !acq::shape_verdict(qcfg::WAVEFORM_CE, qcfg::WAVEFORM_L),
                   "_shape_verdict");
    const std::vector<fr::FrameSpec> both = {fr::get_any("short"), fr::get_any("l-short")};
    const auto m = acq::frame_map(both);
    check::is_true(m.size() == 2 && m[0].first == qcfg::WAVEFORM_CE && m[0].second == 0 &&
                       m[1].first == qcfg::WAVEFORM_L && m[1].second == 1,
                   "frame_map");
}

static void test_errors() {
    const std::vector<fr::FrameSpec> two_ce = {fr::get_any("short"), fr::get_any("full")};
    throws<std::invalid_argument>([&] { (void)acq::frame_map(two_ce); },
                                  "acquire takes at most one frame per waveform, not 'short' and 'full'",
                                  "frame_map, two CE frames");
    throws<std::invalid_argument>([&] { (void)acq::frame_map(std::vector<fr::FrameSpec>{}); },
                                  "acquire needs at least one frame", "frame_map, none");
    qtypes::FreqPath one;
    one.t_s = {0.0};
    one.f_hz = {1500.0};
    one.weight = {1.0};
    throws<acq::UnboundLocal>([&] { (void)acq::path_line(one); },
                              "cannot access local variable 'c' where it is not associated with a value",
                              "_path_line of one point");
    acq::Grid<double> none(0, 5);
    throws<std::out_of_range>([&] { (void)acq::tbd_viterbi(none); }, "index 0 is out of bounds for axis 0 with size 0",
                              "tbd_viterbi of no frames");
    acq::Grid<double> E(4, 3);
    throws<std::overflow_error>([&] { (void)acq::tbd_viterbi(E, 200); }, "Python integer -200 out of bounds for int8",
                                  "tbd_viterbi, a step past int8");
}

int main() {
    try {
        test_viterbi();
        test_frames_and_channels();
        test_grid();
        test_track_before_detect();
        test_small_pieces();
        test_errors();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_acquire_tbd");
}
