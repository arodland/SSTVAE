// native/core/qrss/acquire_stats against sstvae/qrss/acquire.py: the
// array functions (band_bins, spectrogram, cfar_normalise, spectral_shape,
// preamble_p/preamble_scales) through one pinned digest (bitwise), the
// scalar ones (a1_gamma, a1_threshold, a2_threshold, tbd_gumbel,
// _band_cov, _parabola) pinned to the bit, gumbel_p to tolerance (numpy's
// SIMD exp/expm1), and the reference's errors.
// tests/test_qrss_native_acquire_stats.py compares against Python
// directly. The digest:
//
//   out = io.BytesIO()
//   n = 9000
//   z = splitmix(900, 2 * n)                                  # infra_testgen.hpp
//   x = ((unit(z[:n]) - 0.5).astype(np.float32)
//        + 1j * (unit(z[n:]) - 0.5).astype(np.float32)).astype(np.complex64)
//   starts = np.array([-1000.5, 0.0, 0.5, 1.5, 250.0, 1000.0, 3333.3, 7999.0, 8500.0, 9500.0])
//   sel, f = A.band_bins(500, (1400.0, 1600.0))
//   out.write(sel.astype(np.int64).tobytes()); out.write(f.tobytes())
//   P = A.spectrogram(x, 500, starts, sel)
//   out.write(P.tobytes())
//   out.write(A.spectrogram(x, 500, starts[2:4]).tobytes())
//   out.write(A.cfar_normalise(P).tobytes())
//   out.write(A.cfar_normalise(P, None).tobytes())
//   out.write(A.cfar_normalise(P, 7).tobytes())
//   E = A.cfar_normalise(P)
//   rows = np.arange(len(E)) % 3 != 1
//   out.write(A.spectral_shape(E, 7, rows).tobytes())
//   out.write(A.spectral_shape(E).tobytes())
//   nd, nt, nc = 21, 7, 10
//   zs = splitmix(901, 2 * nd * nt * nc)
//   S = ((unit(zs[: nd * nt * nc]) - 0.5) * 3.0
//        + 1j * (unit(zs[nd * nt * nc:]) - 0.5) * 3.0).reshape(nd, nt, nc)
//   S[4, 3, :] += 10.0
//   var = (unit(splitmix(902, nt * nc)) * 2.0 + 0.5).reshape(nt, nc)
//   p, best, st = A.preamble_p(S, var)
//   out.write(p.tobytes()); out.write(best.astype(np.int64).tobytes())
//   for k, lam in st:
//       out.write(np.int64(k).tobytes()); out.write(lam.tobytes())
//   out.write(A.preamble_p(S, var, (1, 3))[0].tobytes())
//   sha256(out.getvalue())

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
#include "qrss/acquire_stats.hpp"
#include "qrss/rational.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
namespace acq = sstvae::qrss::acquire;

static std::span<const int> rxa_scales() { return sstvae::qrss::rxconfig::acquire::A2_SCALES; }

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

static void pair_is(std::pair<double, double> got, double a, double b, const std::string& what) {
    qrss_check::libm_equal(got.first, a, what + " [0]");
    qrss_check::libm_equal(got.second, b, what + " [1]");
}

static void test_digest() {
    std::vector<std::uint8_t> out;
    const std::size_t n = 9000;
    const auto z = qrss_testgen::splitmix(900, 2 * n);
    std::vector<std::complex<float>> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = {static_cast<float>(qrss_testgen::unit(z[i]) - 0.5), static_cast<float>(qrss_testgen::unit(z[n + i]) - 0.5)};
    const std::vector<double> starts = {-1000.5, 0.0, 0.5, 1.5, 250.0, 1000.0, 3333.3, 7999.0, 8500.0, 9500.0};
    const auto bb = acq::band_bins(500, {1400.0, 1600.0});
    append_all(out, bb.sel);
    append_all(out, bb.f_hz);
    const std::span<const std::int64_t> sel(bb.sel);
    const auto P = acq::spectrogram(x, 500, starts, sel);
    append_all(out, P.data);
    append_all(out, acq::spectrogram(x, 500, std::span<const double>(starts).subspan(2, 2)).data);
    append_all(out, acq::cfar_normalise(P).data);
    append_all(out, acq::cfar_normalise(P, std::nullopt).data);
    append_all(out, acq::cfar_normalise(P, 7).data);
    const auto E = acq::cfar_normalise(P);
    std::vector<std::uint8_t> rows(E.rows);
    for (std::size_t i = 0; i < E.rows; ++i) rows[i] = (i % 3 != 1);
    append_all(out, acq::spectral_shape(E, 7, std::span<const std::uint8_t>(rows)));
    append_all(out, acq::spectral_shape(E));

    const std::size_t nd = 21, nt = 7, nc = 10, N = nd * nt * nc;
    const auto zs = qrss_testgen::splitmix(901, 2 * N);
    std::vector<std::complex<double>> S(N);
    for (std::size_t i = 0; i < N; ++i)
        S[i] = {(qrss_testgen::unit(zs[i]) - 0.5) * 3.0, (qrss_testgen::unit(zs[N + i]) - 0.5) * 3.0};
    for (std::size_t c = 0; c < nc; ++c) S[(4 * nt + 3) * nc + c] += 10.0;
    const auto zv = qrss_testgen::splitmix(902, nt * nc);
    std::vector<double> var(nt * nc);
    for (std::size_t i = 0; i < var.size(); ++i) var[i] = qrss_testgen::unit(zv[i]) * 2.0 + 0.5;
    const auto pp = acq::preamble_p(S, nd, nt, nc, var);
    append_all(out, pp.p);
    append_all(out, pp.best);
    for (const auto& s : pp.st) {
        append(out, s.k);
        append_all(out, s.lam);
    }
    const std::vector<int> scales13 = {1, 3};
    append_all(out, acq::preamble_p(S, nd, nt, nc, var, scales13).p);
    check::equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))),
                 std::string("a1149a36127ecbfff9931798b60cd8d011f1ea82addf96e2e0e30c6bb7f9ba29"),
                 "band_bins / spectrogram / cfar_normalise / spectral_shape / preamble_p digest");

    // preamble_scales is preamble_p's st
    const auto st = acq::preamble_scales(S, nd, nt, nc, var);
    bool same = st.size() == pp.st.size();
    for (std::size_t i = 0; same && i < st.size(); ++i) same = st[i].k == pp.st[i].k && st[i].lam == pp.st[i].lam;
    check::is_true(same, "preamble_scales == preamble_p's st");
    // the planted cell's Lambda is ~800 at every scale, so every gammaincc
    // underflows to 0 there and np.argmin takes the first: p 0, best 0
    check::equal(pp.p[4 * nt + 3], 0.0, "the planted cell's p underflows to 0");
    check::equal(pp.best[4 * nt + 3], std::int64_t(0), "a tie of zeros is the first scale");
}

static void test_scalars() {
    // float(A._frame_len(fs)), A.frame_window(500)[1]
    check::equal(acq::frame_len(500), std::int64_t(2000), "frame_len(500)");
    check::equal(acq::frame_len(4000), std::int64_t(16000), "frame_len(4000)");
    const auto w = acq::frame_window(500);
    check::is_true(w.size() == 2000 && w[0] == 0.0 && w[1000] == 1.0, "frame_window(500) is a periodic Hann");
    // [v.hex() for v in A.a1_gamma(M[, 500])]; A.a1_threshold(M).hex()
    pair_is(acq::a1_gamma(1), 0x1p+0, 0x1p+0, "a1_gamma(1)");
    pair_is(acq::a1_gamma(11), 0x1.4f13b13b13b14p+3, 0x1.0cede62433b79p+0, "a1_gamma(11)");
    pair_is(acq::a1_gamma(11, 500), 0x1.4f13b13b13b14p+3, 0x1.0cede62433b79p+0, "a1_gamma(11, 500)");
    pair_is(acq::a1_gamma(100), 0x1.7b259c83087e3p+6, 0x1.0e147ae147ae1p+0, "a1_gamma(100)");
    check::equal(acq::a1_threshold(1), 0x1.26bb1bbb55515p+3, "a1_threshold(1)");
    check::equal(acq::a1_threshold(11), 0x1.c4bce337c580cp+4, "a1_threshold(11)");
    check::equal(acq::a1_threshold(100), 0x1.1d8f14d372d3ap+7, "a1_threshold(100)");
    check::equal(acq::a2_threshold(), 0x1.2ce51da3ef9fap+5, "a2_threshold()");
    check::equal(acq::a2_threshold(1e-7, 2), 0x1.49372c8bfe8d9p+4, "a2_threshold(1e-7, 2)");
    check::equal(acq::a2_threshold(1e-3, 5, 1), 0x1.d969aba15e1dep+3, "a2_threshold(1e-3, 5, 1)");
    // [v.hex() for v in A.tbd_gumbel(n)]
    pair_is(acq::tbd_gumbel(0), 0x1.93f8fcb0ab858p+0, 0x1.8b87eae23ac27p-1, "tbd_gumbel(0)");
    pair_is(acq::tbd_gumbel(10), 0x1.917e89afa1975p+2, 0x1.770608bd4e252p-1, "tbd_gumbel(10)");
    pair_is(acq::tbd_gumbel(31), 0x1.3b0a3d70a3d71p+3, 0x1.676c8b4395810p-1, "tbd_gumbel(31)");
    pair_is(acq::tbd_gumbel(100), 0x1.06d0b73c2fa49p+4, 0x1.4aae541ee393fp-1, "tbd_gumbel(100)");
    pair_is(acq::tbd_gumbel(896), 0x1.7a74bc6a7ef9ep+5, 0x1.33b645a1cac08p-1, "tbd_gumbel(896)");
    pair_is(acq::tbd_gumbel(2000), 0x1.19da405d9d33bp+6, 0x1.24e0dcb77fe3ep-1, "tbd_gumbel(2000)");
    // A.gumbel_p(z, n): np.exp/np.expm1 are numpy's SIMD kernels, so to an ulp or two
    check::close(std::vector<double>{acq::gumbel_p(10.0, 31), acq::gumbel_p(15.0, 100), acq::gumbel_p(50.0, 896)},
                 std::vector<double>{0.5515144727809443, 0.9998878882651835, 0.011259740669304606}, 4e-16,
                 "gumbel_p");
    // [v.hex() for v in A._band_cov(n[, 500])]
    pair_is(acq::band_cov(1), 0x1p+0, 0x1.c71c71c71c719p-6, "band_cov(1)");
    pair_is(acq::band_cov(2), 0x1.71c71c71c71c8p+1, 0x1.8782c00d802d0p-4, "band_cov(2)");
    pair_is(acq::band_cov(5), 0x1.171c71c71c71cp+3, 0x1.601f88bab6822p-2, "band_cov(5)");
    pair_is(acq::band_cov(76), 0x1.258e38e38e38ep+7, 0x1.90ac6c839bc3fp+2, "band_cov(76)");
    pair_is(acq::band_cov(321), 0x1.3795555555555p+9, 0x1.aad5c5cb7f178p+4, "band_cov(321)");
    pair_is(acq::band_cov(1, 500), 0x1p+0, 0x1.c71c71c71c71cp-6, "band_cov(1, 500)");
    pair_is(acq::band_cov(2, 500), 0x1.71c71c71c71c8p+1, 0x1.8782c00d81e40p-4, "band_cov(2, 500)");
    pair_is(acq::band_cov(5, 500), 0x1.171c71c71c71cp+3, 0x1.601f88bab7b56p-2, "band_cov(5, 500)");
    pair_is(acq::band_cov(76, 500), 0x1.258e38e38e38ep+7, 0x1.90ac6c839bf35p+2, "band_cov(76, 500)");
    pair_is(acq::band_cov(321, 500), 0x1.3795555555555p+9, 0x1.aad5c5cb7f270p+4, "band_cov(321, 500)");
    // A._parabola(1.0, 2.0, 1.5); in float32: (*np.float32([1, 2, 1.25])) and
    // (*np.log(np.maximum(np.float32([3, 7, 5]), 1e-9)))
    check::equal(acq::parabola(1.0, 2.0, 1.5), 0x1.5555555555555p-3, "parabola, float64");
    check::equal(acq::parabola(1.0f, 2.0f, 1.25f), 0x1.24924ap-4, "parabola, float32");
    // (numpy's float32 log, 0x1.f2272cp+0 for log(7), is an ulp off the
    // correctly rounded 0x1.f2272ap+0 that std::log(7.0f) gives: its inputs
    // are pinned, not recomputed)
    check::equal(acq::parabola(0x1.193ea8p+0f, 0x1.f2272cp+0f, 0x1.9c0420p+0f), 0x1.b9e184p-3,
                 "parabola, float32 log levels");
    check::equal(acq::parabola(1.0, 0.0, 1.0), 0.0, "parabola, a minimum is 0");
    check::equal(acq::parabola(0.0, 1.0, 1.9), 0.5, "parabola, clipped to 1/2");
}

static void test_edges() {
    // a dead frame normalises to 0; no selected frame gives a flat shape
    acq::Grid32 Z(2, 3);
    check::is_true(acq::cfar_normalise(Z).data == std::vector<float>(6, 0.0f), "cfar_normalise of zeros");
    acq::Grid32 ones(4, 5);
    for (auto& v : ones.data) v = 1.0f;
    const std::vector<std::uint8_t> none(4, 0);
    check::is_true(acq::spectral_shape(ones, 3, std::span<const std::uint8_t>(none)) == std::vector<double>(5, 1.0),
                   "spectral_shape over no frames is 1");
}

static void test_errors() {
    throws<qrss::ZeroDivision>([] { (void)acq::a2_threshold(1e-4, 5, 0); }, "float division by zero",
                               "a2_threshold, no scales");
    throws<std::invalid_argument>([] { (void)acq::tbd_gumbel(-1); }, "math domain error", "tbd_gumbel(-1)");
    const std::vector<double> st0 = {0.0};
    throws<std::out_of_range>([&] { (void)acq::spectrogram(std::span<const std::complex<float>>(), 500, st0); },
                              "index 0 is out of bounds for axis 0 with size 0", "spectrogram of nothing");
    const std::vector<std::complex<float>> x10(10);
    const std::vector<std::int64_t> bad = {2000};
    throws<std::out_of_range>(
        [&] { (void)acq::spectrogram(x10, 500, st0, std::span<const std::int64_t>(bad)); },
        "index 2000 is out of bounds for axis 1 with size 2000", "spectrogram, bin out of range");
    acq::Grid32 E(4, 5);
    const std::vector<std::uint8_t> mask3(3, 1);
    throws<std::out_of_range>(
        [&] { (void)acq::spectral_shape(E, 3, std::span<const std::uint8_t>(mask3)); },
        "boolean index did not match indexed array along axis 0; size of axis is 4 but size of corresponding "
        "boolean axis is 3",
        "spectral_shape, mask length");
    const std::vector<std::complex<double>> S(2 * 3 * 10, {1.0, 0.0});
    const std::vector<double> var(3 * 10, 1.0);
    const std::vector<int> zero = {0};
    throws<qrss::ZeroDivision>([&] { (void)acq::preamble_scales(S, 2, 3, 10, var, zero); },
                               "integer division or modulo by zero", "preamble_scales, scale 0");
}

// The review's probes. The sequential digest is, with S and var built as
// in the digest above:
//   out = b""
//   for k, lam in A.preamble_scales(np.asfortranarray(S), var):
//       out += np.int64(k).tobytes() + lam.tobytes()
//   sha256(out)
// (a Fortran-ordered S, as _preamble_sums' einsum makes it, is summed one
// term at a time by numpy; 43, 0, 31 and 57 cells of the four scales then
// differ from the C-ordered S's pairwise sums).
static void test_review_probes() {
    const std::size_t nd = 21, nt = 7, nc = 10, N = nd * nt * nc;
    const auto zs = qrss_testgen::splitmix(901, 2 * N);
    std::vector<std::complex<double>> S(N);
    for (std::size_t i = 0; i < N; ++i)
        S[i] = {(qrss_testgen::unit(zs[i]) - 0.5) * 3.0, (qrss_testgen::unit(zs[N + i]) - 0.5) * 3.0};
    for (std::size_t c = 0; c < nc; ++c) S[(4 * nt + 3) * nc + c] += 10.0;
    const auto zv = qrss_testgen::splitmix(902, nt * nc);
    std::vector<double> var(nt * nc);
    for (std::size_t i = 0; i < var.size(); ++i) var[i] = qrss_testgen::unit(zv[i]) * 2.0 + 0.5;
    const auto seq = acq::preamble_scales(S, nd, nt, nc, var, rxa_scales(), acq::SumOrder::Sequential);
    std::vector<std::uint8_t> out;
    for (const auto& sc : seq) {
        append(out, sc.k);
        append_all(out, sc.lam);
    }
    check::equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))),
                 std::string("bbaa76400ed67400d18d929cc9e468229e90d8b28579b5cced362cb0cce39cdc"),
                 "preamble_scales, numpy's sequential order (a Fortran-ordered S)");
    const auto pw = acq::preamble_scales(S, nd, nt, nc, var);
    const std::size_t expect_diff[4] = {43, 0, 31, 57};
    for (std::size_t s = 0; s < 4; ++s) {
        std::size_t diff = 0;
        for (std::size_t c = 0; c < pw[s].lam.size(); ++c) diff += pw[s].lam[c] != seq[s].lam[c];
        check::equal(diff, expect_diff[s], "cells where the two orders differ, scale " + std::to_string(s));
    }
    const auto ps = acq::preamble_p(S, nd, nt, nc, var, rxa_scales(), acq::SumOrder::Sequential);
    bool same = true;
    for (std::size_t s = 0; same && s < 4; ++s) same = ps.st[s].lam == seq[s].lam;
    check::is_true(same, "preamble_p passes its order on");

    // An inf sample: real input stays a real product (inf, NaN alternate
    // across bins); complex input takes numpy's complex product (all NaN).
    const std::size_t n = 4000;
    const auto z = qrss_testgen::splitmix(900, n);
    std::vector<double> xr(n);
    for (std::size_t i = 0; i < n; ++i) xr[i] = qrss_testgen::unit(z[i]) - 0.5;
    xr[1500] = INFINITY;
    std::vector<std::complex<double>> xc(xr.begin(), xr.end());
    const std::vector<double> st = {0.0, 1000.0, 2000.0};
    const std::vector<std::int64_t> bins = {0, 1, 2, -1};
    const auto Pr = acq::spectrogram(std::span<const double>(xr), 500, st, std::span<const std::int64_t>(bins));
    const auto Pc = acq::spectrogram(std::span<const std::complex<double>>(xc), 500, st,
                                     std::span<const std::int64_t>(bins));
    check::is_true(std::isinf(Pr(0, 0)) && std::isnan(Pr(0, 1)) && std::isinf(Pr(1, 2)) && std::isnan(Pr(1, 3)),
                   "real input with an inf: inf and NaN bins, as numpy's");
    bool all_nan = true;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 4; ++j) all_nan = all_nan && std::isnan(Pc(i, j));
    check::is_true(all_nan, "complex input with an inf: every bin NaN, as numpy's");
    check::equal(Pr(2, 1), Pc(2, 1), "a finite frame is the same either way");
    check::equal(Pr(2, 1), 87.24764f, "a finite frame's bin 1");

    // int(round(fs)) of a NaN/inf fs is round()'s error, and a NaN fs is
    // not answered from the band_cov cache (NaN is equivalent to every key)
    (void)acq::band_cov(5, 4000.0);
    throws<std::invalid_argument>([] { (void)acq::frame_len(NAN); }, "cannot convert float NaN to integer",
                                  "frame_len(NaN)");
    throws<std::overflow_error>([] { (void)acq::frame_len(INFINITY); }, "cannot convert float infinity to integer",
                                "frame_len(inf)");
    throws<std::invalid_argument>([] { (void)acq::band_cov(5, NAN); }, "cannot convert float NaN to integer",
                                  "band_cov(5, NaN) with a table cached");
    // M * M past int64 is Python's exact product, rounded
    pair_is(acq::a1_gamma(3037000500), 0x1.56fbe7c0f11bap+31, 0x1.0e38e38e24c69p+0, "a1_gamma(3037000500)");
    pair_is(acq::a1_gamma((std::int64_t(1) << 40) + 1), 0x1.e50d7943604f7p+39, 0x1.0e38e38e38d55p+0,
            "a1_gamma(2^40 + 1)");
    // band_cov's cache from several threads at once, with more fs than it
    // holds (8): every answer the serial one, whatever was evicted when
    std::vector<double> fss;
    for (int i = 0; i < 13; ++i) fss.push_back(250.0 + 37.5 * i);
    std::vector<std::pair<double, double>> serial;
    for (const double fs : fss) serial.push_back(acq::band_cov(9, fs));
    std::vector<int> bad(8, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < 8; ++t)
        pool.emplace_back([&, t] {
            for (int rep = 0; rep < 6; ++rep)
                for (std::size_t i = 0; i < fss.size(); ++i) {
                    const std::size_t j = (i * (t + 2) + rep) % fss.size();
                    if (acq::band_cov(9, fss[j]) != serial[j]) ++bad[static_cast<std::size_t>(t)];
                }
        });
    for (auto& th : pool) th.join();
    int nbad = 0;
    for (const int b : bad) nbad += b;
    check::equal(nbad, 0, "band_cov from 8 threads through an evicting cache");

    // a negative local_bins: np.ones(w)'s refusal
    acq::Grid32 E(4, 9);
    for (auto& v : E.data) v = 1.0f;
    throws<std::invalid_argument>([&] { (void)acq::spectral_shape(E, -4); }, "negative dimensions are not allowed",
                                  "spectral_shape, negative local_bins");
}

int main() {
    try {
        test_digest();
        test_scalars();
        test_edges();
        test_errors();
        test_review_probes();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_acquire_stats");
}
