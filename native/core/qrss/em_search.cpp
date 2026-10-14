// The template search and retroactive detection (em.py's second half): see
// em_search.hpp. Every step names the reference line it reproduces.

#include "qrss/em_search.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>   // the AVX2 clone of the search's accumulation (below)
#endif

#include "dsp/fft.hpp"   // pocketfft (numpy's np.fft)
#include "math/npcomplex.hpp"
#include "qrss/ce_rx.hpp"
#include "qrss/chan.hpp"
#include "qrss/config.hpp"
#include "qrss/demod.hpp"
#include "qrss/header.hpp"
#include "qrss/morse.hpp"
#include "qrss/npsum.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/track.hpp"

namespace sstvae::qrss::em {
namespace {

using cplx = std::complex<double>;
namespace np = sstvae::math::np;
namespace pd = pocketfft::detail;

constexpr double PI = 3.141592653589793;   // np.pi
constexpr double TWO_PI = 2 * PI;          // 2 * np.pi, exact
constexpr double INF = std::numeric_limits<double>::infinity();

// np.exp(-2j * np.pi * x) for a real x: the exponent is (0, -2 pi x) (numpy's
// fused complex multiply leaves the real part a zero), and numpy's cexp of a
// pure imaginary is (cos, sin) of it.
inline cplx exp_neg_i2pi(double x) {
    const double a = -TWO_PI * x;
    return {std::cos(a), std::sin(a)};
}

// numpy's float `x % 1.0` (npy_divmod's remainder): fmod, then +1 when
// negative; a zero remainder is +0.
inline double py_mod1(double x) {
    double m = std::fmod(x, 1.0);
    if (m != 0.0 || std::isnan(m)) {
        if (m < 0.0) m += 1.0;
    } else {
        m = 0.0;
    }
    return m;
}

// int(math.floor(x)) / int(math.ceil(x)): Python's refusals of NaN and inf.
std::int64_t py_int_of(double v) {
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    if (!(v >= -9.2233720368547758e18 && v < 9.2233720368547758e18))
        throw std::overflow_error("Python int too large to convert to C int64");
    return static_cast<std::int64_t>(v);
}
std::int64_t py_floor(double x) { return py_int_of(std::floor(x)); }
std::int64_t py_ceil(double x) { return py_int_of(std::ceil(x)); }

// Python's floor modulo of an int by a positive int.
inline std::int64_t floor_mod(std::int64_t a, std::int64_t n) {
    const std::int64_t r = a % n;
    return r < 0 ? r + n : r;
}

// Python's max(a, b): a unless b compares greater.
inline double py_max(double a, double b) { return b > a ? b : a; }

// np.argmax of a float64 vector: the first NaN, else the first maximum.
std::size_t np_argmax(const std::vector<double>& v) {
    if (v.empty()) throw std::invalid_argument("attempt to get argmax of an empty sequence");
    double mp = v[0];
    std::size_t at = 0;
    if (std::isnan(mp)) return 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (!(v[i] <= mp)) {
            mp = v[i];
            at = i;
            if (std::isnan(mp)) break;
        }
    }
    return at;
}

std::string rstrip_spaces(std::string s) {
    s.erase(s.find_last_not_of(' ') + 1);   // npos + 1 is 0: all spaces
    return s;
}

// f"{pid:08x}" of a Python int (a negative one as "-" and 7 digits).
std::string hex08(std::int64_t v) {
    const bool neg = v < 0;
    const std::uint64_t mag = neg ? 0 - static_cast<std::uint64_t>(v) : static_cast<std::uint64_t>(v);
    char buf[32];
    std::snprintf(buf, sizeof buf, "%llx", static_cast<unsigned long long>(mag));
    std::string hex = buf;
    const std::size_t width = neg ? 7 : 8;
    if (hex.size() < width) hex.insert(0, width - hex.size(), '0');
    return (neg ? "-" : "") + hex;
}

// The FFT length _search_chan uses: TS_NFFT doubled until it holds the
// chunk and both lag margins.
std::int64_t fft_length(std::int64_t chunk, std::int64_t lag_max) {
    std::int64_t nfft = TS_NFFT;
    while (nfft < chunk + 2 * lag_max) nfft *= 2;
    return nfft;
}

#if defined(__GNUC__) || defined(__clang__)
#define SSTVAE_EMS_INLINE inline __attribute__((always_inline))
#else
#define SSTVAE_EMS_INLINE inline
#endif

// On x86 without -mfma (the portable build) std::fma is a library call per
// operation; the two loops of the search made of numpy's fused multiplies
// (the spectra's product and np.abs) have a clone compiled for AVX2 + FMA3,
// picked at run time when the CPU has them (math/convolve.cpp's pattern).
// fma, sqrt and division are exactly rounded either way and this file is
// built with -ffp-contract=off, so both clones compute the same bits.
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define SSTVAE_EMS_FMA_CLONES 1
bool cpu_has_avx2_fma() {
    static const bool has = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("fma") != 0 && __builtin_cpu_supports("avx2") != 0;
    }();
    return has;
}
#endif

// p[f] = fr[f] * conj(ft[(f - shift) % n]) for f < n (shift < n): numpy's
// fused complex multiply (np::mul) of FR by the rolled template spectrum.
SSTVAE_EMS_INLINE void rolled_product_impl(const cplx* fr, const cplx* ft, cplx* p, std::size_t n, std::size_t shift) {
    for (std::size_t f = 0; f < shift; ++f) {   // reads ft[f + n - shift]
        const double ar = fr[f].real(), ai = fr[f].imag();
        const cplx b = ft[f + n - shift];
        const double br = b.real(), bi = -b.imag();
        p[f] = cplx(std::fma(ar, br, -(ai * bi)), std::fma(ar, bi, ai * br));
    }
    for (std::size_t f = shift; f < n; ++f) {   // reads ft[f - shift]
        const double ar = fr[f].real(), ai = fr[f].imag();
        const cplx b = ft[f - shift];
        const double br = b.real(), bi = -b.imag();
        p[f] = cplx(std::fma(ar, br, -(ai * bi)), std::fma(ar, bi, ai * br));
    }
}

// lam[j] = (first ? nothing : lam[j] +) np.abs(c[j]) ** 2 * nk for j < w: the
// first row assigned, as numpy's reduction over axis 0 starts from it.
SSTVAE_EMS_INLINE void accumulate_impl(const cplx* c, double nk, double* lam, std::size_t w, bool first) {
    if (first) {
        for (std::size_t j = 0; j < w; ++j) {
            const double a = np::abs(c[j]);
            lam[j] = (a * a) * nk;
        }
    } else {
        for (std::size_t j = 0; j < w; ++j) {
            const double a = np::abs(c[j]);
            lam[j] = lam[j] + (a * a) * nk;
        }
    }
}

#ifdef SSTVAE_EMS_FMA_CLONES
__attribute__((target("avx2,fma"))) void rolled_product_avx2(const cplx* fr, const cplx* ft, cplx* p, std::size_t n,
                                                             std::size_t shift) {
    rolled_product_impl(fr, ft, p, n, shift);
}

// accumulate_impl four lanes at a time: np::abs written out branch-free (the
// same comparisons, selections and exactly rounded operations, lane by lane
// -- numpy's own hypot loop is this SIMD form), the tail by np::abs itself.
__attribute__((target("avx2,fma"))) void accumulate_avx2(const cplx* c, double nk, double* lam, std::size_t w,
                                                         bool first) {
    const __m256d sign = _mm256_set1_pd(-0.0);
    const __m256d inf = _mm256_set1_pd(INF);
    const __m256d nan = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d vk = _mm256_set1_pd(nk);
    std::size_t j = 0;
    for (; j + 4 <= w; j += 4) {
        const __m256d a = _mm256_loadu_pd(reinterpret_cast<const double*>(c + j));       // r0 i0 r1 i1
        const __m256d b = _mm256_loadu_pd(reinterpret_cast<const double*>(c + j + 2));   // r2 i2 r3 i3
        __m256d re = _mm256_permute4x64_pd(_mm256_unpacklo_pd(a, b), 0xD8);              // r0 r1 r2 r3
        __m256d im = _mm256_permute4x64_pd(_mm256_unpackhi_pd(a, b), 0xD8);              // i0 i1 i2 i3
        re = _mm256_andnot_pd(sign, re);
        im = _mm256_andnot_pd(sign, im);
        const __m256d re_inf = _mm256_cmp_pd(re, inf, _CMP_EQ_OQ);
        const __m256d im_inf = _mm256_cmp_pd(im, inf, _CMP_EQ_OQ);
        im = _mm256_blendv_pd(im, inf, re_inf);
        re = _mm256_blendv_pd(re, inf, im_inf);
        const __m256d re_nan = _mm256_cmp_pd(re, re, _CMP_UNORD_Q);
        const __m256d im_nan = _mm256_cmp_pd(im, im, _CMP_UNORD_Q);
        im = _mm256_blendv_pd(im, nan, re_nan);
        re = _mm256_blendv_pd(re, nan, im_nan);
        const __m256d larger = _mm256_blendv_pd(im, re, _mm256_cmp_pd(re, im, _CMP_GT_OQ));    // re > im ? re : im
        const __m256d smaller = _mm256_blendv_pd(re, im, _mm256_cmp_pd(im, re, _CMP_LT_OQ));   // im < re ? im : re
        const __m256d bad =
            _mm256_or_pd(_mm256_cmp_pd(larger, zero, _CMP_EQ_OQ), _mm256_cmp_pd(smaller, inf, _CMP_EQ_OQ));
        const __m256d ratio = _mm256_andnot_pd(bad, _mm256_div_pd(smaller, larger));
        const __m256d mag = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_fmadd_pd(ratio, ratio, one)), larger);
        const __m256d v = _mm256_mul_pd(_mm256_mul_pd(mag, mag), vk);
        _mm256_storeu_pd(lam + j, first ? v : _mm256_add_pd(_mm256_loadu_pd(lam + j), v));
    }
    if (j < w) accumulate_impl(c + j, nk, lam + j, w - j, first);
}
#endif

void rolled_product(const cplx* fr, const cplx* ft, cplx* p, std::size_t n, std::size_t shift) {
#ifdef SSTVAE_EMS_FMA_CLONES
    if (cpu_has_avx2_fma()) {
        rolled_product_avx2(fr, ft, p, n, shift);
        return;
    }
#endif
    rolled_product_impl(fr, ft, p, n, shift);
}

void accumulate(const cplx* c, double nk, double* lam, std::size_t w, bool first) {
#ifdef SSTVAE_EMS_FMA_CLONES
    if (cpu_has_avx2_fma()) {
        accumulate_avx2(c, nk, lam, w, first);
        return;
    }
#endif
    accumulate_impl(c, nk, lam, w, first);
}

// `rows` transforms of length n, row-major, in place, as one pocketfft call:
// it runs several rows at once in SIMD lanes, each lane the arithmetic of a
// lone transform of its row (numpy's np.fft, a row at a time), so the bits
// are the same -- measured equal, and pinned by the parity tests. The
// inverse is scaled by fct. Single-threaded.
void c2c_rows_inplace(cplx* data, std::size_t rows, std::size_t n, bool forward, double fct) {
    if (rows == 0 || n == 0) return;
    const pocketfft::shape_t shape{rows, n};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(n * sizeof(cplx)),
                                     static_cast<std::ptrdiff_t>(sizeof(cplx))};
    pocketfft::c2c(shape, stride, stride, pocketfft::shape_t{1}, forward, data, data, fct, 1);
}

// Rows per batch of the search's inverse transforms (1 MB at nfft 2048).
constexpr std::size_t SEARCH_BATCH = 32;

// One channel's correlation search: the capture's chunk-window FFTs (FR),
// cached by start sample (they depend on nothing else), and the search over
// (m, lag) of one template.
class ChanSearch {
public:
    ChanSearch(std::span<const cplx> z, std::int64_t lag_max, std::int64_t chunk)
        : z_(z), lag_max_(lag_max), chunk_(chunk), nfft_(fft_length(chunk, lag_max)),
          plan_(pd::get_plan<pd::pocketfft_c<double>>(static_cast<std::size_t>(nfft_))) {
        if (lag_max < 0) throw std::invalid_argument("lag_max must be >= 0, not " + std::to_string(lag_max));
    }

    std::int64_t nfft() const { return nfft_; }

    // FR row of a chunk starting at `start`: np.fft.fft(R[k], nfft), R[k][i] =
    // z[start - lag_max + i] for i < chunk + 2 lag_max (0 outside z).
    const std::vector<cplx>& fr(std::int64_t start) {
        auto it = cache_.find(start);
        if (it != cache_.end()) return it->second;
        std::vector<cplx> row(static_cast<std::size_t>(nfft_), cplx(0.0, 0.0));
        const std::int64_t width = chunk_ + 2 * lag_max_;
        const auto nz = static_cast<std::int64_t>(z_.size());
        for (std::int64_t i = 0; i < width; ++i) {
            const std::int64_t idx = start - lag_max_ + i;
            if (idx >= 0 && idx < nz) row[static_cast<std::size_t>(i)] = z_[static_cast<std::size_t>(idx)];
        }
        plan_->exec(reinterpret_cast<pd::cmplx<double>*>(row.data()), 1.0, true);
        return cache_.emplace(start, std::move(row)).first->second;
    }

    // _search_chan's loop over m for template t (FR from the cache).
    ChanPeak search(double N0, const ChunkTemplate& t, std::int64_t m_max, std::span<const std::int64_t> drift_bins) {
        const std::size_t K = t.size();
        if (!drift_bins.empty() && drift_bins.size() != K)
            throw std::invalid_argument("drift_bins holds " + std::to_string(drift_bins.size()) + " values for " +
                                        std::to_string(K) + " chunks");
        if (t.chunk != chunk_) throw std::invalid_argument("chunk_template's chunk does not match the search's");
        const auto N = static_cast<std::size_t>(nfft_);
        const auto W = static_cast<std::size_t>(2 * lag_max_ + 1);
        const auto C = static_cast<std::size_t>(chunk_);
        // FT = np.fft.fft(E, nfft, axis=1); norm = 1.0 / (N0 * en)
        std::vector<cplx> FT(K * N, cplx(0.0, 0.0));
        std::vector<const std::vector<cplx>*> FR(K);
        std::vector<double> norm(K);
        for (std::size_t k = 0; k < K; ++k) {
            std::copy(t.E.begin() + static_cast<std::ptrdiff_t>(k * C), t.E.begin() + static_cast<std::ptrdiff_t>((k + 1) * C),
                      FT.begin() + static_cast<std::ptrdiff_t>(k * N));
            norm[k] = 1.0 / (N0 * t.en[k]);
        }
        c2c_rows_inplace(FT.data(), K, N, true, 1.0);
        for (std::size_t k = 0; k < K; ++k) FR[k] = &fr(t.starts[k]);
        const double inv_n = 1.0 / static_cast<double>(nfft_);   // numpy's ifft fct, 1/n
        ChanPeak best;
        const std::size_t B = std::min(K, SEARCH_BATCH);
        std::vector<cplx> P(B * N);
        std::vector<double> lam(W);
        for (std::int64_t m = -m_max; m <= m_max; ++m) {
            std::fill(lam.begin(), lam.end(), 0.0);
            for (std::size_t k0 = 0; k0 < K; k0 += B) {
                const std::size_t nb = std::min(B, K - k0);
                for (std::size_t i = 0; i < nb; ++i) {
                    const std::size_t k = k0 + i;
                    const std::int64_t mk = m + (drift_bins.empty() ? 0 : drift_bins[k]);
                    // FTm[k, f] = FT[k, (f - mk) % nfft]; P = FR * conj(FTm)
                    const auto shift = static_cast<std::size_t>(floor_mod(mk, nfft_));
                    rolled_product(FR[k]->data(), FT.data() + k * N, P.data() + i * N, N, shift);
                }
                c2c_rows_inplace(P.data(), nb, N, false, inv_n);
                // np.sum(np.abs(C) ** 2 * norm[:, None], axis=0): the rows added in turn
                for (std::size_t i = 0; i < nb; ++i)
                    accumulate(P.data() + i * N, norm[k0 + i], lam.data(), W, k0 + i == 0);
            }
            const std::size_t j = np_argmax(lam);
            if (lam[j] > best.lam) best = ChanPeak{lam[j], static_cast<std::int64_t>(j) - lag_max_, m};
        }
        return best;
    }

private:
    std::span<const cplx> z_;
    std::int64_t lag_max_;
    std::int64_t chunk_;
    std::int64_t nfft_;
    std::shared_ptr<pd::pocketfft_c<double>> plan_;
    std::map<std::int64_t, std::vector<cplx>> cache_;
};

// A hypothesis's channel, derotated and mixed to its frequency, with N0 and
// t0: computed once and kept for every segment (the reference recomputes it
// per segment, identically).
struct CandChan {
    bool skip = false;               // a 250 Hz capture's candidate outside its reach
    std::vector<cplx> z;
    double N0 = 0.0;
    double t0_index = 0.0;
};

// The candidate's z: chan.ch as complex128, the drift's quadratic phase
// removed (when drift is not 0), then mixed down by f0 - chan.mix_audio_hz.
CandChan derotated(const track::ChanCapture& chan, double f0, double drift, double N0) {
    CandChan out;
    const std::size_t n = chan.ch.size();
    out.z.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        out.z[i] = cplx(static_cast<double>(chan.ch[i].real()), static_cast<double>(chan.ch[i].imag()));
    const double fs = static_cast<double>(chan.fs);
    if (drift != 0.0) {   // `if drift:` (NaN is truthy too)
        for (std::size_t i = 0; i < n; ++i) {
            const double t = (static_cast<double>(i) - chan.t0_index) / fs;
            const double turns = (drift / 60.0) * t * t / 2.0;
            out.z[i] = np::mul(out.z[i], exp_neg_i2pi(turns - std::floor(turns)));
        }
    }
    out.N0 = N0;
    const double f_ch = f0 - chan.mix_audio_hz();
    for (std::size_t i = 0; i < n; ++i) {
        const double t = (static_cast<double>(i) - chan.t0_index) / fs;
        out.z[i] = np::mul(out.z[i], exp_neg_i2pi(py_mod1(f_ch * t)));
    }
    out.t0_index = chan.t0_index;
    return out;
}

// max(chan.noise_floor() * CH_FS, 1e-300)
double noise_n0(const track::ChanCapture& chan) {
    return py_max(track::noise_floor(chan) * config::CH_FS, 1e-300);
}

// The accumulator's air-order (S, W) of `segment` as float64 (acc.S[g, idx]).
std::pair<std::vector<double>, std::vector<double>> acc_segment(const store::Accumulator& acc, std::int64_t segment,
                                                                std::int64_t n) {
    const std::span<const std::uint16_t> idx = store::canonical_index(segment, n);
    const std::size_t base = static_cast<std::size_t>(segment) * store::GROUP_LATENTS;
    std::pair<std::vector<double>, std::vector<double>> out;
    out.first.resize(idx.size());
    out.second.resize(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        out.first[i] = static_cast<double>(acc.S[base + idx[i]]);
        out.second[i] = static_cast<double>(acc.W[base + idx[i]]);
    }
    return out;
}

// _search_input / _from_passband over a SearchCap (q and f_mix_hz from opt).
SearchInput input_of(const SearchCap& cap, const frame::FrameSpec& spec, std::optional<std::int64_t> q,
                     const std::optional<Rational>& f_mix_hz) {
    if (const auto* const* pb = std::get_if<const passband::PassbandStore*>(&cap)) {
        const frontend::Capture c = from_passband(**pb, q, spec);
        return search_input(c, f_mix_hz);
    }
    if (const auto* const* p = std::get_if<const types::PassResult*>(&cap)) return search_input(**p);
    if (const auto* const* c = std::get_if<const frontend::Capture*>(&cap)) return search_input(**c, f_mix_hz);
    return search_input(*std::get<const receiver::Prepared*>(cap), f_mix_hz);
}

std::vector<float> to_f32(const std::vector<double>& v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i]);
    return out;
}

const std::vector<double>* opt_ptr(const std::optional<std::vector<double>>& v) { return v ? &*v : nullptr; }

// The search proper over a prepared input (template_search after its
// _search_input).
SearchResult search_prepared(store::Store& store, const store::Accumulator& acc, const SearchInput& in,
                             const frame::FrameSpec& spec, const SearchOptions& opt, const Hooks& hooks) {
    const receiver::Prepared& prep = *in.prep;
    const std::int64_t q = prep.q;
    std::vector<std::int64_t> segs;
    if (opt.segments) {
        segs = *opt.segments;
    } else {
        for (std::int64_t g = 0; g <= acc.mode; ++g)
            if (acc.has_data(g)) segs.push_back(g);
    }
    const std::vector<Hypothesis> cands = candidates(store, acc, opt.freqs);
    if (segs.empty() || cands.empty()) return {};
    const std::optional<header::HeaderFields> h0 = known_header(store, acc, segs[0]);
    std::optional<std::string> call0;
    if (h0) call0 = rstrip_spaces(h0->callsign);
    const std::optional<receiver::CwKnown> cwk = known_keying(store, acc, call0);
    // keying, ook = cwk, or the header's call (FSK) on a frame with windows, or none
    std::optional<ce::Keying> keying;
    bool ook = false;
    if (cwk) {
        keying = ce::window_keying(std::span<const std::uint8_t>(cwk->units), spec);
        ook = cwk->ook;
    } else if (h0 && spec.n_win() != 0) {
        const std::vector<std::uint8_t> units = morse::keying_units(*call0);
        keying = ce::window_keying(std::span<const std::uint8_t>(units), spec);
    }
    const std::int64_t chunk = static_cast<std::int64_t>(std::nearbyint(TS_CHUNK_S * config::CH_FS));
    const std::int64_t lag_max = py_ceil(opt.tau_s * config::CH_FS);
    // this port's refusal (the reference slices the correlation from its end)
    if (lag_max < 0) throw std::invalid_argument("tau_s must be >= 0, not " + frame::py_repr(opt.tau_s));
    const std::int64_t m_max = py_ceil(opt.df_hz * TS_NFFT / config::CH_FS);
    const double dur = spec.keyed_end_pos() * config::T_SYM;
    const double ppm_step = TS_SLIP_SAMPLES / (dur * config::CH_FS) * 1e6;
    const bool ch_rate = prep.fs == config::CH_FS;

    // A 250 Hz capture's channel (channel_for with the capture's own mix): one
    // for every candidate, its noise floor with it.
    std::optional<track::ChanCapture> ch_chan;
    double ch_n0 = 0.0;
    std::vector<std::optional<CandChan>> per_cand(cands.size());

    struct Best {
        double lam = -INF;
        std::int64_t seg = 0;
        double f_hz = 0.0, drift = 0.0, ppm = 0.0, tau0 = 0.0;
        std::int64_t n_chunks = 0;
    } best;
    double n_cells = 0.0;
    const std::int64_t n = spec.n_data();
    for (const std::int64_t g : segs) {
        const auto [S, W] = acc_segment(acc, g, n);
        const types::EmPrior prior = prior_from(S, W, q);
        const std::optional<header::HeaderFields> h = known_header(store, acc, g);
        std::optional<frame::HdrBits> bits;
        if (h && spec.has_header()) bits = frame::HdrBits::of(header::encode(*h));
        const track::Classes cl = track::make_classes(spec, bits ? &*bits : nullptr, &prior);
        for (std::size_t c = 0; c < cands.size(); ++c) {
            check_cancel(hooks);
            const Hypothesis& hyp = cands[c];
            if (!per_cand[c]) {
                if (ch_rate) {
                    if (!ch_chan) {
                        ch_chan = receiver::channel_for(prep, spec, types::Detection{}, {}, in.f_mix);
                        ch_n0 = noise_n0(*ch_chan);
                    }
                    if (std::fabs(hyp.f_hz - ch_chan->mix_audio_hz()) + opt.df_hz > TS_CH_REACH_HZ) {
                        CandChan skip;
                        skip.skip = true;
                        per_cand[c] = std::move(skip);
                    } else {
                        per_cand[c] = derotated(*ch_chan, hyp.f_hz, hyp.drift_hz_per_min, ch_n0);
                    }
                } else {
                    const double lo = -(config::LEAD_IN_MAX_S + config::SPAN * config::T_SYM) - opt.tau_s - 1.0;
                    const double hi = dur * (1 + 3e-4) + opt.tau_s + 1.0;
                    std::optional<std::span<const float>> keep;
                    if (prep.keep) keep = std::span<const float>(*prep.keep);
                    const track::ChanCapture chan =
                        track::make_chan(prep.fe, keep, prep.t0_index, prep.fs, hyp.f_hz, lo, hi);
                    per_cand[c] = derotated(chan, hyp.f_hz, hyp.drift_hz_per_min, noise_n0(chan));
                }
            }
            const CandChan& cc = *per_cand[c];
            if (cc.skip) continue;   // this candidate is not inside the capture
            ChanSearch cs(cc.z, lag_max, chunk);
            for (std::int64_t k = -opt.ppm_steps; k <= opt.ppm_steps; ++k) {
                const double ppm = hyp.ppm + static_cast<double>(k) * ppm_step;
                const ChunkTemplate t = chunk_template(spec, cl.mu, cl.nu, keying ? &*keying : nullptr, ook,
                                                       cc.t0_index, ppm, static_cast<std::int64_t>(cc.z.size()), chunk);
                if (t.size() == 0) continue;
                const ChanPeak pk = cs.search(cc.N0, t, m_max, {});
                n_cells += static_cast<double>((2 * lag_max + 1) * (2 * m_max + 1));
                if (pk.lam > best.lam) {
                    best.lam = pk.lam;
                    best.seg = g;
                    best.f_hz = hyp.f_hz + static_cast<double>(pk.m * config::CH_FS) / TS_NFFT;
                    best.drift = hyp.drift_hz_per_min;
                    best.ppm = ppm;
                    best.tau0 = static_cast<double>(pk.lag);
                    best.n_chunks = static_cast<std::int64_t>(t.size());
                }
            }
        }
    }
    if (!std::isfinite(best.lam)) return {};
    SearchResult out;
    out.z = template_z(best.lam, best.n_chunks, n_cells);
    SearchInfo info;
    info.lam = best.lam;
    info.chunks = best.n_chunks;
    info.cells = n_cells;
    info.segment = best.seg;
    info.f_hz = best.f_hz;
    info.ppm = best.ppm;
    info.tau0 = best.tau0;
    info.drift = best.drift;
    out.info = info;
    if (out.z > opt.z_accept) {
        types::Detection det;
        det.path.t_s = {-config::LEAD_IN_MAX_S - 10.0, 0.0, dur + 10.0};
        det.path.f_hz.resize(3);
        for (std::size_t i = 0; i < 3; ++i) det.path.f_hz[i] = best.f_hz + best.drift * det.path.t_s[i] / 60.0;
        det.path.weight = {1.0, 1.0, 1.0};
        types::Timing tm;
        tm.tau0 = best.tau0;
        tm.ppm = best.ppm;
        tm.gamma = 0.0;
        tm.z = out.z;
        tm.cov = math::Matrix(3, 3, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
        det.f_hz = best.f_hz;
        det.timing = tm;
        det.z_ref = out.z;
        det.method = "template";
        det.lead_in_s = 0.0;
        out.det = std::move(det);
    }
    return out;
}

// receive_template after its _search_input, with the segment decided.
types::PassResult receive_found(store::Store& store, const store::Accumulator& acc, const SearchInput& in,
                                const frame::FrameSpec& spec_in, const types::Detection& det, std::int64_t seg,
                                const std::string& estimator) {
    const receiver::Prepared& prep = *in.prep;
    const auto [S, W] = acc_segment(acc, seg, spec_in.n_data());
    const types::EmPrior prior = prior_from(S, W, prep.q);
    const std::optional<header::HeaderFields> h = known_header(store, acc, seg);
    auto chan = std::make_shared<const track::ChanCapture>(receiver::channel_for(prep, spec_in, det, {}, in.f_mix));
    track::TrackArgs ta;
    ta.timing = det.timing;
    std::optional<receiver::CwKnown> cwk;
    if (h && spec_in.has_header()) {
        const frame::HdrBits hb = frame::HdrBits::of(header::encode(*h));
        ta.classes = track::make_classes(spec_in, &hb, &prior);
        cwk = known_keying(store, acc, rstrip_spaces(h->callsign));
        if (cwk) {
            // the tracker's own window_keying of the (192,) mask: every window alike
            ta.keying = ce::window_keying(std::span<const std::uint8_t>(cwk->units), spec_in);
            ta.ook = cwk->ook;
        }
    } else {
        ta.classes = track::make_classes(spec_in, nullptr, &prior);
    }
    track::TrackResult tr = track::track(chan, spec_in, det, ta);
    // the result keeps cw_known's own mask, as _keying_tuple hands it on
    if (cwk) tr.keying = cwk->units;
    tr.z_ref = det.z_ref;
    // demod.extract(None, tr, spec, prep.q, estimator); prior_from refused a negative q already
    demod::Extraction ex = demod::extract(tr, spec_in, static_cast<std::uint64_t>(prep.q), estimator);
    const std::vector<double> llr64(ex.llr.begin(), ex.llr.end());
    const receiver::HeaderDecode hd =
        receiver::decode_header(llr64, opt_ptr(ex.llr_spread), opt_ptr(ex.llr_block), spec_in);
    std::vector<float> hdr_llr = hd.block ? to_f32(*ex.llr_block) : std::move(ex.llr);
    const frame::FrameSpec spec = hd.plain ? frame::plain(spec_in) : spec_in;
    types::PassResult p;
    p.report = track::report(tr, ex.kappa, ex.suspect);
    p.f_hz = tr.freq(0.0);
    p.uid = types::pass_uid(prep.q, p.f_hz, ex.z);
    p.q = prep.q;
    p.frame = spec.name();
    p.waveform = config::WAVEFORM_CE;
    p.timing = tr.timing;
    p.z = std::move(ex.z);
    p.w = std::move(ex.w);
    p.hdr_llr = std::move(hdr_llr);
    p.header = hd.header;
    p.cw = std::nullopt;
    p.ch = chan->ch;
    p.ch_fs = config::CH_FS;
    p.ch_t0_index = chan->t0_index;
    p.f_mix_hz = chan->f_mix;
    p.psi = receiver::psi_frame(tr);
    p.estimator = estimator;
    return p;
}

}  // namespace

// --- the hypotheses ------------------------------------------------------------------------

std::vector<Hypothesis> candidates(store::Store& store, const store::Accumulator& acc, std::span<const double> extra_hz) {
    std::vector<Hypothesis> out;
    for (const pyjson::Value& m : acc.members) {
        std::optional<types::PassResult> p;
        try {
            p = store.load_pass(m.at("uid").as_string("uid"));
        } catch (const store::FileError&) {   // OSError
            continue;
        } catch (const pyjson::KeyError&) {   // KeyError
            continue;
        }
        if (is_l(*p)) continue;
        out.push_back({p->report.offset_hz, p->report.drift_hz_per_min, p->timing.ppm});
    }
    for (const double f : extra_hz) out.push_back({f, 0.0, 0.0});
    std::vector<Hypothesis> uniq;
    for (const Hypothesis& c : out) {
        bool dup = false;
        for (const Hypothesis& u : uniq) {
            if (std::fabs(c.f_hz - u.f_hz) < 0.25 && std::fabs(c.drift_hz_per_min - u.drift_hz_per_min) < 0.05 &&
                std::fabs(c.ppm - u.ppm) < 1) {
                dup = true;
                break;
            }
        }
        if (!dup) uniq.push_back(c);
    }
    return uniq;
}

// --- the chunked template ------------------------------------------------------------------

ChunkTemplate chunk_template(const frame::FrameSpec& spec, std::span<const double> mu, std::span<const double> nu,
                             const ce::Keying* keying, bool ook, double t0_index, double ppm, std::int64_t n_ch,
                             std::int64_t chunk) {
    if (chunk <= 0) throw std::invalid_argument("chunk must be positive, not " + std::to_string(chunk));
    const double scale = config::T_SYM * config::CH_FS * (1.0 + ppm * 1e-6);
    const std::int64_t lo = py_floor(t0_index + spec.keyed_start_pos() * scale);
    const std::int64_t hi = py_ceil(t0_index + spec.keyed_end_pos() * scale);
    const std::int64_t n0 = std::max<std::int64_t>(lo, 0);
    const std::int64_t n1 = std::min(hi, n_ch);
    if (n1 <= n0) throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    const auto len = static_cast<std::size_t>(n1 - n0);
    std::vector<double> tau(len);
    for (std::size_t i = 0; i < len; ++i) tau[i] = (static_cast<double>(n0 + static_cast<std::int64_t>(i)) - t0_index) / scale;
    const std::vector<cplx> e = ce_rx::mean_template(tau, mu, nu, spec, keying, ook);
    const auto C = static_cast<std::size_t>(chunk);
    const std::size_t n_k = len / C;
    std::vector<double> en_all(n_k);
    std::vector<double> sq(C);
    for (std::size_t k = 0; k < n_k; ++k) {
        for (std::size_t i = 0; i < C; ++i) {
            const double a = np::abs(e[k * C + i]);
            sq[i] = a * a;
        }
        en_all[k] = npsum::sum(std::span<const double>(sq));
    }
    // en.max(initial=0.0): NaN propagates
    double mx = 0.0;
    for (const double v : en_all) {
        if (std::isnan(v)) {
            mx = v;
            break;
        }
        if (v > mx) mx = v;
    }
    const double thr = 1e-6 * py_max(mx, 1e-300);
    ChunkTemplate out;
    out.chunk = chunk;
    for (std::size_t k = 0; k < n_k; ++k) {
        if (!(en_all[k] > thr)) continue;
        out.starts.push_back(n0 + chunk * static_cast<std::int64_t>(k));
        out.E.insert(out.E.end(), e.begin() + static_cast<std::ptrdiff_t>(k * C),
                     e.begin() + static_cast<std::ptrdiff_t>((k + 1) * C));
        out.en.push_back(en_all[k]);
    }
    return out;
}

// --- one channel's search ------------------------------------------------------------------

ChanPeak search_chan(std::span<const std::complex<double>> z, double N0, const ChunkTemplate& t, std::int64_t lag_max,
                     std::int64_t m_max, std::span<const std::int64_t> drift_bins) {
    ChanSearch cs(z, lag_max, t.chunk);
    return cs.search(N0, t, m_max, drift_bins);
}

// --- the search's input ----------------------------------------------------------------------

SearchInput search_input(const types::PassResult& p) {
    if (p.ch.empty()) throw std::invalid_argument("pass " + p.uid + " keeps no capture");
    return {std::make_shared<const receiver::Prepared>(receiver::capture_from_pass(p)), p.f_mix_hz};
}

SearchInput search_input(const frontend::Capture& cap, const std::optional<Rational>& f_mix_hz) {
    auto prep = std::make_shared<const receiver::Prepared>(receiver::prepare(cap));
    if (prep->fs == config::CH_FS) {
        if (!f_mix_hz) throw std::invalid_argument("a 250 Hz capture needs the frequency it was mixed at (f_mix_hz)");
        return {std::move(prep), f_mix_hz};
    }
    return {std::move(prep), std::nullopt};
}

SearchInput search_input(const receiver::Prepared& prep, const std::optional<Rational>& f_mix_hz) {
    // receiver.prepare of a Prepared is the Prepared itself (not owned)
    std::shared_ptr<const receiver::Prepared> p(std::shared_ptr<const receiver::Prepared>(), &prep);
    if (prep.fs == config::CH_FS) {
        if (!f_mix_hz) throw std::invalid_argument("a 250 Hz capture needs the frequency it was mixed at (f_mix_hz)");
        return {std::move(p), f_mix_hz};
    }
    return {std::move(p), std::nullopt};
}

frontend::Capture from_passband(const passband::PassbandStore& pb, std::optional<std::int64_t> q,
                                const frame::FrameSpec& spec) {
    if (!q)
        throw std::invalid_argument(
            "searching a PassbandStore needs the slot: pass q= (or use em.retro_detect to search every stored slot)");
    passband::SlotCapture sc = pb.capture(*q, spec.keyed_end_pos() * config::T_SYM);
    if (!sc.capture) throw std::invalid_argument("the passband store holds nothing for slot q=" + std::to_string(*q));
    return std::move(*sc.capture);
}

// --- the search -------------------------------------------------------------------------------

SearchResult template_search(store::Store& store, const store::Key& key, const SearchCap& cap,
                             const frame::FrameSpec& spec, const SearchOptions& opt, const Hooks& hooks) {
    ce_only(spec, "em.template_search");
    // acc = store.accumulator(key) comes first: a missing accumulator is the
    // KeyError before the capture is read
    const store::Accumulator acc = store.accumulator(key);
    const SearchInput in = input_of(cap, spec, opt.q, opt.f_mix_hz);
    return search_prepared(store, acc, in, spec, opt, hooks);
}

std::optional<types::PassResult> receive_template(store::Store& store, const store::Key& key, const SearchCap& cap,
                                                  const frame::FrameSpec& spec,
                                                  const std::optional<types::Detection>& det,
                                                  const std::string& estimator, std::optional<std::int64_t> segment,
                                                  const SearchOptions& search, const Hooks& hooks) {
    ce_only(spec, "em.receive_template");
    const SearchInput in = input_of(cap, spec, search.q, search.f_mix_hz);
    // the accumulator is read once: the reference reads it in template_search
    // and again here, with nothing written between
    const store::Accumulator acc = store.accumulator(key);
    if (det) return receive_found(store, acc, in, spec, *det, segment.value_or(0), estimator);
    // template_search(store, key, prep, spec, return_stats=True, f_mix_hz=f_mix,
    // **search): the prepared capture and its mix as they are
    SearchOptions o = search;
    o.q.reset();
    o.f_mix_hz = in.f_mix;
    SearchResult r = search_prepared(store, acc, in, spec, o, hooks);
    if (!r.det) return std::nullopt;
    return receive_found(store, acc, in, spec, *r.det, r.info->segment, estimator);
}

// --- retroactive detection -----------------------------------------------------------------------

std::string retro_find_line(std::int64_t q, double z, double f_hz, std::int64_t segment,
                            const std::optional<store::Key>& key, const std::string& rule) {
    std::string line = "slot q=" + std::to_string(q) + ": Z " + frame::py_format(z, "%.1f") + " at " +
                       frame::py_format(f_hz, "%.3f") + " Hz, segment " + std::to_string(segment) + ", ";
    if (key)
        line += "attached to " + key->callsign + " " + hex08(key->picture_id) + " by " +
                (rule.empty() ? std::string("existing membership") : rule);
    else
        line += "stored provisional";
    return line;
}

RetroSteps retro_steps(store::Store& store, const store::Key& key, const passband::PassbandStore* pb,
                       const frame::FrameSpec& spec, const RetroOptions& opt, const Hooks& hooks,
                       associate::SoftHeaderFn soft) {
    RetroSteps s;
    if (pb != nullptr) {
        s.slots = [pb](double dur) { return pb->slots(dur); };
        s.capture = [pb](std::int64_t q, double dur) -> std::pair<SlotInputPtr, double> {
            passband::SlotCapture sc = pb->capture(q, dur);
            if (!sc.capture) return {nullptr, sc.coverage};
            auto in = std::make_shared<SlotInput>();
            in->capture = std::move(sc.capture);
            return {std::move(in), sc.coverage};
        };
    }
    // the slot's capture prepared once, for its search and its receive
    auto prepared = [](const SlotInputPtr& in) -> const std::shared_ptr<const receiver::Prepared>& {
        if (!in) throw std::invalid_argument("retro_detect: no slot capture");
        if (in->prep) return in->prep;
        if (!in->capture) throw std::invalid_argument("retro_detect: a C++ step needs the slot's Capture");
        in->prep = std::make_shared<const receiver::Prepared>(receiver::prepare(*in->capture));
        return in->prep;
    };
    const SearchOptions search = opt.search;
    s.search = [&store, key, spec, search, hooks, prepared](const SlotInputPtr& in) -> SlotSearch {
        const std::shared_ptr<const receiver::Prepared>& prep = prepared(in);
        // template_search(store, key, cap, spec, return_stats=True, **search): cap
        // a Capture, so search.q is not read; f_mix_hz only at 250 Hz
        ce_only(spec, "em.template_search");
        const store::Accumulator acc = store.accumulator(key);
        const SearchInput si = search_input(*prep, search.f_mix_hz);
        const SearchResult r = search_prepared(store, acc, si, spec, search, hooks);
        SlotSearch out;
        out.det = r.det;
        out.z = r.z;
        if (r.info) out.segment = r.info->segment;
        return out;
    };
    const std::string estimator = opt.estimator;
    s.receive = [&store, key, spec, estimator, prepared](const SlotInputPtr& in, const types::Detection& det,
                                                         std::int64_t segment) -> std::optional<types::PassResult> {
        const std::shared_ptr<const receiver::Prepared>& prep = prepared(in);
        // receive_template(store, key, cap, spec, det=det, estimator=estimator, segment=...)
        ce_only(spec, "em.receive_template");
        const SearchInput si = search_input(*prep, std::nullopt);
        const store::Accumulator acc = store.accumulator(key);
        return receive_found(store, acc, si, spec, det, segment, estimator);
    };
    s.associate = [&store, soft = std::move(soft)](const types::PassResult& p) {
        return associate::associate(p, store, soft);
    };
    return s;
}

std::vector<RetroFind> retro_detect_steps(store::Store& store, const store::Key& key, const frame::FrameSpec& spec,
                                          const RetroSteps& steps, const std::optional<std::vector<std::int64_t>>& slots,
                                          const Hooks& hooks) {
    ce_only(spec, "em.retro_detect");
    const store::Accumulator acc = store.accumulator(key);
    const double dur = spec.keyed_end_pos() * config::T_SYM;
    std::vector<std::int64_t> held;
    for (const std::string& u : acc.uids()) held.push_back(store.pass_info(u).q);
    std::vector<std::pair<std::int64_t, double>> stored;
    for (const std::string& u : store.pass_uids()) {
        const store::PassInfo i = store.pass_info(u);
        stored.emplace_back(i.q, i.f_hz + (i.waveform == config::WAVEFORM_L ? receiver::L_EXCLUDE_OFFSET_HZ : 0.0));
    }
    const std::vector<std::int64_t> qs = slots ? *slots : steps.slots(dur);
    const auto total = static_cast<std::int64_t>(qs.size());
    std::vector<RetroFind> out;
    for (std::int64_t n = 0; n < total; ++n) {
        check_cancel(hooks);
        const std::int64_t q = qs[static_cast<std::size_t>(n)];
        const std::string what = "slot q=" + std::to_string(q);
        const auto examined = [&] { report_progress(hooks, n + 1, total, what); };
        if (std::find(held.begin(), held.end(), q) != held.end()) {
            examined();
            continue;
        }
        const auto [cap, cov] = steps.capture(q, dur);
        if (!cap || cov < RETRO_MIN_COVERAGE) {
            examined();
            continue;
        }
        const SlotSearch r = steps.search(cap);
        if (!r.det) {
            log_line(hooks, what + ": nothing (Z " + frame::py_format(r.z, "%.1f") + ")");
            examined();
            continue;
        }
        bool same = false;
        for (const auto& [qq, f] : stored)
            if (qq == q && std::fabs(f - r.det->f_hz) < RETRO_SAME_HZ) {
                same = true;
                break;
            }
        if (same) {
            log_line(hooks, what + ": Z " + frame::py_format(r.z, "%.1f") + " at " +
                                frame::py_format(r.det->f_hz, "%.3f") + " Hz, already stored");
            examined();
            continue;
        }
        if (!r.segment) throw pyjson::KeyError("segment", "'segment'");   // info["segment"]
        std::optional<types::PassResult> p = steps.receive(cap, *r.det, *r.segment);
        if (!p) {
            examined();
            continue;
        }
        associate::Result a = steps.associate(*p);
        stored.emplace_back(q, p->f_hz);
        log_line(hooks, retro_find_line(q, r.z, p->f_hz, *r.segment, a.key, a.rule));
        out.push_back(RetroFind{std::move(*p), std::move(a.key), std::move(a.rule)});
        examined();
    }
    return out;
}

std::vector<RetroFind> retro_detect(store::Store& store, const store::Key& key, const passband::PassbandStore& pb,
                                    const frame::FrameSpec& spec, const RetroOptions& opt, const Hooks& hooks) {
    ce_only(spec, "em.retro_detect");
    const RetroSteps steps = retro_steps(store, key, &pb, spec, opt, hooks);
    return retro_detect_steps(store, key, spec, steps, opt.slots, hooks);
}

}  // namespace sstvae::qrss::em
