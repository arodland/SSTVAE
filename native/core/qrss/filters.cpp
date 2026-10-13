#include "qrss/filters.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace sstvae::qrss::filters {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;   // numpy's pi

// Python's // and % (floor), which C++'s / and % (truncating) are not.
std::int64_t floor_div(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

std::int64_t floor_mod(std::int64_t a, std::int64_t b) { return a - floor_div(a, b) * b; }

// --- Cephes (scipy.special's xsf::cephes::i0 / i0e) ----------------------------

// Chebyshev coefficients for exp(-x) I0(x) on [0, 8].
constexpr double I0_A[30] = {
    -4.41534164647933937950E-18, 3.33079451882223809783E-17,  -2.43127984654795469359E-16,
    1.71539128555513303061E-15,  -1.16853328779934516808E-14, 7.67618549860493561688E-14,
    -4.85644678311192946090E-13, 2.95505266312963983461E-12,  -1.72682629144155570723E-11,
    9.67580903537323691224E-11,  -5.18979560163526290666E-10, 2.65982372468238665035E-9,
    -1.30002500998624804212E-8,  6.04699502254191894932E-8,   -2.67079385394061173391E-7,
    1.11738753912010371815E-6,   -4.41673835845875056359E-6,  1.64484480707288970893E-5,
    -5.75419501008210370398E-5,  1.88502885095841655729E-4,   -5.76375574538582365885E-4,
    1.63947561694133579842E-3,   -4.32430999505057594430E-3,  1.05464603945949983183E-2,
    -2.37374148058994688156E-2,  4.93052842396707084878E-2,   -9.49010970480476444210E-2,
    1.71620901522208775349E-1,   -3.04682672343198398683E-1,  6.76795274409476084995E-1};

// Chebyshev coefficients for exp(-x) sqrt(x) I0(x) on the inverted [8, inf].
constexpr double I0_B[25] = {
    -7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17,
    3.46122286769746109310E-17,  -2.82762398051658348494E-16, -3.42548561967721913462E-16,
    1.77256013305652638360E-15,  3.81168066935262242075E-15,  -9.55484669882830764870E-15,
    -4.15056934728722208663E-14, 1.54008621752140982691E-14,  3.85277838274214270114E-13,
    7.18012445138366623367E-13,  -1.79417853150680611778E-12, -1.32158118404477131188E-11,
    -3.14991652796324136454E-11, 1.18891471078464383424E-11,  4.94060238822496958910E-10,
    3.39623202570838634515E-9,   2.26666899049817806459E-8,   2.04891858946906374183E-7,
    2.89137052083475648297E-6,   6.88975834691682398426E-5,   3.36911647825569408990E-3,
    8.04490411014108831608E-1};

double chbevl(double x, const double* array, int n) {
    const double* p = array;
    double b0 = *p++;
    double b1 = 0.0;
    double b2 = 0.0;
    int i = n - 1;
    do {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + *p++;
    } while (--i);
    return 0.5 * (b0 - b2);
}

// --- numpy pieces ---------------------------------------------------------------

// numpy's pairwise_sum for float64 (8 accumulators, blocks of 128), and
// np.sum's reduction around it: 0.0 + pairwise(a).
double pairwise_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

double np_sum(const std::vector<double>& v) { return 0.0 + pairwise_sum(v.data(), v.size()); }

// np.sinc: y = pi*x, eps where that is zero, sin(y)/y.
double np_sinc(double x) {
    double y = PI * x;
    if (y == 0.0) y = DBL_EPSILON;
    return std::sin(y) / y;
}

// np.linspace(start, stop, num) (endpoint=True, float64).
std::vector<double> np_linspace(double start, double stop, std::int64_t num) {
    std::vector<double> y(static_cast<std::size_t>(num));
    const std::int64_t div = num - 1;
    const double delta = stop - start;
    for (std::int64_t i = 0; i < num; ++i) y[i] = static_cast<double>(i);
    if (div > 0) {
        const double step = delta / static_cast<double>(div);
        if (step == 0.0) {
            for (auto& v : y) v = (v / static_cast<double>(div)) * delta;
        } else {
            for (auto& v : y) v = v * step;
        }
    } else {
        for (auto& v : y) v = v * delta;
    }
    for (auto& v : y) v = v + start;
    if (num > 1) y[num - 1] = stop;
    return y;
}

void check_window_length(std::int64_t M) {
    if (M < 0) throw std::invalid_argument("Window length M must be a non-negative integer");
}

// --- arithmetic in the output type, as scipy's Cython does it ------------------

template <typename T>
struct Arith {
    static T mul(T a, T b) { return a * b; }
    static T add(T a, T b) { return a + b; }
    static bool is_zero(T a) { return a == T(0); }
};

// C99 complex multiply (what Cython's `double complex` compiles to for
// finite values): (ac - bd) + i(ad + bc), each product rounded.
template <typename R>
struct Arith<std::complex<R>> {
    using C = std::complex<R>;
    static C mul(C a, C b) {
        const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
        const R re = ar * br - ai * bi;
        const R im = ar * bi + ai * br;
        return C(re, im);
    }
    static C add(C a, C b) { return C(a.real() + b.real(), a.imag() + b.imag()); }
    static bool is_zero(C a) { return a.real() == R(0) && a.imag() == R(0); }
};

// _upfirdn_apply.pyx `_apply_impl`, mode constant.
template <typename T>
void apply_impl(const T* x, std::int64_t len_x, const T* h_trans_flip, std::int64_t len_h, T* out,
                std::int64_t up, std::int64_t down, T cval, std::int64_t len_out) {
    using A = Arith<T>;
    const std::int64_t h_per_phase = len_h / up;
    const std::int64_t padded_len = len_x + h_per_phase - 1;
    std::int64_t x_idx = 0, y_idx = 0, t = 0;
    const bool zpad = A::is_zero(cval);
    if (len_out == 0) return;

    while (x_idx < len_x) {
        std::int64_t h_idx = t * h_per_phase;
        std::int64_t x_conv_idx = x_idx - h_per_phase + 1;
        T acc = out[y_idx];
        if (x_conv_idx < 0) {
            if (zpad) {
                h_idx -= x_conv_idx;
            } else {
                for (std::int64_t k = x_conv_idx; k < 0; ++k) {
                    acc = A::add(acc, A::mul(cval, h_trans_flip[h_idx]));
                    ++h_idx;
                }
            }
            x_conv_idx = 0;
        }
        for (std::int64_t k = x_conv_idx; k < x_idx + 1; ++k) {
            acc = A::add(acc, A::mul(x[k], h_trans_flip[h_idx]));
            ++h_idx;
        }
        out[y_idx] = acc;
        ++y_idx;
        if (y_idx >= len_out) return;
        t += down;
        x_idx += t / up;
        t = t % up;
    }

    // The flush loop: past the right end (and, for a signal shorter than a
    // phase, the left), x reads as cval, and cval*h is added even when zero.
    while (x_idx < padded_len) {
        std::int64_t h_idx = t * h_per_phase;
        const std::int64_t x_conv_idx = x_idx - h_per_phase + 1;
        T acc = out[y_idx];
        for (std::int64_t k = x_conv_idx; k < x_idx + 1; ++k) {
            const T xval = (k >= len_x || k < 0) ? cval : x[k];
            acc = A::add(acc, A::mul(xval, h_trans_flip[h_idx]));
            ++h_idx;
        }
        out[y_idx] = acc;
        ++y_idx;
        if (y_idx >= len_out) return;
        t += down;
        x_idx += t / up;
        t = t % up;
    }
}

template <typename T>
T scale_by_up(double h, std::int64_t up);

// scipy: h = asarray(firwin(...), dtype=x.dtype); h *= up.
template <>
double scale_by_up<double>(double h, std::int64_t up) {
    return h * static_cast<double>(up);
}
template <>
float scale_by_up<float>(double h, std::int64_t up) {
    return static_cast<float>(h) * static_cast<float>(up);
}
// complex: (h + 0j) * (up + 0j) = (h*up - 0*0) + (h*0 + 0*up)j = (h*up, +0).
template <>
std::complex<double> scale_by_up<std::complex<double>>(double h, std::int64_t up) {
    return Arith<std::complex<double>>::mul(std::complex<double>(h, 0.0),
                                            std::complex<double>(static_cast<double>(up), 0.0));
}
template <>
std::complex<float> scale_by_up<std::complex<float>>(double h, std::int64_t up) {
    return Arith<std::complex<float>>::mul(std::complex<float>(static_cast<float>(h), 0.0f),
                                           std::complex<float>(static_cast<float>(up), 0.0f));
}

std::string shape1(std::size_t n) { return "(" + std::to_string(n) + ",)"; }

}  // namespace

// --- special functions ------------------------------------------------------------

double i0(double x) {
    if (x < 0) x = -x;
    if (x <= 8.0) {
        const double y = (x / 2.0) - 2.0;
        return std::exp(x) * chbevl(y, I0_A, 30);
    }
    return std::exp(x) * chbevl(32.0 / x - 2.0, I0_B, 25) / std::sqrt(x);
}

double i0e(double x) {
    if (x < 0) x = -x;
    if (x <= 8.0) {
        const double y = (x / 2.0) - 2.0;
        return chbevl(y, I0_A, 30);
    }
    return chbevl(32.0 / x - 2.0, I0_B, 25) / std::sqrt(x);
}

// --- windows ------------------------------------------------------------------------

std::vector<double> general_hamming(std::int64_t M, double alpha, bool sym) {
    check_window_length(M);
    if (M <= 1) return std::vector<double>(static_cast<std::size_t>(M), 1.0);
    const bool trunc = !sym;
    const std::int64_t Mx = sym ? M : M + 1;
    const double a[2] = {alpha, 1. - alpha};
    const std::vector<double> fac = np_linspace(-PI, PI, Mx);
    std::vector<double> w(static_cast<std::size_t>(Mx), 0.0);
    for (int k = 0; k < 2; ++k)
        for (std::size_t i = 0; i < w.size(); ++i)
            w[i] = w[i] + a[k] * std::cos(static_cast<double>(k) * fac[i]);
    if (trunc) w.pop_back();
    return w;
}

std::vector<double> hann(std::int64_t M, bool sym) { return general_hamming(M, 0.5, sym); }

std::vector<double> hamming(std::int64_t M, bool sym) { return general_hamming(M, 0.54, sym); }

std::vector<double> kaiser(std::int64_t M, double beta, bool sym) {
    check_window_length(M);
    if (M <= 1) return std::vector<double>(static_cast<std::size_t>(M), 1.0);
    const bool trunc = !sym;
    const std::int64_t Mx = sym ? M : M + 1;
    const double alpha = static_cast<double>(Mx - 1) / 2.0;
    const double denom = i0(beta);
    std::vector<double> w(static_cast<std::size_t>(Mx));
    for (std::int64_t i = 0; i < Mx; ++i) {
        const double q = (static_cast<double>(i) - alpha) / alpha;
        w[i] = i0(beta * std::sqrt(1.0 - q * q)) / denom;
    }
    if (trunc) w.pop_back();
    return w;
}

std::vector<double> get_window(const Window& window, std::int64_t Nx, bool fftbins) {
    if (!(Nx > 0))
        throw std::invalid_argument("Parameter Nx=" + std::to_string(Nx) + " is not a positive integer");
    const bool sym = window.sym.value_or(!fftbins);
    switch (window.kind) {
        case Window::Kind::Hamming: return hamming(Nx, sym);
        case Window::Kind::Hann: return hann(Nx, sym);
        case Window::Kind::Kaiser: return kaiser(Nx, window.beta, sym);
    }
    throw std::invalid_argument("unknown window");
}

// --- firwin --------------------------------------------------------------------------

double kaiser_atten(std::int64_t numtaps, double width) {
    return 2.285 * static_cast<double>(numtaps - 1) * PI * width + 7.95;
}

double kaiser_beta(double a) {
    if (a > 50) return 0.1102 * (a - 8.7);
    if (a > 21) return 0.5842 * std::pow(a - 21, 0.4) + 0.07886 * (a - 21);
    return 0.0;
}

std::vector<double> firwin(std::int64_t numtaps, std::span<const double> cutoff_in, const Window& window_in,
                           PassZero pass_zero, bool scale, std::optional<double> fs_in,
                           std::optional<double> width) {
    const double fs = fs_in ? *fs_in : 2.0;
    const double nyq = 0.5 * fs;
    std::vector<double> cutoff(cutoff_in.begin(), cutoff_in.end());
    for (auto& c : cutoff) c = c / nyq;

    if (cutoff.empty()) throw std::invalid_argument("At least one cutoff frequency must be given.");
    // numpy's min/max propagate NaN, and NaN compares false: scipy lets a
    // NaN cutoff through, and so does this.
    double mn = cutoff[0], mx = cutoff[0];
    for (double c : cutoff) {
        if (std::isnan(c) || std::isnan(mn)) mn = std::numeric_limits<double>::quiet_NaN();
        else if (c < mn) mn = c;
        if (std::isnan(c) || std::isnan(mx)) mx = std::numeric_limits<double>::quiet_NaN();
        else if (c > mx) mx = c;
    }
    if (mn <= 0 || mx >= 1)
        throw std::invalid_argument(
            "Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2.");
    for (std::size_t i = 1; i < cutoff.size(); ++i)
        if (cutoff[i] - cutoff[i - 1] <= 0)
            throw std::invalid_argument(
                "Invalid cutoff frequencies: the frequencies must be strictly increasing.");

    Window window = window_in;
    if (width) {
        const double atten = kaiser_atten(numtaps, *width / nyq);
        window = Window::kaiser(kaiser_beta(atten));
    }

    const std::size_t n = cutoff.size();
    bool pz = true;
    switch (pass_zero) {
        case PassZero::Lowpass:
            if (n != 1)
                throw std::invalid_argument("cutoff must have one element if pass_zero==\"lowpass\", got " +
                                            shape1(n));
            pz = true;
            break;
        case PassZero::Bandstop:
            if (n <= 1)
                throw std::invalid_argument(
                    "cutoff must have at least two elements if pass_zero==\"bandstop\", got " + shape1(n));
            pz = true;
            break;
        case PassZero::Highpass:
            if (n != 1)
                throw std::invalid_argument("cutoff must have one element if pass_zero==\"highpass\", got " +
                                            shape1(n));
            pz = false;
            break;
        case PassZero::Bandpass:
            if (n <= 1)
                throw std::invalid_argument(
                    "cutoff must have at least two elements if pass_zero==\"bandpass\", got " + shape1(n));
            pz = false;
            break;
        case PassZero::Yes: pz = true; break;
        case PassZero::No: pz = false; break;
    }

    const bool pass_nyquist = ((n % 2 == 0) == pz);
    if (pass_nyquist && numtaps % 2 == 0)
        throw std::invalid_argument(
            "A filter with an even number of coefficients must have zero response at the Nyquist "
            "frequency.");

    std::vector<double> edges;
    if (pz) edges.push_back(0.0);
    edges.insert(edges.end(), cutoff.begin(), cutoff.end());
    if (pass_nyquist) edges.push_back(1.0);

    const std::size_t nt = numtaps > 0 ? static_cast<std::size_t>(numtaps) : 0;
    const double alpha = 0.5 * static_cast<double>(numtaps - 1);
    std::vector<double> m(nt);
    for (std::size_t i = 0; i < nt; ++i) m[i] = static_cast<double>(i) - alpha;

    std::vector<double> h(nt, 0.0);
    for (std::size_t j = 0; j + 1 < edges.size(); j += 2) {
        const double left = edges[j], right = edges[j + 1];
        for (std::size_t i = 0; i < nt; ++i) h[i] = h[i] + right * np_sinc(right * m[i]);
        for (std::size_t i = 0; i < nt; ++i) h[i] = h[i] - left * np_sinc(left * m[i]);
    }

    const std::vector<double> win = get_window(window, numtaps, false);
    for (std::size_t i = 0; i < nt; ++i) h[i] = h[i] * win[i];

    if (scale) {
        const double left = edges[0], right = edges[1];
        double scale_frequency;
        if (left == 0) scale_frequency = 0.0;
        else if (right == 1) scale_frequency = 1.0;
        else scale_frequency = 0.5 * (left + right);
        std::vector<double> hc(nt);
        for (std::size_t i = 0; i < nt; ++i) hc[i] = h[i] * std::cos(PI * m[i] * scale_frequency);
        const double s = np_sum(hc);
        for (auto& v : h) v = v / s;
    }
    return h;
}

// --- upfirdn ---------------------------------------------------------------------------

std::int64_t upfirdn_output_len(std::int64_t len_h, std::int64_t in_len, std::int64_t up, std::int64_t down) {
    // Cython's int64 arithmetic: the products and sums wrap, the floor
    // division refuses what C cannot do (a zero divisor, INT64_MIN // -1).
    // Signed overflow is undefined in C++, so the wrapping is done unsigned.
    using U = std::uint64_t;
    const auto num = static_cast<std::int64_t>((static_cast<U>(in_len) - 1) * static_cast<U>(up) +
                                               static_cast<U>(len_h) - 1);
    if (down == 0) throw std::domain_error("integer division or modulo by zero");
    if (down == -1 && num == std::numeric_limits<std::int64_t>::min())
        throw std::overflow_error("value too large to perform division");
    return static_cast<std::int64_t>(static_cast<U>(floor_div(num, down)) + 1);
}

template <typename T>
std::vector<T> upfirdn(std::span<const T> h, std::span<const T> x, std::int64_t up, std::int64_t down,
                       T cval) {
    if (h.empty()) throw std::invalid_argument("h must be 1-D with non-zero length");
    if (up < 1 || down < 1) throw std::invalid_argument("Both up and down must be >= 1");
    const auto len_h0 = static_cast<std::int64_t>(h.size());
    const auto len_x = static_cast<std::int64_t>(x.size());
    const std::int64_t len_out = upfirdn_output_len(len_h0, len_x, up, down);
    if (len_out < 0) throw std::invalid_argument("negative dimensions are not allowed");

    // _pad_h: zero-pad to a multiple of up, then row t holds phase t's taps
    // h[t + up*k], reversed.
    const std::int64_t h_padlen = len_h0 + floor_mod(-len_h0, up);
    const std::int64_t hpp = h_padlen / up;
    std::vector<T> htf(static_cast<std::size_t>(h_padlen), T{});
    for (std::int64_t t = 0; t < up; ++t)
        for (std::int64_t j = 0; j < hpp; ++j) {
            const std::int64_t src = t + up * (hpp - 1 - j);
            if (src < len_h0) htf[t * hpp + j] = h[src];
        }

    std::vector<T> out(static_cast<std::size_t>(len_out), T{});
    apply_impl(x.data(), len_x, htf.data(), h_padlen, out.data(), up, down, cval, len_out);
    return out;
}

std::vector<std::complex<double>> upfirdn(std::span<const double> h, std::span<const std::complex<double>> x,
                                          std::int64_t up, std::int64_t down) {
    std::vector<std::complex<double>> hc(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) hc[i] = std::complex<double>(h[i], 0.0);
    return upfirdn<std::complex<double>>(hc, x, up, down, std::complex<double>{});
}

// --- resample_poly ----------------------------------------------------------------------

template <typename T>
std::vector<T> resample_poly_taps(std::span<const T> x, std::int64_t up, std::int64_t down,
                                  std::span<const T> h, std::int64_t half_len, T cval) {
    const auto n_in = static_cast<std::int64_t>(x.size());
    std::int64_t n_out = n_in * up;
    n_out = floor_div(n_out, down) + (floor_mod(n_out, down) != 0 ? 1 : 0);

    const std::int64_t n_pre_pad = down - floor_mod(half_len, down);
    std::int64_t n_post_pad = 0;
    const std::int64_t n_pre_remove = floor_div(half_len + n_pre_pad, down);
    const auto len_h = static_cast<std::int64_t>(h.size());
    while (upfirdn_output_len(len_h + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove)
        ++n_post_pad;

    std::vector<T> hp(static_cast<std::size_t>(n_pre_pad + len_h + n_post_pad), T{});
    std::copy(h.begin(), h.end(), hp.begin() + n_pre_pad);
    const std::vector<T> y = upfirdn<T>(hp, x, up, down, cval);

    // y[n_pre_remove : n_pre_remove + n_out], clamped as a Python slice.
    const auto ny = static_cast<std::int64_t>(y.size());
    const std::int64_t a = std::min(std::max<std::int64_t>(n_pre_remove, 0), ny);
    const std::int64_t b = std::min(std::max<std::int64_t>(n_pre_remove + n_out, a), ny);
    return std::vector<T>(y.begin() + a, y.begin() + b);
}

template <typename T>
std::vector<T> resample_poly(std::span<const T> x, std::int64_t up, std::int64_t down, const Window& window,
                             T cval) {
    if (up < 1 || down < 1) throw std::invalid_argument("up and down must be >= 1");
    const std::int64_t g = std::gcd(up, down);
    up /= g;
    down /= g;
    if (up == 1 && down == 1) return std::vector<T>(x.begin(), x.end());
    const std::vector<T> h = resample_poly_filter<T>(up, down, window);
    return resample_poly_taps<T>(x, up, down, h, 10 * std::max(up, down), cval);
}

template <typename T>
std::vector<T> resample_poly_filter(std::int64_t up, std::int64_t down, const Window& window) {
    if (up < 1 || down < 1) throw std::invalid_argument("up and down must be >= 1");
    const std::int64_t max_rate = std::max(up, down);
    const double f_c = 1. / static_cast<double>(max_rate);
    const std::int64_t half_len = 10 * max_rate;
    const std::vector<double> h64 = firwin(2 * half_len + 1, f_c, window);
    std::vector<T> h(h64.size());
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = scale_by_up<T>(h64[i], up);
    return h;
}

#define SSTVAE_QRSS_FILTERS_INSTANTIATE(T)                                                               \
    template std::vector<T> upfirdn<T>(std::span<const T>, std::span<const T>, std::int64_t, std::int64_t, \
                                       T);                                                               \
    template std::vector<T> resample_poly<T>(std::span<const T>, std::int64_t, std::int64_t,               \
                                             const Window&, T);                                          \
    template std::vector<T> resample_poly_filter<T>(std::int64_t, std::int64_t, const Window&);       \
    template std::vector<T> resample_poly_taps<T>(std::span<const T>, std::int64_t, std::int64_t,          \
                                                  std::span<const T>, std::int64_t, T);

SSTVAE_QRSS_FILTERS_INSTANTIATE(float)
SSTVAE_QRSS_FILTERS_INSTANTIATE(double)
SSTVAE_QRSS_FILTERS_INSTANTIATE(std::complex<float>)
SSTVAE_QRSS_FILTERS_INSTANTIATE(std::complex<double>)

#undef SSTVAE_QRSS_FILTERS_INSTANTIATE

}  // namespace sstvae::qrss::filters
