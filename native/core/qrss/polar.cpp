#include "qrss/polar.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/header.hpp"
#include "util/sha256.hpp"

namespace sstvae::qrss::polar {

namespace {

constexpr std::size_t UN = static_cast<std::size_t>(N);
constexpr std::size_t UK = static_cast<std::size_t>(K);
constexpr std::size_t UE = static_cast<std::size_t>(E);

// numpy's NPY_LOGE2, the same decimal literal.
constexpr double LOGE2 = 0.693147180559945309417232121458176568;

// numpy's npy_logaddexp (what np.logaddexp's float64 loop calls), with the
// same libm exp and log1p.
double logaddexp(double x, double y) {
    if (x == y) return x + LOGE2;   // infinities of one sign
    const double tmp = x - y;
    if (tmp > 0) return x + std::log1p(std::exp(-tmp));
    if (tmp <= 0) return y + std::log1p(std::exp(tmp));
    return tmp;                     // NaN
}

// log(1 + exp(-a)): polar._softplus_neg, np.logaddexp(0.0, -a).
double softplus_neg(double a) { return logaddexp(0.0, -a); }

// numpy's DOUBLE_pairwise_sum over a contiguous run (blocks of 128, eight
// accumulators): how np.sum(axis=-1) adds each row.
double pairwise_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; ++k) r[k] += a[i + static_cast<std::size_t>(k)];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// np.add.reduce of a row: the identity, plus the pairwise sum.
double row_sum(const double* a, std::size_t n) {
    double s = 0.0;
    s += pairwise_sum(a, n);
    return s;
}

// numpy's sort order for float64: NaN last, all NaNs equal.
bool np_less(double a, double b) { return a < b || (b != b && a == a); }

// np.argsort(v, kind="stable").
std::vector<std::size_t> stable_argsort(const std::vector<double>& v) {
    std::vector<std::size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), std::size_t(0));
    std::stable_sort(idx.begin(), idx.end(),
                     [&v](std::size_t a, std::size_t b) { return np_less(v[a], v[b]); });
    return idx;
}

// np.sign for float64: 0 for either zero, NaN for NaN.
double np_sign(double x) {
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    if (x == 0) return 0.0;
    return x;
}

// np.minimum: NaN propagates from either side.
double np_minimum(double a, double b) { return (a < b || a != a) ? a : b; }

// Prefix counts of INFO_SET over mother indices: a node [lo, lo + n) is
// rate 0 (all leaves frozen) when no information index falls inside it.
const std::array<int, UN + 1>& info_prefix() {
    static const std::array<int, UN + 1> p = [] {
        std::array<bool, UN> is_info{};
        for (int i : INFO_SET) is_info[static_cast<std::size_t>(i)] = true;
        std::array<int, UN + 1> out{};
        for (std::size_t i = 0; i < UN; ++i) out[i + 1] = out[i] + (is_info[i] ? 1 : 0);
        return out;
    }();
    return p;
}

bool rate0(std::size_t lo, std::size_t n) {
    const auto& p = info_prefix();
    return p[lo + n] == p[lo];
}

// polar._List: the path metrics shared down the recursion.
struct State {
    std::size_t L;
    std::vector<double> pm;
};

// polar._scl's return: beta (L x n) and origin (L).
struct Node {
    std::vector<std::uint8_t> beta;
    std::vector<std::size_t> origin;
};

// polar._scl: decode the subtree at mother indices [lo, lo + n) from its
// input LLRs alpha (L x n, row-major), in the reference's operation order.
Node scl(const std::vector<double>& alpha, std::size_t n, std::size_t lo, State& st) {
    const std::size_t L = st.L;
    Node out;
    if (rate0(lo, n)) {
        // st.pm = st.pm + _softplus_neg(alpha).sum(axis=1)
        std::vector<double> sp(n);
        for (std::size_t l = 0; l < L; ++l) {
            for (std::size_t j = 0; j < n; ++j) sp[j] = softplus_neg(alpha[l * n + j]);
            st.pm[l] = st.pm[l] + row_sum(sp.data(), n);
        }
        out.beta.assign(L * n, 0);
        out.origin.resize(L);
        std::iota(out.origin.begin(), out.origin.end(), std::size_t(0));
        return out;
    }
    if (n == 1) {
        // cand = [pm + sp(lam), pm + sp(-lam)]; keep the L smallest, stably.
        std::vector<double> cand(2 * L);
        for (std::size_t l = 0; l < L; ++l) {
            const double lam = alpha[l];
            cand[l] = st.pm[l] + softplus_neg(lam);
            cand[L + l] = st.pm[l] + softplus_neg(-lam);
        }
        const std::vector<std::size_t> order = stable_argsort(cand);
        out.beta.resize(L);
        out.origin.resize(L);
        for (std::size_t l = 0; l < L; ++l) {
            const std::size_t k = order[l];
            st.pm[l] = cand[k];
            out.beta[l] = static_cast<std::uint8_t>(k / L);
            out.origin[l] = k % L;
        }
        return out;
    }
    const std::size_t h = n / 2;
    // f = sign(a1) * sign(a2) * minimum(|a1|, |a2|)   (min-sum)
    std::vector<double> f(L * h);
    for (std::size_t l = 0; l < L; ++l) {
        const double* row = alpha.data() + l * n;
        for (std::size_t j = 0; j < h; ++j) {
            const double a1 = row[j], a2 = row[h + j];
            f[l * h + j] = (np_sign(a1) * np_sign(a2)) * np_minimum(std::fabs(a1), std::fabs(a2));
        }
    }
    Node a = scl(f, h, lo, st);
    // a1, a2 = a1[org_a], a2[org_a]; g = a2 + (1 - 2 beta_a) a1   (exact g)
    std::vector<double>& g = f;   // f is spent; same shape
    for (std::size_t l = 0; l < L; ++l) {
        const double* row = alpha.data() + a.origin[l] * n;
        for (std::size_t j = 0; j < h; ++j) {
            const double a1 = row[j], a2 = row[h + j];
            g[l * h + j] = a2 + (1.0 - 2.0 * static_cast<double>(a.beta[l * h + j])) * a1;
        }
    }
    Node b = scl(g, h, lo + h, st);
    // beta_a = beta_a[org_b]; [beta_a ^ beta_b, beta_b]; org_a[org_b]
    out.beta.resize(L * n);
    out.origin.resize(L);
    for (std::size_t l = 0; l < L; ++l) {
        const std::size_t s = b.origin[l];
        for (std::size_t j = 0; j < h; ++j) {
            const std::uint8_t bb = b.beta[l * h + j];
            out.beta[l * n + j] = static_cast<std::uint8_t>(a.beta[s * h + j] ^ bb);
            out.beta[l * n + h + j] = bb;
        }
        out.origin[l] = a.origin[s];
    }
    return out;
}

std::string shape1(std::size_t n) { return "(" + std::to_string(n) + ",)"; }

}  // namespace

bool info_set_ok() {
    std::vector<std::uint8_t> be;
    be.reserve(2 * UK);
    int prev = -1;
    for (int i : INFO_SET) {
        if (i <= prev || i >= N) return false;
        prev = i;
        be.push_back(static_cast<std::uint8_t>(static_cast<unsigned>(i) >> 8));
        be.push_back(static_cast<std::uint8_t>(i & 0xFF));
    }
    return util::to_hex(util::sha256(std::span<const std::uint8_t>(be))) == config::POLAR_INFO_SET_SHA256;
}

void polar_transform_rows(std::span<std::uint8_t> x, std::size_t n) {
    // No rows: numpy reshapes a size-0 array into any size-0 shape, so the
    // reference refuses nothing here (shape (0, 5) comes back as it went in).
    if (n == 0 || x.empty()) return;
    if (x.size() % n)
        throw std::invalid_argument("polar_transform: " + std::to_string(x.size()) +
                                    " values are not whole rows of " + std::to_string(n));
    for (std::size_t h = n / 2; h >= 1; h /= 2)
        if (n % (2 * h))
            throw std::invalid_argument("polar_transform: a row of " + std::to_string(n) +
                                        " does not split into blocks of " + std::to_string(2 * h));
    const std::size_t rows = x.size() / n;
    // At each stage the first half of every 2h-block takes the XOR of the
    // second; numpy's order is h = n/2 first (the stages commute anyway).
    for (std::size_t h = n / 2; h >= 1; h /= 2)
        for (std::size_t r = 0; r < rows; ++r) {
            std::uint8_t* row = x.data() + r * n;
            for (std::size_t b0 = 0; b0 < n; b0 += 2 * h)
                for (std::size_t j = 0; j < h; ++j) row[b0 + j] ^= row[b0 + h + j];
        }
}

std::vector<std::uint8_t> polar_transform(std::span<const std::uint8_t> v) {
    std::vector<std::uint8_t> x(v.begin(), v.end());
    polar_transform_rows(x, x.size());
    return x;
}

std::vector<std::uint8_t> rate_match(std::span<const std::uint8_t> x) {
    std::vector<std::uint8_t> out(UE);
    rate_match_into<std::uint8_t>(x, out);
    return out;
}

std::vector<double> fold(std::span<const double> llr) {
    if (llr.size() != UE)
        throw std::invalid_argument("expected " + std::to_string(E) + " coded LLRs, got shape " +
                                    shape1(llr.size()));
    // np.bincount(_RM, weights=llr, minlength=N): out[_RM[e]] += llr[e], e ascending.
    std::vector<double> out(UN, 0.0);
    for (std::size_t e = 0; e < UE; ++e) out[static_cast<std::size_t>(mother_index(static_cast<int>(e)))] += llr[e];
    return out;
}

std::vector<std::uint8_t> polar_encode(std::span<const std::uint8_t> info) {
    if (info.size() != UK)
        throw std::invalid_argument("expected " + std::to_string(K) + " information bits, got shape " +
                                    shape1(info.size()));
    for (std::uint8_t b : info)
        if (b > 1) throw std::invalid_argument("information bits must be 0 or 1");
    std::vector<std::uint8_t> u(UN, 0);
    for (std::size_t t = 0; t < UK; ++t) u[static_cast<std::size_t>(INFO_SET[t])] = info[t];
    polar_transform_rows(u, UN);
    return rate_match(u);
}

ListResult decode_list(std::span<const double> llr, std::size_t list_size) {
    if (list_size < 1) throw std::invalid_argument("list_size must be >= 1");
    const std::vector<double> mother = fold(llr);
    for (double v : mother)
        if (!std::isfinite(v)) throw std::invalid_argument("LLRs must be finite");
    const std::size_t L = list_size;
    State st{L, std::vector<double>(L, std::numeric_limits<double>::infinity())};
    st.pm[0] = 0.0;                     // one live path; the rest are placeholders
    std::vector<double> alpha(L * UN);
    for (std::size_t l = 0; l < L; ++l) std::copy(mother.begin(), mother.end(), alpha.begin() + static_cast<std::ptrdiff_t>(l * UN));
    Node root = scl(alpha, UN, 0, st);
    polar_transform_rows(root.beta, UN);    // G is its own inverse
    std::vector<std::size_t> order = stable_argsort(st.pm);
    ListResult res;
    for (std::size_t k : order) {
        if (!std::isfinite(st.pm[k])) continue;
        const std::uint8_t* u = root.beta.data() + k * UN;
        for (int i : INFO_SET) res.info.push_back(u[static_cast<std::size_t>(i)]);
        res.pm.push_back(st.pm[k]);
        ++res.m;
    }
    return res;
}

std::optional<std::vector<std::uint8_t>> polar_decode(std::span<const double> llr, std::size_t list_size,
                                                      const Accept& accept) {
    const ListResult cands = decode_list(llr, list_size);
    for (std::size_t i = 0; i < cands.m; ++i) {
        const auto bits = cands.row(i);
        const bool ok = accept ? accept(bits) : header::unpack(bits).has_value();
        if (ok) return std::vector<std::uint8_t>(bits.begin(), bits.end());
    }
    return std::nullopt;
}

}  // namespace sstvae::qrss::polar
