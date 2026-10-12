#include "qrss/precoder.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "qrss/sequences.hpp"

namespace sstvae::qrss::precoder {

namespace {

// The number of whole rows of `n` in `size` values.
std::size_t rows_of(std::size_t size, std::size_t n) {
    if (n == 0) {
        if (size != 0)
            throw std::invalid_argument("precoder: rows of length 0 cannot hold " +
                                        std::to_string(size) + " values");
        return 0;
    }
    if (size % n)
        throw std::invalid_argument("precoder: " + std::to_string(size) +
                                    " values are not a whole number of rows of " + std::to_string(n));
    return size / n;
}

// numpy's pairwise_sum (loops_utils.h.src) for float64, unit stride:
// under 8 values a plain loop from -0.0; up to 128, eight accumulators
// combined as ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) and then the rest;
// above that, split at a multiple of 8 near the middle.
double pairwise_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = -0.0;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (std::size_t k = 0; k < 8; ++k) r[k] = a[k];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (std::size_t k = 0; k < 8; ++k) r[k] += a[i + k];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// precoder._fwht_inplace on one block of m, then / sqrt(m) as wht does.
void wht_block(double* p, std::size_t m) {
    for (std::size_t h = 1; h < m; h *= 2) {
        for (std::size_t i = 0; i < m; i += 2 * h) {
            for (std::size_t j = i; j < i + h; ++j) {
                const double x = p[j];
                const double y = p[j + h];
                p[j] = x + y;       // v[..., 0, :] += y
                p[j + h] = x - y;   // v[..., 1, :] = x - y
            }
        }
    }
    const double s = std::sqrt(static_cast<double>(m));   // np.sqrt(m)
    for (std::size_t j = 0; j < m; ++j) p[j] = p[j] / s;
}

}  // namespace

std::vector<std::size_t> block_sizes(std::size_t n) {
    std::vector<std::size_t> out(n / BLOCK, BLOCK);
    const std::size_t rem = n % BLOCK;
    for (std::size_t m = BLOCK; m > 0; m /= 2)   // 1 << k for k from bit_length-1 down
        if (rem & m) out.push_back(m);
    return out;
}

std::vector<Run> runs(std::size_t n) {
    // Sizes are BLOCK repeated, then distinct powers of two, descending:
    // the runs are contiguous in that order.
    std::vector<Run> out;
    std::size_t start = 0;
    if (n / BLOCK) {
        out.push_back({0, BLOCK, n / BLOCK});
        start = BLOCK * (n / BLOCK);
    }
    const std::size_t rem = n % BLOCK;
    for (std::size_t m = BLOCK; m > 0; m /= 2) {
        if (rem & m) {
            out.push_back({start, m, 1});
            start += m;
        }
    }
    return out;
}

void wht_inplace(std::span<double> v, std::size_t n) {
    const std::size_t rows = rows_of(v.size(), n);
    const std::vector<Run> rs = runs(n);
    for (std::size_t r = 0; r < rows; ++r) {
        double* row = v.data() + r * n;
        for (const Run& run : rs)
            for (std::size_t b = 0; b < run.count; ++b) wht_block(row + run.start + b * run.m, run.m);
    }
}

std::vector<double> wht(std::span<const double> v, std::size_t n) {
    std::vector<double> out(v.begin(), v.end());
    wht_inplace(out, n);
    return out;
}

std::vector<double> precode(std::span<const double> a, std::uint64_t q, std::size_t n) {
    const std::size_t rows = rows_of(a.size(), n);
    const std::vector<std::int8_t> c = sequences::scrambler(q, n);
    std::vector<double> out(a.size());
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t j = 0; j < n; ++j)   // int8 -> float64, then the product
            out[r * n + j] = static_cast<double>(c[j]) * a[r * n + j];
    wht_inplace(out, n);
    return out;
}

std::vector<double> unprecode(std::span<const double> y, std::uint64_t q, std::size_t n) {
    const std::size_t rows = rows_of(y.size(), n);
    const std::vector<std::int8_t> c = sequences::scrambler(q, n);
    std::vector<double> out = wht(y, n);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t j = 0; j < n; ++j) out[r * n + j] = static_cast<double>(c[j]) * out[r * n + j];
    return out;
}

std::vector<double> block_mean(std::span<const double> v, std::size_t n) {
    const std::size_t rows = rows_of(v.size(), n);
    const std::vector<Run> rs = runs(n);
    std::vector<double> out(v.size());
    for (std::size_t r = 0; r < rows; ++r) {
        const double* row = v.data() + r * n;
        for (const Run& run : rs) {
            for (std::size_t b = 0; b < run.count; ++b) {
                const std::size_t s = run.start + b * run.m;
                // np.add.reduce starts from the identity 0.0, then the
                // mean divides by the count.
                double sum = 0.0;
                sum += pairwise_sum(row + s, run.m);
                const double mean = sum / static_cast<double>(run.m);
                for (std::size_t j = 0; j < run.m; ++j) out[r * n + s + j] = mean;
            }
        }
    }
    return out;
}

std::vector<std::int64_t> block_index(std::size_t n) {
    std::vector<std::int64_t> out;
    out.reserve(n);
    std::int64_t b = 0;
    for (std::size_t m : block_sizes(n)) {
        out.insert(out.end(), m, b);
        ++b;
    }
    return out;
}

}  // namespace sstvae::qrss::precoder
