// numpy's summation, bit for bit: np.sum and np.mean as numpy computes
// them on a 1-D array, so a C++ RMS or mean rounds exactly where the
// Python reference rounds (picture.prepare_segment's RMS is hashed into the
// picture ID; the receiver's statistics feed thresholds).
//
// What numpy does (numpy 2.x, numpy/_core/src/umath/loops_utils.h.src):
//
// * np.add.reduce adds `pairwise_sum` of the operand to the identity 0.
//   pairwise_sum sums fewer than 8 values left to right, up to 128 with
//   eight interleaved accumulators combined ((r0+r1)+(r2+r3))+((r4+r5)+
//   (r6+r7)) and then the tail left to right, and more than 128 by
//   splitting at n/2 rounded down to a multiple of 8 and adding the halves.
//   No step uses a multiply, so no FMA contraction can reach it.
// * When the reduction casts (np.sum(x32, dtype=np.float64), np.mean of an
//   integer or bool array, which sums in float64), the iterator buffers:
//   the operand is cast in chunks of NPY_BUFSIZE = 8192 elements, each
//   chunk is pairwise summed, and the chunk sums are added left to right
//   onto the identity. Without a cast there is one pairwise sum over the
//   whole array, strided or not (checked against numpy 2.5 up to 3e5).
// * np.mean is that sum divided by the count. For float64 it is a float64
//   division. For float32, numpy divides the float32 sum by an intp count,
//   which promotes to float64, and casts the quotient back to float32;
//   `mean` does the same (two roundings, as numpy).
// * complex sums keep separate real and imaginary accumulators, four
//   complex values per eight-accumulator block (numpy's CDOUBLE_pairwise_sum
//   over the interleaved floats with n counting floats).
//
// Not covered: float16 sums (numpy accumulates them in float32 and rounds
// back), reductions along an axis of a 2-D array (numpy's iterator then
// chooses the loop order), and np.mean of float16. Each is its own case;
// add it here, checked against numpy, before relying on it.

#ifndef SSTVAE_QRSS_NPSUM_HPP
#define SSTVAE_QRSS_NPSUM_HPP

#include <complex>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace sstvae::qrss::npsum {

inline constexpr std::size_t PW_BLOCKSIZE = 128;   // numpy's PW_BLOCKSIZE
inline constexpr std::size_t BUFSIZE = 8192;       // numpy's NPY_BUFSIZE, in elements

// numpy's @TYPE@_pairwise_sum: the sum of a[0], a[stride], ... (n values),
// in T arithmetic. T is double or float.
template <typename T>
T pairwise_sum(const T* a, std::size_t n, std::ptrdiff_t stride = 1) {
    static_assert(std::is_floating_point_v<T>);
    if (n < 8) {
        T res = T(-0.0);
        for (std::size_t i = 0; i < n; ++i) res += a[static_cast<std::ptrdiff_t>(i) * stride];
        return res;
    }
    if (n <= PW_BLOCKSIZE) {
        T r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k * stride];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; ++k)
                r[k] += a[(static_cast<std::ptrdiff_t>(i) + k) * stride];
        T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[static_cast<std::ptrdiff_t>(i) * stride];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2, stride) +
           pairwise_sum(a + static_cast<std::ptrdiff_t>(n2) * stride, n - n2, stride);
}

// numpy's C@TYPE@_pairwise_sum. `a` points at interleaved (re, im) floats;
// n counts floats (twice the complex count) and stride is in floats
// between consecutive *floats of one value pair*, as numpy calls it
// (n * 2, steps / 2), so a contiguous complex array is stride 1.
template <typename T>
void pairwise_sum_complex(T& rr, T& ri, const T* a, std::size_t n, std::ptrdiff_t stride = 1) {
    static_assert(std::is_floating_point_v<T>);
    auto at = [&](std::size_t i, int part) { return a[static_cast<std::ptrdiff_t>(i) * stride + part]; };
    if (n < 8) {
        rr = T(-0.0);
        ri = T(-0.0);
        for (std::size_t i = 0; i < n; i += 2) {
            rr += at(i, 0);
            ri += at(i, 1);
        }
        return;
    }
    if (n <= PW_BLOCKSIZE) {
        T r[8];
        for (int k = 0; k < 4; ++k) {
            r[2 * k] = at(2 * static_cast<std::size_t>(k), 0);
            r[2 * k + 1] = at(2 * static_cast<std::size_t>(k), 1);
        }
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 4; ++k) {
                r[2 * k] += at(i + 2 * static_cast<std::size_t>(k), 0);
                r[2 * k + 1] += at(i + 2 * static_cast<std::size_t>(k), 1);
            }
        rr = ((r[0] + r[2]) + (r[4] + r[6]));
        ri = ((r[1] + r[3]) + (r[5] + r[7]));
        for (; i < n; i += 2) {
            rr += at(i, 0);
            ri += at(i, 1);
        }
        return;
    }
    T rr1, ri1, rr2, ri2;
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_complex(rr1, ri1, a, n2, stride);
    pairwise_sum_complex(rr2, ri2, a + static_cast<std::ptrdiff_t>(n2) * stride, n - n2, stride);
    rr = rr1 + rr2;
    ri = ri1 + ri2;
}

// np.sum(x) for float64 or float32 x: one pairwise sum, onto 0.
template <typename T>
T sum(std::span<const T> x) {
    static_assert(std::is_floating_point_v<T>);
    return T(0) + pairwise_sum(x.data(), x.size());
}

inline double sum(std::span<const double> x) { return sum<double>(x); }
inline float sum(std::span<const float> x) { return sum<float>(x); }

// np.sum(z) for complex128 or complex64 z.
template <typename T>
std::complex<T> sum(std::span<const std::complex<T>> z) {
    T rr = T(0), ri = T(0);
    if (!z.empty()) pairwise_sum_complex(rr, ri, reinterpret_cast<const T*>(z.data()), 2 * z.size());
    return {T(0) + rr, T(0) + ri};
}

inline std::complex<double> sum(std::span<const std::complex<double>> z) { return sum<double>(z); }
inline std::complex<float> sum(std::span<const std::complex<float>> z) { return sum<float>(z); }

// np.sum(x, dtype=Acc). Where x's dtype is not Acc, the buffered reduction:
// From cast to Acc (a C++ static_cast, which is numpy's cast for every
// pairing used here: float32 -> float64 exact, float64 -> float32 rounded
// to nearest even, integers and bool -> float64) BUFSIZE values at a time.
// Where it is Acc already there is nothing to cast, numpy does not buffer,
// and np.sum(x, dtype=x.dtype) is np.sum(x): one pairwise sum, not chunks
// (chunking it differs from numpy from 8193 values on).
template <typename Acc, typename From>
Acc sum_as(std::span<const From> x) {
    static_assert(std::is_floating_point_v<Acc>);
    if constexpr (std::is_same_v<Acc, From>) {
        return sum<Acc>(x);
    } else {
        std::vector<Acc> buf(x.size() < BUFSIZE ? x.size() : BUFSIZE);
        Acc res = Acc(0);
        for (std::size_t i0 = 0; i0 < x.size(); i0 += BUFSIZE) {
            const std::size_t m = (x.size() - i0 < BUFSIZE) ? x.size() - i0 : BUFSIZE;
            for (std::size_t j = 0; j < m; ++j) buf[j] = static_cast<Acc>(x[i0 + j]);
            res += pairwise_sum(buf.data(), m);
        }
        return res;
    }
}

// np.mean(x) for float64 x.
inline double mean(std::span<const double> x) {
    return sum(x) / static_cast<double>(x.size());
}

// np.mean(x) for float32 x: float32(float64(sum) / n), as numpy's _mean
// does (np.float32 / np.intp promotes to float64).
inline float mean(std::span<const float> x) {
    return static_cast<float>(static_cast<double>(sum(x)) / static_cast<double>(x.size()));
}

// np.mean(x) where numpy sums in float64 through a cast: an integer or
// bool array (numpy's default dtype for those), or np.mean(x32,
// dtype=np.float64). For float64 x it is np.mean(x), as numpy's is.
template <typename From>
double mean_as_double(std::span<const From> x) {
    return sum_as<double>(x) / static_cast<double>(x.size());
}

}  // namespace sstvae::qrss::npsum

#endif  // SSTVAE_QRSS_NPSUM_HPP
