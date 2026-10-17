#include "qrss/half.hpp"

#include <bit>
#include <cstddef>

namespace sstvae::qrss::half {

namespace {

// Round `sig >> shift` to the nearest integer, ties to even. shift >= 1.
template <typename U>
U round_shift(U sig, unsigned shift) {
    const U q = sig >> shift;
    const U rem = sig & ((U(1) << shift) - 1);
    const U halfway = U(1) << (shift - 1);
    return (rem > halfway || (rem == halfway && (q & 1u))) ? q + 1 : q;
}

}  // namespace

std::uint16_t from_double(double x) {
    const std::uint64_t d = std::bit_cast<std::uint64_t>(x);
    const auto sign = static_cast<std::uint16_t>((d >> 48) & 0x8000u);
    const unsigned exp = static_cast<unsigned>((d >> 52) & 0x7ffu);
    const std::uint64_t man = d & 0x000fffffffffffffULL;

    if (exp == 0x7ffu) {
        if (man == 0) return static_cast<std::uint16_t>(sign | 0x7c00u);   // +-inf
        // NaN: the top 10 significand bits, kept a NaN.
        auto r = static_cast<std::uint16_t>(0x7c00u + (man >> 42));
        if (r == 0x7c00u) ++r;
        return static_cast<std::uint16_t>(sign + r);
    }
    // |x| >= 2^16: past the largest half (65504) by more than half an ulp.
    if (exp >= 1023 + 16) return static_cast<std::uint16_t>(sign | 0x7c00u);
    // |x| < 2^-14: a subnormal half (unit 2^-24) or zero.
    if (exp <= 1023 - 15) {
        // Below 2^-25 (half the smallest subnormal) is zero; exactly 2^-25
        // is a tie, which round_shift sends to the even 0.
        if (exp < 1023 - 25) return sign;
        const std::uint64_t sig = man | (std::uint64_t(1) << 52);   // x = sig * 2^(exp-1075)
        const std::uint64_t q = round_shift(sig, 1051u - exp);      // in units of 2^-24
        // q == 0x400 is the smallest normal, which is the right answer.
        return static_cast<std::uint16_t>(sign + q);
    }
    // Normal: 10 bits of significand, rounded; a carry out of them
    // increments the exponent, and out of exponent 30 makes inf.
    const unsigned h_exp = exp - (1023 - 15);
    const std::uint64_t q = round_shift(man, 42u);
    return static_cast<std::uint16_t>(sign + (static_cast<std::uint64_t>(h_exp) << 10) + q);
}

std::uint16_t from_float(float x) {
    const std::uint32_t f = std::bit_cast<std::uint32_t>(x);
    const auto sign = static_cast<std::uint16_t>((f >> 16) & 0x8000u);
    const unsigned exp = (f >> 23) & 0xffu;
    const std::uint32_t man = f & 0x007fffffu;

    if (exp == 0xffu) {
        if (man == 0) return static_cast<std::uint16_t>(sign | 0x7c00u);
        auto r = static_cast<std::uint16_t>(0x7c00u + (man >> 13));
        if (r == 0x7c00u) ++r;
        return static_cast<std::uint16_t>(sign + r);
    }
    if (exp >= 127 + 16) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exp <= 127 - 15) {
        if (exp < 127 - 25) return sign;
        const std::uint32_t sig = man | (std::uint32_t(1) << 23);   // x = sig * 2^(exp-150)
        const std::uint32_t q = round_shift(sig, 126u - exp);
        return static_cast<std::uint16_t>(sign + q);
    }
    const unsigned h_exp = exp - (127 - 15);
    const std::uint32_t q = round_shift(man, 13u);
    return static_cast<std::uint16_t>(sign + (h_exp << 10) + q);
}

double to_double(std::uint16_t h) {
    const std::uint64_t sign = static_cast<std::uint64_t>(h & 0x8000u) << 48;
    const unsigned e = (h >> 10) & 0x1fu;
    const std::uint64_t m = h & 0x03ffu;
    std::uint64_t d;
    if (e == 0) {
        if (m == 0) {
            d = sign;
        } else {
            // m * 2^-24 with m's top bit at p: 2^(p-24) * 1.f
            const int p = std::bit_width(m) - 1;
            d = sign | (static_cast<std::uint64_t>(p - 24 + 1023) << 52) |
                ((m << (52 - p)) & 0x000fffffffffffffULL);
        }
    } else if (e == 0x1fu) {
        d = sign | 0x7ff0000000000000ULL | (m << 42);
    } else {
        d = sign | (static_cast<std::uint64_t>(e - 15 + 1023) << 52) | (m << 42);
    }
    return std::bit_cast<double>(d);
}

float to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const unsigned e = (h >> 10) & 0x1fu;
    const std::uint32_t m = h & 0x03ffu;
    std::uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = sign;
        } else {
            const int p = std::bit_width(m) - 1;
            f = sign | (static_cast<std::uint32_t>(p - 24 + 127) << 23) | ((m << (23 - p)) & 0x007fffffu);
        }
    } else if (e == 0x1fu) {
        f = sign | 0x7f800000u | (m << 13);
    } else {
        f = sign | (static_cast<std::uint32_t>(e - 15 + 127) << 23) | (m << 13);
    }
    return std::bit_cast<float>(f);
}

std::vector<std::uint16_t> from_double(std::span<const double> x) {
    std::vector<std::uint16_t> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = from_double(x[i]);
    return out;
}

std::vector<double> to_double(std::span<const std::uint16_t> h) {
    std::vector<double> out(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) out[i] = to_double(h[i]);
    return out;
}

}  // namespace sstvae::qrss::half
