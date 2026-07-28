// A test harness small enough not to be a dependency.
//
// Catch2 or doctest would be nicer, but every dependency added here is
// one vcpkg has to resolve on three platforms before the parity harness
// -- the thing that makes the whole port checkable -- can run at all.
// The harness is ~60 lines; the corpus it checks is the valuable part.

#pragma once

#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

namespace sstvae::check {

inline int failures = 0;
inline int checks = 0;

inline void fail(const std::string& what, const std::string& detail) {
    ++failures;
    std::fprintf(stderr, "FAIL %s: %s\n", what.c_str(), detail.c_str());
}

inline void is_true(bool ok, const std::string& what) {
    ++checks;
    if (!ok) fail(what, "expected true");
}

// Values are rendered through this rather than std::to_string directly,
// so a check can compare strings (dtype names, mostly) as readily as
// numbers.
inline std::string to_text(const std::string& s) { return "\"" + s + "\""; }
inline std::string to_text(const char* s) { return to_text(std::string(s)); }
template <typename T>
std::string to_text(const T& v) {
    return std::to_string(v);
}

template <typename A, typename B>
void equal(const A& got, const B& want, const std::string& what) {
    ++checks;
    if (!(got == want))
        fail(what, "got " + to_text(got) + ", want " + to_text(want));
}

// Largest absolute difference between two sequences, reported with the
// index where it occurs -- a bare "1.7e-09 > 1e-12" says nothing about
// whether one element is wrong or all of them are.
template <typename T>
void close(const std::vector<T>& got, const std::vector<T>& want, double tol,
           const std::string& what) {
    ++checks;
    if (got.size() != want.size()) {
        fail(what, "size " + std::to_string(got.size()) + " != " +
                       std::to_string(want.size()));
        return;
    }
    double worst = 0.0;
    std::size_t where = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::abs(got[i] - want[i]);
        if (d > worst) {
            worst = d;
            where = i;
        }
    }
    if (!(worst <= tol)) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "max |diff| = %.6g at index %zu (tolerance %.6g)", worst,
                      where, tol);
        fail(what, buf);
    }
}

inline int report(const char* suite) {
    if (failures == 0) {
        std::printf("ok: %s (%d checks)\n", suite, checks);
        return 0;
    }
    std::fprintf(stderr, "%d of %d checks FAILED in %s\n", failures, checks, suite);
    return 1;
}

}  // namespace sstvae::check
