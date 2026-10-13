// The exceptions the receiver's numpy/scipy numerics raise, by Python type.
//
// core/math/ reproduces numpy and scipy calls (median, interp, polyfit,
// lstsq, solve, inv, eigh, make_lsq_spline) for the QRSSTVAE receiver
// port, and the receiver catches some of their exceptions by type --
// `except (ValueError, np.linalg.LinAlgError)` around make_lsq_spline,
// `except np.linalg.LinAlgError` around eigh, solve and inv -- so the C++
// keeps the distinction:
//
//   Python ValueError     -> std::invalid_argument (pybind11's default mapping)
//   Python TypeError      -> math::TypeError (an std::invalid_argument, so a
//                            caller that only knows "bad input" still catches it)
//   np.linalg.LinAlgError -> math::LinAlgError (std::runtime_error)
//   Python IndexError     -> std::out_of_range (pybind11's default mapping)
//
// Messages are the reference's texts.

#ifndef SSTVAE_MATH_ERRORS_HPP
#define SSTVAE_MATH_ERRORS_HPP

#include <stdexcept>

namespace sstvae::math {

// numpy.linalg.LinAlgError: "Singular matrix", "SVD did not converge in
// Linear Least Squares", ...
struct LinAlgError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Python's TypeError where numpy raises one for a bad argument
// (np.polyfit's "expected non-empty vector for x").
struct TypeError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_ERRORS_HPP
