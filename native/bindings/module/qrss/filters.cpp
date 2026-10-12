// sstvae_native.qrss.filters: bindings of native/core/qrss/filters, the
// scipy filters sstvae/qrss uses. Signatures, defaults, dtype rules and
// error texts follow scipy 1.18 (scipy.signal.upfirdn, firwin,
// resample_poly, get_window and windows.hann/hamming/kaiser;
// scipy.special.i0/i0e), so tests/qrss_native/filters*.py can assign
// these over the names sstvae.qrss takes from scipy.
//
// What is ported is what sstvae/qrss calls: mode/padtype 'constant' and
// the hamming, hann and kaiser windows. Anything else scipy accepts
// raises NotImplementedError naming what is missing, never a silently
// different answer; anything scipy itself refuses raises scipy's error.
// The C++ checks happen in scipy's order, which for firwin means a
// refusal decided here (a bad pass_zero, window or numtaps type) is held
// back until the core has had the chance to raise an earlier one.

#include "qrss/registry.hpp"

#include <pybind11/complex.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/filters.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace flt = sstvae::qrss::filters;
using namespace pybind11::literals;

py::module_ numpy() { return py::module_::import("numpy"); }

std::string pyrepr(const py::handle& o) { return py::repr(o).cast<std::string>(); }

[[noreturn]] void py_raise(PyObject* type, const std::string& msg) {
    PyErr_SetString(type, msg.c_str());
    throw py::error_already_set();
}

std::int64_t floor_mod(std::int64_t a, std::int64_t b) {
    const std::int64_t r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}

// A Python exception decided now and raised later, so that an earlier
// check in scipy's order still wins.
struct Deferred {
    int stage = 0;   // see firwin_py
    PyObject* type = nullptr;
    py::object value;
    [[noreturn]] void raise_now() const {
        PyErr_SetObject(type, value.ptr());
        throw py::error_already_set();
    }
};

// --- windows ------------------------------------------------------------------

// scipy.signal.windows._windows._WIN_FUNCS: every name get_window knows,
// so an unknown name gets scipy's ValueError and a known but unported one
// NotImplementedError.
const std::set<std::string>& scipy_window_names() {
    static const std::set<std::string> names = {
        "bart", "barthann", "bartlett", "bkh", "black", "blackharr", "blackman", "blackmanharris",
        "blk", "bman", "bmn", "bohman", "box", "boxcar", "brt", "brthan", "bth", "cheb", "chebwin",
        "cosine", "dpss", "exponential", "flat", "flattop", "flt", "gauss", "gaussian", "general cosine",
        "general gauss", "general gaussian", "general hamming", "general_cosine", "general_gauss",
        "general_gaussian", "general_hamming", "ggs", "gss", "halfcosine", "ham", "hamm", "hamming", "han",
        "hann", "kaiser", "kaiser bessel derived", "kaiser_bessel_derived", "kbd", "ksr", "lanczos", "nut",
        "nutl", "nuttall", "ones", "par", "parz", "parzen", "poisson", "rect", "rectangular", "sinc",
        "taylor", "taylorwin", "tri", "triang", "triangle", "tuk", "tukey"};
    return names;
}

bool is_real_number(const py::handle& o) {
    static const py::object real = py::module_::import("numbers").attr("Real");
    return py::isinstance(o, real);
}

bool is_integral(const py::handle& o) {
    static const py::object integral = py::module_::import("numbers").attr("Integral");
    return py::isinstance(o, integral);
}

// scipy.signal.get_window's reading of `window` (after its Nx and fftbins
// checks): a float is a Kaiser beta; a name or (name, args...) tuple, with
// an optional _symmetric/_periodic suffix.
struct ParsedWindow {
    std::optional<flt::Window> window;
    std::optional<Deferred> error;
};

ParsedWindow parse_window(const py::object& window) {
    ParsedWindow out;
    auto fail = [&](PyObject* type, const std::string& msg) {
        out.error = Deferred{0, type, py::str(msg)};
        return out;
    };
    const std::string wrepr = "window=" + pyrepr(window);
    if (!py::isinstance<py::str>(window) && !py::isinstance<py::tuple>(window)) {
        PyObject* f = PyNumber_Float(window.ptr());
        if (!f) {
            PyErr_Clear();
            return fail(PyExc_ValueError, "Parameter " + wrepr + " must be a tuple, a string or a float!");
        }
        const double beta = PyFloat_AsDouble(f);
        Py_DECREF(f);
        out.window = flt::Window::kaiser(beta);
        return out;
    }
    std::string name;
    py::tuple args;
    if (py::isinstance<py::tuple>(window)) {
        const auto t = window.cast<py::tuple>();
        if (t.empty()) return fail(PyExc_IndexError, "tuple index out of range");
        if (!py::isinstance<py::str>(t[0])) return fail(PyExc_ValueError, "First tuple entry of parameter " + wrepr + " is not a str!");
        name = t[0].cast<std::string>();
        args = t[py::slice(1, static_cast<py::ssize_t>(t.size()), 1)];
    } else {
        name = window.cast<std::string>();
    }
    std::optional<bool> sym;
    auto ends_with = [&](const std::string& suffix) {
        return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (ends_with("_symmetric")) {
        sym = true;
        name.resize(name.size() - 10);
    } else if (ends_with("_periodic")) {
        sym = false;
        name.resize(name.size() - 9);
    }
    if (!scipy_window_names().count(name))
        return fail(PyExc_ValueError, "Invalid window name '" + name + "' in parameter " + wrepr + "!");
    flt::Window::Kind kind;
    if (name == "hamming" || name == "hamm" || name == "ham") kind = flt::Window::Kind::Hamming;
    else if (name == "hann" || name == "han") kind = flt::Window::Kind::Hann;
    else if (name == "kaiser" || name == "ksr") kind = flt::Window::Kind::Kaiser;
    else return fail(PyExc_NotImplementedError,
                     "sstvae_native.qrss.filters: window '" + name + "' is not ported (hamming, hann and kaiser are)");
    const bool has_args = kind == flt::Window::Kind::Kaiser;
    if (!args.empty() && !has_args)
        return fail(PyExc_ValueError, "'" + name + "' does not allow parameters, but " + wrepr + "!");
    if (args.empty() && has_args)
        return fail(PyExc_ValueError, "'" + name + "' must have parameters, but " + wrepr + "!");
    double beta = 0.0;
    if (has_args) {
        if (args.size() > 1) return fail(PyExc_TypeError, "kaiser() got multiple values for argument 'sym'");
        if (!is_real_number(args[0]))
            return fail(PyExc_TypeError, "kaiser beta must be a real number, not " + pyrepr(args[0]));
        beta = py::float_(args[0]).cast<double>();
    }
    out.window = flt::Window{kind, beta, sym};
    return out;
}

flt::Window window_or_raise(const py::object& window) {
    ParsedWindow p = parse_window(window);
    if (p.error) p.error->raise_now();
    return *p.window;
}

// numpy's np.ones(M) refuses a bool, which the windows reach for M <= 1.
std::string bool_length_error(const py::object& M) {
    return "expected a sequence of integers or a single integer, got '" + py::str(M).cast<std::string>() + "'";
}

// scipy.signal.windows' _len_guards, then numpy's refusal of a non-integer
// length (linspace/ones go through operator.index, and ones refuses a bool).
std::int64_t window_length(const py::object& M) {
    const py::object Mi = py::int_(M);
    if (M.not_equal(Mi) || Mi < py::int_(0))
        py_raise(PyExc_ValueError, "Window length M must be a non-negative integer");
    if (py::isinstance<py::bool_>(M)) py_raise(PyExc_TypeError, bool_length_error(M));
    if (!is_integral(M))
        py_raise(PyExc_TypeError, "'" + py::str(py::type::of(M).attr("__name__")).cast<std::string>() +
                                   "' object cannot be interpreted as an integer");
    return Mi.cast<std::int64_t>();
}

void no_xp(const py::object& xp, const py::object& device) {
    if (!xp.is_none() || !device.is_none())
        py_raise(PyExc_NotImplementedError, "sstvae_native.qrss.filters: only numpy (xp=None, device=None)");
}

// --- dtypes -------------------------------------------------------------------

enum class Kind { F32, F64, C64, C128 };

Kind kind_of(const py::dtype& dt) {
    switch (dt.char_()) {
        case 'f': return Kind::F32;
        case 'd': return Kind::F64;
        case 'F': return Kind::C64;
        case 'D': return Kind::C128;
        default:
            py_raise(PyExc_NotImplementedError,
                  "sstvae_native.qrss.filters: dtype " + pyrepr(dt) + " is not ported "
                  "(float32, float64, complex64 and complex128 are)");
    }
    throw std::logic_error("unreachable");
}

template <typename T>
T as_value(const py::object& v);
template <>
double as_value<double>(const py::object& v) { return py::float_(v).cast<double>(); }
template <>
float as_value<float>(const py::object& v) { return static_cast<float>(py::float_(v).cast<double>()); }
template <>
std::complex<double> as_value<std::complex<double>>(const py::object& v) { return v.cast<std::complex<double>>(); }
template <>
std::complex<float> as_value<std::complex<float>>(const py::object& v) {
    const auto c = v.cast<std::complex<double>>();
    return {static_cast<float>(c.real()), static_cast<float>(c.imag())};
}

// x.shape[axis], with Python's IndexError for an axis out of range (as
// scipy reads it), and the axis normalised.
std::pair<std::int64_t, std::int64_t> axis_len(const py::array& x, const py::object& axis) {
    const auto n = x.attr("shape").attr("__getitem__")(axis).cast<std::int64_t>();
    const auto a = floor_mod(axis.cast<std::int64_t>(), static_cast<std::int64_t>(x.ndim()));
    return {n, a};
}

// fn(row) -> std::vector<T> of out_len over every 1-D lane along `axis` of
// x (already of dtype T); the result has x's shape with out_len on axis.
template <typename T, typename Fn>
py::array along_axis(const py::array& x, std::int64_t axis, std::int64_t out_len, Fn&& fn) {
    auto np = numpy();
    const bool one_d = x.ndim() == 1;
    py::object moved = one_d ? py::object(x) : np.attr("moveaxis")(x, axis, -1);
    auto xa = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(moved);
    if (!xa) throw py::error_already_set();
    std::vector<py::ssize_t> shape(xa.shape(), xa.shape() + xa.ndim());
    const py::ssize_t n = shape.back();
    shape.back() = static_cast<py::ssize_t>(out_len);
    py::array_t<T> out(shape);
    std::size_t rows = 1;
    for (std::size_t i = 0; i + 1 < shape.size(); ++i) rows *= static_cast<std::size_t>(shape[i]);
    const T* src = xa.data();
    T* dst = out.mutable_data();
    {
        py::gil_scoped_release unlock;
        for (std::size_t r = 0; r < rows; ++r) {
            const std::vector<T> y = fn(std::span<const T>(src + r * n, static_cast<std::size_t>(n)));
            if (static_cast<std::int64_t>(y.size()) != out_len)
                throw std::logic_error("sstvae_native.qrss.filters: output length disagrees with its formula");
            std::copy(y.begin(), y.end(), dst + r * static_cast<std::size_t>(out_len));
        }
    }
    if (one_d) return std::move(out);
    return np.attr("ascontiguousarray")(np.attr("moveaxis")(out, -1, axis));
}

template <typename T>
std::vector<T> flat(const py::array& a) {
    auto c = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(a);
    if (!c) throw py::error_already_set();
    return std::vector<T>(c.data(), c.data() + c.size());
}

const char* const UPFIRDN_MODES[] = {"constant", "wrap", "edge", "smooth", "symmetric", "reflect",
                                     "antisymmetric", "antireflect", "line"};

bool is_upfirdn_mode(const std::string& m) {
    return std::find_if(std::begin(UPFIRDN_MODES), std::end(UPFIRDN_MODES),
                        [&](const char* s) { return m == s; }) != std::end(UPFIRDN_MODES);
}

// --- upfirdn ------------------------------------------------------------------

template <typename T>
py::array upfirdn_typed(const py::array& h, const py::array& x, std::int64_t up, std::int64_t down,
                        std::int64_t axis, std::int64_t out_len, const py::object& cval) {
    const std::vector<T> taps = flat<T>(h);
    const T cv = as_value<T>(cval);
    return along_axis<T>(x, axis, out_len, [&](std::span<const T> row) {
        return flt::upfirdn<T>(taps, row, up, down, cv);
    });
}

py::array upfirdn_py(const py::object& h_in, const py::object& x_in, const py::object& up_in,
                     const py::object& down_in, const py::object& axis_in, const py::object& mode,
                     const py::object& cval) {
    auto np = numpy();
    py::array x = np.attr("asarray")(x_in);
    py::array h = np.attr("asarray")(h_in);
    if (h.ndim() != 1 || h.size() == 0) py_raise(PyExc_ValueError, "h must be 1-D with non-zero length");
    const py::dtype out_t = np.attr("result_type")(h.dtype(), x.dtype(), np.attr("float32"));
    h = np.attr("asarray")(h, out_t);
    const auto up = py::int_(up_in).cast<std::int64_t>();
    const auto down = py::int_(down_in).cast<std::int64_t>();
    if (up < 1 || down < 1) py_raise(PyExc_ValueError, "Both up and down must be >= 1");
    const auto [n_in, axis] = axis_len(x, axis_in);
    const std::int64_t out_len = flt::upfirdn_output_len(h.size(), n_in, up, down);
    if (out_len < 0) py_raise(PyExc_ValueError, "negative dimensions are not allowed");
    const std::string m = mode.attr("lower")().cast<std::string>();
    if (!is_upfirdn_mode(m)) py_raise(PyExc_ValueError, "Unknown mode: " + m);
    if (m != "constant")
        py_raise(PyExc_NotImplementedError,
              "sstvae_native.qrss.filters.upfirdn: mode '" + m + "' is not ported (only 'constant')");
    const Kind k = kind_of(out_t);
    x = np.attr("asarray")(x, out_t);
    switch (k) {
        case Kind::F32: return upfirdn_typed<float>(h, x, up, down, axis, out_len, cval);
        case Kind::F64: return upfirdn_typed<double>(h, x, up, down, axis, out_len, cval);
        case Kind::C64: return upfirdn_typed<std::complex<float>>(h, x, up, down, axis, out_len, cval);
        case Kind::C128: return upfirdn_typed<std::complex<double>>(h, x, up, down, axis, out_len, cval);
    }
    throw std::logic_error("unreachable");
}

// --- firwin -------------------------------------------------------------------

// The stage of scipy's firwin at which each refusal happens, in order.
enum Stage { S_CUTOFF = 1, S_WIDTH, S_PASS_ZERO, S_NYQUIST, S_NX, S_WINDOW, S_WINDOW_LENGTH, S_NONE = 99 };

int stage_of(const std::string& msg) {
    auto starts = [&](const char* p) { return msg.rfind(p, 0) == 0; };
    if (starts("At least one cutoff") || starts("Invalid cutoff")) return S_CUTOFF;
    if (starts("cutoff must have")) return S_PASS_ZERO;
    if (starts("A filter with an even number")) return S_NYQUIST;
    if (starts("Parameter Nx=")) return S_NX;
    return S_WINDOW;
}

py::array_t<double> firwin_py(const py::object& numtaps, const py::object& cutoff, const py::object& width,
                              const py::object& window, const py::object& pass_zero, const py::object& scale,
                              const py::object& fs) {
    auto np = numpy();
    std::optional<double> fs_v;
    if (!fs.is_none()) {
        if (!np.attr("isscalar")(fs).cast<bool>())
            py_raise(PyExc_ValueError, "Sampling frequency fs must be a single scalar.");
        fs_v = py::float_(fs).cast<double>();
    }
    py::array c = np.attr("asarray")(cutoff, "dtype"_a = "float64");
    if (c.ndim() > 1) py_raise(PyExc_ValueError, "The cutoff argument must be at most one-dimensional.");
    const std::vector<double> cut = flat<double>(c);

    std::vector<Deferred> deferred;
    std::optional<double> width_v;
    if (!width.is_none()) {
        PyObject* f = PyNumber_Float(width.ptr());
        if (f) {
            width_v = PyFloat_AsDouble(f);
            Py_DECREF(f);
        } else {
            PyObject *type, *value, *tb;
            PyErr_Fetch(&type, &value, &tb);
            PyErr_NormalizeException(&type, &value, &tb);
            deferred.push_back({S_WIDTH, type, py::reinterpret_steal<py::object>(value)});
            Py_XDECREF(tb);
        }
    }

    flt::PassZero pz = flt::PassZero::Yes;
    if (pass_zero.is(py::bool_(true))) {
        pz = flt::PassZero::Yes;
    } else if (pass_zero.is(py::bool_(false))) {
        pz = flt::PassZero::No;
    } else if (py::isinstance<py::str>(pass_zero) && pass_zero.equal(py::str("lowpass"))) {
        pz = flt::PassZero::Lowpass;
    } else if (py::isinstance<py::str>(pass_zero) && pass_zero.equal(py::str("bandstop"))) {
        pz = flt::PassZero::Bandstop;
    } else if (py::isinstance<py::str>(pass_zero) && pass_zero.equal(py::str("highpass"))) {
        pz = flt::PassZero::Highpass;
    } else if (py::isinstance<py::str>(pass_zero) && pass_zero.equal(py::str("bandpass"))) {
        pz = flt::PassZero::Bandpass;
    } else {
        deferred.push_back({S_PASS_ZERO, PyExc_ValueError,
                            py::str("Parameter pass_zero=" + pyrepr(pass_zero) +
                                    " not in (True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop')")});
    }

    // numtaps: an Integral is used as is; anything else reaches get_window's
    // "not a positive integer" after the parity checks, so it stands in by
    // its parity until then.
    const std::string nx_msg = "Parameter Nx=" + pyrepr(numtaps) + " is not a positive integer";
    std::int64_t nt;
    if (is_integral(numtaps)) {
        nt = py::int_(numtaps).cast<std::int64_t>();
        if (py::isinstance<py::bool_>(numtaps) && nt > 0)   // True: the window's np.ones(True)
            deferred.push_back({S_WINDOW_LENGTH, PyExc_TypeError, py::str(bool_length_error(numtaps))});
    } else if (is_real_number(numtaps)) {
        nt = numtaps.attr("__mod__")(2).equal(py::int_(0)) ? 0 : 1;
        deferred.push_back({S_NX, PyExc_ValueError, py::str(nx_msg)});
    } else {
        py_raise(PyExc_TypeError, "firwin: numtaps must be a number, not " + pyrepr(numtaps));
    }

    // width replaces the window entirely, so a bad window only counts without it.
    flt::Window win = flt::Window::hamming();
    if (width.is_none()) {
        ParsedWindow p = parse_window(window);
        if (p.error) {
            p.error->stage = S_WINDOW;
            deferred.push_back(*p.error);
        } else {
            win = *p.window;
        }
    }
    const int truth = PyObject_IsTrue(scale.ptr());
    if (truth < 0) throw py::error_already_set();
    const bool sc = truth == 1;

    const Deferred* first = nullptr;
    for (const auto& d : deferred)
        if (!first || d.stage < first->stage) first = &d;

    std::vector<double> h;
    try {
        h = flt::firwin(nt, cut, win, pz, sc, fs_v, width_v);
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        const int s = stage_of(msg);
        if (first && first->stage <= s) first->raise_now();
        py_raise(PyExc_ValueError, s == S_NX ? nx_msg : msg);
    }
    if (first) first->raise_now();
    return to_numpy(h);
}

// --- resample_poly ------------------------------------------------------------

const char* const RESAMPLE_FUNCS[] = {"mean", "median", "minimum", "maximum"};

template <typename T>
py::array resample_designed(const py::array& x, std::int64_t up, std::int64_t down, std::int64_t axis,
                            std::int64_t n_out, const flt::Window& w, const py::object& cval) {
    const std::vector<T> h = flt::resample_poly_filter<T>(up, down, w);
    const std::int64_t half_len = 10 * std::max(up, down);
    const T cv = as_value<T>(cval);
    return along_axis<T>(x, axis, n_out, [&](std::span<const T> row) {
        return flt::resample_poly_taps<T>(row, up, down, h, half_len, cv);
    });
}

template <typename T>
py::array resample_given(const py::array& x, std::int64_t up, std::int64_t down, std::int64_t axis,
                         std::int64_t n_out, const py::array& h_arr, std::int64_t half_len,
                         const py::object& cval) {
    const std::vector<T> h = flat<T>(h_arr);
    const T cv = as_value<T>(cval);
    return along_axis<T>(x, axis, n_out, [&](std::span<const T> row) {
        return flt::resample_poly_taps<T>(row, up, down, h, half_len, cv);
    });
}

py::object resample_poly_py(const py::object& x_in, const py::object& up_in, const py::object& down_in,
                            const py::object& axis_in, const py::object& window, const py::object& padtype,
                            const py::object& cval_in) {
    auto np = numpy();
    py::array x = np.attr("asarray")(x_in);
    if (up_in.not_equal(py::int_(up_in))) py_raise(PyExc_ValueError, "up must be an integer");
    if (down_in.not_equal(py::int_(down_in))) py_raise(PyExc_ValueError, "down must be an integer");
    std::int64_t up = py::int_(up_in).cast<std::int64_t>();
    std::int64_t down = py::int_(down_in).cast<std::int64_t>();
    if (up < 1 || down < 1) py_raise(PyExc_ValueError, "up and down must be >= 1");
    if (!cval_in.is_none() && padtype.not_equal(py::str("constant"))) {
        PyErr_SetObject(PyExc_ValueError, py::make_tuple("cval has no effect when padtype is ", padtype).ptr());
        throw py::error_already_set();
    }
    const std::int64_t g = std::gcd(up, down);
    up /= g;
    down /= g;
    if (up == 1 && down == 1) return np.attr("array")(x, "copy"_a = true);
    const auto [n_in, axis] = axis_len(x, axis_in);
    std::int64_t n_out = n_in * up;
    n_out = n_out / down + (n_out % down != 0 ? 1 : 0);

    // The filter: an array is the taps; anything else designs one.
    const bool given = py::isinstance<py::list>(window) || py::isinstance<py::array>(window) ||
                       py::isinstance(window, np.attr("generic"));
    py::array h_given;
    std::int64_t half_len = 0;
    flt::Window w = flt::Window::kaiser(5.0);
    if (given) {
        h_given = np.attr("array")(window, "copy"_a = true);
        if (h_given.ndim() > 1) py_raise(PyExc_ValueError, "window must be 1-D");
        if (h_given.ndim() == 0) py_raise(PyExc_IndexError, "tuple index out of range");
        const auto size = static_cast<std::int64_t>(h_given.size());
        half_len = size >= 1 ? (size - 1) / 2 : -1;   // Python's (size - 1) // 2
        h_given = h_given.attr("__imul__")(up);
    } else {
        w = window_or_raise(window);
    }

    // padtype: only 'constant' is ported.
    const bool is_func = py::isinstance<py::str>(padtype) &&
                         std::any_of(std::begin(RESAMPLE_FUNCS), std::end(RESAMPLE_FUNCS),
                                     [&](const char* s) { return padtype.equal(py::str(s)); });
    const bool is_mode = py::isinstance<py::str>(padtype) && is_upfirdn_mode(padtype.cast<std::string>());
    if (!is_func && !is_mode) {
        std::string modes;
        for (const char* s : UPFIRDN_MODES) modes += (modes.empty() ? "" : ", ") + std::string(s);
        py_raise(PyExc_ValueError, "padtype must be one of: maximum, mean, median, minimum, " + modes);
    }
    if (is_func || padtype.cast<std::string>() != "constant")
        py_raise(PyExc_NotImplementedError, "sstvae_native.qrss.filters.resample_poly: padtype " + pyrepr(padtype) +
                                             " is not ported (only 'constant')");
    const py::object cval = cval_in.is_none() ? py::object(py::int_(0)) : cval_in;

    if (given) {
        const py::dtype out_t = np.attr("result_type")(h_given.dtype(), x.dtype(), np.attr("float32"));
        const Kind k = kind_of(out_t);
        h_given = np.attr("asarray")(h_given, out_t);
        x = np.attr("asarray")(x, out_t);
        switch (k) {
            case Kind::F32: return resample_given<float>(x, up, down, axis, n_out, h_given, half_len, cval);
            case Kind::F64: return resample_given<double>(x, up, down, axis, n_out, h_given, half_len, cval);
            case Kind::C64:
                return resample_given<std::complex<float>>(x, up, down, axis, n_out, h_given, half_len, cval);
            case Kind::C128:
                return resample_given<std::complex<double>>(x, up, down, axis, n_out, h_given, half_len, cval);
        }
    }
    // A designed filter takes x's dtype when x is floating; otherwise float64.
    const char kc = x.dtype().kind();
    const py::dtype h_t = (kc == 'f' || kc == 'c') ? x.dtype() : py::dtype("float64");
    const py::dtype out_t = np.attr("result_type")(h_t, x.dtype(), np.attr("float32"));
    if (!out_t.equal(h_t)) py_raise(PyExc_NotImplementedError, "sstvae_native.qrss.filters: dtype " + pyrepr(x.dtype()) +
                                                              " is not ported");
    const Kind k = kind_of(out_t);
    x = np.attr("asarray")(x, out_t);
    switch (k) {
        case Kind::F32: return resample_designed<float>(x, up, down, axis, n_out, w, cval);
        case Kind::F64: return resample_designed<double>(x, up, down, axis, n_out, w, cval);
        case Kind::C64: return resample_designed<std::complex<float>>(x, up, down, axis, n_out, w, cval);
        case Kind::C128: return resample_designed<std::complex<double>>(x, up, down, axis, n_out, w, cval);
    }
    throw std::logic_error("unreachable");
}

// --- i0, i0e ------------------------------------------------------------------

// A scipy.special ufunc with loops f->f and d->d (the float loop computes
// in double): a scalar or 0-d input gives a numpy scalar.
py::object special_ufunc(const py::object& x_in, double (*fn)(double), const char* name) {
    auto np = numpy();
    py::array x = np.attr("asarray")(x_in);
    const bool f32 = np.attr("can_cast")(x.dtype(), np.attr("float32"), "safe").cast<bool>();
    const bool f64 = np.attr("can_cast")(x.dtype(), np.attr("float64"), "safe").cast<bool>();
    if (!f32 && !f64)
        py_raise(PyExc_TypeError, std::string("ufunc '") + name +
                                   "' not supported for the input types, and the inputs could not be safely "
                                   "coerced to any supported types according to the casting rule ''safe''");
    const std::vector<double> v = flat<double>(x);
    std::vector<py::ssize_t> shape(x.shape(), x.shape() + x.ndim());
    if (f32) {
        py::array_t<float> out(shape);
        float* d = out.mutable_data();
        // v[i] holds a value float32 represents exactly (the input cast safely to it).
        for (std::size_t i = 0; i < v.size(); ++i) d[i] = static_cast<float>(fn(v[i]));
        if (x.ndim() == 0) return np.attr("float32")(out);
        return out;
    }
    py::array_t<double> out(shape);
    double* d = out.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = fn(v[i]);
    if (x.ndim() == 0) return np.attr("float64")(out);
    return out;
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("filters", "scipy filters as sstvae.qrss uses them (native/core/qrss/filters)");

    m.def("upfirdn", &upfirdn_py, py::arg("h"), py::arg("x"), py::arg("up") = 1, py::arg("down") = 1,
          py::arg("axis") = -1, py::arg("mode") = "constant", py::arg("cval") = 0);

    m.def("_output_len", [](std::int64_t len_h, std::int64_t in_len, std::int64_t up, std::int64_t down) {
        return flt::upfirdn_output_len(len_h, in_len, up, down);
    }, py::arg("len_h"), py::arg("in_len"), py::arg("up"), py::arg("down"));

    m.def("firwin", &firwin_py, py::arg("numtaps"), py::arg("cutoff"), py::kw_only(),
          py::arg("width") = py::none(), py::arg("window") = "hamming", py::arg("pass_zero") = true,
          py::arg("scale") = true, py::arg("fs") = py::none());

    m.def("kaiser_atten", [](std::int64_t numtaps, double width) { return flt::kaiser_atten(numtaps, width); },
          py::arg("numtaps"), py::arg("width"));
    m.def("kaiser_beta", [](double a) { return flt::kaiser_beta(a); }, py::arg("a"));

    m.def("resample_poly", &resample_poly_py, py::arg("x"), py::arg("up"), py::arg("down"), py::arg("axis") = 0,
          py::arg("window") = py::make_tuple("kaiser", 5.0), py::arg("padtype") = "constant",
          py::arg("cval") = py::none());

    m.def("get_window", [](const py::object& window, const py::object& Nx, const py::object& fftbins,
                           const py::object& xp, const py::object& device) {
        if (!(Nx > py::int_(0) && is_integral(Nx)))
            py_raise(PyExc_ValueError, "Parameter Nx=" + pyrepr(Nx) + " is not a positive integer");
        if (!py::isinstance<py::bool_>(fftbins))
            py_raise(PyExc_ValueError, "Parameter fftbins=" + pyrepr(fftbins) + " is not of type bool!");
        no_xp(xp, device);
        const flt::Window w = window_or_raise(window);
        if (py::isinstance<py::bool_>(Nx)) py_raise(PyExc_TypeError, bool_length_error(Nx));
        return to_numpy(flt::get_window(w, py::int_(Nx).cast<std::int64_t>(), fftbins.cast<bool>()));
    }, py::arg("window"), py::arg("Nx"), py::arg("fftbins") = true, py::kw_only(), py::arg("xp") = py::none(),
       py::arg("device") = py::none());

    m.def("hann", [](const py::object& M, bool sym, const py::object& xp, const py::object& device) {
        no_xp(xp, device);
        return to_numpy(flt::hann(window_length(M), sym));
    }, py::arg("M"), py::arg("sym") = true, py::kw_only(), py::arg("xp") = py::none(), py::arg("device") = py::none());
    m.def("hamming", [](const py::object& M, bool sym, const py::object& xp, const py::object& device) {
        no_xp(xp, device);
        return to_numpy(flt::hamming(window_length(M), sym));
    }, py::arg("M"), py::arg("sym") = true, py::kw_only(), py::arg("xp") = py::none(), py::arg("device") = py::none());
    m.def("kaiser", [](const py::object& M, double beta, bool sym, const py::object& xp, const py::object& device) {
        no_xp(xp, device);
        return to_numpy(flt::kaiser(window_length(M), beta, sym));
    }, py::arg("M"), py::arg("beta"), py::arg("sym") = true, py::kw_only(), py::arg("xp") = py::none(),
       py::arg("device") = py::none());

    m.def("i0", [](const py::object& x) { return special_ufunc(x, &flt::i0, "i0"); }, py::arg("x"));
    m.def("i0e", [](const py::object& x) { return special_ufunc(x, &flt::i0e, "i0e"); }, py::arg("x"));
}

const Registrar reg("filters", 20, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
