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
#include <pybind11/gil_safe_call_once.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
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

// A Python int as int64: OverflowError (Python's own) when it does not fit.
std::int64_t to_int64(const py::handle& pyint) {
    const long long v = PyLong_AsLongLong(pyint.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return static_cast<std::int64_t>(v);
}

// The result of a C-API number call (a new reference, or NULL with the error set).
py::object py_number(PyObject* r) {
    if (!r) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(r);
}

bool truth(const py::handle& o) {
    const int t = PyObject_IsTrue(o.ptr());
    if (t < 0) throw py::error_already_set();
    return t == 1;
}

// A Python exception decided now and raised later, so that an earlier
// check in scipy's order still wins.
struct Deferred {
    int stage = 0;   // see firwin_py
    py::object type;
    py::object value;
    [[noreturn]] void raise_now() const {
        PyErr_SetObject(type.ptr(), value.ptr());
        throw py::error_already_set();
    }
};

Deferred deferred_err(int stage, PyObject* type, const std::string& msg) {
    return {stage, py::reinterpret_borrow<py::object>(type), py::str(msg)};
}

// Runs fn; the Python exception it raises, if any, kept for later.
template <typename Fn>
std::optional<Deferred> capture(int stage, Fn&& fn) {
    try {
        fn();
    } catch (py::error_already_set& e) {
        return Deferred{stage, e.type(), e.value()};
    }
    return std::nullopt;
}

// numbers.Integral, looked up once. Held for the life
// of the process and never destroyed (a static py::object would be
// released after the interpreter has gone), and initialised under
// pybind11's GIL-aware once: an import can release the GIL, which a plain
// function-local static would turn into a deadlock between two threads.
const py::object& numbers_integral() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> store;
    return store.call_once_and_store_result([] { return py::module_::import("numbers").attr("Integral"); })
        .get_stored();
}

bool is_integral(const py::handle& o) { return py::isinstance(o, numbers_integral()); }

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

// Kaiser's beta as scipy.signal.windows.kaiser uses it:
// `special.i0(beta * sqrt(...)) / special.i0(asarray(beta, float64))`.
// Read only once the window is longer than one sample, since kaiser
// returns np.ones(M) for M <= 1 before it looks at beta. A Python float or
// int is the float64 it converts to. Anything else goes through numpy's
// own multiply by a float64 array, so a beta numpy cannot multiply (a
// str, a datetime, a Decimal) is refused with numpy's error; a product
// that is not float64 (complex, longdouble, object such as a Fraction,
// timedelta) has no i0 loop, and that TypeError is i0's. What multiplies
// to float64 is a float64 beta, read by asarray(beta, float64). A beta of
// one element (a 0-d array, a numpy scalar, [5.0]) makes the scalar's
// window; one numpy would broadcast over the window is not ported.
double kaiser_beta_value(const py::object& beta) {
    if (PyFloat_Check(beta.ptr()) || PyLong_Check(beta.ptr())) {
        const double v = PyFloat_AsDouble(beta.ptr());
        if (v == -1.0 && PyErr_Occurred()) throw py::error_already_set();
        return v;
    }
    auto np = py::module_::import("numpy");
    const py::object product = beta * np.attr("ones")(1);
    if (!py::isinstance<py::array>(product))
        py_raise(PyExc_TypeError, "kaiser beta must be a real number, not " + pyrepr(beta));
    const py::array p = py::reinterpret_borrow<py::array>(product);
    if (!p.dtype().equal(py::dtype::of<double>()))
        py_raise(PyExc_TypeError,
                 "ufunc 'i0' not supported for the input types, and the inputs could not be safely coerced to any "
                 "supported types according to the casting rule ''safe''");
    const py::array b = np.attr("asarray")(beta, "dtype"_a = "float64");
    if (p.ndim() != 1 || b.size() != 1)
        py_raise(PyExc_NotImplementedError, "sstvae_native.qrss.filters: a kaiser beta of shape " +
                                                pyrepr(b.attr("shape")) + " is not ported (one value is)");
    return b.attr("reshape")(-1).attr("__getitem__")(0).cast<double>();
}

// scipy.signal.get_window's reading of `window` (after its Nx and fftbins
// checks): a float is a Kaiser beta; a name or (name, args...) tuple, with
// an optional _symmetric/_periodic suffix. A Kaiser beta given in a tuple
// is kept as the object and read by `make` (kaiser reads it only when
// the window is longer than one sample).
struct ParsedWindow {
    std::optional<flt::Window> window;
    py::object beta;   // the tuple's beta, unread
    std::optional<Deferred> error;

    // The window for length M, reading beta now if the length needs it.
    flt::Window make(std::int64_t M) const {
        flt::Window w = *window;
        if (beta && M > 1) w.beta = kaiser_beta_value(beta);
        return w;
    }
};

ParsedWindow parse_window(const py::object& window) {
    ParsedWindow out;
    auto fail = [&](PyObject* type, const std::string& msg) {
        out.error = deferred_err(0, type, msg);
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
    if (has_args) {
        if (args.size() > 1) return fail(PyExc_TypeError, "kaiser() got multiple values for argument 'sym'");
        out.beta = args[0];
    }
    out.window = flt::Window{kind, 0.0, sym};
    return out;
}

ParsedWindow window_or_raise(const py::object& window) {
    ParsedWindow p = parse_window(window);
    if (p.error) p.error->raise_now();
    return p;
}

// numpy's np.ones(M) refusal of anything but an integer (bools included).
[[noreturn]] void ones_refuses(const py::handle& M) {
    py_raise(PyExc_TypeError, "expected a sequence of integers or a single integer, got '" + pyrepr(M) + "'");
}

bool is_bool_like(const py::handle& o) {
    if (PyBool_Check(o.ptr())) return true;
    auto np = py::module_::import("numpy");
    return py::isinstance(o, np.attr("bool_")) ||
           (py::isinstance<py::array>(o) && py::reinterpret_borrow<py::array>(o).dtype().kind() == 'b');
}

// scipy.signal.windows' window length: _len_guards(M), then np.ones(M) for
// M <= 1 (an integer, never a bool), else _extend's `not sym` and the
// length numpy reads, operator.index(M) for np.linspace (hann, hamming)
// or any real for np.arange (kaiser, whose float M equal to an integer
// makes the same window). `sym` is evaluated only for M > 1, as scipy does.
struct WindowLength {
    std::int64_t M;
    bool sym;
};

WindowLength window_length(const py::object& M, const py::object& sym, bool any_real) {
    const py::object Mi = py::int_(M);
    if (Mi.not_equal(M) || M < py::int_(0))
        py_raise(PyExc_ValueError, "Window length M must be a non-negative integer");
    if (M <= py::int_(1)) {
        if (is_bool_like(M)) ones_refuses(M);
        PyObject* i = PyNumber_Index(M.ptr());
        if (!i) {
            PyErr_Clear();
            ones_refuses(M);
        }
        return {to_int64(py::reinterpret_steal<py::object>(i)), true};
    }
    const bool s = truth(sym);
    // kaiser takes np.arange(0, M), which accepts any real; general_cosine
    // takes np.linspace(-pi, pi, M), whose operator.index sees _extend's
    // M + 1 for a periodic window (so np.array(5.0) is refused with the
    // text of the np.float64 that the addition makes of it).
    const py::object Mx = (any_real || s) ? M : py_number(PyNumber_Add(M.ptr(), py::int_(1).ptr()));
    const py::object n = any_real ? Mi : py_number(PyNumber_Index(Mx.ptr()));
    const long long v = PyLong_AsLongLong(n.ptr());
    if (v == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        py_raise(PyExc_ValueError, "Maximum allowed size exceeded");   // numpy's, for a length past int64
    }
    if (any_real || s) return {static_cast<std::int64_t>(v), s};
    // linspace's own refusal, reachable only through an M + 1 that wrapped
    if (v < 0)
        py_raise(PyExc_ValueError, ("Number of samples, " + std::to_string(v) + ", must be non-negative."));
    if (v == 0) return {0, true};   // np.uint64(2**64 - 1) + 1: an empty linspace, truncated
    return {static_cast<std::int64_t>(v - 1), false};
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

// cval as Cython converts it to the output type (_upfirdn_apply's DTYPE_t
// argument): PyFloat_AsDouble for a real type, PyComplex_AsCComplex for a
// complex one, each narrowed to float for the 32-bit types. So the errors
// are theirs ("must be real number, not str").
double c_double(const py::object& v) {
    const double d = PyFloat_AsDouble(v.ptr());
    if (d == -1.0 && PyErr_Occurred()) throw py::error_already_set();
    return d;
}

std::complex<double> c_complex(const py::object& v) {
    const Py_complex c = PyComplex_AsCComplex(v.ptr());
    if (c.real == -1.0 && PyErr_Occurred()) throw py::error_already_set();
    return {c.real, c.imag};
}

template <typename T>
T as_value(const py::object& v);
template <>
double as_value<double>(const py::object& v) { return c_double(v); }
template <>
float as_value<float>(const py::object& v) { return static_cast<float>(c_double(v)); }
template <>
std::complex<double> as_value<std::complex<double>>(const py::object& v) { return c_complex(v); }
template <>
std::complex<float> as_value<std::complex<float>>(const py::object& v) {
    const auto c = c_complex(v);
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

// The stage of scipy's firwin at which each refusal happens, in order:
// the cutoff checks; `float(width)` and kaiser_atten/kaiser_beta; pass_zero;
// the Nyquist test's `numtaps % 2 == 0`; alpha's `numtaps - 1`;
// np.arange(0, numtaps); get_window's Nx test; its window name; np.ones
// for a window of length one; kaiser's beta.
enum Stage {
    S_CUTOFF = 1, S_WIDTH, S_PASS_ZERO, S_NYQUIST, S_ALPHA, S_ARANGE, S_NX, S_WINDOW, S_WINDOW_LENGTH, S_BETA
};

int stage_of(const std::string& msg) {
    auto starts = [&](const char* p) { return msg.rfind(p, 0) == 0; };
    if (starts("At least one cutoff") || starts("Invalid cutoff")) return S_CUTOFF;
    if (starts("cutoff must have")) return S_PASS_ZERO;
    if (starts("A filter with an even number")) return S_NYQUIST;
    if (starts("Parameter Nx=")) return S_NX;
    return S_WINDOW;
}

// scipy's kaiser_atten(numtaps, width) and kaiser_beta(atten) in Python
// arithmetic, for a numtaps that is not an integer (whose own operators
// decide what happens).
void kaiser_atten_beta_py(const py::object& numtaps, double width) {
    const py::object a = py::float_(2.285) * (numtaps - py::int_(1)) * py::float_(3.141592653589793) *
                             py::float_(width) +
                         py::float_(7.95);
    if (a > py::int_(50)) {
        (void)(py::float_(0.1102) * (a - py::float_(8.7)));
    } else if (a > py::int_(21)) {
        const py::object d = a - py::int_(21);
        (void)(py::float_(0.5842) * py_number(PyNumber_Power(d.ptr(), py::float_(0.4).ptr(), Py_None)) +
               py::float_(0.07886) * d);
    }
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
    const double nyq = 0.5 * fs_v.value_or(2.0);
    py::array c = np.attr("asarray")(cutoff, "dtype"_a = "float64");
    if (c.ndim() > 1) py_raise(PyExc_ValueError, "The cutoff argument must be at most one-dimensional.");
    const std::vector<double> cut = flat<double>(c);

    std::vector<Deferred> deferred;
    auto defer = [&](std::optional<Deferred> d) {
        if (d) deferred.push_back(std::move(*d));
    };

    const bool integral = is_integral(numtaps);
    std::optional<double> width_v;
    if (!width.is_none()) {
        defer(capture(S_WIDTH, [&] { width_v = py::float_(width).cast<double>(); }));
        if (width_v && !integral)
            defer(capture(S_WIDTH, [&] { kaiser_atten_beta_py(numtaps, *width_v / nyq); }));
    }

    // pass_zero as scipy reads it: `in` the string pairs, then `is` True/False.
    flt::PassZero pz = flt::PassZero::Yes;
    auto pz_err = capture(S_PASS_ZERO, [&] {
        if (py::make_tuple("bandstop", "lowpass").contains(pass_zero)) {
            pz = pass_zero.equal(py::str("lowpass")) ? flt::PassZero::Lowpass : flt::PassZero::Bandstop;
        } else if (py::make_tuple("bandpass", "highpass").contains(pass_zero)) {
            pz = pass_zero.equal(py::str("highpass")) ? flt::PassZero::Highpass : flt::PassZero::Bandpass;
        } else if (pass_zero.is(py::bool_(true))) {
            pz = flt::PassZero::Yes;
        } else if (pass_zero.is(py::bool_(false))) {
            pz = flt::PassZero::No;
        } else {
            py_raise(PyExc_ValueError, "Parameter pass_zero=" + pyrepr(pass_zero) +
                                           " not in (True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop')");
        }
    });
    const bool pz_ok = !pz_err;
    defer(std::move(pz_err));
    const bool pz_true = pz == flt::PassZero::Yes || pz == flt::PassZero::Lowpass || pz == flt::PassZero::Bandstop;
    const bool pass_nyquist = pz_ok && ((cut.size() % 2 == 0) == pz_true);

    // numtaps. An integer is used as is (the core computes from it). Anything
    // else is carried through scipy's arithmetic until get_window refuses it
    // as Nx, and each operation on the way may refuse it first; the core
    // gets a stand-in of the same parity.
    const std::string nx_msg = "Parameter Nx=" + pyrepr(numtaps) + " is not a positive integer";
    std::int64_t nt = 1;
    if (integral) {
        const py::int_ ni(numtaps);
        const long long v = PyLong_AsLongLong(ni.ptr());
        if (v == -1 && PyErr_Occurred()) {
            // Past int64: scipy's np.arange(0, numtaps) refuses it.
            PyErr_Clear();
            const bool even = py_number(PyNumber_Remainder(ni.ptr(), py::int_(2).ptr())).equal(py::int_(0));
            nt = (ni < py::int_(0) ? -1 : 1) * (even ? 2 : 1);
            deferred.push_back(deferred_err(S_ARANGE, PyExc_ValueError, "Maximum allowed size exceeded"));
        } else {
            nt = static_cast<std::int64_t>(v);
        }
        if (PyBool_Check(numtaps.ptr()) && nt > 0)   // True: the window's np.ones(True)
            defer(capture(S_WINDOW_LENGTH, [&] { ones_refuses(numtaps); }));
    } else {
        if (pass_nyquist)
            defer(capture(S_NYQUIST, [&] {
                const py::object r = py_number(PyNumber_Remainder(numtaps.ptr(), py::int_(2).ptr()));
                nt = r.equal(py::int_(0)) ? 2 : 1;
            }));
        defer(capture(S_ALPHA, [&] { (void)(py::float_(0.5) * (numtaps - py::int_(1))); }));
        defer(capture(S_ARANGE, [&] {
            // np.arange(0, numtaps) for its refusals (NaN, inf, a complex, a
            // length past int64), but not the array a huge finite numtaps would
            // allocate: scipy would allocate it and then refuse numtaps as Nx.
            PyObject* f = PyNumber_Float(numtaps.ptr());
            if (f) {
                const double v = PyFloat_AsDouble(f);
                Py_DECREF(f);
                if (std::isfinite(v) && std::fabs(v) >= 0x1p63)
                    py_raise(PyExc_ValueError, "Maximum allowed size exceeded");
                if (std::isfinite(v) && std::fabs(v) > 1e7) return;
            } else {
                PyErr_Clear();
            }
            (void)np.attr("arange")(0, numtaps, "dtype"_a = "float64");
        }));
        // get_window: `Nx > 0 and isinstance(Nx, numbers.Integral)`
        auto cmp = capture(S_NX, [&] { (void)(numtaps > py::int_(0)); });
        deferred.push_back(cmp ? *cmp : deferred_err(S_NX, PyExc_ValueError, nx_msg));
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
            if (p.beta && integral && nt > 1)
                defer(capture(S_BETA, [&] { win.beta = kaiser_beta_value(p.beta); }));
        }
    }
    const bool sc = truth(scale);

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
        w = window_or_raise(window).make(2 * 10 * std::max(up, down) + 1);
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
// in double): a scalar or 0-d input gives a numpy scalar, and an array the
// memory layout a ufunc allocates (np.empty_like's 'K': an F-ordered input
// gives an F-ordered output).
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
    py::array c_order;
    if (f32) {
        py::array_t<float> out(shape);
        float* d = out.mutable_data();
        // v[i] holds a value float32 represents exactly (the input cast safely to it).
        for (std::size_t i = 0; i < v.size(); ++i) d[i] = static_cast<float>(fn(v[i]));
        c_order = out;
    } else {
        py::array_t<double> out(shape);
        double* d = out.mutable_data();
        for (std::size_t i = 0; i < v.size(); ++i) d[i] = fn(v[i]);
        c_order = out;
    }
    if (x.ndim() == 0) return np.attr(f32 ? "float32" : "float64")(c_order);
    if (x.ndim() == 1 || (x.flags() & py::array::c_style)) return c_order;
    py::object out = np.attr("empty_like")(x, "dtype"_a = c_order.dtype());
    np.attr("copyto")(out, c_order);
    return out;
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("filters", "scipy filters as sstvae.qrss uses them (native/core/qrss/filters)");

    m.def("upfirdn", &upfirdn_py, py::arg("h"), py::arg("x"), py::arg("up") = 1, py::arg("down") = 1,
          py::arg("axis") = -1, py::arg("mode") = "constant", py::arg("cval") = 0);

    // Cython's int64 floor division: ZeroDivisionError and OverflowError, not a trap.
    m.def("_output_len", [](std::int64_t len_h, std::int64_t in_len, std::int64_t up, std::int64_t down) {
        if (down == 0) py_raise(PyExc_ZeroDivisionError, "integer division or modulo by zero");
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
        const ParsedWindow p = window_or_raise(window);
        const bool sym = p.window->sym.value_or(!fftbins.cast<bool>());
        const WindowLength L = window_length(Nx, py::bool_(sym), p.window->kind == flt::Window::Kind::Kaiser);
        return to_numpy(flt::get_window(p.make(L.M), L.M, fftbins.cast<bool>()));
    }, py::arg("window"), py::arg("Nx"), py::arg("fftbins") = true, py::kw_only(), py::arg("xp") = py::none(),
       py::arg("device") = py::none());

    m.def("hann", [](const py::object& M, const py::object& sym, const py::object& xp, const py::object& device) {
        no_xp(xp, device);
        const WindowLength L = window_length(M, sym, false);
        return to_numpy(flt::hann(L.M, L.sym));
    }, py::arg("M"), py::arg("sym") = true, py::kw_only(), py::arg("xp") = py::none(), py::arg("device") = py::none());
    m.def("hamming", [](const py::object& M, const py::object& sym, const py::object& xp, const py::object& device) {
        no_xp(xp, device);
        const WindowLength L = window_length(M, sym, false);
        return to_numpy(flt::hamming(L.M, L.sym));
    }, py::arg("M"), py::arg("sym") = true, py::kw_only(), py::arg("xp") = py::none(), py::arg("device") = py::none());
    m.def("kaiser", [](const py::object& M, const py::object& beta, const py::object& sym, const py::object& xp,
                       const py::object& device) {
        no_xp(xp, device);
        const WindowLength L = window_length(M, sym, true);
        return to_numpy(flt::kaiser(L.M, L.M > 1 ? kaiser_beta_value(beta) : 0.0, L.sym));
    }, py::arg("M"), py::arg("beta"), py::arg("sym") = true, py::kw_only(), py::arg("xp") = py::none(),
       py::arg("device") = py::none());

    m.def("i0", [](const py::object& x) { return special_ufunc(x, &flt::i0, "i0"); }, py::arg("x"));
    m.def("i0e", [](const py::object& x) { return special_ufunc(x, &flt::i0e, "i0e"); }, py::arg("x"));
}

const Registrar reg("filters", 20, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
