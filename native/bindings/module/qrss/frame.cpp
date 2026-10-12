// sstvae_native.qrss.frame and sstvae_native.qrss.lframe: bindings of
// native/core/qrss/frame (CE and L in one FrameSpec type). Function names,
// argument names, dtypes, read-only flags and ValueError texts follow
// sstvae/qrss/frame.py and sstvae/qrss/lframe.py, one submodule each.
//
// A `spec` argument is a native FrameSpec or anything with the reference
// dataclass's fields (waveform, name, n_pre, n_hdr, n_data, cw_after,
// hdr_rho), so the reference's own FrameSpec/LFrameSpec objects -- presets
// or frames a test built -- pass straight through. tests/qrss_native/frame.py
// turns what comes back (native specs, layouts as dicts) into the
// reference's dataclasses where Python needs them.

#include "qrss/registry.hpp"

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/frame.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;

py::module_ numpy() {
    static py::module_ np = py::module_::import("numpy");
    return np;
}

// A frame name between Python and the C++: UTF-8 with lone surrogates passed
// through, so any str the reference takes as a name survives the round trip
// (and no preset name, all ASCII, can match one that has them).
std::string utf8_of(const py::handle& s) {
    return std::string(py::bytes(s.attr("encode")("utf-8", "surrogatepass")));
}

py::str str_of(std::string_view s) {
    PyObject* o = PyUnicode_DecodeUTF8(s.data(), static_cast<py::ssize_t>(s.size()), "surrogatepass");
    if (!o) throw py::error_already_set();
    return py::reinterpret_steal<py::str>(o);
}

// Raise `type` with a C++ message that may carry such a name.
[[noreturn]] void raise_text(PyObject* type, std::string_view what) {
    PyErr_SetObject(type, str_of(what).ptr());
    throw py::error_already_set();
}

// operator.index(h) as int64 (numpy ints included; OverflowError past int64).
std::int64_t as_index(const py::handle& h) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Index(h.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

// int(h) as int64: what FrameSpec.__post_init__ does to each cw_after entry.
std::int64_t as_int(const py::handle& h) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Long(h.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

fr::Waveform as_waveform(const py::handle& h) {
    const std::int64_t w = as_index(h);
    if (w != cfg::WAVEFORM_CE && w != cfg::WAVEFORM_L)
        throw py::value_error("waveform must be " + std::to_string(cfg::WAVEFORM_CE) + " (CE) or " +
                              std::to_string(cfg::WAVEFORM_L) + " (L), not " + std::to_string(w));
    return static_cast<fr::Waveform>(w);
}

fr::FrameSpec make(const py::handle& waveform, const py::handle& name, const py::handle& n_pre,
                   const py::handle& n_hdr, const py::handle& n_data, const py::handle& cw_after,
                   const py::handle& hdr_rho) {
    std::vector<std::int64_t> cw;
    for (const py::handle c : py::reinterpret_borrow<py::iterable>(cw_after)) cw.push_back(as_int(c));
    std::optional<double> rho;
    if (!hdr_rho.is_none()) {
        py::object f = py::reinterpret_steal<py::object>(PyNumber_Float(hdr_rho.ptr()));  // float(rho)
        if (!f) throw py::error_already_set();
        rho = PyFloat_AsDouble(f.ptr());
    }
    const fr::Waveform w = as_waveform(waveform);
    std::string nm = utf8_of(py::str(name));   // f"{self.name}" in the refusals
    const std::int64_t pre = as_index(n_pre), hdr = as_index(n_hdr), data = as_index(n_data);
    try {
        return fr::FrameSpec::make(w, std::move(nm), pre, hdr, data, std::move(cw), rho);
    } catch (const std::invalid_argument& e) {
        raise_text(PyExc_ValueError, e.what());
    }
}

// A native FrameSpec, or the reference's (any object with its fields).
fr::FrameSpec to_spec(const py::handle& obj) {
    if (py::isinstance<fr::FrameSpec>(obj)) return obj.cast<fr::FrameSpec>();
    return make(obj.attr("waveform"), obj.attr("name"), obj.attr("n_pre"), obj.attr("n_hdr"),
                obj.attr("n_data"), obj.attr("cw_after"), obj.attr("hdr_rho"));
}

std::vector<std::int64_t> shape_of(const py::array& a) {
    return std::vector<std::int64_t>(a.shape(), a.shape() + a.ndim());
}

const fr::HdrBits* ptr(const std::optional<fr::HdrBits>& b) { return b ? &*b : nullptr; }

// np.asarray(x, dtype=np.float64), C-contiguous.
py::array_t<double> float64_array(const py::object& obj) {
    py::object a = numpy().attr("asarray")(obj, py::arg("dtype") = "float64");
    auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
    if (!v) throw py::error_already_set();
    return v;
}

// hdr_bits as spread_signs reads them, np.asarray(hdr_bits, dtype=np.float64),
// so a conversion fails with numpy's own error. None stays None: numpy makes
// it a 0-d nan, which the C++ refuses with the same text.
std::optional<fr::HdrBits> bits_f64(const py::object& obj) {
    if (obj.is_none()) return std::nullopt;
    const auto a = float64_array(obj);
    fr::HdrBits b;
    b.one_d = a.ndim() == 1;
    b.len = a.ndim() >= 1 ? static_cast<std::size_t>(a.shape(0)) : 1;
    if (b.one_d) b.values.assign(a.data(), a.data() + a.size());
    return b;
}

// {spec.name!r}: the repr of the spec object's own name (a reference spec may
// carry a name that is not a str; a native one's is).
std::string name_repr(const py::object& spec) { return py::repr(spec.attr("name")).cast<std::string>(); }

// assemble's header checks, made on hdr_bits as given, as the reference makes
// them: np.asarray(hdr_bits) and its own (b != 0) & (b != 1) when the frame
// has a header -- so a dtype whose values would only become 0/1 on the way to
// float64 (strings, complex with an imaginary part, long double) is refused
// as Python refuses it -- and len(hdr_bits) when it has none. Then the bits
// the C++ takes: their float64 values with a header, None without.
std::optional<fr::HdrBits> assemble_bits(const fr::FrameSpec& s, const py::object& spec, const py::object& obj) {
    if (!s.has_header()) {
        if (!obj.is_none() && py::len(obj) != 0)
            throw py::value_error("frame " + name_repr(spec) + " has no header but hdr_bits were given");
        return std::nullopt;
    }
    if (obj.is_none()) throw py::value_error("frame " + name_repr(spec) + " has a header: hdr_bits is required");
    py::array b = numpy().attr("asarray")(obj);
    bool ok = b.ndim() == 1 && b.shape(0) == cfg::N_HDR_BITS;
    if (ok) {
        // np.any((b != 0) & (b != 1)), with Python's operators and truth value
        const auto ne = [&](int v) {
            py::object r = py::reinterpret_steal<py::object>(PyObject_RichCompare(b.ptr(), py::int_(v).ptr(), Py_NE));
            if (!r) throw py::error_already_set();
            return r;
        };
        ok = !static_cast<bool>(py::bool_(numpy().attr("any")(ne(0) & ne(1))));
    }
    if (!ok) throw py::value_error("hdr_bits must be " + std::to_string(cfg::N_HDR_BITS) + " values of 0/1");
    return bits_f64(obj);
}

template <typename T>
py::array_t<T> shaped(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
    py::array_t<T> out(shape);
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

py::array_t<std::int64_t> ro(const std::vector<std::int64_t>& v) { return to_numpy(v, false); }

py::dict layout_dict(const fr::Layout& lay) {
    py::dict d;
    d["pre"] = ro(lay.pre);
    d["hdr_ref"] = ro(lay.hdr_ref);
    if (lay.waveform == fr::Waveform::L) {
        d["hdr_sym"] = ro(lay.hdr_sym);
    } else {
        d["hdr_bits"] = ro(lay.hdr_bits);
        d["hdr_spare"] = ro(lay.hdr_spare);
    }
    d["data_ref"] = ro(lay.data_ref);
    d["data"] = ro(lay.data);
    d["ref"] = ro(lay.ref);
    d["known_idx"] = ro(lay.known_idx);
    if (lay.waveform == fr::Waveform::L)
        d["known_val"] = to_numpy(lay.known_val_l, false);
    else
        d["known_val"] = to_numpy(lay.known_val_ce, false);
    d["pos"] = ro(lay.pos);
    py::array_t<std::int64_t> win({static_cast<py::ssize_t>(lay.win_pos.size()), py::ssize_t(2)});
    for (std::size_t w = 0; w < lay.win_pos.size(); ++w) {
        win.mutable_at(static_cast<py::ssize_t>(w), 0) = lay.win_pos[w][0];
        win.mutable_at(static_cast<py::ssize_t>(w), 1) = lay.win_pos[w][1];
    }
    win.attr("setflags")(py::arg("write") = false);
    d["win_pos"] = win;
    py::array_t<bool> is_ref(static_cast<py::ssize_t>(lay.is_ref.size()));
    for (std::size_t k = 0; k < lay.is_ref.size(); ++k) is_ref.mutable_data()[k] = lay.is_ref[k] != 0;
    is_ref.attr("setflags")(py::arg("write") = false);
    d["is_ref"] = is_ref;
    return d;
}

// y[..., idx] for any plain numeric dtype: a gather of fixed-size items.
template <std::size_t N>
struct Item {
    unsigned char b[N];
};

template <std::size_t N>
py::array gather_items(const py::array& y, std::size_t row_len, const std::vector<std::int64_t>& idx) {
    const auto* p = static_cast<const Item<N>*>(y.data());
    const auto g = fr::gather(std::span<const Item<N>>(p, static_cast<std::size_t>(y.size())), row_len, idx);
    std::vector<py::ssize_t> shape(y.shape(), y.shape() + y.ndim());
    shape.back() = static_cast<py::ssize_t>(idx.size());
    py::array out(y.dtype(), shape);
    if (!g.empty()) std::memcpy(out.mutable_data(), g.data(), g.size() * N);
    return out;
}

// y[..., idx]: the C++ gather for numeric items of 1-16 bytes, and numpy's own
// indexing for anything else (objects, strings, datetimes, long double
// complex), which the reference gathers just the same.
py::array gather_any(const py::array& y, std::size_t row_len, const std::vector<std::int64_t>& idx) {
    const char kind = y.dtype().kind();
    if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f' || kind == 'c') {
        switch (y.itemsize()) {
            case 1: return gather_items<1>(y, row_len, idx);
            case 2: return gather_items<2>(y, row_len, idx);
            case 4: return gather_items<4>(y, row_len, idx);
            case 8: return gather_items<8>(y, row_len, idx);
            case 16: return gather_items<16>(y, row_len, idx);
            default: break;
        }
    }
    return y[py::make_tuple(py::ellipsis(), to_numpy(idx))].cast<py::array>();
}

// np.asarray(y) with at least one axis, C-contiguous; the last axis checked.
py::array symbol_rows(const py::object& y_sym, const fr::FrameSpec& spec) {
    py::array a = numpy().attr("asarray")(y_sym);
    if (a.ndim() == 0) throw py::index_error("tuple index out of range");   // y.shape[-1]
    fr::check_y_symbols(spec, a.shape(a.ndim() - 1));
    return numpy().attr("ascontiguousarray")(a);
}

std::vector<py::ssize_t> with_last(const py::array& y, py::ssize_t last) {
    std::vector<py::ssize_t> shape(y.shape(), y.shape() + y.ndim());
    shape.back() = last;
    return shape;
}

// lframe.unpair on np.asarray(z). The C++ takes every dtype that becomes
// complex128 exactly as numpy's float64 arithmetic would take it (real dtypes
// get +0 imaginary parts, as z.imag is zeros); for the rest -- long double,
// whose products numpy rounds once in extended precision, objects, and the
// non-numeric dtypes numpy refuses with its own error -- numpy runs the
// reference's own arithmetic.
py::array unpair_any(const py::object& obj) {
    py::array z = numpy().attr("asarray")(obj);
    if (z.ndim() == 0) throw py::index_error("tuple index out of range");   // z.shape[-1]
    const auto shape = with_last(z, 2 * z.shape(z.ndim() - 1));
    const char kind = z.dtype().kind();
    const py::ssize_t size = z.itemsize();
    if (kind == 'b' || kind == 'i' || kind == 'u' || (kind == 'f' && size <= 8) || (kind == 'c' && size <= 16)) {
        auto c = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>::ensure(z);
        if (!c) throw py::error_already_set();
        return shaped(fr::unpair(std::span<const std::complex<double>>(c.data(), static_cast<std::size_t>(c.size()))),
                      shape);
    }
    py::array out(py::dtype("float64"), shape);
    const py::object s = numpy().attr("sqrt")(2.0);
    out[py::make_tuple(py::ellipsis(), py::slice(py::int_(0), py::none(), py::int_(2)))] = s * z.attr("real");
    out[py::make_tuple(py::ellipsis(), py::slice(py::int_(1), py::none(), py::int_(2)))] = s * z.attr("imag");
    return out;
}

py::object keyed_end(const fr::FrameSpec& s) {
    if (s.last_window_ends_frame()) return py::float_(s.keyed_end_pos());
    return py::int_(static_cast<std::int64_t>(s.keyed_end_pos()));
}

py::tuple int_tuple(const std::vector<std::int64_t>& v) {
    py::tuple t(v.size());
    for (std::size_t k = 0; k < v.size(); ++k) t[k] = py::int_(v[k]);
    return t;
}

std::string spec_repr(const fr::FrameSpec& s) {
    const char* cls = s.waveform() == fr::Waveform::L ? "LFrameSpec" : "FrameSpec";
    return std::string(cls) + "(name=" + py::repr(str_of(s.name())).cast<std::string>() +
           ", n_pre=" + std::to_string(s.n_pre()) +
           ", n_hdr=" + std::to_string(s.n_hdr()) + ", n_data=" + std::to_string(s.n_data()) +
           ", cw_after=" + py::repr(int_tuple(s.cw_after())).cast<std::string>() +
           ", hdr_rho=" + fr::py_repr(s.hdr_rho()) + ")";
}

py::dict preset_dict(cfg::PresetGroup g) {
    py::dict d;
    for (const fr::FrameSpec& s : fr::presets(g)) d[py::str(s.name())] = s;
    return d;
}

// The unknown-name refusal with the name shown by Python's own repr (the
// C++'s escapes ASCII and Latin-1 exactly but takes every other code point
// as printable; Python consults the Unicode database).
[[noreturn]] void unknown(const py::handle& name, fr::Waveform w) {
    throw py::value_error(fr::unknown_frame(py::repr(name).cast<std::string>(), w));
}

fr::Waveform waveform_of_name(const py::str& name) {
    return name.attr("startswith")("l-").cast<bool>() ? fr::Waveform::L : fr::Waveform::CE;
}

// get(name_or_spec) for one waveform's table: a native spec of that waveform
// passes through; any other non-string is refused as unknown, shown by repr.
fr::FrameSpec get_in(const py::object& obj, fr::Waveform w) {
    if (py::isinstance<fr::FrameSpec>(obj) && obj.cast<const fr::FrameSpec&>().waveform() == w)
        return obj.cast<fr::FrameSpec>();
    if (!py::isinstance<py::str>(obj)) {
        (void)py::hash(obj);   // `name_or_spec in PRESETS`: an unhashable one is a TypeError
        unknown(obj, w);
    }
    const std::string name = utf8_of(obj);
    try {
        return w == fr::Waveform::L ? fr::get_l(name) : fr::get(name);
    } catch (const std::invalid_argument&) {
        unknown(obj, w);
    }
}

void translate_wrong_waveform() {
    py::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const fr::WrongWaveform& e) {
            const py::str text = str_of(e.what());
            PyErr_SetObject(PyExc_AssertionError, text.ptr());
        }
    });
}

void bind_spec_class(py::module_& m) {
    py::class_<fr::FrameSpec>(m, "FrameSpec",
                              "One frame shape of either waveform (frame.FrameSpec / lframe.LFrameSpec); "
                              "build with make() or of().")
        .def_property_readonly("waveform", [](const fr::FrameSpec& s) { return static_cast<int>(s.waveform()); })
        .def_property_readonly("name", [](const fr::FrameSpec& s) { return str_of(s.name()); })
        .def_property_readonly("n_pre", &fr::FrameSpec::n_pre)
        .def_property_readonly("n_hdr", &fr::FrameSpec::n_hdr)
        .def_property_readonly("n_data", &fr::FrameSpec::n_data)
        .def_property_readonly("cw_after", [](const fr::FrameSpec& s) { return int_tuple(s.cw_after()); })
        .def_property_readonly("hdr_rho", &fr::FrameSpec::hdr_rho)
        .def_property_readonly("t_sym", &fr::FrameSpec::t_sym)
        .def_property_readonly("cw_window", &fr::FrameSpec::cw_window)
        .def_property_readonly("has_header", &fr::FrameSpec::has_header)
        .def_property_readonly("n_data_payload", &fr::FrameSpec::n_data_payload)
        .def_property_readonly("n_data_sym", &fr::FrameSpec::n_data_sym)
        .def_property_readonly("n_sym", &fr::FrameSpec::n_sym)
        .def_property_readonly("n_win", &fr::FrameSpec::n_win)
        .def_property_readonly("n_pos", &fr::FrameSpec::n_pos)
        .def_property_readonly("n_ref", &fr::FrameSpec::n_ref)
        .def_property_readonly("duration_s", &fr::FrameSpec::duration_s)
        .def_property_readonly("win_start_pos", [](const fr::FrameSpec& s) { return int_tuple(s.win_start_pos()); })
        .def_property_readonly("keyed_end_pos", &keyed_end)
        .def_property_readonly("keyed_start_pos", &fr::FrameSpec::keyed_start_pos)
        .def("pos_of", [](const fr::FrameSpec& s, const py::object& sv) -> py::object {
            // np.asarray(s, dtype=np.int64); an int for a scalar
            py::object a0 = numpy().attr("asarray")(sv, py::arg("dtype") = "int64");
            auto a = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(a0);
            if (!a) throw py::error_already_set();
            std::vector<std::int64_t> out(static_cast<std::size_t>(a.size()));
            for (std::size_t k = 0; k < out.size(); ++k) out[k] = s.pos_of(a.data()[k]);
            if (a.ndim() == 0) return py::int_(out[0]);
            return shaped(out, std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim()));
        }, py::arg("s"))
        .def("time_s", [](const fr::FrameSpec& s, const py::object& pos) -> py::object {
            // np.asarray(pos, dtype=np.float64) * t_sym: a numpy scalar for a scalar
            const auto a = float64_array(pos);
            std::vector<double> out(static_cast<std::size_t>(a.size()));
            for (std::size_t k = 0; k < out.size(); ++k) out[k] = s.time_s(a.data()[k]);
            py::array r = shaped(out, std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim()));
            if (a.ndim() == 0) return r[py::tuple()];
            return r;
        }, py::arg("pos"))
        .def("window_spans_s", [](const fr::FrameSpec& s) {
            py::list out;
            for (const auto& [a, b] : s.window_spans_s()) out.append(py::make_tuple(a, b));
            return out;
        })
        .def("fields", [](const fr::FrameSpec& s) {
            return py::make_tuple(static_cast<int>(s.waveform()), str_of(s.name()), s.n_pre(), s.n_hdr(), s.n_data(),
                                  int_tuple(s.cw_after()), s.hdr_rho());
        }, "(waveform, name, n_pre, n_hdr, n_data, cw_after, hdr_rho)")
        .def("__eq__", [](const fr::FrameSpec& a, const py::object& b) -> py::object {
            if (!py::isinstance<fr::FrameSpec>(b)) return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            return py::bool_(a == b.cast<const fr::FrameSpec&>());
        })
        .def("__hash__", [](const fr::FrameSpec& s) {
            return py::hash(py::make_tuple(static_cast<int>(s.waveform()), str_of(s.name()), s.n_pre(), s.n_hdr(),
                                           s.n_data(), int_tuple(s.cw_after()), s.hdr_rho()));
        })
        .def("__repr__", &spec_repr);

    m.def("make", [](const py::object& waveform, const py::object& name, const py::object& n_pre,
                     const py::object& n_hdr, const py::object& n_data, const py::object& cw_after,
                     const py::object& hdr_rho) {
        return make(waveform, name, n_pre, n_hdr, n_data, cw_after, hdr_rho);
    }, py::arg("waveform"), py::arg("name"), py::arg("n_pre"), py::arg("n_hdr"), py::arg("n_data"),
       py::arg("cw_after"), py::arg("hdr_rho") = py::none(),
       "FrameSpec(...) / LFrameSpec(...) of `waveform`, validated as __post_init__ does");
    m.def("of", [](const py::object& spec) { return to_spec(spec); }, py::arg("spec"),
          "A native FrameSpec from the reference's FrameSpec or LFrameSpec (or a native one)");
}

void bind_frame(py::module_& q) {
    py::module_ m = q.def_submodule("frame", "Port of sstvae.qrss.frame (CE); FrameSpec covers both waveforms");
    translate_wrong_waveform();
    bind_spec_class(m);

    m.def("_data_symbols", [](const py::object& n) { return fr::data_symbols(as_index(n)); }, py::arg("n_data"));
    m.def("get", [](const py::object& name_or_spec) { return get_in(name_or_spec, fr::Waveform::CE); },
          py::arg("name_or_spec"));
    m.def("plain", [](const py::object& spec) { return fr::plain(to_spec(spec)); }, py::arg("spec"));
    m.def("layout", [](const py::object& spec) {
        const fr::FrameSpec s = to_spec(spec);
        fr::require_waveform(s, fr::Waveform::CE, "frame.layout");
        return layout_dict(*fr::layout(s));
    }, py::arg("spec"), "frame.layout's fields as a dict of read-only arrays");
    m.def("spread_signs", [](const py::object& spec, const py::object& hdr_bits) {
        const fr::FrameSpec s = to_spec(spec);
        const auto b = bits_f64(hdr_bits);
        return to_numpy(fr::spread_signs(s, ptr(b)));
    }, py::arg("spec"), py::arg("hdr_bits"));
    m.def("spread_symbols", [](const py::object& spec, const py::object& hdr_bits) {
        const fr::FrameSpec s = to_spec(spec);
        const auto b = bits_f64(hdr_bits);
        return to_numpy(fr::spread_symbols(s, ptr(b)));
    }, py::arg("spec"), py::arg("hdr_bits"));
    m.def("assemble", [](const py::object& spec, const py::object& hdr_bits, const py::object& x) {
        const fr::FrameSpec s = to_spec(spec);
        fr::require_waveform(s, fr::Waveform::CE, "frame.assemble");
        const auto xa = float64_array(x);   // x first, then the header (the reference's order)
        fr::check_x_shape(s, shape_of(xa));
        const auto b = assemble_bits(s, spec, hdr_bits);
        return to_numpy(fr::assemble(s, ptr(b), std::span<const double>(xa.data(), static_cast<std::size_t>(xa.size()))));
    }, py::arg("spec"), py::arg("hdr_bits"), py::arg("x"));
    m.def("disassemble", [](const py::object& y_sym, const py::object& spec) {
        const fr::FrameSpec s = to_spec(spec);
        fr::require_waveform(s, fr::Waveform::CE, "frame.disassemble");
        const py::array y = symbol_rows(y_sym, s);
        const auto lay = fr::layout(s);
        const auto n = static_cast<std::size_t>(s.n_sym());
        return py::make_tuple(gather_any(y, n, lay->hdr_bits), gather_any(y, n, lay->data));
    }, py::arg("y_sym"), py::arg("spec"));

    // The refusal texts' formatting, exposed so the parity test can hold it
    // against Python's repr directly.
    m.def("_repr_float", [](double v) { return fr::py_repr(v); }, py::arg("v"));
    m.def("_repr_str", [](const std::string& s) { return fr::py_repr(std::string_view(s)); }, py::arg("s"));

    m.attr("PRESETS") = preset_dict(cfg::PresetGroup::Presets);
    m.attr("LEGACY") = preset_dict(cfg::PresetGroup::Legacy);
    m.attr("PROTOTYPES") = preset_dict(cfg::PresetGroup::Prototypes);
}

void bind_lframe(py::module_& q) {
    py::module_ m = q.def_submodule("lframe", "Port of sstvae.qrss.lframe (L)");
    m.attr("FrameSpec") = q.attr("frame").attr("FrameSpec");
    m.attr("make") = q.attr("frame").attr("make");
    m.attr("of") = q.attr("frame").attr("of");

    m.def("_refs_below", [](const py::object& j) { return fr::refs_below(as_index(j)); }, py::arg("j"));
    m.def("_data_symbols", [](const py::object& n, const py::object& j0) {
        return fr::data_symbols_l(as_index(n), as_index(j0));
    }, py::arg("n_payload"), py::arg("j0"));
    m.def("get", [](const py::object& name_or_spec) { return get_in(name_or_spec, fr::Waveform::L); },
          py::arg("name_or_spec"));
    m.def("get_any", [](const py::object& name_or_spec) {
        if (py::isinstance<fr::FrameSpec>(name_or_spec)) return name_or_spec.cast<fr::FrameSpec>();
        if (!py::isinstance<py::str>(name_or_spec)) return get_in(name_or_spec, fr::Waveform::CE);   // frame.get's
        try {
            return fr::get_any(utf8_of(name_or_spec));
        } catch (const std::invalid_argument&) {
            unknown(name_or_spec, waveform_of_name(name_or_spec.cast<py::str>()));
        }
    }, py::arg("name_or_spec"));
    m.def("twin", [](const py::object& spec, const py::object& waveform) {
        // The reference compares with ==, so a float, a bool or a numpy scalar
        // selects as its value does and anything else is "the other one".
        const fr::FrameSpec s = to_spec(spec);
        const int own = static_cast<int>(s.waveform());
        const int w = py::int_(own).equal(waveform) ? own
                      : waveform.equal(py::int_(cfg::WAVEFORM_L)) ? cfg::WAVEFORM_L
                                                                   : -1;
        try {
            return fr::twin(s, w);
        } catch (const std::invalid_argument&) {
            // the name looked up: "l-" + spec.name, or spec.name[2:] for get_any
            const py::str name = str_of(s.name());
            if (w == cfg::WAVEFORM_L) unknown(py::str("l-") + name, fr::Waveform::L);
            const py::str rest = name[py::slice(py::int_(2), py::none(), py::none())];
            unknown(rest, waveform_of_name(rest));
        }
    }, py::arg("spec"), py::arg("waveform"));
    m.def("layout", [](const py::object& spec) {
        const fr::FrameSpec s = to_spec(spec);
        fr::require_waveform(s, fr::Waveform::L, "lframe.layout");
        return layout_dict(*fr::layout(s));
    }, py::arg("spec"), "lframe.layout's fields as a dict of read-only arrays");
    m.def("pair", [](const py::object& v) {
        const auto a = float64_array(v);
        if (a.ndim() != 1)
            throw py::value_error("pair takes an even number of values, not shape " + fr::shape_repr(shape_of(a)));
        return to_numpy(fr::pair(std::span<const double>(a.data(), static_cast<std::size_t>(a.size()))));
    }, py::arg("v"));
    m.def("unpair", [](const py::object& z) { return unpair_any(z); }, py::arg("z"));
    m.def("data_latents", [](const py::object& spec, const py::object& x, const py::object& hdr_bits) -> py::object {
        const fr::FrameSpec s = to_spec(spec);
        py::object a = numpy().attr("asarray")(x, py::arg("dtype") = "float64");
        const auto xa = float64_array(a);
        fr::check_x_shape(s, shape_of(xa));
        if (s.hdr_rho() == 0) return a;   // np.asarray(x) itself, as the reference returns it
        const auto b = bits_f64(hdr_bits);
        return to_numpy(fr::data_latents(s, std::span<const double>(xa.data(), static_cast<std::size_t>(xa.size())),
                                         ptr(b)));
    }, py::arg("spec"), py::arg("x"), py::arg("hdr_bits") = py::none());
    m.def("assemble", [](const py::object& spec, const py::object& hdr_bits, const py::object& x) {
        const fr::FrameSpec s = to_spec(spec);
        fr::require_waveform(s, fr::Waveform::L, "lframe.assemble");
        const auto b = assemble_bits(s, spec, hdr_bits);   // the header first, then x (the reference's order)
        const auto xa = float64_array(x);
        fr::check_x_shape(s, shape_of(xa));
        return to_numpy(fr::assemble_l(s, ptr(b), std::span<const double>(xa.data(), static_cast<std::size_t>(xa.size()))));
    }, py::arg("spec"), py::arg("hdr_bits"), py::arg("x"));
    m.def("disassemble", [](const py::object& y_sym, const py::object& spec) {
        const fr::FrameSpec s = to_spec(spec);
        fr::require_waveform(s, fr::Waveform::L, "lframe.disassemble");
        const py::array y = symbol_rows(y_sym, s);
        const auto lay = fr::layout(s);
        const auto n = static_cast<std::size_t>(s.n_sym());
        return py::make_tuple(unpair_any(gather_any(y, n, lay->hdr_sym)), unpair_any(gather_any(y, n, lay->data)));
    }, py::arg("y_sym"), py::arg("spec"));

    m.attr("N_HDR_SLOTS") = fr::N_HDR_SLOTS_L;
    m.attr("PRESETS") = preset_dict(cfg::PresetGroup::LPresets);
}

const Registrar reg_frame("frame", 20, &bind_frame);
const Registrar reg_lframe("lframe", 21, &bind_lframe);

}  // namespace
}  // namespace sstvae_bindings::qrss
