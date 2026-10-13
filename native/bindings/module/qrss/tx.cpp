// sstvae_native.qrss.tx: bindings of native/core/qrss/tx (the slot
// functions, transmit_audio and _transmit_audio_l) and the streaming form
// the app uses, PassAudio, with qrss_transmit.py's two WAV forms. Names,
// signatures (keyword-only arguments included), dtypes and ValueError
// texts follow sstvae/qrss/tx.py.
//
// A source is any object with BeaconFile's attributes (latents_i8, ...)
// or StoredPicture's (segs, ...) -- the reference's own dataclasses pass
// straight through -- and a header any object with HeaderFields'. A frame
// is a native FrameSpec or the reference's FrameSpec / LFrameSpec, through
// sstvae_native.qrss.frame.of. tests/qrss_native/tx.py wraps the one
// HeaderFields that comes back.
//
// transmit_audio converts its arguments as the reference reads them:
// float() of the carrier, the lead-in, pre_s, post_s, amplitude and
// chunk_s, the truth value of ook, and the carrier's exact fraction from
// sstvae_native.qrss.ce.carrier_fraction (a Fraction as it is, a float or
// str limited to 2^20). Refusals quote the caller's own object, format(x).

#include "qrss/registry.hpp"

#include <algorithm>
#include <cerrno>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/frame.hpp"
#include "qrss/picture_chain.hpp"
#include "qrss/rational.hpp"
#include "qrss/tx.hpp"
#include "qrss/wav.hpp"

namespace sstvae_bindings::qrss {
namespace {

using namespace picture_chain;
namespace qcfg = sstvae::qrss::config;
namespace fr = sstvae::qrss::frame;
namespace txm = sstvae::qrss::tx;
namespace wavm = sstvae::qrss::wav;
using sstvae::qrss::Rational;
using polar_util::fmt;

// sstvae_native.qrss.frame.of, looked up once the frame binding exists (it
// binds first). Leaked on purpose: nothing may run its destructor after
// the interpreter is gone.
py::object* frame_of_fn = nullptr;
// sstvae_native.qrss.ce.carrier_fraction (ce binds before tx), likewise.
py::object* carrier_fraction_fn = nullptr;

fr::FrameSpec spec_of(const py::handle& spec) {
    if (py::isinstance<fr::FrameSpec>(spec)) return spec.cast<fr::FrameSpec>();
    return (*frame_of_fn)(spec).cast<fr::FrameSpec>();
}

std::optional<std::int64_t> opt_int(const py::handle& v) {
    if (v.is_none()) return std::nullopt;
    return int64_of(v);
}

// The `segment` argument as _segment_and_header reads it: int(segment) for
// a stored picture, and for a beacon file Python's `segment not in (None,
// src.segment)` -- an equality test, so 1.0 is the file's segment 1 and 1.5
// is not (refused quoting str(1.5), which `opt_shown` supplies).
std::optional<std::int64_t> segment_arg(const txm::Source& s, const py::handle& segment) {
    if (segment.is_none()) return std::nullopt;
    if (const auto* bf = std::get_if<bfm::BeaconFile>(&s)) {
        const py::int_ own(bf->segment);
        if (segment.ptr() == own.ptr()) return bf->segment;
        const int eq = PyObject_RichCompareBool(segment.ptr(), own.ptr(), Py_EQ);
        if (eq < 0) throw py::error_already_set();
        // Any segment but the file's (without overflowing a clamped int64).
        if (eq) return bf->segment;
        return bf->segment == std::numeric_limits<std::int64_t>::max() ? bf->segment - 1 : bf->segment + 1;
    }
    return int64_of(segment);
}

std::optional<std::string> opt_shown(const py::handle& v) {
    if (v.is_none()) return std::nullopt;
    return str_of(v);
}

// use_int8 as the reference reads it: `use_int8 is False` refuses a beacon
// file, and its truth value picks the int8 version of a stored picture --
// so None and a falsy value other than False both mean "as stored".
std::optional<bool> use_int8_of(const py::handle& v) {
    if (v.is_none()) return std::nullopt;
    if (v.ptr() == Py_False) return false;
    if (truthy(v)) return true;
    return std::nullopt;
}

py::object symbols_array(const txm::Symbols& s) {
    if (s.waveform == fr::Waveform::L) return to_numpy(std::span<const std::complex<double>>(s.l));
    return to_numpy(std::span<const double>(s.ce));
}

py::object opt_bits(const std::optional<std::vector<std::uint8_t>>& v) {
    if (!v) return py::none();
    return to_numpy(std::span<const std::uint8_t>(*v));
}

// q as precode reads it: int(q), then the uint64 range check.
std::uint64_t q_of(const py::handle& qv) {
    const py::int_ qi = py::reinterpret_borrow<py::int_>(steal(PyNumber_Long(qv.ptr())));
    return to_u64(qi, "q must fit in uint64, not " + str_of(qi));
}

// transmit_audio's arguments, as Python objects.
struct PassArgs {
    py::object src, q, carrier_hz, spec, segment, header, lead_in_s, ook, pre_s, post_s, amplitude, chunk_s;
};

// ce.carrier_fraction(carrier_hz) as a Rational: a numerator or a
// denominator past int64 is numpy's refusal of the period den * 8000.
Rational fraction_of(const py::handle& carrier_hz) {
    const py::tuple nd = (*carrier_fraction_fn)(carrier_hz);
    int over = 0;
    const long long num = PyLong_AsLongLongAndOverflow(nd[0].ptr(), &over);
    if (num == -1 && PyErr_Occurred()) throw py::error_already_set();
    int over_d = 0;
    const long long den = PyLong_AsLongLongAndOverflow(nd[1].ptr(), &over_d);
    if (den == -1 && PyErr_Occurred()) throw py::error_already_set();
    if (over != 0 || over_d != 0 || num == std::numeric_limits<long long>::min()) {
        PyErr_SetString(PyExc_OverflowError, "Python int too large to convert to C long");
        throw py::error_already_set();
    }
    return Rational(num, den);
}

// transmit_audio (l_entry false) or _transmit_audio_l (true) up to its
// audio: every check in the reference's order, then the pass.
std::unique_ptr<txm::PassAudio> make_pass(const PassArgs& a, bool l_entry) {
    const fr::FrameSpec f = spec_of(a.spec);
    std::optional<bool> ook;
    if (!l_entry) {
        if (!a.ook.is_none()) ook = truthy(a.ook);   // `if ook:` / bool(ook)
        txm::check_ook(f, ook);
    }
    const double hz = float_of(a.carrier_hz);
    const std::string shown = fmt(a.carrier_hz);
    txm::PassSpec pass;
    pass.carrier = txm::Carrier(hz, std::nullopt, shown);
    if (l_entry)
        txm::check_pilot(hz, shown);
    else
        txm::check_pass_carrier(f, pass.carrier, ook);
    const txm::Source s = source_of(a.src);
    pass.spec = f;
    pass.segment = segment_arg(s, a.segment);
    pass.header = opt_header_of(a.header);
    pass.lead_in_s = float_of(a.lead_in_s);
    pass.ook = ook;
    pass.pre_s = float_of(a.pre_s);
    pass.post_s = float_of(a.post_s);
    pass.amplitude = float_of(a.amplitude);
    pass.chunk_s = float_of(a.chunk_s);
    pass.shown.segment = opt_shown(a.segment);
    pass.shown.lead_in_s = fmt(a.lead_in_s);
    pass.shown.pre_s = fmt(a.pre_s);
    txm::PassPlan plan = l_entry ? txm::plan_pass_l(s, pass) : txm::plan_pass(s, pass);
    const std::uint64_t qq = q_of(a.q);   // in slot_symbols, after its own checks
    plan.pass.carrier = txm::Carrier(hz, fraction_of(a.carrier_hz), shown);
    py::gil_scoped_release nogil;
    return std::make_unique<txm::PassAudio>(plan, qq);
}

// The whole pass as a new float64 array.
py::array_t<double> render_all(const txm::PassAudio& audio) {
    py::array_t<double> out(static_cast<py::ssize_t>(audio.size()));
    double* p = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        audio.render(0, std::span<double>(p, static_cast<std::size_t>(audio.size())));
    }
    return out;
}

// float32 or float64, as np.dtype(dtype) names it.
bool is_float32(const py::handle& dtype) {
    const py::dtype dt = py::dtype::from_args(py::reinterpret_borrow<py::object>(dtype));
    if (dt.kind() == 'f' && dt.itemsize() == 4) return true;
    if (dt.kind() == 'f' && dt.itemsize() == 8) return false;
    throw py::value_error("dtype must be float32 or float64, not " + str_of(dt));
}

py::array render_block(const txm::PassAudio& audio, std::int64_t i0, std::int64_t m, bool f32) {
    if (m < 0) throw py::value_error("negative dimensions are not allowed");
    audio.check_block(i0, m);   // IndexError before the buffer is allocated, not MemoryError
    if (f32) {
        py::array_t<float> out(static_cast<py::ssize_t>(m));
        float* p = out.mutable_data();
        py::gil_scoped_release nogil;
        audio.render(i0, std::span<float>(p, static_cast<std::size_t>(m)));
        return out;
    }
    py::array_t<double> out(static_cast<py::ssize_t>(m));
    double* p = out.mutable_data();
    py::gil_scoped_release nogil;
    audio.render(i0, std::span<double>(p, static_cast<std::size_t>(m)));
    return out;
}

// open()'s path argument: os.fspath(path) as a str.
std::string path_of(const py::handle& path) {
    return py::str(py::module_::import("os").attr("fspath")(path)).cast<std::string>();
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("tx", "Port of sstvae.qrss.tx");
    frame_of_fn = new py::object(q.attr("frame").attr("of"));
    const py::object full = q.attr("frame").attr("get")("full");

    // (segment, header, hdr bits, keying, ook). The header is the object
    // given, for a stored picture, or the beacon file's own fields as a
    // tuple (the adapter makes it a HeaderFields).
    m.def("_segment_and_header", [](const py::object& src, const py::object& segment, const py::object& header) {
        const txm::Source s = source_of(src);
        const txm::SegmentAndHeader r =
            txm::segment_and_header(s, segment_arg(s, segment), opt_header_of(header), opt_shown(segment));
        py::object h = py::none();
        if (std::holds_alternative<bfm::BeaconFile>(s))
            h = header_tuple(*r.header);
        else if (r.header)
            h = header;
        return py::make_tuple(r.segment, h, opt_bits(r.hdr_bits), opt_bits(r.keying), py::bool_(r.ook));
    }, py::arg("src"), py::arg("segment"), py::arg("header"));

    m.def("slot_air_latents", [](const py::object& src, const py::object& segment, const py::object& use_int8) {
        const txm::Source s = is_beacon(src) ? txm::Source(beacon_of(src)) : txm::Source(picture_of(src));
        return to_numpy(txm::slot_air_latents(s, opt_int(segment), use_int8_of(use_int8)));
    }, py::arg("src"), py::arg("segment") = py::none(), py::arg("use_int8") = py::none());

    m.def("slot_symbols", [](const py::object& src, const py::object& qv, const py::object& spec,
                             const py::object& segment, const py::object& header, const py::object& use_int8) {
        const txm::Source s = source_of(src);
        const fr::FrameSpec f = spec_of(spec);
        const txm::SlotPlan plan =
            txm::plan_symbols(s, f, segment_arg(s, segment), opt_header_of(header), use_int8_of(use_int8),
                              opt_shown(segment));
        // precode's int(q) and its range check come last, as in the reference.
        const py::int_ qi = py::reinterpret_borrow<py::int_>(steal(PyNumber_Long(qv.ptr())));
        const std::uint64_t qq = to_u64(qi, "q must fit in uint64, not " + str_of(qi));
        return symbols_array(txm::assemble_symbols(plan, f, qq));
    }, py::arg("src"), py::arg("q"), py::arg("spec") = full, py::kw_only(), py::arg("segment") = py::none(),
       py::arg("header") = py::none(), py::arg("use_int8") = py::none());

    m.def("slot_keying", [](const py::object& src, const py::object& header) -> py::object {
        const txm::Source s = is_beacon(src) ? txm::Source(beacon_of(src)) : txm::Source(sstvae::qrss::picture::StoredPicture{});
        return opt_bits(txm::slot_keying(s, opt_header_of(header)));
    }, py::arg("src"), py::arg("header") = py::none());

    m.def("slot_samples", [](const py::object& spec, const py::object& pre_s, const py::object& post_s,
                             const py::object& fs) {
        return txm::slot_samples(spec_of(spec), float_of(pre_s), float_of(post_s), float_of(fs));
    }, py::arg("spec"), py::arg("pre_s") = qcfg::TX_PRE_S, py::arg("post_s") = qcfg::TX_POST_S,
       py::arg("fs") = sstvae::qrss::config::FS);

    m.def("check_pilot", [](const py::object& carrier_hz) {
        return txm::check_pilot(float_of(carrier_hz), str_of(carrier_hz));
    }, py::arg("carrier_hz"));

    // --- the pass's audio
    carrier_fraction_fn = new py::object(q.attr("ce").attr("carrier_fraction"));
    py::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const wavm::WriteError& e) {
            const py::str name(e.path);
            errno = e.err;
            PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError, name.ptr());
        }
    });

    m.def("transmit_audio", [](const py::object& src, const py::object& qv, const py::object& carrier_hz,
                               const py::object& spec, const py::object& segment, const py::object& header,
                               const py::object& lead_in_s, const py::object& ook, const py::object& pre_s,
                               const py::object& post_s, const py::object& amplitude, const py::object& chunk_s) {
        return render_all(*make_pass({src, qv, carrier_hz, spec, segment, header, lead_in_s, ook, pre_s, post_s,
                                      amplitude, chunk_s}, false));
    }, py::arg("src"), py::arg("q"), py::arg("carrier_hz") = qcfg::TX_CARRIER_HZ, py::kw_only(), py::arg("spec") = full,
       py::arg("segment") = py::none(), py::arg("header") = py::none(), py::arg("lead_in_s") = 0.0,
       py::arg("ook") = py::none(), py::arg("pre_s") = qcfg::TX_PRE_S, py::arg("post_s") = qcfg::TX_POST_S,
       py::arg("amplitude") = qcfg::TX_AMPLITUDE, py::arg("chunk_s") = qcfg::TX_CHUNK_S);

    m.def("_transmit_audio_l", [](const py::object& src, const py::object& qv, const py::object& carrier_hz,
                                  const py::object& spec, const py::object& segment, const py::object& header,
                                  const py::object& lead_in_s, const py::object& pre_s, const py::object& post_s,
                                  const py::object& amplitude, const py::object& chunk_s) {
        return render_all(*make_pass({src, qv, carrier_hz, spec, segment, header, lead_in_s, py::none(), pre_s, post_s,
                                      amplitude, chunk_s}, true));
    }, py::arg("src"), py::arg("q"), py::arg("carrier_hz"), py::arg("spec"), py::arg("segment"), py::arg("header"),
       py::arg("lead_in_s"), py::arg("pre_s"), py::arg("post_s"), py::arg("amplitude"), py::arg("chunk_s"));

    py::class_<txm::PassAudio>(m, "PassAudio",
                               "One pass as 8 kHz audio, produced in blocks on request (tx.transmit_audio, "
                               "streamed). Any block size gives the same samples.")
        .def(py::init([](const py::object& src, const py::object& qv, const py::object& carrier_hz,
                         const py::object& spec, const py::object& segment, const py::object& header,
                         const py::object& lead_in_s, const py::object& ook, const py::object& pre_s,
                         const py::object& post_s, const py::object& amplitude, const py::object& chunk_s) {
                 return make_pass({src, qv, carrier_hz, spec, segment, header, lead_in_s, ook, pre_s, post_s,
                                   amplitude, chunk_s}, false);
             }),
             py::arg("src"), py::arg("q"), py::arg("carrier_hz") = qcfg::TX_CARRIER_HZ, py::kw_only(), py::arg("spec") = full,
             py::arg("segment") = py::none(), py::arg("header") = py::none(), py::arg("lead_in_s") = 0.0,
             py::arg("ook") = py::none(), py::arg("pre_s") = qcfg::TX_PRE_S, py::arg("post_s") = qcfg::TX_POST_S,
             py::arg("amplitude") = qcfg::TX_AMPLITUDE, py::arg("chunk_s") = qcfg::TX_CHUNK_S)
        .def("__len__", [](const txm::PassAudio& a) { return a.size(); })
        .def_property_readonly("size", &txm::PassAudio::size, "tx.slot_samples(spec, pre_s, post_s)")
        .def_property_readonly("n_pre", &txm::PassAudio::n_pre, "samples before t0")
        .def_property_readonly("position", &txm::PassAudio::position)
        .def_property_readonly("chunk", &txm::PassAudio::chunk, "the reference's chunk, in samples")
        .def_property_readonly("waveform_l", &txm::PassAudio::waveform_l)
        .def_property_readonly("carrier", [](const txm::PassAudio& a) {
            return py::make_tuple(py::int_(a.carrier().num()), py::int_(a.carrier().den()));
        }, "(num, den): the carrier the audio is mixed to, Hz")
        .def("render", [](const txm::PassAudio& a, std::int64_t i0, std::int64_t m, const py::object& dtype) {
            return render_block(a, i0, m, is_float32(dtype));
        }, py::arg("i0"), py::arg("m"), py::arg("dtype") = "float64",
           "samples [i0, i0 + m) of the pass (IndexError outside it)")
        .def("read", [](txm::PassAudio& a, std::int64_t m, const py::object& dtype) {
            if (m < 0) throw py::value_error("negative dimensions are not allowed");
            const std::int64_t k = std::min(m, a.size() - a.position());
            py::array out = render_block(a, a.position(), k, is_float32(dtype));
            a.seek(a.position() + k);
            return out;
        }, py::arg("m"), py::arg("dtype") = "float64",
           "the next m samples from position (fewer at the end), moving on past them")
        .def("seek", &txm::PassAudio::seek, py::arg("i"))
        .def("write_wav", [](const txm::PassAudio& a, const py::object& path, bool as_float, double peak) {
            const std::string p = path_of(path);
            py::gil_scoped_release nogil;
            txm::write_wav(a, p, as_float, peak);
        }, py::arg("path"), py::arg("float") = false, py::arg("peak") = 0.95,
           "qrss_transmit.py's WAV: peak-normalised int16, or float32 with float=True");

    // sstvae.wavio's two writers, for an array.
    m.def("write_wav", [](const py::object& path, const py::object& x, double peak) {
        const std::string p = path_of(path);
        const std::vector<double> v = from_numpy<double>(numpy().attr("asarray")(x, py::arg("dtype") = "float64"));
        py::gil_scoped_release nogil;
        wavm::write_int16(p, v, peak);
    }, py::arg("path"), py::arg("x"), py::arg("peak") = 0.95);
    m.def("write_wav_float", [](const py::object& path, const py::object& x) {
        const std::string p = path_of(path);
        const std::vector<double> v = from_numpy<double>(numpy().attr("asarray")(x, py::arg("dtype") = "float64"));
        py::gil_scoped_release nogil;
        wavm::write_float32(p, v);
    }, py::arg("path"), py::arg("x"));
}

const Registrar reg("tx", 62, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
