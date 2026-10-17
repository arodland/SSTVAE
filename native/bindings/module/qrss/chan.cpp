// sstvae_native.qrss.chan: bindings of native/core/qrss/chan (track.py's
// make_chan, ChanCapture.noise_psd_at and ChanCapture.noise_floor).
//
//   NOISE_WIN_S
//   make_chan(fe, keep, t0_index_fe, fs_fe, f_audio_hz, t_lo_s, t_hi_s, exclude_hz=())
//       -> a "ChanCapture" dict (rx_types.hpp; tests/qrss_native/_rx.py's to_ref
//          makes it the reference's track.ChanCapture)
//   noise_psd_at(chan, t_s, noise_psd=None)   np.float64 for a scalar t_s, else
//                                             an array of t_s's shape
//   noise_floor(chan, noise_psd=None) -> float
//   noise_psd_table(chan, noise_psd=None) -> (t_s, psd)
//
// `chan` is the reference's ChanCapture or such a dict. Its noise table is
// its `_psd`, as in the reference: used as it is when set, computed and
// written back (an attribute, or the dict's key) when None, so the
// reference's object caches exactly what its own method would have.
// `noise_psd` is the function computing the table: None is the C++
// frontend.noise_psd, anything else is called as the reference calls
// frontend.noise_psd -- noise_psd(chan.ch, chan.fs, exclude_hz=[...],
// win_s=NOISE_WIN_S, audio_hz=...) -- and its result read as float32.

#include "qrss/rx_types.hpp"

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "math/npstats.hpp"
#include "qrss/chan.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace tr = sstvae::qrss::track;
using cf = std::complex<float>;

struct Doubles {
    std::vector<double> v;
    py::tuple shape;
};

Doubles doubles(const py::handle& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o, py::arg("dtype") = "float64"));
    if (!a) throw py::error_already_set();
    return {from_numpy<double>(a), py::tuple(a.attr("shape"))};
}

py::object shaped(const std::vector<double>& v, const py::tuple& shape) {
    py::object a = to_numpy(v).attr("reshape")(shape);
    if (shape.empty()) return a[py::tuple()];
    return a;
}

void set_psd(const py::object& chan, const py::object& value) {
    if (py::isinstance<py::dict>(chan))
        py::reinterpret_borrow<py::dict>(chan)["_psd"] = value;
    else
        py::setattr(chan, "_psd", value);
}

// The chan's noise table: its _psd when set, else computed (by `fn`, or the
// C++ noise_psd for None) and written back.
std::shared_ptr<const tr::NoisePsd> table_of(const py::object& chan, const py::object& fn) {
    if (rx::has_field(chan, "_psd")) {
        const py::object psd = rx::field(chan, "_psd");
        if (!psd.is_none()) {
            py::tuple t = py::tuple(psd);
            if (t.size() != 2) throw py::value_error("ChanCapture._psd must be (t_s, psd)");
            return std::make_shared<const tr::NoisePsd>(tr::NoisePsd{rx::vec<double>(t[0]), rx::vec<double>(t[1])});
        }
    }
    const tr::ChanCapture c = rx::chan_capture(chan);
    tr::NoisePsdFn f;
    if (!fn.is_none()) {
        const py::object py_ch = rx::field(chan, "ch"), py_fs = rx::field(chan, "fs");
        f = [&fn, py_ch, py_fs](std::span<const cf>, std::int64_t, std::span<const double> ex, double win_s,
                                std::optional<double> audio_hz) {
            py::list exl;
            for (const double x : ex) exl.append(x);
            const py::object r = fn(py_ch, py_fs, py::arg("exclude_hz") = exl, py::arg("win_s") = win_s,
                                    py::arg("audio_hz") = audio_hz ? py::object(py::float_(*audio_hz)) : py::object(py::none()));
            return rx::vec<float>(r);
        };
    }
    const auto table = tr::noise_psd_table(c, f);
    set_psd(chan, py::make_tuple(to_numpy(table->t_s), to_numpy(table->psd)));
    return table;
}

py::object noise_psd_at_py(const py::object& chan, const py::object& t_s, const py::object& fn) {
    const auto table = table_of(chan, fn);
    const Doubles t = doubles(t_s);
    return shaped(sstvae::math::interp(std::span<const double>(t.v), table->t_s, table->psd), t.shape);
}

double noise_floor_py(const py::object& chan, const py::object& fn) {
    const auto table = table_of(chan, fn);
    (void)sstvae::math::interp(0.0, table->t_s, table->psd);   // the reference's noise_psd_at(0.0)
    return sstvae::math::median(std::span<const double>(table->psd));
}

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule("chan", "native/core/qrss/chan (track.make_chan, ChanCapture's noise floor)");
    m.attr("NOISE_WIN_S") = tr::NOISE_WIN_S;

    m.def("make_chan", [](const py::object& fe, const py::object& keep, const py::object& t0_index_fe,
                          const py::object& fs_fe, const py::object& f_audio_hz, const py::object& t_lo_s,
                          const py::object& t_hi_s, const py::object& exclude_hz) {
        const auto v = rx::vec<cf>(fe);
        const auto k = rx::opt_vec<float>(keep);
        std::vector<double> ex;
        for (const py::handle h : py::reinterpret_borrow<py::iterable>(exclude_hz)) ex.push_back(rx::as_double(h));
        const std::int64_t fs = rx::as_int(fs_fe);
        const double t0 = rx::as_double(t0_index_fe), f = rx::as_double(f_audio_hz);
        const double lo = rx::as_double(t_lo_s), hi = rx::as_double(t_hi_s);
        std::optional<tr::ChanCapture> c;
        {
            py::gil_scoped_release nogil;
            std::optional<std::span<const float>> ks;
            if (k) ks = std::span<const float>(*k);
            c.emplace(tr::make_chan(v, ks, t0, fs, f, lo, hi, ex));
        }
        return rx::to_py(*c);
    }, py::arg("fe"), py::arg("keep"), py::arg("t0_index_fe"), py::arg("fs_fe"), py::arg("f_audio_hz"),
          py::arg("t_lo_s"), py::arg("t_hi_s"), py::arg("exclude_hz") = py::tuple());

    m.def("noise_psd_at", &noise_psd_at_py, py::arg("chan"), py::arg("t_s"), py::arg("noise_psd") = py::none());
    m.def("noise_floor", &noise_floor_py, py::arg("chan"), py::arg("noise_psd") = py::none());
    m.def("noise_psd_table", [](const py::object& chan, const py::object& fn) {
        const auto table = table_of(chan, fn);
        return py::make_tuple(to_numpy(table->t_s), to_numpy(table->psd));
    }, py::arg("chan"), py::arg("noise_psd") = py::none());
}

const Registrar reg("chan", 67, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
