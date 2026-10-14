// sstvae_native.qrss.passband: bindings of native/core/qrss/passband (the
// 48 h passband store, frontend.PassbandStore), its FLAC packer
// (core/qrss/flac) and the directory lock (core/qrss/dirlock,
// live.dir_lock). tests/qrss_native/passband.py assigns them over the
// reference's:
//
//   PassbandStore(root, fs=FE_FS, scale=8192.0, keep_hours=48.0, pack=False)
//     .root (pathlib.Path) .fs .scale .keep_hours .per_hour .pack (read-only)
//     .write(t0, x, expire=True)   ._write_at(n0, x, expire)
//     .read(t_start, t_end, return_mask=False) -> complex64 [, bool]
//     .write_capture(cap, expire=True) -> (t_first, gain)
//     .capture(q, dur_s, lead_s=PB_LEAD_S, tail_s=PB_TAIL_S) -> (frontend.Capture | None, coverage)
//     .slots(dur_s) .hours() .raw_hours() .expire(now=None) .pack_hour(hour)
//     ._maybe_pack(now)
//     ._packer: None until a packer thread has started, then a handle with
//       .join(timeout=None), .is_alive(), .name, .daemon (the store has one
//       packer at a time, and the handle is always that one)
//   _merge_ranges(rs) -> [[lo, hi], ...]     slot_t0(q)     PB_* constants
//   dir_lock(path) -> lock file object (close(), closed, name, fileno(),
//     context manager) or None
//   _hour_of_name(name), _flac_encode/_flac_read/_flac_info (tests)
//
// The store's capture() returns the reference's frontend.Capture record
// (imported from sstvae.qrss.frontend, which --native-qrss never replaces),
// as its callers (em._from_passband, em.retro_detect) read its fields.
// Errors: OSError (FileNotFoundError, ...) for a file that cannot be made,
// opened or written, naming it; ValueError (an fs <= 0 among them),
// IndexError, OverflowError and KeyError where the reference raises them; a FLAC
// hour that does not decode is RuntimeError, as soundfile's errors are.
// The I/O runs with the GIL released; no Python object is touched there.

#include "qrss/rx_types.hpp"
#include "qrss/fspath.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/dirlock.hpp"
#include "qrss/flac.hpp"
#include "qrss/frontend.hpp"
#include "qrss/passband.hpp"
#include "util/fileio.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace pb = sstvae::qrss::passband;
namespace flac = sstvae::qrss::flac;
namespace dl = sstvae::qrss::dirlock;
namespace cfg = sstvae::qrss::config;
using cf = std::complex<float>;

// --- errors -----------------------------------------------------------------------

// (a FileError's code is an errno already, rawfile.hpp; util::errno_of is
// the one mapping either way)
[[noreturn]] void raise_os(const std::error_code& ec, const std::string& path_bytes) {
    fspath::raise_os_error(sstvae::util::errno_of(ec), fspath::fs_text(path_bytes));
}

// fn(), with the store's file errors raised as OSError naming the file.
template <typename F>
auto os_guard(F&& fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const pb::FileError& e) {
        raise_os(e.code(), e.path());
    }
}

[[noreturn]] void raise(PyObject* type, const py::handle& text) {
    PyErr_SetObject(type, text.ptr());
    throw py::error_already_set();
}

// --- the store ----------------------------------------------------------------------

struct Store {
    std::unique_ptr<pb::PassbandStore> s;
    py::object root;   // pathlib.Path(root), as the reference keeps it
};

struct Packer {
    py::object owner;   // the Store
};

pb::PassbandStore& store_of(const py::handle& self) { return *self.cast<Store&>().s; }

py::object capture_class() { return py::module_::import("sstvae.qrss.frontend").attr("Capture"); }

std::int64_t as_int(const py::handle& o) { return rx::as_int(o); }

py::list int_list(const std::vector<std::int64_t>& v) {
    py::list out;
    for (const std::int64_t x : v) out.append(py::int_(x));
    return out;
}

void bind_store(py::module_& m) {
    py::class_<Packer>(m, "_Packer", "the store's packer thread (threading.Thread's join / is_alive)")
        .def("join", [](Packer& p, const py::object& timeout) {
            std::optional<double> t;
            if (!timeout.is_none()) t = rx::as_double(timeout);
            pb::PassbandStore& s = store_of(p.owner);
            py::gil_scoped_release nogil;
            s.join_packer(t);
        }, py::arg("timeout") = py::none())
        .def("is_alive", [](Packer& p) { return store_of(p.owner).packer_alive(); })
        .def_property_readonly("name", [](const Packer&) { return std::string("passband-pack"); })
        .def_property_readonly("daemon", [](const Packer&) { return true; });

    py::class_<Store>(m, "PassbandStore", py::dynamic_attr(),
                      "frontend.PassbandStore: the whole FE stream on disk for keep_hours")
        .def(py::init([](const py::object& root, const py::object& fs, const py::object& scale,
                         const py::object& keep_hours, const py::object& pack) {
                 const fspath::FsPath p = fspath::pathlib_path(root);
                 pb::StoreOptions opt;
                 const std::int64_t f = as_int(fs);
                 if (f > std::numeric_limits<int>::max() || f < std::numeric_limits<int>::min())
                     throw std::overflow_error("fs is out of range");
                 opt.fs = static_cast<int>(f);
                 opt.scale = rx::as_double(scale);
                 opt.keep_hours = rx::as_double(keep_hours);
                 opt.pack = rx::as_bool(pack);
                 Store st;
                 st.root = py::module_::import("pathlib").attr("Path")(p.name);
                 st.s = os_guard([&] {
                     py::gil_scoped_release nogil;
                     return std::make_unique<pb::PassbandStore>(p.bytes, opt);
                 });
                 return st;
             }),
             py::arg("root"), py::arg("fs") = cfg::FE_FS, py::arg("scale") = 8192.0, py::arg("keep_hours") = 48.0,
             py::arg("pack") = false)
        .def_property_readonly("root", [](const Store& st) { return st.root; })
        .def_property_readonly("fs", [](const Store& st) { return st.s->fs(); })
        .def_property_readonly("scale", [](const Store& st) { return st.s->scale(); })
        .def_property_readonly("keep_hours", [](const Store& st) { return st.s->keep_hours(); })
        .def_property_readonly("per_hour", [](const Store& st) { return st.s->per_hour(); })
        .def_property_readonly("pack", [](const Store& st) { return st.s->pack(); })
        .def_property_readonly("_packer", [](const py::object& self) -> py::object {
            if (!store_of(self).packer_started()) return py::none();
            return py::cast(Packer{self});
        })
        .def("write", [](Store& st, const py::object& t0, const py::object& x, const py::object& expire) {
            const double t = rx::as_double(t0);
            const std::vector<cf> v = rx::vec<cf>(x);
            const bool e = rx::as_bool(expire);
            os_guard([&] {
                py::gil_scoped_release nogil;
                st.s->write(t, v, e);
            });
        }, py::arg("t0"), py::arg("x"), py::arg("expire") = true)
        .def("_write_at", [](Store& st, const py::object& n0, const py::object& x, const py::object& expire) {
            const std::int64_t n = as_int(n0);
            const std::vector<cf> v = rx::vec<cf>(x);
            const bool e = rx::as_bool(expire);
            os_guard([&] {
                py::gil_scoped_release nogil;
                st.s->write_at(n, v, e);
            });
        }, py::arg("n0"), py::arg("x"), py::arg("expire"))
        .def("read", [](Store& st, const py::object& t_start, const py::object& t_end,
                        const py::object& return_mask) -> py::object {
            const double a = rx::as_double(t_start), b = rx::as_double(t_end);
            const bool want_mask = rx::as_bool(return_mask);
            pb::Masked r = os_guard([&] {
                py::gil_scoped_release nogil;
                return st.s->read(a, b);
            });
            py::array x = to_numpy(r.x);
            if (!want_mask) return x;
            return py::make_tuple(x, rx::arr_bool(r.mask));
        }, py::arg("t_start"), py::arg("t_end"), py::arg("return_mask") = false)
        .def("write_capture", [](Store& st, const py::object& cap, const py::object& expire) {
            const py::object fs_o = rx::field(cap, "fs");
            if (as_int(fs_o) != st.s->fs()) {
                const py::str msg = py::str("the store holds {} Hz FE, not a {} Hz capture").format(st.s->fs(), fs_o);
                raise(PyExc_ValueError, msg);
            }
            sstvae::qrss::frontend::Capture c;
            c.fs = st.s->fs();
            c.fe = rx::vec<cf>(rx::field(cap, "fe"));
            c.q = as_int(rx::field(cap, "q"));
            c.t0_index = rx::as_double(rx::field(cap, "t0_index"));
            const bool e = rx::as_bool(expire);
            const pb::Written w = os_guard([&] {
                py::gil_scoped_release nogil;
                return st.s->write_capture(c, e);
            });
            return py::make_tuple(w.t_first, w.gain);
        }, py::arg("cap"), py::arg("expire") = true)
        .def("capture", [](Store& st, const py::object& q, const py::object& dur_s, const py::object& lead_s,
                           const py::object& tail_s) {
            const std::int64_t qq = as_int(q);
            const double d = rx::as_double(dur_s), l = rx::as_double(lead_s), t = rx::as_double(tail_s);
            pb::SlotCapture r = os_guard([&] {
                py::gil_scoped_release nogil;
                return st.s->capture(qq, d, l, t);
            });
            if (!r.capture) return py::tuple(py::make_tuple(py::none(), r.coverage));
            const auto& c = *r.capture;
            py::object cap = capture_class()(py::arg("fe") = to_numpy(c.fe), py::arg("q") = c.q,
                                             py::arg("t0_index") = c.t0_index, py::arg("fs") = c.fs);
            return py::tuple(py::make_tuple(cap, r.coverage));
        }, py::arg("q"), py::arg("dur_s"), py::arg("lead_s") = pb::PB_LEAD_S, py::arg("tail_s") = pb::PB_TAIL_S)
        .def("slots", [](Store& st, const py::object& dur_s) {
            const double d = rx::as_double(dur_s);
            return int_list(os_guard([&] {
                py::gil_scoped_release nogil;
                return st.s->slots(d);
            }));
        }, py::arg("dur_s"))
        .def("hours", [](Store& st) {
            return int_list(os_guard([&] {
                py::gil_scoped_release nogil;
                return st.s->hours();
            }));
        })
        .def("raw_hours", [](Store& st) {
            return int_list(os_guard([&] {
                py::gil_scoped_release nogil;
                return st.s->raw_hours();
            }));
        })
        .def("expire", [](Store& st, const py::object& now) {
            std::optional<double> t;
            if (!now.is_none()) t = rx::as_double(now);
            return int_list(os_guard([&] {
                py::gil_scoped_release nogil;
                return st.s->expire(t);
            }));
        }, py::arg("now") = py::none())
        .def("pack_hour", [](Store& st, const py::object& hour) {
            // (a hour that is not an int names "fe_<hour>.i16" with a fraction: no raw hour)
            PyObject* i = PyNumber_Index(hour.ptr());
            if (!i) {
                PyErr_Clear();
                return false;
            }
            const std::int64_t h = py::reinterpret_steal<py::object>(i).cast<std::int64_t>();
            return os_guard([&] {
                py::gil_scoped_release nogil;
                return st.s->pack_hour(h);
            });
        }, py::arg("hour"))
        .def("_maybe_pack", [](Store& st, const py::object& now) {
            const double t = rx::as_double(now);
            os_guard([&] {
                py::gil_scoped_release nogil;
                st.s->maybe_pack(t);
            });
        }, py::arg("now"))
        .def("_raw_path", [](Store& st, std::int64_t h) { return fspath::fs_text(st.s->raw_path(h)); })
        .def("_flac", [](Store& st, std::int64_t h) { return fspath::fs_text(st.s->flac_path(h)); });
}

// --- the directory lock ------------------------------------------------------------

struct LockFile {
    std::optional<dl::DirLock> lock;
    py::object name;
};

void bind_lock(py::module_& m) {
    py::class_<LockFile>(m, "_LockFile", "live.dir_lock's open lock file: close() releases the lock")
        .def("close", [](LockFile& f) {
            if (f.lock) f.lock->close();
        })
        .def_property_readonly("closed", [](const LockFile& f) { return !f.lock || !f.lock->is_open(); })
        .def_property_readonly("name", [](const LockFile& f) { return f.name; })
        .def_property_readonly("mode", [](const LockFile&) { return std::string("a+"); })
        .def("fileno", [](const LockFile& f) {
            if (!f.lock || !f.lock->is_open()) throw py::value_error("I/O operation on closed file");
            return f.lock->native_handle();
        })
        .def("__enter__", [](py::object self) { return self; })
        .def("__exit__", [](LockFile& f, const py::args&) {
            if (f.lock) f.lock->close();
            return false;
        });

    m.def("dir_lock", [](const py::object& path) -> py::object {
        const fspath::FsPath p = fspath::pathlib_path(path);
        std::optional<dl::DirLock> l = os_guard([&] {
            py::gil_scoped_release nogil;
            return dl::DirLock::try_acquire(p.bytes);
        });
        if (!l) return py::none();
        LockFile f;
        f.name = fspath::fs_text(l->path());
        f.lock = std::move(l);
        return py::cast(std::move(f));
    }, py::arg("path"));
}

// --- FLAC (tests and measurements) ---------------------------------------------------

void bind_flac(py::module_& m) {
    m.def("_flac_encode", [](const py::object& path, const py::object& data, int fs, int blocksize,
                             int max_fixed_order, int max_partition_order, bool stereo, int seek_every) {
        const fspath::FsPath p = fspath::fs_path(path);
        auto a = py::array_t<std::int16_t, py::array::c_style | py::array::forcecast>::ensure(data);
        if (!a) throw py::error_already_set();
        if (a.ndim() != 2) throw py::value_error("expected (frames, channels) int16");
        const int ch = static_cast<int>(a.shape(1));
        const std::uint64_t frames = static_cast<std::uint64_t>(a.shape(0));
        const std::int16_t* d = a.data();
        flac::EncodeOptions opt;
        opt.blocksize = blocksize;
        opt.max_fixed_order = max_fixed_order;
        opt.max_partition_order = max_partition_order;
        opt.stereo_decorrelation = stereo;
        opt.seek_every_frames = seek_every;
        flac::EncodeStats st = os_guard([&] {
            py::gil_scoped_release nogil;
            return flac::encode_file(p.bytes, fs, ch, frames, [&](std::uint64_t first, std::size_t n, std::int16_t* out) {
                std::memcpy(out, d + first * static_cast<std::uint64_t>(ch), n * static_cast<std::size_t>(ch) * 2);
                return true;
            }, opt);
        });
        py::dict out;
        out["bytes"] = st.bytes;
        out["frames"] = st.frames;
        out["stereo"] = py::make_tuple(st.stereo[0], st.stereo[1], st.stereo[2], st.stereo[3]);
        py::list sub;
        for (const auto v : st.subframes) sub.append(v);
        out["subframes"] = sub;
        out["md5"] = py::bytes(reinterpret_cast<const char*>(st.md5.data()), st.md5.size());
        return out;
    }, py::arg("path"), py::arg("data"), py::arg("fs"), py::arg("blocksize") = 4096, py::arg("max_fixed_order") = 4,
          py::arg("max_partition_order") = flac::EncodeOptions{}.max_partition_order, py::arg("stereo") = true, py::arg("seek_every") = 16);

    m.def("_flac_read", [](const py::object& path, std::uint64_t start, const py::object& stop) {
        const fspath::FsPath p = fspath::fs_path(path);
        std::uint64_t n = UINT64_MAX;   // (to the end)
        if (!stop.is_none()) {
            const std::int64_t s = as_int(stop);
            n = s > static_cast<std::int64_t>(start) ? static_cast<std::uint64_t>(s) - start : 0;
        }
        std::vector<std::int16_t> y;
        int ch = 0;
        os_guard([&] {
            py::gil_scoped_release nogil;
            flac::Reader r(p.bytes);
            ch = r.info().channels;
            y = r.read(start, n);
        });
        py::array_t<std::int16_t> out({static_cast<py::ssize_t>(ch ? y.size() / static_cast<std::size_t>(ch) : 0),
                                       static_cast<py::ssize_t>(ch)});
        std::copy(y.begin(), y.end(), out.mutable_data());
        return out;
    }, py::arg("path"), py::arg("start") = 0, py::arg("stop") = py::none());

    m.def("_flac_info", [](const py::object& path) {
        const fspath::FsPath p = fspath::fs_path(path);
        flac::Info info = os_guard([&] {
            py::gil_scoped_release nogil;
            flac::Reader r(p.bytes);
            return r.info();
        });
        py::dict out;
        out["channels"] = info.channels;
        out["sample_rate"] = info.sample_rate;
        out["bits_per_sample"] = info.bits_per_sample;
        out["frames"] = info.frames;
        out["md5"] = py::bytes(reinterpret_cast<const char*>(info.md5.data()), info.md5.size());
        return out;
    }, py::arg("path"));
}

// --- the module ---------------------------------------------------------------------

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule(
        "passband", "frontend.PassbandStore, its FLAC packer and live.dir_lock (native/core/qrss/passband)");
    m.attr("PB_PACK_AFTER_S") = pb::PB_PACK_AFTER_S;
    m.attr("PB_LEAD_S") = pb::PB_LEAD_S;
    m.attr("PB_TAIL_S") = pb::PB_TAIL_S;

    m.def("slot_t0", [](const py::object& q) { return pb::slot_t0(as_int(q)); }, py::arg("q"));
    m.def("_merge_ranges", [](const py::iterable& rs) {
        std::vector<pb::Range> v;
        for (const py::handle r : rs) {
            const py::tuple t = py::tuple(py::reinterpret_borrow<py::object>(r));
            if (t.size() != 2)
                throw py::value_error("too many values to unpack (expected 2, got " + std::to_string(t.size()) + ")");
            PyObject* lo = PyNumber_Index(t[0].ptr());
            if (!lo) throw py::error_already_set();
            PyObject* hi = PyNumber_Index(t[1].ptr());
            if (!hi) {
                Py_DECREF(lo);
                throw py::error_already_set();
            }
            const py::object l = py::reinterpret_steal<py::object>(lo), h = py::reinterpret_steal<py::object>(hi);
            v.push_back({l.cast<std::int64_t>(), h.cast<std::int64_t>()});
        }
        py::list out;
        for (const pb::Range& r : pb::merge_ranges(std::move(v))) {
            py::list p;
            p.append(r[0]);
            p.append(r[1]);
            out.append(p);
        }
        return out;
    }, py::arg("rs"));
    m.def("_hour_of_name", [](const py::object& name) -> py::object {
        const auto h = pb::hour_of_name(rx::as_string(name));
        return h ? py::object(py::int_(*h)) : py::object(py::none());
    }, py::arg("name"));

    bind_store(m);
    bind_lock(m);
    bind_flac(m);
}

const Registrar reg("passband", 90, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
