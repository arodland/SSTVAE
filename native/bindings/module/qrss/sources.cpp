// sstvae_native.qrss.sources: bindings of native/core/qrss/sources (live.py's
// audio sources). Names and arguments follow the reference's:
//
//   Stamper(hold_s=STAMP_HOLD_S): hold_s, push(x, t), flush()
//       -- tests/qrss_native/sources.py assigns it over live.Stamper
//   StdinSource(fmt="f32", chunk_s=0.25, stream=None, hold_s=STAMP_HOLD_S,
//               *, clock=None): name, eof, chunks(timeout=1.0)
//       and, beyond the reference: next(timeout), close(), error
//   WavSource(path, start_unix, speed=0.0, chunk_s=1.0, *, clock=None,
//             sleep=None): x, start, speed, chunk, name, eof,
//             chunks(timeout=1.0)
//
// push and flush return [(x, t_end)] with x float64 (the reference keeps
// the dtype it is given; every source gives it float64). StdinSource reads
// a file descriptor, not a Python stream: `stream` is the reference's
// (sys.stdin.buffer when None) and its fileno() is what the C++ reads, so
// bytes already in that stream's buffer are not seen (nothing has read
// them when a listener starts); an int is taken as a descriptor. The
// stream is kept alive, never closed.
//
// `clock` (and WavSource's `sleep`): a callable standing in for
// time.time() (time.sleep()), for the parity tests; None is the C++'s own.
// StdinSource calls its clock on the reader thread, taking the GIL for it,
// so every call that waits on the reader releases the GIL first: chunks()
// and next() while they wait, close() and the destructor while they join.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "qrss/fspath.hpp"
#include "qrss/live.hpp"
#include "qrss/registry.hpp"
#include "qrss/sources.hpp"
#include "qrss/wavread.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace lv = sstvae::qrss::live;
namespace wav = sstvae::qrss::wav;
using namespace pybind11::literals;

py::list chunks_py(const std::vector<lv::Chunk>& v) {
    py::list out;
    for (const lv::Chunk& c : v) out.append(py::make_tuple(to_numpy(c.x), c.t_end));
    return out;
}

// A Python callable as a seam called from any thread: the GIL taken for
// the call and for the last reference's release.
std::shared_ptr<py::object> hold(py::object f) {
    return std::shared_ptr<py::object>(new py::object(std::move(f)), [](py::object* p) {
        py::gil_scoped_acquire g;
        delete p;
    });
}

lv::Clock clock_of(const py::object& f) {
    if (f.is_none()) return {};
    auto h = hold(f);
    return [h]() -> double {
        if (Py_IsFinalizing()) throw std::runtime_error("the interpreter is finalizing");
        py::gil_scoped_acquire g;
        const py::object r = (*h)();
        return r.cast<double>();
    };
}

lv::Sleep sleep_of(const py::object& f) {
    if (f.is_none()) return {};
    auto h = hold(f);
    return [h](double s) {
        py::gil_scoped_acquire g;
        (*h)(s);
    };
}

// FormatError as the exception the reference raises (as receiver_slot's binding)
[[noreturn]] void raise_format_error(const wav::FormatError& e) {
    using K = wav::FormatError::Kind;
    py::object cls;
    if (e.kind == K::struct_error) {
        cls = py::module_::import("struct").attr("error");
    } else {
        PyObject* c = e.kind == K::zero_division   ? PyExc_ZeroDivisionError
                      : e.kind == K::type          ? PyExc_TypeError
                      : e.kind == K::unbound_local ? PyExc_UnboundLocalError
                                                   : PyExc_ValueError;
        cls = py::reinterpret_borrow<py::object>(c);
    }
    PyErr_SetString(cls.ptr(), e.what());
    throw py::error_already_set();
}

void warn_wav(const std::vector<wav::Warning>& warnings) {
    for (const wav::Warning& w : warnings) {
        const py::object cat = w.kind == wav::Warning::Kind::runtime
                                   ? py::reinterpret_borrow<py::object>(PyExc_RuntimeWarning)
                                   : py::module_::import("scipy.io.wavfile").attr("WavFileWarning");
        if (PyErr_WarnEx(cat.ptr(), w.text.c_str(), 1) < 0) throw py::error_already_set();
    }
}

// What ended a stream, as a Python object: the exception itself when it was
// Python's, OSError for a read error, RuntimeError(what) otherwise.
py::object error_py(const std::exception_ptr& e) {
    if (!e) return py::none();
    try {
        std::rethrow_exception(e);
    } catch (py::error_already_set& x) {
        return x.value();
    } catch (const std::system_error& x) {
        if (x.code().category() == std::generic_category())
            return py::reinterpret_borrow<py::object>(PyExc_OSError)(x.code().value(), x.what());
        return py::reinterpret_borrow<py::object>(PyExc_OSError)(x.what());
    } catch (const std::exception& x) {
        return py::reinterpret_borrow<py::object>(PyExc_RuntimeError)(x.what());
    }
}

struct PyStdin {
    std::unique_ptr<lv::StreamSource> src;
    py::object stream;
    ~PyStdin() {
        if (src) {
            py::gil_scoped_release nogil;
            src.reset();
        }
    }
};

struct PyWav {
    std::unique_ptr<lv::WavSource> src;
    py::object x;          // the samples, read-only
    py::object name;
    double speed = 0.0;
    std::int64_t chunk = 0;
    ~PyWav() {
        if (src) {
            py::gil_scoped_release nogil;
            src.reset();
        }
    }
};

// chunks(timeout): live.py's generator as an iterator -- (x, t_end), or
// (None, None) when timeout passed with none, until the end.
template <typename W>
struct ChunkIter {
    std::shared_ptr<W> w;
    double timeout;
};

template <typename W>
py::object next_py(W& w, double timeout) {
    std::optional<lv::Chunk> c;
    {
        py::gil_scoped_release nogil;
        c = w.src->next(timeout);
    }
    if (c) return py::make_tuple(to_numpy(c->x), c->t_end);
    if (w.src->eof()) throw py::stop_iteration();
    return py::make_tuple(py::none(), py::none());
}

template <typename W>
void bind_iter(py::module_& m, const char* name) {
    py::class_<ChunkIter<W>>(m, name)
        .def("__iter__", [](ChunkIter<W>& it) -> ChunkIter<W>& { return it; },
             py::return_value_policy::reference_internal)
        .def("__next__", [](ChunkIter<W>& it) { return next_py(*it.w, it.timeout); });
}

void bind(py::module_& qrss) {
    py::module_ m = qrss.def_submodule("sources", "native/core/qrss/sources (live.py's audio sources)");
    m.attr("STAMP_HOLD_S") = lv::STAMP_HOLD_S;
    m.attr("RESYNC_S") = lv::RESYNC_S;

    py::class_<lv::Stamper>(m, "Stamper")
        .def(py::init<double>(), "hold_s"_a = lv::STAMP_HOLD_S)
        .def_property_readonly("hold_s", &lv::Stamper::hold_s)
        .def(
            "push",
            [](lv::Stamper& s, const py::handle& x, double t) {
                std::vector<double> v = from_numpy<double>(x);
                std::vector<lv::Chunk> out;
                {
                    py::gil_scoped_release nogil;
                    out = s.push(std::move(v), t);
                }
                return chunks_py(out);
            },
            "x"_a, "t"_a)
        .def("flush", [](lv::Stamper& s) {
            std::vector<lv::Chunk> out;
            {
                py::gil_scoped_release nogil;
                out = s.flush();
            }
            return chunks_py(out);
        });

    bind_iter<PyStdin>(m, "_StdinChunks");
    bind_iter<PyWav>(m, "_WavChunks");

    py::class_<PyStdin, std::shared_ptr<PyStdin>>(m, "StdinSource")
        .def(py::init([](const std::string& fmt, double chunk_s, py::object stream, double hold_s,
                         const py::object& clock) {
                 if (stream.is_none()) stream = py::module_::import("sys").attr("stdin").attr("buffer");
                 int fd = 0;
                 if (py::isinstance<py::int_>(stream))
                     fd = stream.cast<int>();
                 else
                     fd = stream.attr("fileno")().cast<int>();
                 auto w = std::make_shared<PyStdin>();
                 w->stream = stream;
                 lv::StreamOptions o;
                 o.fmt = fmt;
                 o.chunk_s = chunk_s;
                 o.hold_s = hold_s;
                 o.clock = clock_of(clock);
                 w->src = std::make_unique<lv::StreamSource>(lv::InputHandle::from_fd(fd), std::move(o));
                 return w;
             }),
             "fmt"_a = "f32", "chunk_s"_a = 0.25, "stream"_a = py::none(), "hold_s"_a = lv::STAMP_HOLD_S,
             py::kw_only(), "clock"_a = py::none())
        .def_property_readonly("name", [](PyStdin& w) { return w.src->name(); })
        .def_property_readonly("eof", [](PyStdin& w) { return w.src->eof(); })
        .def_property_readonly("stream", [](PyStdin& w) { return w.stream; })
        .def("chunks", [](const std::shared_ptr<PyStdin>& w, double timeout) {
                 return ChunkIter<PyStdin>{w, timeout};
             },
             "timeout"_a = 1.0)
        .def("next", [](PyStdin& w, double timeout) -> py::object {
                 std::optional<lv::Chunk> c;
                 {
                     py::gil_scoped_release nogil;
                     c = w.src->next(timeout);
                 }
                 if (c) return py::make_tuple(to_numpy(c->x), c->t_end);
                 return py::none();
             },
             "timeout"_a = 1.0)
        .def("close", [](PyStdin& w) {
            py::gil_scoped_release nogil;
            w.src->close();
        })
        .def_property_readonly("error", [](PyStdin& w) { return error_py(w.src->error()); });

    py::class_<PyWav, std::shared_ptr<PyWav>>(m, "WavSource")
        .def(py::init([](const py::object& path, double start_unix, double speed, double chunk_s,
                         const py::object& clock, const py::object& sleep) {
                 // wavio.read_wav(str(path)), its warnings and refusals the reference's
                 const fspath::FsPath p = fspath::str_path(path);
                 std::vector<wav::Warning> warnings;
                 std::vector<double> x;
                 try {
                     py::gil_scoped_release nogil;
                     x = wav::read_wav(p.bytes, &warnings);
                 } catch (const wav::ReadError& e) {
                     fspath::raise_os_error(e.err, p.name);
                 } catch (const wav::FormatError& e) {
                     warn_wav(warnings);
                     raise_format_error(e);
                 }
                 warn_wav(warnings);
                 lv::WavOptions o;
                 o.speed = speed;
                 o.chunk_s = chunk_s;
                 o.clock = clock_of(clock);
                 o.sleep = sleep_of(sleep);
                 o.shown_as = std::string();
                 auto w = std::make_shared<PyWav>();
                 w->src = std::make_unique<lv::WavSource>(std::move(x), std::string(), start_unix, std::move(o));
                 // f"{path} (replay from ...)": the path as Python formats it
                 w->name = py::str(path) + py::str(w->src->name());
                 w->x = to_numpy(w->src->samples(), false);
                 w->speed = speed;
                 w->chunk = static_cast<std::int64_t>(chunk_s * 8000.0);
                 return w;
             }),
             "path"_a, "start_unix"_a, "speed"_a = 0.0, "chunk_s"_a = 1.0, py::kw_only(),
             "clock"_a = py::none(), "sleep"_a = py::none())
        .def_property_readonly("x", [](PyWav& w) { return w.x; })
        .def_property_readonly("start", [](PyWav& w) { return w.src->start(); })
        .def_property_readonly("speed", [](PyWav& w) { return w.speed; })
        .def_property_readonly("chunk", [](PyWav& w) { return w.chunk; })
        .def_property_readonly("name", [](PyWav& w) { return w.name; })
        .def_property_readonly("eof", [](PyWav& w) { return w.src->eof(); })
        .def("chunks", [](const std::shared_ptr<PyWav>& w, double timeout) {
                 return ChunkIter<PyWav>{w, timeout};
             },
             "timeout"_a = 1.0);
}

const Registrar reg("sources", 93, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
