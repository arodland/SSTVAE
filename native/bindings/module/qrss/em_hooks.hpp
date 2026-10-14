// The hooks of qrss/em.hpp (em::Hooks: log, progress, cancel) for a call
// made from Python: what sstvae_native.qrss.em and the bindings built on it
// (the EM loop's, the retroactive search's) hand the C++.
//
//   log       a Python callable taking one str (Python's `log=print`
//             convention: the line exactly as the reference hands it), or
//             None for no log;
//   progress  a Python callable taking (done, total, what), or None;
//   cancel    a sstvae_native.qrss.em.CancelFlag (bound in em.cpp; set()
//             from any thread, Python's or not), or None: never cancelled.
//
// The hooks are called on the thread that made the call, from inside the
// GIL-released region: each takes the GIL for its call and releases it
// after, so a Python hook may run any Python (including starting another
// native call; never the same call re-entrantly). An exception a Python hook
// raises ends the native call with that exception (py::error_already_set
// carries it through the C++ and re-raises it), at the same consistent
// point em.hpp describes for a cancel. em::Cancelled raises as
// sstvae_native.qrss.em.Cancelled (a RuntimeError).
//
// The Python objects are owned by the returned PyHooks (and by every copy of
// its std::functions) and released with the GIL held wherever the last copy
// dies, so the hooks may outlive the call that made them. Building a
// PyHooks needs the GIL; keep the PyHooks (whose `cancel_owner` keeps the
// CancelFlag, and so the atomic em::Hooks::cancel points at, alive) until the
// native call has returned.

#pragma once

#include "qrss/store_types.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "qrss/em.hpp"

namespace sstvae_bindings::qrss::emh {

namespace em = sstvae::qrss::em;

// A cancel flag Python holds and sets (sstvae_native.qrss.em.CancelFlag).
struct CancelFlag {
    std::atomic<bool> flag{false};
};

// A Python object owned by a C++ closure, released with the GIL held.
inline std::shared_ptr<py::object> held_object(py::object o) {
    return std::shared_ptr<py::object>(new py::object(std::move(o)), [](py::object* p) {
        py::gil_scoped_acquire gil;
        delete p;
    });
}

struct PyHooks {
    em::Hooks hooks;
    py::object cancel_owner;   // the CancelFlag em::Hooks::cancel points into (None: none)
};

// em::Hooks from Python's log / progress / cancel (see above). TypeError for
// a log or progress that is not callable, or a cancel that is not a
// CancelFlag. Needs the GIL.
inline PyHooks hooks_from_py(const py::object& log, const py::object& progress, const py::object& cancel) {
    PyHooks out;
    if (!log.is_none()) {
        if (!PyCallable_Check(log.ptr())) throw py::type_error("log must be callable or None");
        auto fn = held_object(log);
        out.hooks.log = [fn](const std::string& line) {
            py::gil_scoped_acquire gil;
            (*fn)(rx::to_str(line));
        };
    }
    if (!progress.is_none()) {
        if (!PyCallable_Check(progress.ptr())) throw py::type_error("progress must be callable or None");
        auto fn = held_object(progress);
        out.hooks.progress = [fn](std::int64_t done, std::int64_t total, const std::string& what) {
            py::gil_scoped_acquire gil;
            (*fn)(done, total, rx::to_str(what));
        };
    }
    out.cancel_owner = cancel;
    if (!cancel.is_none()) {
        if (!py::isinstance<CancelFlag>(cancel)) throw py::type_error("cancel must be a CancelFlag or None");
        out.hooks.cancel = &cancel.cast<CancelFlag&>().flag;
    }
    return out;
}

}  // namespace sstvae_bindings::qrss::emh
