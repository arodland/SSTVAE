// sstvae_native.qrss.store: the multi-pass store (native/core/qrss/store),
// port of sstvae/qrss/store.py, with its names, arguments and defaults.
//
//   N_GROUPS GROUP_LATENTS SENT DAY_S SLOT_S ZW_CACHE SAME_PASS_HZ
//   default_root() -> pathlib.Path
//   key_name(key) -> str            slot_time(q) -> float
//   mean_w_db(w) -> float           canonical_index(segment, n) -> intp[n] (read-only)
//   Accumulator(key, mode, codec_id, S, W, S_ext, W_ext, members=[], foreign=[],
//               parts={}, created=0.0, updated=0.0)
//       .empty(key, mode, codec_id, now)  (static)
//       .key .mode .codec_id .S .W .S_ext .W_ext .members .foreign .parts
//       .created .updated .uids
//       .member(uid) ._sums(store, skip=None, ext=True) .rebuild(store)
//       .loo(uid, store) .has_data(g) .median_w(g) .seg_snr_db() .meta()
//   Store(root=None, clock=time.time, *, replace=None, zw_cache=None)
//       .root (pathlib.Path) .clock ._zw (an OrderedDict copy of the cache)
//       every method of store.py's Store, private ones included
//       (_pass_path, _acc_paths, _prov_path -> pathlib.Path; _commit).
//
// How the objects behave against the reference's:
//
// - Accumulator.S, .W, .S_ext and .W_ext are float32 (3, 52800) views of the
//   C++ arrays (writable, kept alive by the accumulator); assigning one
//   copies into it (np.asarray(x, float32), which must hold 3 x 52800
//   values). store.py's rebuild *replaces* the arrays, so a view taken
//   before a rebuild keeps the old values there and sees the new ones here.
// - .members, .foreign, .parts and member()/twin()'s dicts are fresh Python
//   objects at each access: reading them is as in Python, changing one in
//   place does not reach the accumulator (assign the attribute instead).
// - pass_zw returns copies of the cached arrays, pass_info a fresh dict.
// - load_pass returns the PassResult as rx_types.hpp's "__type__" dict;
//   tests/qrss_native/_store.py's facade makes it the reference's.
// - clock is called (with the GIL) for every timestamp; replace, when given,
//   instead of the C++ os.replace, as replace(src, dst) with str paths;
//   zw_cache, when given, is an int or a callable asked for the cache size at
//   each read (the facade reads store.py's ZW_CACHE there, so a test's
//   monkeypatch of it reaches the C++).
//
// The GIL is released around every core call (store_types.hpp's `nogil`):
// a Store may be used from several Python threads, its own mutex
// serialising them. An Accumulator method or a Store method taking an
// accumulator works on a copy made with the GIL held and writes its results
// back (S, W, updated) with it held again.

#include "qrss/store_types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "qrss/store.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace store = sstvae::qrss::store;
using Acc = store::Accumulator;
using st::nogil;
using st::PyStore;
using st::Value;

store::Store& core(PyStore& s) { return *s.core; }

std::optional<std::string> opt_str(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return rx::as_string(o);
}

std::optional<std::int64_t> opt_int(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return rx::as_int(o);
}

// What the sums read of an accumulator (store.py's _sums: key, mode,
// members, S_ext, W_ext), copied with the GIL held.
Acc sums_copy(const Acc& a) {
    Acc c;
    c.key = a.key;
    c.mode = a.mode;
    c.codec_id = a.codec_id;
    c.members = a.members;
    c.S_ext = a.S_ext;
    c.W_ext = a.W_ext;
    return c;
}

py::tuple pair_py(const std::pair<std::vector<double>, std::vector<double>>& p) {
    return py::make_tuple(rx::arr(p.first), rx::arr(p.second));
}

std::vector<store::Part> parts_of(const py::handle& o) {
    std::vector<store::Part> out;
    if (o.is_none()) return out;
    for (const auto& [k, v] : py::reinterpret_borrow<py::dict>(o)) {
        const py::sequence sw = py::reinterpret_borrow<py::sequence>(v);
        if (sw.size() != 2) throw py::value_error("a foreign part is (S, W)");
        out.push_back({rx::as_string(k), st::npz_array(sw[0]), st::npz_array(sw[1])});
    }
    return out;
}

py::dict parts_py(const std::vector<store::Part>& parts) {
    py::dict d;
    for (const store::Part& p : parts) d[rx::to_str(p.sha)] = py::make_tuple(st::npz_py(p.S), st::npz_py(p.W));
    return d;
}

py::object time_time() { return py::module_::import("time").attr("time"); }

void bind_accumulator(py::module_& m) {
    py::class_<Acc>(m, "Accumulator", "One picture's accumulated latents, canonical [3, 52800] float32.")
        .def(py::init([](const py::object& key, const py::object& mode, const py::object& codec_id,
                         const py::object& S, const py::object& W, const py::object& S_ext, const py::object& W_ext,
                         const py::object& members, const py::object& foreign, const py::object& parts,
                         const py::object& created, const py::object& updated) {
                 Acc a;
                 a.key = st::key_index(key);
                 a.mode = rx::as_int(mode);
                 a.codec_id = rx::as_int(codec_id);
                 st::set_plane(a.S, S);
                 st::set_plane(a.W, W);
                 st::set_plane(a.S_ext, S_ext);
                 st::set_plane(a.W_ext, W_ext);
                 if (!members.is_none()) a.members = st::values_of(members);
                 if (!foreign.is_none()) a.foreign = st::values_of(foreign);
                 a.parts = parts_of(parts);
                 a.created = rx::as_double(created);
                 a.updated = rx::as_double(updated);
                 return a;
             }),
             py::arg("key"), py::arg("mode"), py::arg("codec_id"), py::arg("S"), py::arg("W"), py::arg("S_ext"),
             py::arg("W_ext"), py::arg("members") = py::none(), py::arg("foreign") = py::none(),
             py::arg("parts") = py::none(), py::arg("created") = 0.0, py::arg("updated") = 0.0)
        .def_static("empty", [](const py::object& key, const py::object& mode, const py::object& codec_id,
                                const py::object& now) {
            return Acc::empty(st::key_index(key), rx::as_int(mode), rx::as_int(codec_id), rx::as_double(now));
        }, py::arg("key"), py::arg("mode"), py::arg("codec_id"), py::arg("now"))
        .def_property("key", [](const Acc& a) { return st::key_py(a.key); },
                      [](Acc& a, const py::object& k) { a.key = st::key_index(k); })
        .def_property("mode", [](const Acc& a) { return a.mode; },
                      [](Acc& a, const py::object& v) { a.mode = rx::as_int(v); })
        .def_property("codec_id", [](const Acc& a) { return a.codec_id; },
                      [](Acc& a, const py::object& v) { a.codec_id = rx::as_int(v); })
        .def_property("S", [](const py::object& self) { return st::plane_view(self.cast<Acc&>().S, self); },
                      [](Acc& a, const py::object& v) { st::set_plane(a.S, v); })
        .def_property("W", [](const py::object& self) { return st::plane_view(self.cast<Acc&>().W, self); },
                      [](Acc& a, const py::object& v) { st::set_plane(a.W, v); })
        .def_property("S_ext", [](const py::object& self) { return st::plane_view(self.cast<Acc&>().S_ext, self); },
                      [](Acc& a, const py::object& v) { st::set_plane(a.S_ext, v); })
        .def_property("W_ext", [](const py::object& self) { return st::plane_view(self.cast<Acc&>().W_ext, self); },
                      [](Acc& a, const py::object& v) { st::set_plane(a.W_ext, v); })
        .def_property("members", [](const Acc& a) { return st::to_py(a.members); },
                      [](Acc& a, const py::object& v) { a.members = st::values_of(v); })
        .def_property("foreign", [](const Acc& a) { return st::to_py(a.foreign); },
                      [](Acc& a, const py::object& v) { a.foreign = st::values_of(v); })
        .def_property("parts", [](const Acc& a) { return parts_py(a.parts); },
                      [](Acc& a, const py::object& v) { a.parts = parts_of(v); })
        .def_property("created", [](const Acc& a) { return a.created; },
                      [](Acc& a, const py::object& v) { a.created = rx::as_double(v); })
        .def_property("updated", [](const Acc& a) { return a.updated; },
                      [](Acc& a, const py::object& v) { a.updated = rx::as_double(v); })
        .def_property_readonly("uids", [](const Acc& a) {
            py::list out;
            for (const Value& mm : a.members) out.append(st::to_py(mm.at("uid")));
            return out;
        })
        .def("member", [](const Acc& a, const py::object& uid) {
            return st::opt_py(a.member(rx::as_string(uid)));
        }, py::arg("uid"))
        .def("_sums", [](const Acc& a, PyStore& s, const py::object& skip, bool ext) {
            const Acc c = sums_copy(a);
            const std::optional<std::string> sk = opt_str(skip);
            const Acc::Sums r = nogil([&] {
                return c.sums(core(s), sk ? std::optional<std::string_view>(*sk) : std::nullopt, ext);
            });
            return py::make_tuple(st::plane_py(r.S), st::plane_py(r.W));
        }, py::arg("store"), py::arg("skip") = py::none(), py::arg("ext") = true)
        .def("rebuild", [](Acc& a, PyStore& s) {
            Acc c = sums_copy(a);
            nogil([&] { c.rebuild(core(s)); });
            a.S.resize(store::PLANE);   // (a no-op: S and W always hold PLANE values)
            a.W.resize(store::PLANE);
            std::copy(c.S.begin(), c.S.end(), a.S.begin());   // in place: views stay valid
            std::copy(c.W.begin(), c.W.end(), a.W.begin());
        }, py::arg("store"))
        .def("loo", [](const Acc& a, const py::object& uid, PyStore& s) {
            const Acc c = sums_copy(a);
            const std::string u = rx::as_string(uid);
            return pair_py(nogil([&] { return c.loo(u, core(s)); }));
        }, py::arg("uid"), py::arg("store"))
        .def("has_data", [](const Acc& a, const py::object& g) { return a.has_data(rx::as_int(g)); }, py::arg("g"))
        .def("median_w", [](const Acc& a, const py::object& g) { return a.median_w(rx::as_int(g)); }, py::arg("g"))
        .def("seg_snr_db", [](const Acc& a) {
            py::list out;
            for (const auto& v : a.seg_snr_db()) out.append(v ? py::object(py::float_(*v)) : py::object(py::none()));
            return out;
        })
        .def("meta", [](const Acc& a) { return st::to_py(a.meta()); })
        .def("__copy__", [](const Acc& a) { return Acc(a); })
        .def("__deepcopy__", [](const Acc& a, const py::object&) { return Acc(a); }, py::arg("memo"))
        .def("__repr__", [](const Acc& a) {
            return py::str("Accumulator(key={!r}, mode={!r}, codec_id={!r}, members={!r}, foreign={!r}, "
                           "created={!r}, updated={!r})")
                .format(st::key_py(a.key), a.mode, a.codec_id, st::to_py(a.members), st::to_py(a.foreign),
                        a.created, a.updated);
        });
}

std::unique_ptr<PyStore> make_store(const py::object& root, const py::object& clock, const py::object& replace,
                                    const py::object& zw_cache) {
    auto s = std::make_unique<PyStore>();
    s->clock = clock;
    s->replace = replace;
    s->zw_cache = zw_cache;
    std::optional<std::string> r;
    if (!root.is_none()) r = fspath::pathlib_path(root).bytes;
    store::Store::Seams seams;
    PyStore* p = s.get();
    seams.clock = [p]() {
        py::gil_scoped_acquire gil;
        return rx::as_double(p->clock());
    };
    if (!replace.is_none()) {
        seams.replace = [p](const std::string& from, const std::string& to) {
            py::gil_scoped_acquire gil;
            p->replace(fspath::fs_text(from), fspath::fs_text(to));
        };
    }
    if (!zw_cache.is_none()) {
        if (PyLong_Check(zw_cache.ptr())) {
            const std::int64_t n = rx::as_int(zw_cache);
            const std::size_t cap = n < 0 ? 0 : static_cast<std::size_t>(n);
            seams.zw_cache = [cap]() { return cap; };
        } else {
            seams.zw_cache = [p]() {
                py::gil_scoped_acquire gil;
                const std::int64_t n = rx::as_int(p->zw_cache());
                return n < 0 ? std::size_t{0} : static_cast<std::size_t>(n);
            };
        }
    }
    s->core = nogil([&] { return std::make_unique<store::Store>(r, std::move(seams)); });
    return s;
}

void bind_store(py::module_& m) {
    py::class_<PyStore>(m, "Store", "The on-disk multi-pass store (sstvae/qrss/store.py's Store, in C++).")
        .def(py::init(&make_store), py::arg("root") = py::none(), py::arg("clock") = time_time(), py::kw_only(),
             py::arg("replace") = py::none(), py::arg("zw_cache") = py::none())
        .def_property_readonly("root", [](PyStore& s) { return st::path_py(core(s).root()); })
        .def_property("clock", [](PyStore& s) { return s.clock; },
                      [](PyStore& s, const py::object& c) { s.clock = c; })
        .def_property_readonly("_zw", [](PyStore& s) {
            const auto items = nogil([&] { return core(s).zw_items(); });
            py::object od = py::module_::import("collections").attr("OrderedDict")();
            for (const auto& [uid, zw] : items) od[rx::to_str(uid)] = py::make_tuple(rx::arr(zw->z), rx::arr(zw->w));
            return od;
        })
        // --- passes ---
        .def("_pass_path", [](PyStore& s, const py::object& uid) {
            return st::path_py(core(s).pass_path(rx::as_string(uid)));
        }, py::arg("uid"))
        .def("add_pass", [](PyStore& s, const py::object& p) {
            const auto r = rx::pass_result(p);
            nogil([&] { core(s).add_pass(r); });
        }, py::arg("p"))
        .def("has_pass", [](PyStore& s, const py::object& uid) {
            const std::string u = rx::as_string(uid);
            return nogil([&] { return core(s).has_pass(u); });
        }, py::arg("uid"))
        .def("load_pass", [](PyStore& s, const py::object& uid) {
            const std::string u = rx::as_string(uid);
            return rx::to_py(nogil([&] { return core(s).load_pass(u); }));
        }, py::arg("uid"))
        .def("pass_uids", [](PyStore& s) {
            py::list out;
            for (const std::string& u : nogil([&] { return core(s).pass_uids(); })) out.append(rx::to_str(u));
            return out;
        })
        .def("pass_zw", [](PyStore& s, const py::object& uid) {
            const std::string u = rx::as_string(uid);
            const store::ZWPtr zw = nogil([&] { return core(s).pass_zw(u); });
            return py::make_tuple(rx::arr(zw->z), rx::arr(zw->w));
        }, py::arg("uid"))
        .def("pass_info", [](PyStore& s, const py::object& uid) {
            const std::string u = rx::as_string(uid);
            return st::info_py(nogil([&] { return core(s).pass_info(u); }));
        }, py::arg("uid"))
        .def("update_pass", [](PyStore& s, const py::object& p) {
            const auto r = rx::pass_result(p);
            return st::keys_py(nogil([&] { return core(s).update_pass(r); }));
        }, py::arg("p"))
        // --- accumulators ---
        .def("_acc_paths", [](PyStore& s, const py::object& key) {
            const auto [j, n] = core(s).acc_paths(st::key_of(key));
            return py::make_tuple(st::path_py(j), st::path_py(n));
        }, py::arg("key"))
        .def("has_accumulator", [](PyStore& s, const py::object& key) {
            const store::Key k = st::key_of(key);
            return nogil([&] { return core(s).has_accumulator(k); });
        }, py::arg("key"))
        .def("accumulator", [](PyStore& s, const py::object& key) {
            const store::Key k = st::key_of(key);
            return nogil([&] { return core(s).accumulator(k); });
        }, py::arg("key"))
        .def("create", [](PyStore& s, const py::object& key, const py::object& mode, const py::object& codec_id) {
            const store::Key k = st::key_of(key);
            const std::int64_t md = rx::as_int(mode), cid = rx::as_int(codec_id);
            return nogil([&] { return core(s).create(k, md, cid); });
        }, py::arg("key"), py::arg("mode"), py::arg("codec_id"))
        .def("save_accumulator", [](PyStore& s, const Acc& acc) {
            const Acc c = acc;
            nogil([&] { core(s).save_accumulator(c); });
        }, py::arg("acc"))
        .def("_commit", [](PyStore& s, Acc& acc) {
            Acc c = acc;
            nogil([&] { core(s).commit(c); });
            acc.S.resize(store::PLANE);
            acc.W.resize(store::PLANE);
            std::copy(c.S.begin(), c.S.end(), acc.S.begin());
            std::copy(c.W.begin(), c.W.end(), acc.W.begin());
            acc.updated = c.updated;
        }, py::arg("acc"))
        .def("open_keys", [](PyStore& s) { return st::keys_py(nogil([&] { return core(s).open_keys(); })); })
        .def("due", [](PyStore& s, const py::object& q) {
            const std::int64_t qq = rx::as_int(q);
            py::list out;
            for (const store::Due& d : nogil([&] { return core(s).due(qq); })) out.append(st::due_py(d));
            return out;
        }, py::arg("q"))
        .def("find_member", [](PyStore& s, const py::object& uid) {
            const std::string u = rx::as_string(uid);
            return st::opt_key(nogil([&] { return core(s).find_member(u); }));
        }, py::arg("uid"))
        .def("attach", [](PyStore& s, const py::object& key, const py::object& uid, const py::object& segment,
                          const py::object& method, const py::object& mode, const py::object& codec_id) {
            const store::Key k = st::key_of(key);
            const std::string u = rx::as_string(uid);
            const std::int64_t g = rx::as_int(segment);
            const std::string meth = rx::as_string(py::str(method));
            const auto md = opt_int(mode), cid = opt_int(codec_id);
            return nogil([&] { return core(s).attach(k, u, g, meth, md, cid); });
        }, py::arg("key"), py::arg("uid"), py::arg("segment"), py::arg("method"), py::kw_only(),
           py::arg("mode") = py::none(), py::arg("codec_id") = py::none())
        .def("twin", [](PyStore& s, const Acc& acc, const py::object& q, const py::object& f_hz,
                        const py::object& segment, const py::object& uid, const py::object& waveform) {
            Acc c;
            c.key = acc.key;
            c.members = acc.members;
            const std::int64_t qq = rx::as_int(q), g = rx::as_int(segment);
            const double f = rx::as_double(f_hz);
            const std::optional<std::string> u = opt_str(uid);
            const auto wf = opt_int(waveform);
            return st::opt_py(nogil([&] {
                return core(s).twin(c, qq, f, g, u ? std::optional<std::string_view>(*u) : std::nullopt, wf);
            }));
        }, py::arg("acc"), py::arg("q"), py::arg("f_hz"), py::arg("segment"), py::arg("uid") = py::none(),
           py::arg("waveform") = py::none())
        .def("detach", [](PyStore& s, const py::object& key, const py::object& uid) {
            const store::Key k = st::key_of(key);
            const std::string u = rx::as_string(uid);
            return nogil([&] { return core(s).detach(k, u); });
        }, py::arg("key"), py::arg("uid"))
        .def("delete_accumulator", [](PyStore& s, const py::object& key) {
            const store::Key k = st::key_of(key);
            nogil([&] { core(s).delete_accumulator(k); });
        }, py::arg("key"))
        .def("merge", [](PyStore& s, const py::object& key, const py::object& S, const py::object& W,
                         const py::object& source) {
            const store::Key k = st::key_of(key);
            const std::vector<float> SS = rx::vec<float>(S), WW = rx::vec<float>(W);
            const std::string src = rx::as_string(source);
            return nogil([&] { return core(s).merge(k, SS, WW, src); });
        }, py::arg("key"), py::arg("S"), py::arg("W"), py::arg("source") = "")
        // --- provisional clusters ---
        .def("_prov_path", [](PyStore& s, const py::object& cid) {
            return st::path_py(core(s).prov_path(rx::as_string(cid)));
        }, py::arg("cid"))
        .def("prov_clusters", [](PyStore& s) { return st::to_py(nogil([&] { return core(s).prov_clusters(); })); })
        .def("prov_cluster_of", [](PyStore& s, const py::object& uid) {
            const std::string u = rx::as_string(uid);
            return st::opt_py(nogil([&] { return core(s).prov_cluster_of(u); }));
        }, py::arg("uid"))
        .def("prov_add", [](PyStore& s, const py::object& uid, const py::object& cluster) {
            const std::string u = rx::as_string(uid);
            const std::optional<std::string> c = opt_str(cluster);
            return rx::to_str(nogil([&] { return core(s).prov_add(u, c); }));
        }, py::arg("uid"), py::arg("cluster") = py::none())
        .def("prov_remove", [](PyStore& s, const py::object& uid) {
            const std::string u = rx::as_string(uid);
            nogil([&] { core(s).prov_remove(u); });
        }, py::arg("uid"))
        // --- housekeeping ---
        .def("log", [](PyStore& s, const py::object& event) {
            if (!PyDict_Check(event.ptr()))
                throw py::type_error("'" + st::class_name(event) + "' object is not a mapping");
            const Value e = st::to_value(event, st::Unknown::as_float);
            nogil([&] { core(s).log(e); });
        }, py::arg("event"))
        .def("expire", [](PyStore& s, const py::object& now, const py::object& days) {
            const std::optional<double> t = now.is_none() ? std::nullopt : std::optional<double>(rx::as_double(now));
            const double d = rx::as_double(days);
            const store::Expired r = nogil([&] { return core(s).expire(t, d); });
            py::dict out;
            out["acc"] = st::keys_py(r.acc);
            out["prov"] = st::to_py(r.prov);
            py::list passes;
            for (const std::string& u : r.passes) passes.append(rx::to_str(u));
            out["passes"] = passes;
            return out;
        }, py::arg("now") = py::none(), py::arg("days") = 7.0);
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("store", "Port of sstvae/qrss/store.py (native/core/qrss/store)");
    m.attr("N_GROUPS") = store::N_GROUPS;
    m.attr("GROUP_LATENTS") = store::GROUP_LATENTS;
    m.attr("SENT") = store::SENT;
    m.attr("DAY_S") = store::DAY_S;
    m.attr("SLOT_S") = store::SLOT_S;
    m.attr("ZW_CACHE") = store::ZW_CACHE;
    m.attr("SAME_PASS_HZ") = store::SAME_PASS_HZ;

    m.def("default_root", [] { return st::path_py(store::default_root()); });
    m.def("key_name", [](const py::object& key) { return rx::to_str(store::key_name(st::key_of(key))); },
          py::arg("key"));
    m.def("slot_time", [](const py::object& q_) { return store::slot_time(rx::as_int(q_)); }, py::arg("q"));
    m.def("mean_w_db", [](const py::object& w) {
        const std::vector<double> v = rx::vec<double>(w);
        return store::mean_w_db(std::span<const double>(v));
    }, py::arg("w"));
    m.def("canonical_index", [](const py::object& segment, const py::object& n) {
        if (!PyIndex_Check(n.ptr())) {
            // The reference's order for an n that is not an integer: its range
            // check (Python's comparisons, so its errors), the segment, then the
            // slice [:n], which refuses it.
            const py::int_ lo(0), hi(store::SENT);
            const int a = PyObject_RichCompareBool(lo.ptr(), n.ptr(), Py_LT);
            if (a < 0) throw py::error_already_set();
            const int b = a ? PyObject_RichCompareBool(n.ptr(), hi.ptr(), Py_LE) : 0;
            if (b < 0) throw py::error_already_set();
            if (!a || !b)
                throw py::value_error("a pass carries 1.." + std::to_string(store::SENT) + " latents, not " +
                                      rx::as_string(py::str(n)));
            (void)store::canonical_index(rx::as_int(segment), 1);
            throw py::type_error("slice indices must be integers or None or have an __index__ method");
        }
        const auto idx = store::canonical_index(rx::as_int(segment), rx::as_int(n));
        py::array_t<py::ssize_t> out(static_cast<py::ssize_t>(idx.size()));
        py::ssize_t* d = out.mutable_data();
        for (std::size_t i = 0; i < idx.size(); ++i) d[i] = idx[i];
        out.attr("setflags")(py::arg("write") = false);
        return out;
    }, py::arg("segment"), py::arg("n"));

    bind_accumulator(m);
    bind_store(m);
}

const Registrar reg("store", 95, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
