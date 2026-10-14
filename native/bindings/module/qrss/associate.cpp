// sstvae_native.qrss.associate: association rules 1 to 4
// (native/core/qrss/associate), port of sstvae/qrss/associate.py with its
// names, arguments and defaults.
//
//   T_ACCEPT RHO_MIN KURTOSIS VALIDATE_MEDIAN_W HDR_REJECT_T HDR_REJECT_EXPECT
//   PAIR_HZ PAIR_DAYS HDR_AGREE_Z HDR_OTHER_FLIP
//   corr_stat(z, w, m, W) -> (t, M)    t an np.float64, or the float 0.0 as
//                                      the reference returns when nothing
//                                      correlates
//   expected_t(w, W, kurtosis=3.0) -> float
//   corr_accept(t, t_exp) -> bool      an np.bool_ where the reference's
//                                      comparisons give one (t or t_exp numpy)
//   _acc_reference(acc, g, n) -> (m, W)
//   _cluster_sums(store, uids) -> (m, W)
//   _best_accumulator(store, z, w, exclude=()) -> (t, key, g) | None
//   validate_header_match(store, key, mode, segment, z, w) -> (ok, details)
//   _attach_by_header(store, h, uids, method) -> [uid]
//   header_agrees(h, llr) -> bool
//   _soft_header(store, uid, llr) -> (HeaderFields "__type__" dict, [uid]) | None
//   _rule(store, key, uid, rule) -> str
//   associate(p, store, soft_header=None) -> (key | None, rule)
//   resolve_provisional(store) -> [(cluster, key)]
//
// `store` is sstvae_native.qrss.store.Store (or the rows' facade, a
// subclass), `acc` its Accumulator, `h` a HeaderFields (or anything with its
// fields), `p` a PassResult. associate's `soft_header`, when given, replaces
// rule 2's search: called as soft_header(store, uid, llr) like the
// reference's _soft_header, returning (h, uids) or None
// (tests/qrss_native/_store.py passes a test's monkeypatch of
// associate._soft_header through it). The rule strings are a GUI contract
// (associate.hpp).
//
// The GIL is released around each core call; associate and
// resolve_provisional hold the store's mutex for the whole call, and the
// soft_header seam takes the GIL back to call Python.

#include "qrss/store_types.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/associate.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace as = sstvae::qrss::associate;
namespace store = sstvae::qrss::store;
using st::nogil;
using st::PyStore;

store::Store& core(PyStore& s) { return *s.core; }

std::vector<std::string> strings(const py::handle& o) {
    std::vector<std::string> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o))) out.push_back(rx::as_string(h));
    return out;
}

py::list strings_py(const std::vector<std::string>& v) {
    py::list out;
    for (const std::string& s : v) out.append(rx::to_str(s));
    return out;
}

py::object np_float(double x) { return rx::np_module().attr("float64")(x); }

bool is_numpy(const py::handle& o) {
    static py::object* generic = new py::object(rx::np_module().attr("generic"));
    return py::isinstance(o, *generic);
}

py::tuple pair_py(const std::pair<std::vector<double>, std::vector<double>>& p) {
    return py::make_tuple(rx::arr(p.first), rx::arr(p.second));
}

// A Python object owned by a C++ closure, released with the GIL held
// wherever the closure dies.
std::shared_ptr<py::object> held(py::object o) {
    return std::shared_ptr<py::object>(new py::object(std::move(o)), [](py::object* p) {
        py::gil_scoped_acquire gil;
        delete p;
    });
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("associate", "Port of sstvae/qrss/associate.py (native/core/qrss/associate)");
    m.attr("T_ACCEPT") = as::T_ACCEPT;
    m.attr("RHO_MIN") = as::RHO_MIN;
    m.attr("KURTOSIS") = as::KURTOSIS;
    m.attr("VALIDATE_MEDIAN_W") = as::VALIDATE_MEDIAN_W;
    m.attr("HDR_REJECT_T") = as::HDR_REJECT_T;
    m.attr("HDR_REJECT_EXPECT") = as::HDR_REJECT_EXPECT;
    m.attr("PAIR_HZ") = as::PAIR_HZ;
    m.attr("PAIR_DAYS") = as::PAIR_DAYS;
    m.attr("HDR_AGREE_Z") = as::HDR_AGREE_Z;
    m.attr("HDR_OTHER_FLIP") = as::HDR_OTHER_FLIP;

    m.def("corr_stat", [](const py::object& z, const py::object& w, const py::object& m_, const py::object& W) {
        const auto zz = rx::vec<double>(z), ww = rx::vec<double>(w), mm = rx::vec<double>(m_), WW = rx::vec<double>(W);
        const as::Corr c = nogil([&] { return as::corr_stat(zz, ww, mm, WW); });
        return py::make_tuple(c.numpy ? np_float(c.t) : py::object(py::float_(c.t)), c.M);
    }, py::arg("z"), py::arg("w"), py::arg("m"), py::arg("W"));
    m.def("expected_t", [](const py::object& w, const py::object& W, const py::object& kurtosis) {
        const auto ww = rx::vec<double>(w), WW = rx::vec<double>(W);
        const double k = rx::as_double(kurtosis);
        return nogil([&] { return as::expected_t(ww, WW, k); });
    }, py::arg("w"), py::arg("W"), py::arg("kurtosis") = as::KURTOSIS);
    // `t > T_ACCEPT and t >= RHO_MIN * t_exp`: the comparison that decides
    // is an np.bool_ when an operand of it is a numpy scalar.
    m.def("corr_accept", [](const py::object& t, const py::object& t_exp) -> py::object {
        const double tt = rx::as_double(t), te = rx::as_double(t_exp);
        const bool first = tt > as::T_ACCEPT;
        const bool v = first && tt >= as::RHO_MIN * te;
        const bool numpy = first ? (is_numpy(t) || is_numpy(t_exp)) : is_numpy(t);
        if (numpy) return rx::np_module().attr("bool_")(v);
        return py::bool_(v);
    }, py::arg("t"), py::arg("t_exp"));

    m.def("_acc_reference", [](const store::Accumulator& acc, const py::object& g, const py::object& n) {
        return pair_py(as::acc_reference(acc, rx::as_int(g), rx::as_int(n)));
    }, py::arg("acc"), py::arg("g"), py::arg("n"));
    m.def("_cluster_sums", [](PyStore& s, const py::object& uids) {
        const std::vector<std::string> u = strings(uids);
        return pair_py(nogil([&] { return as::cluster_sums(core(s), u); }));
    }, py::arg("store"), py::arg("uids"));
    m.def("_best_accumulator", [](PyStore& s, const py::object& z, const py::object& w,
                                  const py::object& exclude) -> py::object {
        const auto zz = rx::vec<double>(z), ww = rx::vec<double>(w);
        std::vector<store::Key> ex;
        for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(exclude))) ex.push_back(st::key_of(h));
        const auto best = nogil([&] { return as::best_accumulator(core(s), zz, ww, ex); });
        if (!best) return py::none();
        return py::make_tuple(np_float(best->t), st::key_py(best->key), best->segment);
    }, py::arg("store"), py::arg("z"), py::arg("w"), py::arg("exclude") = py::tuple());
    m.def("validate_header_match", [](PyStore& s, const py::object& key, const py::object& mode,
                                      const py::object& segment, const py::object& z, const py::object& w) {
        const store::Key k = st::key_of(key);
        const std::int64_t md = rx::as_int(mode), g = rx::as_int(segment);
        const auto zz = rx::vec<double>(z), ww = rx::vec<double>(w);
        const as::Validation v = nogil([&] { return as::validate_header_match(core(s), k, md, g, zz, ww); });
        return py::make_tuple(v.ok, st::to_py(v.details));
    }, py::arg("store"), py::arg("key"), py::arg("mode"), py::arg("segment"), py::arg("z"), py::arg("w"));
    m.def("_attach_by_header", [](PyStore& s, const py::object& h, const py::object& uids, const py::object& method) {
        const auto hf = rx::header_fields(h);
        const std::vector<std::string> u = strings(uids);
        const std::string meth = rx::as_string(method);
        return strings_py(nogil([&] { return as::attach_by_header(core(s), hf, u, meth); }));
    }, py::arg("store"), py::arg("h"), py::arg("uids"), py::arg("method"));
    m.def("header_agrees", [](const py::object& h, const py::object& llr) {
        const auto hf = rx::header_fields(h);
        if (llr.is_none()) return as::header_agrees(hf, std::nullopt);
        const std::vector<double> l = rx::vec<double>(llr);
        return nogil([&] { return as::header_agrees(hf, std::span<const double>(l)); });
    }, py::arg("h"), py::arg("llr"));
    m.def("_soft_header", [](PyStore& s, const py::object& uid, const py::object& llr) -> py::object {
        const std::string u = rx::as_string(uid);
        const std::vector<double> l = rx::vec<double>(llr);
        const auto r = nogil([&] { return as::soft_header(core(s), u, l); });
        if (!r) return py::none();
        return py::make_tuple(rx::to_py(r->h), strings_py(r->uids));
    }, py::arg("store"), py::arg("uid"), py::arg("llr"));
    m.def("_rule", [](PyStore& s, const py::object& key, const py::object& uid, const py::object& rule) {
        const store::Key k = st::key_of(key);
        const std::string u = rx::as_string(uid), r = rx::as_string(rule);
        return rx::to_str(nogil([&] { return as::rule(core(s), k, u, r); }));
    }, py::arg("store"), py::arg("key"), py::arg("uid"), py::arg("rule"));

    m.def("associate", [](const py::object& p, const py::object& store_obj, const py::object& soft_header) {
        PyStore& s = store_obj.cast<PyStore&>();
        const auto pr = rx::pass_result(p);
        as::SoftHeaderFn soft;
        if (!soft_header.is_none()) {
            auto fn = held(soft_header);
            auto self = held(store_obj);
            soft = [fn, self](store::Store&, const std::string& uid,
                              std::span<const double> llr) -> std::optional<as::SoftHeader> {
                py::gil_scoped_acquire gil;
                const py::object r = (*fn)(*self, rx::to_str(uid), rx::arr(std::vector<double>(llr.begin(), llr.end())));
                if (r.is_none()) return std::nullopt;
                const py::sequence hu = py::reinterpret_borrow<py::sequence>(r);
                if (hu.size() != 2) throw py::value_error("_soft_header returns (header, uids) or None");
                return as::SoftHeader{rx::header_fields(hu[0]), strings(hu[1])};
            };
        }
        const as::Result r = nogil([&] { return as::associate(pr, core(s), soft); });
        return py::make_tuple(st::opt_key(r.key), rx::to_str(r.rule));
    }, py::arg("p"), py::arg("store"), py::arg("soft_header") = py::none());
    m.def("resolve_provisional", [](PyStore& s) {
        const auto r = nogil([&] { return as::resolve_provisional(core(s)); });
        py::list out;
        for (const auto& [c, k] : r) out.append(py::make_tuple(st::to_py(c), st::key_py(k)));
        return out;
    }, py::arg("store"));
}

const Registrar reg("associate", 96, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
