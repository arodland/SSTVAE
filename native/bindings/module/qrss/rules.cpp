// sstvae_native.qrss.rules: bindings of native/core/qrss/rules (the
// receiver rules a command line can set back, sstvae/qrss/rules.py).
//
//   EDGE_NOISE, GATING, RULES
//   old(rule) -> bool          the C environment, which os.environ writes through to
//   set_old(rule, on=True)     ValueError "unknown rule 'X': one of ..." as the reference
//   apply(args)                --old-edge-noise / --old-gating off an argparse namespace
//
// set_old writes through os.environ as well as setting the C environment,
// so Python's os.environ (which a spawned process inherits through, and
// which monkeypatch restores) and the C++'s getenv never disagree. A rule
// that is neither a str nor one of RULES gets the reference's ValueError
// with its repr, not a TypeError.

#include "qrss/rx_types.hpp"

#include <string>

#include "qrss/rules.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace rules = sstvae::qrss::rules;

py::tuple rule_names() {
    py::tuple t(rules::RULES.size());
    for (std::size_t i = 0; i < rules::RULES.size(); ++i) t[i] = rx::to_str(rules::RULES[i]);
    return t;
}

// The rule's name when it is one of RULES, else the reference's ValueError.
std::string known_rule(const py::handle& rule) {
    if (py::isinstance<py::str>(rule)) {
        const std::string name = rx::as_string(rule);
        for (const std::string_view r : rules::RULES)
            if (r == name) return name;
    }
    std::string msg = "unknown rule " + py::repr(rule).cast<std::string>() + ": one of ";
    for (std::size_t i = 0; i < rules::RULES.size(); ++i) {
        if (i) msg += ", ";
        msg += rules::RULES[i];
    }
    throw py::value_error(msg);
}

void set_old_py(const py::object& rule, const py::object& on) {
    const std::string name = known_rule(rule);
    const bool set = rx::as_bool(on);
    py::object env = py::module_::import("os").attr("environ");   // not "environ": a macro on Windows
    if (set)
        env[py::str(name)] = py::str("1");
    else
        env.attr("pop")(py::str(name), py::none());
    rules::set_old(name, set);   // the same value os.environ has just written
}

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule("rules", "native/core/qrss/rules (sstvae/qrss/rules.py)");
    m.attr("EDGE_NOISE") = rx::to_str(rules::EDGE_NOISE);
    m.attr("GATING") = rx::to_str(rules::GATING);
    m.attr("RULES") = rule_names();

    m.def("old", [](const std::string& rule) { return rules::old(rule); }, py::arg("rule"));
    m.def("set_old", &set_old_py, py::arg("rule"), py::arg("on") = true);
    m.def("apply", [](const py::object& args) {
        if (rx::as_bool(py::getattr(args, "old_edge_noise", py::bool_(false))))
            set_old_py(rx::to_str(rules::EDGE_NOISE), py::bool_(true));
        if (rx::as_bool(py::getattr(args, "old_gating", py::bool_(false))))
            set_old_py(rx::to_str(rules::GATING), py::bool_(true));
    }, py::arg("args"));
}

const Registrar reg("rules", 7, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
