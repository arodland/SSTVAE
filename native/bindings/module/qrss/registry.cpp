#include "qrss/registry.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "qrss/qrss_config.hpp"

namespace sstvae_bindings::qrss {

std::vector<Entry>& registry() {
    static std::vector<Entry> entries;
    return entries;
}

void bind_all(py::module_& m) {
    py::module_ q = m.def_submodule(
        "qrss", "The QRSSTVAE port (native/core/qrss), one submodule per sstvae.qrss module.");
    auto entries = registry();
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.order != b.order ? a.order < b.order : std::string(a.name) < b.name;
    });
    py::list names;
    for (const Entry& e : entries) names.append(e.name);
    q.attr("__qrss_modules__") = names;
    q.attr("FORMAT_VERSION") = sstvae::qrss::config::FORMAT_VERSION;
    for (const Entry& e : entries) e.fn(q);
}

}  // namespace sstvae_bindings::qrss
