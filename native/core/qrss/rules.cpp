#include "qrss/rules.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include "qrss/pytext.hpp"

namespace sstvae::qrss::rules {

bool old(std::string_view rule) {
    const std::string name(rule);
    const char* v = std::getenv(name.c_str());
    if (v == nullptr) return false;
    const std::string_view s(v);
    return !(s.empty() || s == "0");
}

std::string unknown_rule(std::string_view rule) {
    std::string out = "unknown rule " + pytext::repr(rule) + ": one of ";
    for (std::size_t i = 0; i < RULES.size(); ++i) {
        if (i) out += ", ";
        out += RULES[i];
    }
    return out;
}

void check(std::string_view rule) {
    for (const std::string_view r : RULES)
        if (r == rule) return;
    throw std::invalid_argument(unknown_rule(rule));
}

void set_old(std::string_view rule, bool on) {
    check(rule);
    const std::string name(rule);
#ifdef _WIN32
    _putenv_s(name.c_str(), on ? "1" : "");   // "" removes the variable
#else
    if (on)
        setenv(name.c_str(), "1", 1);
    else
        unsetenv(name.c_str());
#endif
}

void apply(bool old_edge_noise, bool old_gating) {
    if (old_edge_noise) set_old(EDGE_NOISE);
    if (old_gating) set_old(GATING);
}

}  // namespace sstvae::qrss::rules
