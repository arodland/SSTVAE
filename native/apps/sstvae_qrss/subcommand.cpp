#include "subcommand.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace sstvae::qrss::cli {

namespace {

// Filled by the Register objects' static initialisers, before main runs.
std::vector<Subcommand>& registry() {
    static std::vector<Subcommand> r;
    return r;
}

}  // namespace

Subcommand::Register::Register(const char* name, const char* script, const char* summary, Main run) {
    registry().push_back({name, script, summary, run});
}

std::vector<Subcommand> Subcommand::all() {
    std::vector<Subcommand> out = registry();
    std::sort(out.begin(), out.end(),
              [](const Subcommand& a, const Subcommand& b) { return std::string(a.name) < b.name; });
    return out;
}

const Subcommand* Subcommand::find(const std::string& name) {
    for (const Subcommand& s : registry())
        if (name == s.name) return &s;
    return nullptr;
}

}  // namespace sstvae::qrss::cli
