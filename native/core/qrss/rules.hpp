// sstvae/qrss/rules.py: the receiver rules a command line can set back to
// their earlier form (QRSS_OLD_EDGE_NOISE, QRSS_OLD_GATING).
//
// Not a format module: none of this changes what goes on the air.
//
// The switches live in the process environment, as in the reference, so a
// spawned process follows its parent and the Python reference and the C++
// read the same switch in one process (Python's os.environ writes through
// to the C environment, which `old` reads). `set_old` writes the C
// environment; a Python caller that also reads os.environ must write
// through os.environ instead (the binding does both).
//
// Thread safety: `old` reads the environment with std::getenv, which races
// with `set_old` (setenv) on another thread. Set the rules once at startup,
// before any receiver runs -- as the CLIs do. The receiver's own entry
// points take an explicit flag beside the environment (noise_psd's
// NoisePsdOptions::old_edge_noise) for callers that hold the rules in their
// own configuration.

#ifndef SSTVAE_QRSS_RULES_HPP
#define SSTVAE_QRSS_RULES_HPP

#include <string>
#include <string_view>

#include "qrss/rx_config.hpp"

namespace sstvae::qrss::rules {

inline constexpr std::string_view EDGE_NOISE = rxconfig::rules::EDGE_NOISE;
inline constexpr std::string_view GATING = rxconfig::rules::GATING;
inline constexpr const auto& RULES = rxconfig::rules::RULES;

// Whether `rule` is set back to its earlier form: the environment variable
// is set to anything but "" and "0".
bool old(std::string_view rule);

// The refusal for a name not in RULES:
// "unknown rule 'X': one of QRSS_OLD_EDGE_NOISE, QRSS_OLD_GATING".
std::string unknown_rule(std::string_view rule);

// std::invalid_argument (unknown_rule) unless `rule` is one of RULES.
void check(std::string_view rule);

// Set `rule` back to its earlier form (on: the variable is "1") or to the
// current one (off: the variable is removed), for this process and any it
// starts afterwards. std::invalid_argument for an unknown rule.
void set_old(std::string_view rule, bool on = true);

// rules.apply for a command line's --old-edge-noise / --old-gating: sets
// back the rules asked for (a false flag leaves its rule as it is).
void apply(bool old_edge_noise, bool old_gating);

}  // namespace sstvae::qrss::rules

#endif  // SSTVAE_QRSS_RULES_HPP
