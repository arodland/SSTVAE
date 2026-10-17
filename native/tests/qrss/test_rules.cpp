// native/core/qrss/rules against sstvae/qrss/rules.py: the two switches
// are environment variables, "" and "0" meaning off as in the reference,
// set_old writes "1" or removes the variable, an unknown rule is refused
// with the reference's text, apply sets back only the flags given, and
// noise_psd reads QRSS_OLD_EDGE_NOISE when no explicit flag is passed.
// tests/test_qrss_native_rules.py checks the same against the reference in
// one process (and that a child process follows).

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string_view>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

#include "infra_testgen.hpp"
#include "qrss/frontend.hpp"
#include "qrss/rules.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace rules = sstvae::qrss::rules;

static void put(const std::string& name, const char* value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value ? value : "");
#else
    if (value)
        setenv(name.c_str(), value, 1);
    else
        unsetenv(name.c_str());
#endif
}

static void test_names() {
    check::equal(std::string(rules::EDGE_NOISE), std::string("QRSS_OLD_EDGE_NOISE"), "EDGE_NOISE");
    check::equal(std::string(rules::GATING), std::string("QRSS_OLD_GATING"), "GATING");
    check::equal(std::size(rules::RULES), std::size_t{2}, "two rules");
    check::equal(rules::unknown_rule("X"), std::string("unknown rule 'X': one of QRSS_OLD_EDGE_NOISE, QRSS_OLD_GATING"),
                 "the refusal");
    bool refused = false;
    try {
        rules::set_old("qrss_old_gating");
    } catch (const std::invalid_argument& e) {
        refused = std::string(e.what()) == rules::unknown_rule("qrss_old_gating");
    }
    check::is_true(refused, "set_old refuses an unknown rule (case matters)");
    refused = false;
    try {
        rules::check("");
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check::is_true(refused, "check refuses the empty name");
}

static void test_old() {
    for (const std::string_view r : rules::RULES) {
        const std::string name(r);
        const std::pair<const char*, bool> values[] = {
            {nullptr, false}, {"", false}, {"0", false}, {"1", true}, {"yes", true}, {"00", true}, {" ", true},
            {"false", true}, {"0 ", true}};
        for (const auto& [v, want] : values) {
            put(name, v);
            check::equal(rules::old(name), want, name + "=" + (v ? std::string("'") + v + "'" : std::string("unset")));
        }
        rules::set_old(name);
        const char* v = std::getenv(name.c_str());
        check::is_true(v && std::string(v) == "1", name + ": set_old writes 1");
        check::is_true(rules::old(name), name + ": and reads back on");
        rules::set_old(name, false);
        check::is_true(std::getenv(name.c_str()) == nullptr || std::string(std::getenv(name.c_str())).empty(),
                       name + ": set_old(false) removes it");
        check::is_true(!rules::old(name), name + ": and reads back off");
        rules::set_old(name, false);   // unsetting an unset rule is quiet
        check::is_true(!rules::old(name), name + ": twice");
    }
    check::is_true(!rules::old("SOME_OTHER_VARIABLE_THAT_IS_NOT_SET"), "an unset variable is off");
}

static void test_apply() {
    rules::set_old(rules::EDGE_NOISE, false);
    rules::set_old(rules::GATING, false);
    rules::apply(true, false);
    check::is_true(rules::old(rules::EDGE_NOISE) && !rules::old(rules::GATING), "apply(edge)");
    rules::apply(false, true);
    check::is_true(rules::old(rules::EDGE_NOISE) && rules::old(rules::GATING), "a false flag leaves its rule");
    rules::apply(false, false);
    check::is_true(rules::old(rules::EDGE_NOISE) && rules::old(rules::GATING), "apply(nothing) changes nothing");
    rules::set_old(rules::EDGE_NOISE, false);
    rules::set_old(rules::GATING, false);
}

static void test_reaches_noise_psd() {
    // the band's edge, where the two rules differ (test_frontend.cpp pins both)
    const auto a = qrss_testgen::splitmix(104, 7500), b = qrss_testgen::splitmix(105, 7500);
    std::vector<std::complex<float>> ch(7500);
    for (std::size_t i = 0; i < ch.size(); ++i)
        ch[i] = {static_cast<float>(qrss_testgen::unit(a[i]) * 2 - 1),
                 static_cast<float>(qrss_testgen::unit(b[i]) * 2 - 1)};
    const std::vector<double> ex{100.0};
    namespace fe = sstvae::qrss::frontend;
    fe::NoisePsdOptions env, now, then;
    env.audio_hz = now.audio_hz = then.audio_hz = 330.25;
    now.old_edge_noise = false;
    then.old_edge_noise = true;
    const auto v_now = fe::noise_psd(ch, 250, ex, 10.0, now), v_then = fe::noise_psd(ch, 250, ex, 10.0, then);
    check::is_true(v_now != v_then, "the two rules differ at 330.25 Hz");
    rules::set_old(rules::EDGE_NOISE, false);
    check::is_true(fe::noise_psd(ch, 250, ex, 10.0, env) == v_now, "rule unset: the current rule");
    rules::set_old(rules::EDGE_NOISE);
    check::is_true(fe::noise_psd(ch, 250, ex, 10.0, env) == v_then, "rule set: the old rule");
    check::is_true(fe::noise_psd(ch, 250, ex, 10.0, now) == v_now, "an explicit flag beats the environment");
    rules::set_old(rules::EDGE_NOISE, false);
}

int main() {
    try {
        test_names();
        test_old();
        test_apply();
        test_reaches_noise_psd();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_rules");
}
