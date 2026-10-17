// sstvae-qrss's argparse (native/apps/sstvae_qrss/argparse.hpp): its mutually
// exclusive groups against CPython's argparse, through the pins
// gen_argparse_pins.py wrote (argparse_pins.hpp): the usage line of each
// parser, and the status and stderr of each command line.
//
// The parser lives in the CLI's object library, which the ctests globbed
// here do not link (they link sstvae_qrss), so its one source file is
// compiled into this test directly; it needs nothing but sstvae_qrss.
// The subcommands' own refusals, groups included, are held against the
// qrss_*.py scripts by tests/test_qrss_native_cli*.py.

#include <cstdio>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../apps/sstvae_qrss/argparse.cpp"   // NOLINT(bugprone-suspicious-include)
#include "argparse_pins.hpp"
#include "check.hpp"

namespace check = sstvae::check;
using sstvae::qrss::cli::ArgParser;
using sstvae::qrss::cli::ArgSpec;

namespace {

std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    if (s.empty()) return out;
    std::size_t i = 0;
    for (;;) {
        const std::size_t j = s.find('\n', i);
        out.push_back(s.substr(i, j == std::string::npos ? std::string::npos : j - i));
        if (j == std::string::npos) return out;
        i = j + 1;
    }
}

ArgParser build(int parser) {
    ArgParser ap(argparse_pins::PROG, "");
    for (const auto& a : argparse_pins::ARGS) {
        if (a.parser != parser) continue;
        if (a.kind == 'f') {
            ap.flag(a.name);
        } else if (a.kind == 'p') {
            ap.positional(a.name);
        } else {
            ArgSpec spec;
            if (a.type == 'i') spec.integer();
            if (a.type == 'f') spec.real();
            ap.option(a.name, spec);
        }
    }
    for (const auto& g : argparse_pins::GROUPS)
        if (g.parser == parser) {
            std::vector<std::string> flags;
            std::istringstream words(g.flags);
            for (std::string w; words >> w;) flags.push_back(w);
            ap.exclusive(flags, g.required);
        }
    return ap;
}

void test_usage() {
    for (const auto& u : argparse_pins::USAGES)
        check::equal(build(u.parser).usage(), std::string(u.usage), "usage of parser " + std::to_string(u.parser));
}

void test_cases() {
    for (const auto& c : argparse_pins::CASES) {
        const ArgParser ap = build(c.parser);
        std::ostringstream out, err;
        int status = -1;
        const auto args = ap.parse(split_lines(c.argv), out, err, status);
        std::string label = "parser " + std::to_string(c.parser) + " [";
        for (const std::string& a : split_lines(c.argv)) label += " " + a;
        label += " ]";
        check::equal(status, c.status, label + " status");
        check::equal(args.has_value(), c.status == 0, label + " parsed");
        check::equal(err.str(), std::string(c.err), label + " stderr");
        check::equal(out.str(), std::string(), label + " stdout");
    }
}

// The flags a group holds come back in the order they were added, however
// the group names them (argparse's group holds them in add order, there
// being no other way to put them in one).
void test_group_order() {
    ArgParser ap("prog", "");
    ap.flag("--a").flag("--b").exclusive({"--b", "--a"});
    check::equal(ap.usage(), std::string("usage: prog [-h] [--a | --b]"), "group named out of order");
    std::ostringstream out, err;
    int status = -1;
    ap.parse({"--b", "--a"}, out, err, status);
    check::equal(err.str(), std::string("usage: prog [-h] [--a | --b]\nprog: error: argument --a: not allowed with "
                                        "argument --b\n"),
                 "conflict named by the later argument");
}

// What argparse refuses when the group is built, and what this does not
// model, refused rather than half-done.
void test_refusals() {
    auto refuses = [](auto build_group, const std::string& what) {
        bool threw = false;
        try {
            ArgParser ap("prog", "");
            ap.positional("x").flag("--a").option("--r", ArgSpec().must());
            build_group(ap);
        } catch (const std::logic_error&) {
            threw = true;
        }
        check::is_true(threw, what);
    };
    refuses([](ArgParser& ap) { ap.exclusive({}); }, "an empty group");
    refuses([](ArgParser& ap) { ap.exclusive({"--a", "x"}); }, "a positional, by its name");
    refuses([](ArgParser& ap) { ap.exclusive({"--a", "--r"}); }, "a required option");
    refuses([](ArgParser& ap) { ap.exclusive({"--a", "--zz"}); }, "an unknown flag");
    refuses([](ArgParser& ap) { ap.exclusive({"-h"}); }, "help");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    try {
        test_usage();
        test_cases();
        test_group_order();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_argparse");
}
