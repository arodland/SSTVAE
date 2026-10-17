// sstvae-qrss's argparse (native/apps/sstvae_qrss/argparse.hpp): its optional
// positional (add_argument(name, nargs="?"), qrss_decode.py's OUT.png)
// against CPython's argparse, through the pins gen_argparse_nargs_pins.py
// wrote (argparse_nargs_pins.hpp): the usage line of each parser, and the
// status, stderr and parsed values of each command line -- "--" in every
// position it can take, and an optional positional before, between and
// after options.
//
// As test_argparse.cpp: the parser lives in the CLI's object library, which
// the globbed ctests do not link, so its one source file is compiled in.

#include <cstdio>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../apps/sstvae_qrss/argparse.cpp"   // NOLINT(bugprone-suspicious-include)
#include "argparse_nargs_pins.hpp"
#include "check.hpp"

namespace check = sstvae::check;
namespace pins = argparse_nargs_pins;
using sstvae::qrss::cli::ArgParser;
using sstvae::qrss::cli::Args;
using sstvae::qrss::cli::ArgSpec;

namespace {

std::vector<std::string> split_argv(int argc, const std::string& s) {
    std::vector<std::string> out;
    if (argc == 0) return out;
    std::size_t i = 0;
    for (;;) {
        const std::size_t j = s.find('\n', i);
        out.push_back(s.substr(i, j == std::string::npos ? std::string::npos : j - i));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    check::equal(out.size(), static_cast<std::size_t>(argc), "argv of " + s);
    return out;
}

std::string dest(const std::string& name) {
    std::size_t k = 0;
    while (k < name.size() && name[k] == '-') ++k;
    std::string d = name.substr(k);
    for (char& c : d)
        if (c == '-') c = '_';
    return d;
}

ArgParser build(int parser) {
    ArgParser ap(pins::PROG, "");
    for (const auto& a : pins::ARGS) {
        if (a.parser != parser) continue;
        if (a.kind == 'f') {
            ap.flag(a.name);
        } else if (a.kind == 'p') {
            ap.positional(a.name);
        } else if (a.kind == 'q') {
            ap.optional_positional(a.name);
        } else {
            ArgSpec spec;
            if (a.type == 'i') spec.integer();
            ap.option(a.name, spec);
        }
    }
    for (const auto& g : pins::GROUPS)
        if (g.parser == parser) {
            std::vector<std::string> flags;
            std::istringstream words(g.flags);
            for (std::string w; words >> w;) flags.push_back(w);
            ap.exclusive(flags, g.required);
        }
    return ap;
}

// The parsed values as the generator prints them: "dest=value" per
// argument, in table order.
std::string values(int parser, const Args& args) {
    std::string out;
    for (const auto& a : pins::ARGS) {
        if (a.parser != parser) continue;
        const std::string d = dest(a.name);
        std::string v;
        if (a.kind == 'f') {
            v = args.on(d) ? "True" : "False";
        } else if (a.type == 'i') {
            const auto n = args.integer(d);
            v = n ? n->decimal : "None";
        } else {
            const auto& t = args.text(d);
            v = t ? *t : "None";
        }
        if (!out.empty()) out += "\n";
        out += d + "=" + v;
    }
    return out;
}

void test_usage() {
    for (const auto& u : pins::USAGES)
        check::equal(build(u.parser).usage(), std::string(u.usage), "usage of parser " + std::to_string(u.parser));
}

void test_cases() {
    for (const auto& c : pins::CASES) {
        const ArgParser ap = build(c.parser);
        std::ostringstream out, err;
        int status = -1;
        const std::vector<std::string> argv = split_argv(c.argc, c.argv);
        const auto args = ap.parse(argv, out, err, status);
        std::string label = "parser " + std::to_string(c.parser) + " [";
        for (const std::string& a : argv) label += " '" + a + "'";
        label += " ]";
        check::equal(status, c.status, label + " status");
        check::equal(args.has_value(), c.status == 0, label + " parsed");
        check::equal(err.str(), std::string(c.err), label + " stderr");
        check::equal(out.str(), std::string(), label + " stdout");
        if (args) check::equal(values(c.parser, *args), std::string(c.values), label + " values");
    }
}

// What argparse would take and this does not model, refused when the
// parser is built: a positional after the optional one (argparse would
// match it greedily around the optional one's empty match).
void test_refusals() {
    bool threw = false;
    try {
        ArgParser ap("prog", "");
        ap.optional_positional("a").positional("b");
    } catch (const std::logic_error&) {
        threw = true;
    }
    check::is_true(threw, "a positional after the optional one");
    threw = false;
    try {
        ArgParser ap("prog", "");
        ap.optional_positional("a").optional_positional("b");
    } catch (const std::logic_error&) {
        threw = true;
    }
    check::is_true(threw, "two optional positionals");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    try {
        test_usage();
        test_cases();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_argparse_nargs");
}
