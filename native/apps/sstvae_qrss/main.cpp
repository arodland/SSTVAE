// sstvae-qrss: dispatches to the subcommand named by the first argument.
// The subcommands are the files beside this one (subcommand.hpp says how
// they register and what each one promises).

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef SSTVAE_HAVE_FETCH
#include <QCoreApplication>

#include "checkpoint/qt_fetcher.hpp"
#endif

#include "subcommand.hpp"

using sstvae::qrss::cli::Subcommand;

namespace {

constexpr const char* PROG = "sstvae-qrss";

// Subcommands this build does not have, and why: the codec-only ones in a
// build without the codec, and the ones not ported yet. A registered
// subcommand of the same name always wins, so a lane that adds one (the
// transmit lane: transmit.cpp) need not touch this table.
struct NotBuilt {
    std::string_view name;
    std::string_view why;
};
constexpr NotBuilt NOT_BUILT[] = {
    {"encode", "built without the codec (SSTVAE_BUILD_CODEC=OFF); use qrss_encode.py, or rebuild with the codec"},
    {"transmit", "not in this build yet; use qrss_transmit.py"},
    {"receive", "not in this build yet; use qrss_receive.py"},
    {"listen", "not in this build yet; use qrss_listen.py"},
    {"decode", "not in this build yet; use qrss_decode.py"},
};

// The Python class an escaped exception would most likely have been.
const char* python_class(const std::exception& e) {
    if (dynamic_cast<const std::invalid_argument*>(&e)) return "ValueError";
    if (dynamic_cast<const std::overflow_error*>(&e)) return "OverflowError";
    if (dynamic_cast<const std::out_of_range*>(&e)) return "IndexError";
    if (dynamic_cast<const std::bad_alloc*>(&e)) return "MemoryError";
    return "RuntimeError";
}

void overview(std::ostream& os) {
    os << "usage: " << PROG << " <command> [arguments]\n\n"
       << "QRSSTVAE from the command line: each command takes the arguments of\n"
       << "the qrss_*.py script it names and behaves as that script does.\n\n"
       << "commands:\n";
    for (const Subcommand& s : Subcommand::all()) {
        std::string head = std::string("  ") + s.name;
        head.resize(14, ' ');
        os << head << s.summary << " (" << s.script << ")\n";
    }
    os << "\n'" << PROG << " <command> --help' describes one command.\n";
}

}  // namespace

int main(int argc, char** argv) {
#ifdef SSTVAE_HAVE_FETCH
    // Qt only for the first-run model download (encode with no --model, as
    // sstvae-decode). Its own argv, so Qt never sees ours.
    int qt_argc = 1;
    char* qt_argv[] = {argv[0], nullptr};
    QCoreApplication app(qt_argc, qt_argv);
    // QCoreApplication runs setlocale(LC_ALL, "") on Unix, and Python never
    // touches LC_NUMERIC: put it back, or under a decimal-comma locale strtod
    // reads "--freq 1500.3" as 1500 and snprintf writes "1500,3" where the
    // script prints "1500.3" (Qt's documented remedy).
    std::setlocale(LC_NUMERIC, "C");
    sstvae::checkpoint::install_qt_fetcher([](std::int64_t got, std::int64_t total) {
        if (total <= 0) return;
        std::fprintf(stderr, "\rfetching model: %lld / %lld KB", static_cast<long long>(got / 1024),
                     static_cast<long long>(total / 1024));
        if (got >= total) std::fprintf(stderr, "\n");
    });
#endif
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) {
        overview(std::cerr);
        return 2;
    }
    if (args[0] == "-h" || args[0] == "--help") {
        overview(std::cout);
        return 0;
    }
    const Subcommand* sub = Subcommand::find(args[0]);
    if (!sub) {
        for (const NotBuilt& nb : NOT_BUILT) {
            if (args[0] == nb.name) {
                std::cerr << PROG << " " << args[0] << ": " << nb.why << "\n";
                return 2;
            }
        }
        std::cerr << PROG << ": unknown command '" << args[0] << "'\n\n";
        overview(std::cerr);
        return 2;
    }
    int status = 1;
    try {
        status = sub->run(std::string(PROG) + " " + sub->name, std::vector<std::string>(args.begin() + 1, args.end()),
                          std::cout, std::cerr);
    } catch (const std::exception& e) {
        // A subcommand ends every path the script can take itself; this is
        // the backstop for one it missed. The script would end in a
        // traceback and exit 1, never in std::terminate's abort.
        std::cout.flush();
        std::cerr << python_class(e) << ": " << e.what() << "\n";
        return 1;
    }
    std::cout.flush();
    return status;
}
