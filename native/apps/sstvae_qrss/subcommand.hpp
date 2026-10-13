// sstvae-qrss: the QRSSTVAE command line, one subcommand per qrss_*.py CLI
// (plan stage 1 item 17, stages 2-3 for the receive side).
//
// Each subcommand is its own file in this directory and registers itself
// with a static `Subcommand::Register`; main.cpp only dispatches. CMake
// globs the directory, so adding one touches nothing shared:
//
//   encode.cpp     qrss_encode.py    (only with the codec: SSTVAE_BUILD_CODEC)
//   beacon.cpp     qrss_beacon.py
//   transmit.cpp   qrss_transmit.py
//   receive.cpp, listen.cpp, decode.cpp later (plan stages 2c and 3)
//
// The contract each one keeps with its Python script: the same argument
// names, defaults and choices (argparse.hpp parses them argparse's way and
// refuses with argparse's texts, exit 2), the same stdout lines, the same
// SystemExit texts on stderr with exit 1 (common.hpp), no output file on a
// refusal, and byte-identical output files. tests/test_qrss_native_cli.py
// runs each script and its subcommand side by side.
//
// A subcommand a build leaves out (encode without the codec) is named in
// main.cpp's NOT_BUILT table, so `sstvae-qrss encode` says why rather than
// "unknown command".

#ifndef SSTVAE_APPS_SSTVAE_QRSS_SUBCOMMAND_HPP
#define SSTVAE_APPS_SSTVAE_QRSS_SUBCOMMAND_HPP

#include <ostream>
#include <string>
#include <vector>

namespace sstvae::qrss::cli {

struct Subcommand {
    // `prog` is the program as usage lines print it ("sstvae-qrss
    // encode"), `args` the subcommand's own arguments. Writes what the
    // script prints to `out` (stdout) and `err` (stderr) and returns the
    // exit status: 0, 1 for a refusal, 2 for a usage error. Streams rather
    // than stdout/stderr so a ctest can run a subcommand in-process.
    using Main = int (*)(const std::string& prog, const std::vector<std::string>& args, std::ostream& out,
                         std::ostream& err);

    const char* name;      // "encode"
    const char* script;    // the Python CLI it mirrors: "qrss_encode.py"
    const char* summary;   // one line for `sstvae-qrss --help`
    Main run;

    // All registered subcommands, sorted by name.
    static std::vector<Subcommand> all();
    static const Subcommand* find(const std::string& name);

    struct Register {
        Register(const char* name, const char* script, const char* summary, Main run);
    };
};

}  // namespace sstvae::qrss::cli

#endif  // SSTVAE_APPS_SSTVAE_QRSS_SUBCOMMAND_HPP
