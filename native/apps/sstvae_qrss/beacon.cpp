// sstvae-qrss beacon: qrss_beacon.py. One segment of a stored picture as a
// clock-chip beacon sends it (beaconfile.hpp has the layout).
//
// The script, step by step, with how it ends at each one:
//   sp = picture.load_qrsp(input)            uncaught: a traceback, exit 1
//   HeaderFields(...); beaconfile.from_picture(sp, segment, h, ook)
//                                            ValueError -> SystemExit(text)
//   beaconfile.write(output, bf)             uncaught OSError
//   print("wrote ...")

#include <cstdint>
#include <exception>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "argparse.hpp"
#include "common.hpp"
#include "qrss/beaconfile.hpp"
#include "qrss/header.hpp"
#include "qrss/picture.hpp"
#include "qrss/pynum.hpp"
#include "subcommand.hpp"

namespace sstvae::qrss::cli {
namespace {

namespace bfm = sstvae::qrss::beaconfile;
namespace pc = sstvae::qrss::picture;

// qrss_beacon.py's docstring, its --help description.
constexpr const char* DESCRIPTION =
    "Write a beacon file for a clock-chip (Si5351) QRSSTVAE beacon.\n"
    "\n"
    "    sstvae-qrss beacon pic.qrsp seg0.bin --callsign K1ABC --grid FN42\n"
    "\n"
    "One segment of a stored picture as the beacon sends it: the header's\n"
    "coded bits, the callsign's Morse keying, and the latents in air order as\n"
    "int8 (x20). A full segment is 50,970 bytes, which fits a 24LC512 EEPROM.\n"
    "The beacon computes the preamble, scrambler, precoder and phase itself.";

ArgParser parser(const std::string& prog) {
    ArgParser ap(prog, DESCRIPTION);
    ap.positional("input", "stored picture (.qrsp)");
    ap.positional("output", "beacon file (.bin)");
    ap.option("--callsign", ArgSpec().must().described("1-8 characters of A-Z, 0-9 and /; sent in the header "
                                                         "and in Morse in every callsign window"));
    ap.option("--grid", ArgSpec().described("4-character Maidenhead locator"));
    ap.option("--segment", ArgSpec().integer().otherwise("0"));
    ap.flag("--ook", "identify by on-off keying instead of frequency-shift keying");
    return ap;
}

int run(const std::string& prog, const std::vector<std::string>& argv, std::ostream& out, std::ostream& err) {
    int status = 0;
    const std::optional<Args> args = parser(prog).parse(argv, out, err, status);
    if (!args) return status;
    const std::string input = *args->text("input");
    const std::string output = *args->text("output");
    const pynum::PyInt segment = *args->integer("segment");

    pc::StoredPicture sp;
    try {
        sp = load_qrsp(input);
    } catch (const PyError& e) {
        return uncaught(err, e);
    }

    bfm::BeaconFile bf;
    try {
        // A segment past int64 is refused by its range check, the first one
        // from_picture makes after sp.check() (which load_qrsp has passed).
        if (!segment.value)
            throw std::invalid_argument("segment must be 0.." + std::to_string(sp.mode) + ", not " +
                                        segment.decimal);
        header::HeaderFields h;
        h.callsign = *args->text("callsign");
        h.grid = args->text("grid");
        h.picture_id = sp.picture_id;
        h.mode = sp.mode;
        h.segment = *segment.value;
        h.codec_id = sp.codec_id;
        bf = bfm::from_picture(sp, *segment.value, h, args->on("ook"));
    } catch (const std::invalid_argument& e) {
        return system_exit(err, e.what());
    }

    try {
        write_file(output, bf.to_bytes());
    } catch (const PyError& e) {
        return uncaught(err, e);
    }
    out << "wrote " << output << ": picture " << sp.id_hex() << " segment " << segment.decimal << ", "
        << bf.callsign << " " << (bf.grid && !bf.grid->empty() ? *bf.grid : "-") << ", "
        << (bf.ook ? "OOK" : "FSK") << " ID, " << bfm::file_size(static_cast<std::int64_t>(bf.latents_i8.size()))
        << " bytes\n";
    return 0;
}

const Subcommand::Register reg("beacon", "qrss_beacon.py", "write a clock-chip beacon file for one segment", &run);

}  // namespace
}  // namespace sstvae::qrss::cli
