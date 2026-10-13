// sstvae-qrss transmit: qrss_transmit.py. One pass of a stored picture
// (.qrsp) or a beacon file (.bin) as transmit audio, an 8 kHz WAV.
//
// The script, step by step, with how it ends at each one:
//   parse_slot(--slot)                       ValueError -> SystemExit("--slot '<text>': <why>")
//   --waveform L with --ook                  SystemExit
//   open(input).read(4) == MAGIC             uncaught OSError
//   .bin: beaconfile.read(input)             ValueError -> SystemExit(text)
//         --callsign other than the file's, or --ook with an FSK file: SystemExit
//   .qrsp: picture.load_qrsp(input)          ValueError -> SystemExit(text); any other uncaught
//         no --callsign                      SystemExit
//   tx.transmit_audio(...)                   ValueError -> SystemExit(text)
//   write_wav_float / write_wav(output, x)   uncaught OSError
//   print("wrote ...")
//
// After the input is read this is qrss/api.hpp's plan_pass, the same
// sequence the desktop app runs in process. The audio is a tx::PassAudio
// streamed into the file (tx::write_wav), never held whole: byte for byte
// what the script writes, with the refusals all made before the output is
// opened, so a refusal leaves no file.

#include <cstdint>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "argparse.hpp"
#include "common.hpp"
#include "qrss/api.hpp"
#include "qrss/beaconfile.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/picture.hpp"
#include "qrss/pynum.hpp"
#include "qrss/pytext.hpp"
#include "qrss/slot.hpp"
#include "qrss/tx.hpp"
#include "qrss/wav.hpp"
#include "subcommand.hpp"

namespace sstvae::qrss::cli {
namespace {

namespace bfm = sstvae::qrss::beaconfile;
namespace fr = sstvae::qrss::frame;
namespace pc = sstvae::qrss::picture;
namespace txm = sstvae::qrss::tx;

// qrss_transmit.py's docstring, its --help description.
constexpr const char* DESCRIPTION =
    "Write one QRSSTVAE pass as transmit audio (8 kHz WAV).\n"
    "\n"
    "    sstvae-qrss transmit seg0.bin tx.wav --slot 2026-10-09T06:00Z\n"
    "    sstvae-qrss transmit pic.qrsp tx.wav --slot 2026-10-09T06:00Z --callsign K1ABC\n"
    "    sstvae-qrss transmit pic.qrsp tx.wav --slot 2026-10-09T06:00Z --callsign K1ABC \\\n"
    "        --waveform L --freq 1500\n"
    "\n"
    "The WAV starts 12 s before t0 = slot + 1 s (11 s before the quarter\n"
    "hour) and ends 3 s after the frame. The slot fixes the scrambler, so a\n"
    "WAV is for that quarter hour only. Every pass keys the callsign windows\n"
    "(a legal requirement), so a .qrsp needs --callsign. --lead-in sends up to\n"
    "10 s of plain carrier before the preamble, for transmitters that drift\n"
    "when keyed. --float writes float32 with no normalisation (exact round\n"
    "trips); otherwise the file is peak-normalised int16.\n"
    "\n"
    "--waveform CE (the default) is the constant-envelope carrier, about 50 Hz\n"
    "wide, centred on --freq. --waveform L is the linear signal for SSB rigs:\n"
    "a pilot tone at --freq with the data just below it, about 22 Hz wide in\n"
    "all and 3.2 dB peak to average, so set the rig's drive by its peaks (ALC\n"
    "not engaged, speech processor off). qrss_listen.py and qrss_receive.py hear\n"
    "L, beside CE, unless they are run with --just-ce.";

ArgParser parser(const std::string& prog) {
    ArgParser ap(prog, DESCRIPTION);
    ap.positional("input", "stored picture (.qrsp) or beacon file (.bin)");
    ap.positional("output", "WAV file");
    ap.option("--slot", ArgSpec().must().described("quarter hour, ISO UTC, e.g. 2026-10-09T06:00Z"));
    ap.option("--callsign", ArgSpec().described("for a .qrsp: 1-8 of A-Z, 0-9, /"));
    ap.option("--grid", ArgSpec().described("for a .qrsp: 4-character locator"));
    ap.option("--segment", ArgSpec().integer().otherwise("0").described("for a .qrsp"));
    ap.option("--freq", ArgSpec().real().otherwise("1500.0").described(
                            "audio carrier, Hz (for --waveform L, the pilot tone)"));
    ap.option("--waveform", ArgSpec().one_of({"CE", "L"}).otherwise("CE").described(
                                "CE: constant envelope (default); L: linear, for SSB rigs"));
    ap.option("--lead-in", ArgSpec().real().otherwise("0.0").described(
                               "seconds of plain carrier before the preamble (0-" +
                               std::to_string(config::LEAD_IN_MAX_S) + ")"));
    ap.flag("--ook", "on-off keyed callsign windows for a .qrsp (a .bin carries its own "
                     "choice: --ook with an FSK .bin is refused)");
    // sorted(frame.PRESETS)
    ap.option("--frame", ArgSpec().one_of({"full", "medium", "short", "tiny"}).otherwise("full").described(
                             "frame shape; anything but full is a test frame"));
    ap.flag("--float", "float32 WAV, not normalised");
    return ap;
}

// Python's f"{x:g}" and f"{x:.1f}" for a finite float (C's %g and %.1f,
// with '.' for the point whatever LC_NUMERIC says).
std::string fmt(const char* spec, double x) { return fr::py_format(x, spec); }

int run(const std::string& prog, const std::vector<std::string>& argv, std::ostream& out, std::ostream& err) {
    int status = 0;
    const std::optional<Args> args = parser(prog).parse(argv, out, err, status);
    if (!args) return status;
    const std::string input = *args->text("input");
    const std::string output = *args->text("output");
    const std::string slot_text = *args->text("slot");
    const std::optional<std::string> callsign = args->text("callsign");
    const pynum::PyInt segment = *args->integer("segment");
    const double freq = *args->real("freq");
    const bool waveform_l = *args->text("waveform") == "L";
    const double lead_in = *args->real("lead_in");
    const bool ook_flag = args->on("ook");

    std::int64_t q = 0;
    try {
        q = slot::parse_slot(slot_text);
    } catch (const std::invalid_argument& e) {
        return system_exit(err, "--slot " + pytext::repr(slot_text) + ": " + e.what());
    }
    const fr::FrameSpec& spec = waveform_l ? fr::get_l("l-" + *args->text("frame")) : fr::get(*args->text("frame"));
    if (waveform_l && ook_flag)
        return system_exit(err, "--ook is waveform CE's: waveform L keys its callsign windows by FSK of its pilot");

    // open(input, "rb").read(4): the input's first bytes, or its OSError.
    std::vector<std::uint8_t> head;
    try {
        head = read_file(input);
    } catch (const PyError& e) {
        return uncaught(err, e);
    }
    const bool is_bin = head.size() >= 4 && std::string(head.begin(), head.begin() + 4) == config::BEACON_MAGIC;

    api::PassRequest req;
    req.q = q;
    req.waveform = waveform_l ? api::Waveform::L : api::Waveform::CE;
    req.freq_hz = freq;
    req.callsign = callsign;
    req.grid = args->text("grid");
    req.frame = *args->text("frame");
    req.lead_in_s = lead_in;
    req.ook = ook_flag;
    std::optional<api::Source> src;
    std::optional<txm::PassAudio> audio;
    try {
        if (is_bin) {
            src = bfm::BeaconFile::from_bytes(head);
        } else {
            try {
                src = load_qrsp(input);
            } catch (const PyError& e) {
                if (e.cls() == "ValueError") return system_exit(err, e.what());
                return uncaught(err, e);
            }
            // A segment past int64: after the callsign's check (api::plan_pass
            // makes it) and the carrier's, _segment_and_header's range check.
            if (callsign && !segment.value) {
                txm::check_pass_carrier(spec, txm::Carrier(freq), ook_flag);
                throw std::invalid_argument("segment must be 0.." +
                                            std::to_string(std::get<pc::StoredPicture>(*src).mode) + ", not " +
                                            segment.decimal);
            }
            req.segment = segment.value.value_or(0);
        }
        audio.emplace(api::pass_stream(*src, req));
    } catch (const std::invalid_argument& e) {
        return system_exit(err, e.what());
    }
    const auto* bf = std::get_if<bfm::BeaconFile>(&*src);
    const std::string call = bf ? bf->callsign : *callsign;
    const bool keyed = bf ? bf->ook : ook_flag;

    try {
        txm::write_wav(*audio, output, args->on("float"));
    } catch (const wav::WriteError& e) {
        return uncaught(err, OsError(e.err, e.path));
    }
    const std::string lead = lead_in != 0.0 ? ", " + fmt("%g", lead_in) + " s lead-in" : "";
    out << "wrote " << output << ": slot q=" << q << ", waveform " << (waveform_l ? "L" : "CE") << ", frame "
        << spec.name() << " (" << fmt("%.1f", spec.duration_s()) << " s), " << call << " ("
        << (keyed ? "OOK" : "FSK") << " callsign windows), " << (waveform_l ? "pilot" : "carrier") << " "
        << fmt("%g", freq) << " Hz" << lead << ", "
        << fmt("%.1f", static_cast<double>(audio->size()) / 8000.0) << " s of audio starting 12 s before t0\n";
    return 0;
}

const Subcommand::Register reg("transmit", "qrss_transmit.py", "write one pass as transmit audio (8 kHz WAV)",
                               &run);

}  // namespace
}  // namespace sstvae::qrss::cli
