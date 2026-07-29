// sstvae-decode: a received WAV in, a picture out.
//
// The C++ counterpart of `sstvae_decode.py`, and Phase 2's exit
// criterion: it exercises the whole receive chain -- WAV reading and
// rate conversion, sync, demodulation, the beacon, and the codec -- in
// one command that can be diffed against the reference on the same
// input.
//
// Headless by construction. It links no GUI toolkit, which is why
// `core/` uses stb rather than QImage (native/third_party/stb/README.md).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "audio/wavio.hpp"
#ifdef SSTVAE_HAVE_FETCH
#include <QCoreApplication>

#include "checkpoint/qt_fetcher.hpp"
#endif

#include "checkpoint/checkpoint.hpp"
#include "codec/codec.hpp"
#include "config.hpp"
#include "images/images.hpp"
#include "modem/modem.hpp"

using namespace sstvae;

namespace {

struct Options {
    std::string input;
    std::string output;
    std::string model;          // directory of .onnx parts, or a single file
    std::string precision{checkpoint::DEFAULT_PRECISION};
    std::optional<double> search_start;
    std::optional<double> search_end;
    bool blind = false;
};

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout,
        "usage: sstvae-decode <input.wav> <output.png> [options]\n"
        "\n"
        "  --model PATH        directory of exported .onnx files, or one .onnx\n"
        "                      (its sibling part is found beside it). Omit to\n"
        "                      use the published artifacts.\n"
        "  --precision P       fp32, fp16 (default) or int8\n"
        "  --search-start SEC  limit preamble search to after this time\n"
        "  --search-end SEC    limit preamble search to before this time\n"
        "  --blind             preamble-free decode; position comes from the\n"
        "                      beacon, for a recording that never contained\n"
        "                      the start of the transmission\n"
        "\n"
        "--model is optional: with no --model the published artifacts are\n"
        "used, from the local cache or downloaded once on first run.\n");
    std::exit(code);
}

Options parse(int argc, char** argv) {
    Options o;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* what) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "-h" || a == "--help") usage(0);
        else if (a == "--model") o.model = value("--model");
        else if (a == "--precision") o.precision = value("--precision");
        else if (a == "--search-start") o.search_start = std::stod(value("--search-start"));
        else if (a == "--search-end") o.search_end = std::stod(value("--search-end"));
        else if (a == "--blind") o.blind = true;
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage(2);
        } else positional.push_back(a);
    }
    if (positional.size() != 2) usage(2);
    o.input = positional[0];
    o.output = positional[1];
    return o;
}

// Resolve "encoder"/"decoder" against whatever --model was given, or
// against the published artifacts when it was omitted.
codec::Resolver model_resolver(const std::string& model, const std::string& precision) {
    return [model, precision](const std::string& part) -> std::string {
        // Was a local approximation of checkpoint.py's rules; now the
        // real thing, which also means an omitted --model resolves to
        // the published artifact from the cache or the Hub.
        return checkpoint::resolve_onnx(part, model, precision);
    };
}

}  // namespace

int main(int argc, char** argv) {
#ifdef SSTVAE_HAVE_FETCH
    // Qt only for the first-run download. Everything else here -- an
    // explicit --model, and any artifact already in the cache -- works
    // in a build without it, which is why this is a compile-time option
    // rather than a hard dependency of the CLI.
    QCoreApplication app(argc, argv);
    checkpoint::install_qt_fetcher([](std::int64_t got, std::int64_t total) {
        if (total <= 0) return;
        std::fprintf(stderr, "\rfetching model: %lld / %lld KB",
                     static_cast<long long>(got / 1024),
                     static_cast<long long>(total / 1024));
        if (got >= total) std::fprintf(stderr, "\n");
    });
#endif
    const Options o = parse(argc, argv);
    try {
        const std::vector<double> x = audio::read_wav(o.input);

        const modem::Modem md;
        std::vector<double> latents, weights;
        std::optional<beacon::BeaconResult> pkt;
        std::string callsign;

        if (o.blind) {
            const modem::BlindDemodResult r = md.demodulate_blind(x);
            latents = r.latents;
            weights = r.weights;
            pkt = r.beacon;
            callsign = r.callsign;
            std::printf("blind decode: %s\n",
                        r.frame_offset ? "located via the beacon"
                                       : "no beacon, position unknown");
        } else {
            std::optional<std::pair<double, double>> search;
            if (o.search_start || o.search_end) {
                search = std::pair<double, double>{
                    o.search_start.value_or(0.0),
                    o.search_end.value_or(static_cast<double>(x.size()) / config::FS)};
            }
            const modem::DemodResult r = md.demodulate(x, search);
            latents = r.latents;
            weights = r.weights;
            pkt = r.beacon;
            callsign = r.callsign;
            std::printf("mode %.*s, %d/%d frames, freq offset %+.1f Hz, "
                        "sync metric %.2f, SNR %.1f dB\n",
                        static_cast<int>(r.mode.name.size()), r.mode.name.data(),
                        r.frames_received, r.mode.n_frames,
                        r.freq_offset, r.sync_metric, r.snr_db);
        }

        if (pkt) {
            std::printf("beacon: frame %d, callsign %s\n", pkt->frame_index,
                        callsign.empty() ? "(none sent)" : ("'" + callsign + "'").c_str());
        } else {
            std::printf("beacon: no superframe decoded (short/noisy reception)\n");
        }

        codec::OnnxCodec cd(model_resolver(o.model, o.precision));
        const images::Picture pic = cd.decode(codec::pad_to_full(latents),
                                              codec::pad_to_full(weights));
        images::save_png(pic, o.output);
        std::printf("wrote %s\n", o.output.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
