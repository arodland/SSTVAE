// Exercise the soundcard path for real: list devices, or send a
// transmission out one and decode what comes back in the other.
//
// This exists because `core/audio/qt/` is the one part of the port with
// no unit tests worth writing. Everything with logic in it lives in
// `core/audio/audio.hpp` and is covered by `tests/test_audio.cpp`
// against a fake device; what is left here is device enumeration and
// moving bytes, and the only way to find out whether *that* works is to
// move bytes through a real device.
//
// It is also the tool for the on-air shakedown: `--loopback` against a
// radio's own transmit and receive devices, with the rig on a dummy
// load, exercises the same path a QSO does.
//
// No loopback hardware needed on Linux -- a null sink plus a *remapped*
// monitor gives one, and the remap matters because Qt does not
// enumerate PulseAudio/PipeWire monitor sources:
//
//   pactl load-module module-null-sink sink_name=sstvae-null
//   pactl load-module module-remap-source source_name=sstvae_loop
//       master=sstvae-null.monitor channels=1
//       source_properties=device.description=SSTVAE-Loopback
//
//   (the second command is one line; it is broken here because a
//   trailing backslash in a // comment is a line continuation)
//
//   sstvae-audio-check --loopback --out sstvae-null --in SSTVAE-Loopback
//
// then `pactl unload-module N` for each. Measured on that setup: mode A,
// 220/220 frames, callsign recovered, 27-29 dB across runs -- with the
// device running at 48 kHz, so the capture resampler was in the path
// rather than bypassed. The Python app's validated Qt-to-Qt loopback
// reads +27.4 dB, so this is the same number through the same devices.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <QCoreApplication>

#include "audio/qt/qtaudio.hpp"
#include "config.hpp"
#include "modem/modem.hpp"
#include "rx/ringbuffer.hpp"
#include "tx/engine.hpp"

using namespace sstvae;

namespace {

struct Options {
    std::string in_device;
    std::string out_device;
    std::string mode = "A";
    std::string callsign = "TEST";
    double level = 0.9;
    bool loopback = false;
};

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout,
        "usage: sstvae-audio-check [--loopback] [options]\n"
        "\n"
        "With no --loopback, lists the audio devices Qt can see and exits.\n"
        "\n"
        "  --loopback        transmit, capture, and decode what came back\n"
        "  --in NAME         capture device (substring of its description)\n"
        "  --out NAME        playback device\n"
        "  --mode A|B|C      transmission length; A (~32 s) by default\n"
        "  --callsign CALL   what to put on the beacon carrier\n"
        "  --level L         output peak, 0..1 (default 0.9)\n"
        "\n"
        "Exits non-zero if a --loopback run fails to decode.\n");
    std::exit(code);
}

Options parse(int argc, char** argv) {
    Options o;
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
        else if (a == "--loopback") o.loopback = true;
        else if (a == "--in") o.in_device = value("--in");
        else if (a == "--out") o.out_device = value("--out");
        else if (a == "--mode") o.mode = value("--mode");
        else if (a == "--callsign") o.callsign = value("--callsign");
        else if (a == "--level") o.level = std::stod(value("--level"));
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage(2);
        }
    }
    return o;
}

void list_devices() {
    std::printf("capture devices:\n");
    for (const std::string& d : audio::qt::input_device_names()) {
        std::printf("  %s%s\n", d.c_str(),
                    d == audio::qt::default_input_name() ? "  (default)" : "");
    }
    std::printf("playback devices:\n");
    for (const std::string& d : audio::qt::output_device_names()) {
        std::printf("  %s%s\n", d.c_str(),
                    d == audio::qt::default_output_name() ? "  (default)" : "");
    }
}

// Deterministic unit-RMS latents. The picture they decode to is
// meaningless; what matters is whether they survive the round trip.
std::vector<double> test_latents(int n) {
    std::vector<double> v(static_cast<std::size_t>(n));
    std::uint64_t s = 5;
    for (double& x : v) {
        s += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        x = static_cast<double>((z ^ (z >> 31)) >> 11) / 9007199254740992.0 - 0.5;
    }
    double ms = 0.0;
    for (double x : v) ms += x * x;
    const double rms = std::sqrt(ms / static_cast<double>(v.size()));
    for (double& x : v) x /= rms;
    return v;
}

int loopback(const Options& o) {
    auto report = [](const std::string& msg) {
        std::printf("  %s\n", msg.c_str());
        std::fflush(stdout);
    };

    const config::ModeSpec& mode = modem::mode_by_name(o.mode);
    const modem::Modem m;
    const std::vector<double> wave = tx::condition_for_output(
        m.modulate(test_latents(mode.n_latents), mode, true, o.callsign), o.level);

    // Long enough for the whole transmission plus the slack at each end.
    rx::RingBuffer ring(mode.duration_s + 20.0);
    audio::qt::InputStream in(o.in_device, ring, config::FS, report);
    std::printf("capturing from \"%s\" at %d Hz\n", in.device_name().c_str(),
                in.device_rate());

    std::printf("sending mode %s, %.0f s ...\n", o.mode.c_str(),
                static_cast<double>(wave.size()) / config::FS);
    std::fflush(stdout);
    const bool played = audio::qt::play(o.out_device, wave, config::FS, {}, {}, report);
    if (!played) {
        std::fprintf(stderr, "playback did not complete\n");
        return 1;
    }

    // Let the tail of the transmission make it through the capture
    // buffer before tearing the stream down.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    in.stop();

    std::uint64_t total = 0;
    const std::vector<double> got = ring.snapshot(&total);
    std::printf("captured %.1f s\n", static_cast<double>(total) / config::FS);
    if (total == 0) {
        std::fprintf(stderr, "nothing was captured -- is --in really hearing --out?\n");
        return 1;
    }

    try {
        const modem::DemodResult r = m.demodulate(got);
        std::printf("decoded: mode %s, %d/%d frames, callsign \"%s\", SNR %.1f dB\n",
                    std::string(r.mode.name).c_str(), r.frames_received,
                    r.mode.n_frames, r.callsign.c_str(), r.snr_db);
        if (r.frames_received < r.mode.n_frames) {
            std::fprintf(stderr, "warning: %d frames missing\n",
                         r.mode.n_frames - r.frames_received);
        }
        return r.callsign == o.callsign ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decode failed: %s\n", e.what());
        return 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    // QMediaDevices needs an application object to exist. Its event loop
    // is deliberately never run: capture has its own, and playback is
    // push-mode precisely so it works from a thread without one.
    QCoreApplication app(argc, argv);
    const Options o = parse(argc, argv);
    try {
        if (!o.loopback) {
            list_devices();
            return 0;
        }
        return loopback(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
