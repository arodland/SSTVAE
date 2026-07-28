// The C++ modem, end to end, against itself.
//
// Deliberately corpus-free. Parity with the Python reference is checked
// by tests/test_native_parity.py, which has the reference available to
// compare against; storing a mode A transmission here would add ~2 MB of
// golden data to say something that suite already says better.
//
// What this adds is coverage in `ctest`, where the reference is not
// available at all: the whole transmit and receive chain, run for real,
// asserted on properties that hold regardless of implementation --
// the mode decodes, every frame arrives, the callsign survives, and the
// latents come back. A build that compiled and passed the boundary
// vectors could still fail this.

#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "modem/modem.hpp"

using namespace sstvae;

namespace {

// Deterministic pseudo-random latents. Not numpy's generator and not
// trying to be: nothing here is compared against Python, so all this
// needs is to be reproducible and to look nothing like a constant.
std::vector<double> test_latents(int n, std::uint64_t seed) {
    std::vector<double> out(static_cast<std::size_t>(n));
    std::uint64_t s = seed;
    for (double& v : out) {
        // splitmix64, then map to roughly N(0,1) via a sum of uniforms.
        double acc = 0.0;
        for (int i = 0; i < 4; ++i) {
            s += 0x9E3779B97F4A7C15ULL;
            std::uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            acc += static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5;
        }
        v = acc * 1.7;
    }
    // Unit RMS: the on-air contract.
    double ms = 0.0;
    for (double v : out) ms += v * v;
    const double rms = std::sqrt(ms / static_cast<double>(out.size()));
    for (double& v : out) v /= rms;
    return out;
}

// Best-fit gain between sent and received latents.
//
// An SNR floor alone does not catch an amplitude error: dropping the
// 1/sqrt(2) from the equalizer scales every latent by 0.707, which
// still measures ~10.7 dB -- above any threshold the clip-limited clean
// loopback (~10 dB) leaves room for. Verified by perturbing exactly
// that and watching the SNR check pass. The gain is checked separately
// because it is the quantity that is actually wrong.
double latent_gain(const std::vector<double>& sent, const std::vector<double>& got,
                   const std::vector<double>& weights) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < sent.size(); ++i) {
        if (weights[i] <= 0) continue;
        num += sent[i] * got[i];
        den += sent[i] * sent[i];
    }
    return den > 0 ? num / den : 0.0;
}

double latent_snr_db(const std::vector<double>& sent,
                     const std::vector<double>& got,
                     const std::vector<double>& weights) {
    double err = 0.0, sig = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < sent.size(); ++i) {
        if (weights[i] <= 0) continue;
        const double d = sent[i] - got[i];
        err += d * d;
        sig += sent[i] * sent[i];
        ++n;
    }
    if (n == 0 || err == 0) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10((sig / static_cast<double>(n)) /
                             (err / static_cast<double>(n)));
}

void test_roundtrip(const config::ModeSpec& mode, const std::string& callsign) {
    const std::string what = "modem/mode " + std::string(mode.name);
    const modem::Modem md;
    const std::vector<double> latents =
        test_latents(mode.n_latents, static_cast<std::uint64_t>(mode.index) + 1);

    const std::vector<double> wave = md.modulate(latents, mode, true, callsign);
    check::equal(wave.size(),
                 static_cast<std::size_t>(config::LEADIN_SAMPLES +
                                          config::PREAMBLE_SAMPLES +
                                          config::HEADER_SAMPLES +
                                          mode.n_frames * config::FRAME_SAMPLES +
                                          config::LEADOUT_SAMPLES),
                 what + ": waveform length");

    // Unit RMS is the transmit contract; tx_condition ends on it.
    double ms = 0.0;
    for (double v : wave) ms += v * v;
    const double rms = std::sqrt(ms / static_cast<double>(wave.size()));
    check::is_true(std::abs(rms - 1.0) < 1e-9, what + ": waveform is unit RMS");

    const modem::DemodResult r = md.demodulate(wave);
    check::equal(r.mode.index, mode.index, what + ": recovered mode");
    check::equal(r.frames_received, mode.n_frames, what + ": every frame received");
    check::equal(r.callsign, callsign, what + ": callsign survived the beacon");
    check::is_true(r.beacon.has_value(), what + ": beacon decoded");
    check::is_true(std::abs(r.freq_offset) < 1.0,
                   what + ": no spurious frequency offset on a clean loopback");

    const double snr = latent_snr_db(latents, r.latents, r.weights);
    // The clean-loopback floor is set by clip-and-filter distortion at
    // the configured headroom, not by the modem. ~10 dB at 0.5 dB
    // headroom; 8 dB leaves room without being meaningless.
    check::is_true(snr > 8.0, what + ": latent SNR " + std::to_string(snr) +
                                  " dB through a clean loopback");
    const double gain = latent_gain(latents, r.latents, r.weights);
    check::is_true(std::abs(gain - 1.0) < 0.05,
                   what + ": latent gain " + std::to_string(gain) +
                       " (an amplitude error is invisible to the SNR check)");

    // Weights are confidences in 0..1, and every transmitted latent
    // should have one -- the zeros are the permanently dropped slots.
    std::size_t weighted = 0;
    bool in_range = true;
    for (double w : r.weights) {
        if (w < 0.0 || w > 1.0) in_range = false;
        if (w > 0) ++weighted;
    }
    check::is_true(in_range, what + ": weights are within 0..1");
    check::equal(weighted, static_cast<std::size_t>(mode.n_tx_latents),
                 what + ": one weight per transmitted latent");
}

void test_blind_roundtrip() {
    const config::ModeSpec& mode = config::MODES[0];
    const modem::Modem md;
    const std::vector<double> latents = test_latents(mode.n_latents, 42);
    const std::vector<double> wave = md.modulate(latents, mode, true, "N6MTS");

    // Blind decode sees no preamble and no header: it must recover its
    // absolute position from the beacon alone.
    const modem::BlindDemodResult r = md.demodulate_blind(wave);
    check::is_true(r.beacon.has_value(), "modem/blind: beacon decoded");
    check::equal(r.callsign, std::string("N6MTS"), "modem/blind: callsign");
    check::is_true(r.frame_offset.has_value(), "modem/blind: absolute frame offset");
    check::is_true(r.frame0_start.has_value(), "modem/blind: frame 0 located");

    // The latents land in canonical (group-aware) slots via the beacon's
    // frame index, so a correct blind decode recovers the picture even
    // though it never saw the header.
    const double snr = latent_snr_db(latents, r.latents, r.weights);
    check::is_true(snr > 8.0, "modem/blind: latent SNR " + std::to_string(snr) + " dB");
    const double gain = latent_gain(latents, r.latents, r.weights);
    check::is_true(std::abs(gain - 1.0) < 0.05,
                   "modem/blind: latent gain " + std::to_string(gain));
}

}  // namespace

int main() {
    try {
        // Mode A only for the per-mode round trip: B and C are the same
        // code over more frames, and each costs seconds of FFT work.
        test_roundtrip(config::MODES[0], "KC2G");
        test_blind_roundtrip();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("modem round trip");
}
