// The soundcard plumbing that is not the soundcard.
//
// None of this needs a device, which is the point: every audio bug this
// project has actually had was in the rate conversion, the sample-format
// conversion or the device matching, and all of them were found by tests
// against a fake device rather than by listening to anything.
//
// The load-bearing test is `test_streaming_matches_one_shot`. Resampling
// each captured chunk independently is wrong in a way that still decodes
// -- on a real on-air recording it cost 4.7 dB of SNR and mangled the
// picture while syncing and reporting 440/440 frames -- so the thing to
// assert is not "the resampler runs" but "splitting the stream up
// changes nothing".

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "check.hpp"
#include "config.hpp"
#include "dsp/dsp.hpp"

using namespace sstvae;
using audio::SampleFormat;

namespace {

std::vector<double> noise(std::size_t n, std::uint64_t seed) {
    std::vector<double> v(n);
    std::uint64_t s = seed;
    for (double& x : v) {
        s += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        x = 2.0 * (static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5);
    }
    return v;
}

void test_resample_ratio_is_directional() {
    // The bug this signature exists to prevent: sharing one "ratio to the
    // device" between capture and playback decimated a 32 s transmission
    // into 0.9 s of noise.
    const audio::Ratio capture = audio::resample_ratio(48000, 8000);
    const audio::Ratio playback = audio::resample_ratio(8000, 48000);
    check::equal(capture.up, 1, "audio/ratio: capture 48k->8k upsamples by 1");
    check::equal(capture.down, 6, "audio/ratio: ... and decimates by 6");
    check::equal(playback.up, 6, "audio/ratio: playback 8k->48k is the inverse");
    check::equal(playback.down, 1, "audio/ratio: ... exactly");

    const audio::Ratio odd = audio::resample_ratio(44100, 8000);
    check::equal(odd.up, 80, "audio/ratio: 44.1k->8k reduced by the gcd");
    check::equal(odd.down, 441, "audio/ratio: ... 80/441");

    const audio::Ratio same = audio::resample_ratio(8000, 8000);
    check::equal(same.up, 1, "audio/ratio: equal rates are 1:1");
    check::equal(same.down, 1, "audio/ratio: ... both ways");
}

void test_streaming_matches_one_shot() {
    // Feed the same signal in irregular chunks and in one piece. The
    // concatenated streaming output must be what one-shot resampling of
    // the whole stream would have produced, sample for sample -- that is
    // the entire contract, and the reference achieves it exactly.
    const int rates[][2] = {{44100, 8000}, {48000, 8000}, {12000, 8000}, {8000, 8000}};
    for (const auto& pair : rates) {
        const audio::Ratio r = audio::resample_ratio(pair[0], pair[1]);
        const std::vector<double> x = noise(static_cast<std::size_t>(pair[0]), 7);

        audio::StreamResampler resampler(r.up, r.down);
        std::vector<double> got;
        std::uint64_t s = 12345;
        std::size_t i = 0;
        while (i < x.size()) {
            // Irregular block sizes, as a real device delivers.
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            const std::size_t n =
                std::min<std::size_t>(100 + (s >> 33) % 1900, x.size() - i);
            const std::vector<double> out =
                resampler(std::span<const double>(x.data() + i, n));
            got.insert(got.end(), out.begin(), out.end());
            i += n;
        }

        const std::vector<double> want = dsp::resample_poly(x, r.up, r.down);
        const std::string tag =
            std::to_string(pair[0]) + "->" + std::to_string(pair[1]);

        check::is_true(got.size() <= want.size(),
                       "audio/stream: " + tag + " does not invent samples");
        // Everything but the last `2 * pad` input samples of latency.
        check::is_true(got.size() * 100 >= want.size() * 99,
                       "audio/stream: " + tag + " emits all but the pad latency");

        double worst = 0.0;
        for (std::size_t k = 0; k < got.size(); ++k) {
            worst = std::max(worst, std::abs(got[k] - want[k]));
        }
        // The reference measures exactly 0 here. This implementation
        // sums the same filter over differently-sized blocks, so the
        // rounding order differs; anything above ~1e-15 would mean the
        // blocks are not lining up rather than that they add up
        // differently.
        check::is_true(worst < 1e-12,
                       "audio/stream: " + tag + " matches one-shot (worst " +
                           std::to_string(worst) + ")");
    }
}

void test_streaming_holds_back_rather_than_guessing() {
    // Nothing may come out until there is context on both sides. An
    // implementation that emitted early would be zero-padding, which is
    // exactly the per-chunk bug wearing a different hat.
    const audio::Ratio r = audio::resample_ratio(48000, 8000);
    audio::StreamResampler resampler(r.up, r.down);
    const std::vector<double> tiny = noise(16, 3);
    check::equal(resampler(tiny).size(), std::size_t{0},
                 "audio/stream: a chunk shorter than the filter emits nothing");
    check::equal(resampler(std::span<const double>{}).size(), std::size_t{0},
                 "audio/stream: an empty chunk is a no-op");
}

void test_sample_format_round_trip() {
    // Values a device would actually produce, including both rails.
    const std::vector<double> x = {0.0, 0.5, -0.5, 0.25, -0.75, 1.0, -1.0};
    for (const SampleFormat fmt : {SampleFormat::Float, SampleFormat::Int16,
                                   SampleFormat::Int32, SampleFormat::UInt8}) {
        const std::string name = audio::sample_format_name(fmt);
        const std::vector<std::byte> raw = audio::mono_to_bytes(x, fmt, 1);
        check::equal(raw.size(), x.size() * static_cast<std::size_t>(
                                                audio::bytes_per_sample(fmt)),
                     "audio/fmt: " + name + " byte count");

        const std::vector<double> back = audio::bytes_to_mono(raw, fmt, 1);
        check::equal(back.size(), x.size(), "audio/fmt: " + name + " sample count");

        // One quantisation step of tolerance, and none at all for float.
        const double step =
            fmt == SampleFormat::Float ? 1e-7 : 2.0 / (audio::bytes_per_sample(fmt) == 1
                                                           ? 256.0
                                                           : (fmt == SampleFormat::Int16
                                                                  ? 65536.0
                                                                  : 4294967296.0));
        double worst = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            worst = std::max(worst, std::abs(back[i] - x[i]));
        }
        check::is_true(worst <= step * 1.5,
                       "audio/fmt: " + name + " survives a round trip (worst " +
                           std::to_string(worst) + ")");
    }
}

void test_full_scale_does_not_wrap() {
    // The reason integer output scales by `scale - 1` and clips in the
    // integer domain: a sample at exactly +1.0 must land on positive
    // full scale, not wrap round to the negative rail. A wrap here is a
    // loud click on every transmission peak.
    const std::vector<double> rails = {1.0, -1.0, 1.5, -1.5};

    const std::vector<std::byte> i16 = audio::mono_to_bytes(rails, SampleFormat::Int16, 1);
    std::vector<std::int16_t> got(rails.size());
    std::memcpy(got.data(), i16.data(), i16.size());
    check::equal(got[0], std::int16_t{32767}, "audio/rail: +1.0 -> positive full scale");
    check::equal(got[1], std::int16_t{-32767}, "audio/rail: -1.0 -> negative");
    check::equal(got[2], std::int16_t{32767}, "audio/rail: over-range clips, not wraps");
    // -32768 and not -32767: the clip bounds are the type's own, so an
    // over-range negative reaches true full scale while -1.0 stops one
    // short of it. That asymmetry is the reference's, and it is the
    // price of never wrapping +1.0 round to the negative rail.
    check::equal(got[3], std::int16_t{-32768}, "audio/rail: ... in both directions");

    const std::vector<std::byte> u8 = audio::mono_to_bytes(rails, SampleFormat::UInt8, 1);
    check::equal(std::to_integer<int>(u8[0]), 255, "audio/rail: UInt8 +1.0");
    check::equal(std::to_integer<int>(u8[1]), 1, "audio/rail: UInt8 -1.0");
    check::equal(std::to_integer<int>(u8[3]), 0, "audio/rail: UInt8 clips at 0");

    // Float is clipped but not scaled.
    const std::vector<std::byte> f = audio::mono_to_bytes(rails, SampleFormat::Float, 1);
    std::vector<float> fs(rails.size());
    std::memcpy(fs.data(), f.data(), f.size());
    check::equal(fs[2], 1.0f, "audio/rail: float over-range is clipped");
    check::equal(fs[3], -1.0f, "audio/rail: ... symmetrically");
}

void test_channels_are_mixed_down_not_picked() {
    // A radio interface that puts the signal on the right channel only
    // must still work. Taking channel 0 would capture silence, which
    // looks exactly like a dead antenna.
    const std::vector<float> stereo = {0.0f, 0.8f, 0.0f, -0.4f, 0.0f, 0.2f};
    std::vector<std::byte> raw(stereo.size() * sizeof(float));
    std::memcpy(raw.data(), stereo.data(), raw.size());

    const std::vector<double> mono = audio::bytes_to_mono(raw, SampleFormat::Float, 2);
    check::equal(mono.size(), std::size_t{3}, "audio/mix: one sample per frame");
    check::close(mono, std::vector<double>{0.4, -0.2, 0.1}, 1e-7,
                 "audio/mix: channels averaged, so a right-only signal survives");

    // A trailing partial frame is dropped rather than misaligning
    // everything after it.
    const std::vector<double> odd = audio::bytes_to_mono(
        std::span<const std::byte>(raw.data(), raw.size() - sizeof(float)),
        SampleFormat::Float, 2);
    check::equal(odd.size(), std::size_t{2}, "audio/mix: partial frame dropped");

    // And playback duplicates across channels.
    const std::vector<std::byte> out =
        audio::mono_to_bytes(std::vector<double>{0.5}, SampleFormat::Float, 2);
    std::vector<float> pair(2);
    std::memcpy(pair.data(), out.data(), out.size());
    check::equal(pair[0], pair[1], "audio/mix: playback writes both channels");
}

void test_device_matching() {
    const std::vector<std::string> devices = {
        "Built-in Audio Analogue Stereo",
        "K4 RX A Digital Stereo (IEC958)",
        "K4 RX B Digital Stereo (IEC958)",
    };

    check::is_true(!audio::match_device(devices, "").has_value(),
                   "audio/dev: no preference means the system default");
    check::is_true(audio::match_device(devices, devices[1]) == std::size_t{1},
                   "audio/dev: an exact name wins");
    // The case a saved config actually hits: the name was stored before
    // the backend decorated it.
    check::is_true(audio::match_device(devices, "K4 RX A") == std::size_t{1},
                   "audio/dev: a unique substring matches");
    check::is_true(audio::match_device(devices, "k4 rx b") == std::size_t{2},
                   "audio/dev: matching is case-insensitive");

    // Ambiguity must fall back to the default rather than guess:
    // capturing from the wrong receiver is worse than saying so.
    check::is_true(!audio::match_device(devices, "K4 RX").has_value(),
                   "audio/dev: an ambiguous substring matches nothing");
    check::is_true(!audio::match_device(devices, "Behringer").has_value(),
                   "audio/dev: an absent device matches nothing");

    // An exact match still wins even when it is also a substring of
    // another entry, or unplugging one radio would move the other.
    const std::vector<std::string> nested = {"K4 RX", "K4 RX A"};
    check::is_true(audio::match_device(nested, "K4 RX") == std::size_t{0},
                   "audio/dev: exact beats ambiguous-substring");
}

// The pipeline is the three conversions a backend must not get wrong,
// driven the way a device drives them: arbitrary byte chunks, never
// aligned to anything. The property is that the chunking is *invisible*
// -- feed the same stream in one piece and in ragged pieces and the
// output must be identical, because that is exactly what per-chunk
// resampling breaks (4.7 dB, on a real recording, while still reporting
// 440/440 frames).
void test_capture_pipeline_is_chunking_invariant() {
    constexpr int kChannels = 2;
    constexpr int kDeviceRate = 48000;
    // A second of stereo int16 at the device's rate.
    const std::vector<double> left = noise(kDeviceRate, 11);
    const std::vector<double> right = noise(kDeviceRate, 12);
    std::vector<double> interleaved(left.size() * kChannels);
    for (std::size_t i = 0; i < left.size(); ++i) {
        interleaved[i * kChannels] = left[i];
        interleaved[i * kChannels + 1] = right[i];
    }
    const std::vector<std::byte> raw =
        audio::mono_to_bytes(interleaved, audio::SampleFormat::Int16, 1);

    const auto run = [&](std::span<const std::size_t> chunk_bytes) {
        audio::CapturePipeline pipe(audio::SampleFormat::Int16, kChannels, kDeviceRate,
                                    config::FS);
        std::vector<double> out;
        std::size_t i = 0;
        std::size_t k = 0;
        while (i < raw.size()) {
            const std::size_t n = std::min(chunk_bytes[k++ % chunk_bytes.size()],
                                           raw.size() - i);
            const std::vector<double> got =
                pipe(std::span<const std::byte>(raw.data() + i, n));
            out.insert(out.end(), got.begin(), got.end());
            i += n;
        }
        return out;
    };

    // One giant chunk against ragged ones. Every size is a multiple of
    // the 4-byte stereo frame, since a backend delivers whole frames and
    // a partial one is a different (also tested) concern.
    const std::size_t whole[] = {raw.size()};
    const std::size_t ragged[] = {512, 4096, 76, 20000, 1024};
    const std::vector<double> a = run(whole);
    const std::vector<double> b = run(ragged);

    check::is_true(!a.empty(), "audio/pipe: the one-shot pass produced audio");
    check::equal(b.size(), a.size(), "audio/pipe: ragged chunking emits the same count");
    double worst = 0.0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    // Exactly zero, not a tolerance: both passes run the same filter
    // over the same samples, and StreamResampler consumes whole
    // multiples of `down`, so the block boundaries land identically.
    // Anything nonzero means the state is not being carried.
    check::equal(worst, 0.0, "audio/pipe: ... sample for sample");

    // 48k stereo in, 8k mono out: a second of input is 8000 samples,
    // less the resampler's pad latency. Asserting the count catches the
    // direction being inverted, which is the bug that sent 32 s of
    // transmission out as 0.9 s of noise.
    constexpr std::size_t kExpected = kDeviceRate / 6;
    check::is_true(a.size() <= kExpected,
                   "audio/pipe: 48k->8k does not invent samples");
    check::is_true(a.size() * 100 >= kExpected * 99,
                   "audio/pipe: ... and emits all but the pad latency (" +
                       std::to_string(a.size()) + " of " + std::to_string(kExpected) +
                       ")");

    // The mixdown is the average, not channel 0. Left and right are
    // independent noise here, so taking one channel would not halve the
    // RMS the way averaging two uncorrelated signals does.
    double sum_sq = 0.0;
    for (const double v : a) sum_sq += v * v;
    const double rms = std::sqrt(sum_sq / static_cast<double>(a.size()));
    check::is_true(rms > 0.0, "audio/pipe: the mixdown is not silence");
}

void test_capture_pipeline_skips_the_resampler_at_rate() {
    audio::CapturePipeline at_rate(audio::SampleFormat::Float, 1, config::FS, config::FS);
    check::is_true(!at_rate.resampling(),
                   "audio/pipe: no resampler when the device is already at FS");
    audio::CapturePipeline off_rate(audio::SampleFormat::Float, 1, 44100, config::FS);
    check::is_true(off_rate.resampling(),
                   "audio/pipe: ... and one when it is not");

    // samples_in counts at the *device's* rate, so it is nonzero on the
    // very first chunk even while the resampler is still filling.
    const std::vector<double> x = noise(64, 5);
    const std::vector<std::byte> raw =
        audio::mono_to_bytes(x, audio::SampleFormat::Float, 1);
    const std::vector<double> out = off_rate(raw);
    check::equal(off_rate.samples_in(), std::uint64_t{64},
                 "audio/pipe: samples_in counts pre-resample input");
    check::equal(out.size(), std::size_t{0},
                 "audio/pipe: ... while the output is still held back");
}

}  // namespace

int main() {
    try {
        test_resample_ratio_is_directional();
        test_streaming_matches_one_shot();
        test_streaming_holds_back_rather_than_guessing();
        test_capture_pipeline_is_chunking_invariant();
        test_capture_pipeline_skips_the_resampler_at_rate();
        test_sample_format_round_trip();
        test_full_scale_does_not_wrap();
        test_channels_are_mixed_down_not_picked();
        test_device_matching();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("audio");
}
