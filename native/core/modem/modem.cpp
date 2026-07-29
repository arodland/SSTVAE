#include "modem/modem.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

#include "dsp/dsp.hpp"
#include "framing/framing.hpp"
#include "ofdm/ofdm.hpp"

namespace sstvae::modem {
namespace {

using config::BEACON_CARRIER;
using config::CHIPS_PER_FRAME;
using config::CLIP_HEADROOM_DB;
using config::DATA_SYMS_PER_FRAME;
using config::DEMOD_BACKOFF;
using config::FRAME_SAMPLES;
using config::FRAMES_PER_GROUP;
using config::FS;
using config::HEADER_SAMPLES;
using config::LATENT_GROUPS;
using config::LATENTS_PER_FRAME;
using config::LEADIN_SAMPLES;
using config::LEADOUT_SAMPLES;
using config::M;
using config::NC;
using config::NC_LATENT;
using config::NCP;
using config::NSYM;
using config::PREAMBLE_CP;
using config::PREAMBLE_SAMPLES;
using config::RS;
using config::SNR_REF_BW_HZ;
using config::SYMS_PER_FRAME;

constexpr double PI = std::numbers::pi;

// np.median: for an even count, the mean of the two middle values.
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const std::size_t n = v.size();
    const std::size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    const double hi = v[mid];
    if (n % 2 == 1) return hi;
    const double lo =
        *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (lo + hi);
}

// Mean per-carrier phase increment of a gain vector (a timing proxy).
double bin_phase_step(std::span<const cdouble> h) {
    cdouble acc{0.0, 0.0};
    for (std::size_t i = 1; i < h.size(); ++i) acc += h[i] * std::conj(h[i - 1]);
    return std::arg(acc);
}

// Catmull-Rom over four surrounding pilots. The 6.9 Hz pilot rate
// oversamples even 2 Hz Doppler fading, but linear interpolation alone
// loses ~14 dB tracking it.
void catmull_rom(std::span<const cdouble> p0, std::span<const cdouble> p1,
                 std::span<const cdouble> p2, std::span<const cdouble> p3,
                 double u, std::span<cdouble> out) {
    const double u2 = u * u;
    const double u3 = u2 * u;
    for (int k = 0; k < NC; ++k) {
        const std::size_t i = static_cast<std::size_t>(k);
        out[i] = 0.5 * (2.0 * p1[i] + (p2[i] - p0[i]) * u +
                        (2.0 * p0[i] - 5.0 * p1[i] + 4.0 * p2[i] - p3[i]) * u2 +
                        (3.0 * p1[i] - p0[i] - 3.0 * p2[i] + p3[i]) * u3);
    }
}

// One data symbol's equalization: matched-filter combining, per-latent
// confidence, and the beacon chip. Shared between demodulate() and
// demodulate_blind(), which do the identical arithmetic.
struct EqualizedSymbol {
    std::array<double, NC_LATENT * 2> slots;
    std::array<double, NC_LATENT * 2> weights;
    double beacon_chip;
};

EqualizedSymbol equalize(std::span<const cdouble> raw_sym,
                         std::span<const cdouble> h, double floor, double med_h) {
    EqualizedSymbol out{};
    const double sqrt2 = std::sqrt(2.0);
    for (int k = 0; k < NC_LATENT; ++k) {
        const std::size_t i = static_cast<std::size_t>(k);
        const double mag = std::max(std::abs(h[i]), floor);
        const cdouble y = raw_sym[i] * std::conj(h[i]) / (mag * mag);
        const double w = std::min(std::abs(h[i]) / med_h, 1.0);
        out.slots[2 * i] = y.real() * sqrt2;
        out.slots[2 * i + 1] = y.imag() * sqrt2;
        out.weights[2 * i] = w;
        out.weights[2 * i + 1] = w;
    }
    const std::size_t b = static_cast<std::size_t>(BEACON_CARRIER);
    const double mag_b = std::max(std::abs(h[b]), floor);
    out.beacon_chip = (raw_sym[b] * std::conj(h[b]) / (mag_b * mag_b)).real();
    return out;
}

}  // namespace

const ModeSpec& mode_by_name(std::string_view name) {
    for (const ModeSpec& m : config::MODES) {
        if (m.name == name) return m;
    }
    throw std::out_of_range("unknown mode \"" + std::string(name) + "\"");
}

double estimate_snr_db(std::span<const cdouble> h_pilot, int n_frames,
                       std::span<const char> received) {
    std::vector<int> idx;
    for (int f = 0; f < n_frames; ++f)
        if (received.empty() || received[static_cast<std::size_t>(f)]) idx.push_back(f);
    if (idx.size() < 2) return std::nan("");

    // Only *adjacent* received frames contribute a noise sample: the
    // estimator treats the frame-to-frame change in channel gain as
    // noise, which is only meaningful across consecutive frames.
    double noise_acc = 0.0;
    std::size_t noise_n = 0;
    for (std::size_t i = 0; i + 1 < idx.size(); ++i) {
        if (idx[i + 1] - idx[i] != 1) continue;
        for (int k = 0; k < NC; ++k) {
            const cdouble d =
                h_pilot[static_cast<std::size_t>(idx[i + 1]) * NC + static_cast<std::size_t>(k)] -
                h_pilot[static_cast<std::size_t>(idx[i]) * NC + static_cast<std::size_t>(k)];
            noise_acc += std::norm(d);
            ++noise_n;
        }
    }
    if (noise_n == 0) return std::nan("");

    double signal_acc = 0.0;
    for (int f : idx)
        for (int k = 0; k < NC; ++k)
            signal_acc += std::norm(
                h_pilot[static_cast<std::size_t>(f) * NC + static_cast<std::size_t>(k)]);

    const double noise_var = 0.5 * (noise_acc / static_cast<double>(noise_n));
    const double signal_var =
        signal_acc / static_cast<double>(idx.size() * static_cast<std::size_t>(NC));
    if (noise_var <= 0) return std::numeric_limits<double>::infinity();
    if (signal_var <= 0) return -std::numeric_limits<double>::infinity();

    // Per-carrier SNR is in a ~RS-wide (50 Hz) noise bandwidth -- the
    // DFT correlator's matched-filter bandwidth -- scaled to the
    // reference bandwidth assuming roughly even power across carriers.
    const double snr_50hz = signal_var / noise_var;
    return 10.0 * std::log10(snr_50hz * (NC * RS / SNR_REF_BW_HZ));
}

Modem::Modem() {
    const auto& p = ofdm::pilot_sequence();
    pilot_.assign(p.begin(), p.end());
}

std::vector<double> Modem::modulate(std::span<const double> latents,
                                    const ModeSpec& mode, bool normalize,
                                    const std::string& callsign,
                                    double clip_headroom_db) const {
    if (latents.size() != static_cast<std::size_t>(mode.n_latents))
        throw std::invalid_argument("modulate: wrong latent count for this mode");

    std::vector<double> lat(latents.begin(), latents.end());
    if (normalize) {
        double ms = 0.0;
        for (double v : lat) ms += v * v;
        const double rms = std::sqrt(ms / static_cast<double>(lat.size()));
        if (rms > 0)
            for (double& v : lat) v /= rms;
    }

    const std::vector<double> slots = framing::interleave(lat, mode);
    const int n_f = mode.n_frames;
    const std::vector<double> chips = beacon::chip_stream(0, n_f, callsign);

    std::vector<cdouble> symbols(static_cast<std::size_t>(n_f) * SYMS_PER_FRAME * NC,
                                 cdouble{});
    for (int f = 0; f < n_f; ++f) {
        const std::size_t base = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
        for (int k = 0; k < NC; ++k)
            symbols[base + static_cast<std::size_t>(k)] =
                pilot_[static_cast<std::size_t>(k)];

        const std::span<const double> frame_slots(
            slots.data() + static_cast<std::size_t>(f) * LATENTS_PER_FRAME,
            LATENTS_PER_FRAME);
        const std::vector<cdouble> data = framing::slots_to_symbols(frame_slots);
        for (int s = 0; s < DATA_SYMS_PER_FRAME; ++s) {
            const std::size_t row = base + static_cast<std::size_t>(s + 1) * NC;
            for (int k = 0; k < NC_LATENT; ++k)
                symbols[row + static_cast<std::size_t>(k)] =
                    data[static_cast<std::size_t>(s) * NC_LATENT +
                         static_cast<std::size_t>(k)];
            symbols[row + static_cast<std::size_t>(BEACON_CARRIER)] =
                cdouble(chips[static_cast<std::size_t>(f) * CHIPS_PER_FRAME +
                              static_cast<std::size_t>(s)],
                        0.0);
        }
    }

    const std::vector<cdouble> hdr = framing::header_symbol(mode);
    std::vector<cdouble> hdr2(hdr.begin(), hdr.end());
    hdr2.insert(hdr2.end(), hdr.begin(), hdr.end());

    const std::vector<double> preamble = ofdm::preamble_waveform();
    const std::vector<double> hdr_wave = ofdm::modulate_symbols(hdr2, 2);
    const std::vector<double> body =
        ofdm::modulate_symbols(symbols, static_cast<std::size_t>(n_f) * SYMS_PER_FRAME);

    std::vector<double> x;
    x.reserve(LEADIN_SAMPLES + preamble.size() + hdr_wave.size() + body.size() +
              LEADOUT_SAMPLES);
    x.insert(x.end(), LEADIN_SAMPLES, 0.0);
    x.insert(x.end(), preamble.begin(), preamble.end());
    x.insert(x.end(), hdr_wave.begin(), hdr_wave.end());
    x.insert(x.end(), body.begin(), body.end());
    x.insert(x.end(), LEADOUT_SAMPLES, 0.0);
    return dsp::tx_condition(x, clip_headroom_db);
}

DemodResult Modem::demodulate(std::span<const double> x,
                              std::optional<std::pair<double, double>> search_s) const {
    std::vector<cdouble> z = dsp::to_baseband(x);

    std::optional<sync::SearchWindow> search;
    if (search_s)
        search = sync::SearchWindow{static_cast<std::int64_t>(search_s->first * FS),
                                    static_cast<std::int64_t>(search_s->second * FS)};
    const sync::Acquisition acq = sync::acquire(z, 0.5, 2, search);
    z = dsp::freq_correct(z, acq.freq_offset);

    const std::int64_t u0 = acq.preamble_start + PREAMBLE_CP;
    const auto w0 = ofdm::demod_window(z, u0, DEMOD_BACKOFF);
    const auto w1 = ofdm::demod_window(z, u0 + M, DEMOD_BACKOFF);
    std::vector<cdouble> h_pre(NC);
    for (int k = 0; k < NC; ++k) {
        const std::size_t i = static_cast<std::size_t>(k);
        h_pre[i] = (w0[i] + w1[i]) / (2.0 * pilot_[i]);
    }

    // Header: two identical BPSK symbols, matched-filter combined so
    // faded carriers contribute little instead of amplifying noise as
    // zero-forcing would.
    const std::int64_t h0 = acq.preamble_start + PREAMBLE_SAMPLES;
    std::vector<double> soft(NC, 0.0);
    for (int s = 0; s < 2; ++s) {
        const auto y = ofdm::demod_window(z, h0 + s * NSYM + NCP, DEMOD_BACKOFF);
        for (int k = 0; k < NC; ++k) {
            const std::size_t i = static_cast<std::size_t>(k);
            soft[i] += (y[i] * std::conj(h_pre[i])).real();
        }
    }
    const auto spec_opt = framing::decode_header(soft);
    if (!spec_opt) throw SyncError("header decode failed");
    const ModeSpec spec = *spec_opt;

    const int n_f = spec.n_frames;
    std::vector<cdouble> raw(static_cast<std::size_t>(n_f) * SYMS_PER_FRAME * NC,
                             cdouble{});
    std::vector<cdouble> h_pilot(static_cast<std::size_t>(n_f) * NC, cdouble{});
    std::vector<char> received(static_cast<std::size_t>(n_f), 0);
    const double phi_ref = bin_phase_step(h_pre);

    // Sample-clock drift tracking. The raw per-frame timing estimate
    // also sees the channel's group delay, which swings by many samples
    // as multipath taps fade; real clock drift is < 0.1 samples/frame. A
    // slow EMA keeps the fading wiggle out while following the drift
    // ramp; shifts are small and incremental.
    double tau_ema = 0.0;
    std::vector<double> pilot_powers;
    std::int64_t p = h0 + HEADER_SAMPLES;
    for (int f = 0; f < n_f; ++f) {
        if (p + FRAME_SAMPLES > static_cast<std::int64_t>(z.size())) break;
        const std::size_t fbase = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
        for (int s = 0; s < SYMS_PER_FRAME; ++s) {
            const auto sym = ofdm::demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF);
            std::copy(sym.begin(), sym.end(),
                      raw.begin() + static_cast<std::ptrdiff_t>(fbase +
                                                               static_cast<std::size_t>(s) * NC));
        }
        for (int k = 0; k < NC; ++k) {
            const std::size_t i = static_cast<std::size_t>(k);
            h_pilot[static_cast<std::size_t>(f) * NC + i] = raw[fbase + i] / pilot_[i];
        }
        received[static_cast<std::size_t>(f)] = 1;
        p += FRAME_SAMPLES;

        double power = 0.0;
        for (int k = 0; k < NC; ++k)
            power += std::norm(raw[fbase + static_cast<std::size_t>(k)]);
        power /= NC;
        pilot_powers.push_back(power);
        if (power > 0.1 * median(pilot_powers)) {
            const double phi = bin_phase_step(
                std::span<const cdouble>(h_pilot.data() + static_cast<std::size_t>(f) * NC, NC));
            // Wrap the difference to (-pi, pi] before scaling.
            const double d = std::arg(std::polar(1.0, phi - phi_ref));
            const double tau = -d * FS / (2 * PI * RS);
            tau_ema += 0.02 * (tau - tau_ema);
            if (std::abs(tau_ema) >= 2) {
                // np.clip(round(x), -2, 2); round() is half-to-even.
                double r = std::nearbyint(tau_ema);
                r = std::clamp(r, -2.0, 2.0);
                const int step = static_cast<int>(r);
                p += step;
                tau_ema -= step;
            }
        }
    }

    // Equalize data symbols with pilots interpolated across the frame.
    std::vector<double> latents(static_cast<std::size_t>(spec.n_tx_latents), 0.0);
    std::vector<double> weights(static_cast<std::size_t>(spec.n_tx_latents), 0.0);

    std::vector<double> mags;
    for (int f = 0; f < n_f; ++f)
        if (received[static_cast<std::size_t>(f)])
            for (int k = 0; k < NC; ++k)
                mags.push_back(std::abs(
                    h_pilot[static_cast<std::size_t>(f) * NC + static_cast<std::size_t>(k)]));
    const double med_h = mags.empty() ? 1.0 : median(mags);
    const double floor = std::max(0.05 * med_h, 1e-9);

    auto row = [&h_pilot](int i) {
        return std::span<const cdouble>(
            h_pilot.data() + static_cast<std::size_t>(i) * NC, NC);
    };
    auto pilot_at = [&](int i, int fallback) {
        if (i >= 0 && i < n_f && received[static_cast<std::size_t>(i)]) return row(i);
        return row(fallback);
    };

    std::vector<double> beacon_soft(static_cast<std::size_t>(n_f) * CHIPS_PER_FRAME, 0.0);
    std::vector<cdouble> h(NC);
    for (int f = 0; f < n_f; ++f) {
        if (!received[static_cast<std::size_t>(f)]) continue;
        const auto p0 = pilot_at(f - 1, f);
        const auto p1 = row(f);
        const auto p2 = pilot_at(f + 1, f);
        const int p3_fallback =
            (f + 1 < n_f && received[static_cast<std::size_t>(f + 1)]) ? f + 1 : f;
        const auto p3 = pilot_at(f + 2, p3_fallback);

        const std::size_t fbase = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
        for (int s = 1; s < SYMS_PER_FRAME; ++s) {
            const double u = static_cast<double>(s) / SYMS_PER_FRAME;
            catmull_rom(p0, p1, p2, p3, u, h);
            const std::span<const cdouble> raw_sym(
                raw.data() + fbase + static_cast<std::size_t>(s) * NC, NC);
            const EqualizedSymbol eq = equalize(raw_sym, h, floor, med_h);

            const std::size_t lo = static_cast<std::size_t>(f) * LATENTS_PER_FRAME +
                                   static_cast<std::size_t>(s - 1) * NC_LATENT * 2;
            for (std::size_t i = 0; i < eq.slots.size(); ++i) {
                latents[lo + i] = eq.slots[i];
                weights[lo + i] = eq.weights[i];
            }
            beacon_soft[static_cast<std::size_t>(f) * CHIPS_PER_FRAME +
                        static_cast<std::size_t>(s - 1)] = eq.beacon_chip;
        }
    }

    for (double& v : latents) v = std::clamp(v, -10.0, 10.0);
    const auto lat_full = framing::deinterleave(latents, spec);
    const auto w_full = framing::deinterleave(weights, spec);
    const auto beacon_result = beacon::decode(beacon_soft);

    int n_received = 0;
    for (char r : received) n_received += r;

    return DemodResult{lat_full.latents,
                       w_full.latents,
                       spec,
                       acq.freq_offset,
                       acq.metric,
                       n_received,
                       beacon_result,
                       beacon_result ? beacon_result->callsign : std::string(),
                       acq.preamble_start,
                       estimate_snr_db(h_pilot, n_f, received)};
}

BlindDemodResult Modem::demodulate_blind(
    std::span<const double> x,
    std::optional<std::pair<double, double>> search_s) const {
    std::vector<cdouble> z = dsp::to_baseband(x);

    std::optional<sync::SearchWindow> search;
    if (search_s)
        search = sync::SearchWindow{static_cast<std::int64_t>(search_s->first * FS),
                                    static_cast<std::int64_t>(search_s->second * FS)};
    const sync::BlindAcquisition ba = sync::acquire_blind(z, 55.0, 1.7, 8, 4.0, search);
    z = dsp::freq_correct(z, ba.freq_offset);

    const std::int64_t p0 = ba.frame_start - NCP;  // CP-start of local frame 0
    const std::int64_t L_lo = static_cast<std::int64_t>(
        std::ceil(-static_cast<double>(p0) / FRAME_SAMPLES));
    const std::int64_t L_hi = static_cast<std::int64_t>(std::floor(
        (static_cast<double>(z.size()) - FRAME_SAMPLES - static_cast<double>(p0)) /
        FRAME_SAMPLES));
    if (L_lo > L_hi)
        throw SyncError("blind lock too close to buffer edge to demod any full frame");
    const int n_f = static_cast<int>(L_hi - L_lo + 1);
    const std::int64_t p_start = p0 + L_lo * FRAME_SAMPLES;

    std::vector<cdouble> raw(static_cast<std::size_t>(n_f) * SYMS_PER_FRAME * NC,
                             cdouble{});
    std::vector<cdouble> h_pilot(static_cast<std::size_t>(n_f) * NC, cdouble{});
    std::int64_t p = p_start;
    for (int f = 0; f < n_f; ++f) {
        const std::size_t fbase = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
        for (int s = 0; s < SYMS_PER_FRAME; ++s) {
            const auto sym = ofdm::demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF);
            std::copy(sym.begin(), sym.end(),
                      raw.begin() + static_cast<std::ptrdiff_t>(
                                        fbase + static_cast<std::size_t>(s) * NC));
        }
        for (int k = 0; k < NC; ++k) {
            const std::size_t i = static_cast<std::size_t>(k);
            h_pilot[static_cast<std::size_t>(f) * NC + i] = raw[fbase + i] / pilot_[i];
        }
        p += FRAME_SAMPLES;
    }

    std::vector<double> mags;
    mags.reserve(h_pilot.size());
    for (const cdouble& v : h_pilot) mags.push_back(std::abs(v));
    const double med_h = median(mags);
    const double floor = std::max(0.05 * med_h, 1e-9);

    auto pilot_at = [&h_pilot, n_f](int i) {
        const int c = std::clamp(i, 0, n_f - 1);
        return std::span<const cdouble>(
            h_pilot.data() + static_cast<std::size_t>(c) * NC, NC);
    };

    std::vector<double> beacon_soft(static_cast<std::size_t>(n_f) * CHIPS_PER_FRAME, 0.0);
    std::vector<double> slot_values(
        static_cast<std::size_t>(n_f) * LATENTS_PER_FRAME, 0.0);
    std::vector<double> slot_weights(
        static_cast<std::size_t>(n_f) * LATENTS_PER_FRAME, 0.0);
    std::vector<cdouble> h(NC);
    for (int f = 0; f < n_f; ++f) {
        const auto p0_ = pilot_at(f - 1);
        const auto p1_ = pilot_at(f);
        const auto p2_ = pilot_at(f + 1);
        const auto p3_ = pilot_at(f + 2);
        const std::size_t fbase = static_cast<std::size_t>(f) * SYMS_PER_FRAME * NC;
        for (int s = 1; s < SYMS_PER_FRAME; ++s) {
            const double u = static_cast<double>(s) / SYMS_PER_FRAME;
            catmull_rom(p0_, p1_, p2_, p3_, u, h);
            const std::span<const cdouble> raw_sym(
                raw.data() + fbase + static_cast<std::size_t>(s) * NC, NC);
            const EqualizedSymbol eq = equalize(raw_sym, h, floor, med_h);
            const std::size_t lo = static_cast<std::size_t>(f) * LATENTS_PER_FRAME +
                                   static_cast<std::size_t>(s - 1) * NC_LATENT * 2;
            for (std::size_t i = 0; i < eq.slots.size(); ++i) {
                slot_values[lo + i] = eq.slots[i];
                slot_weights[lo + i] = eq.weights[i];
            }
            beacon_soft[static_cast<std::size_t>(f) * CHIPS_PER_FRAME +
                        static_cast<std::size_t>(s - 1)] = eq.beacon_chip;
        }
    }

    const auto beacon_result = beacon::decode(beacon_soft);
    const ModeSpec& mode_c = config::MODES[config::N_MODES - 1];
    std::vector<double> latents_full(static_cast<std::size_t>(mode_c.n_latents), 0.0);
    std::vector<double> weights_full(static_cast<std::size_t>(mode_c.n_latents), 0.0);

    std::optional<int> frame_offset;
    std::optional<std::int64_t> frame0_start;
    if (beacon_result) {
        frame_offset = beacon_result->frame_index -
                       static_cast<int>(beacon_result->chip_offset / CHIPS_PER_FRAME);
        for (int f = 0; f < n_f; ++f) {
            const int abs_frame = *frame_offset + f;
            if (abs_frame < 0 || abs_frame >= LATENT_GROUPS * FRAMES_PER_GROUP) continue;
            const auto fs = framing::slot_range_for_frame(abs_frame);
            for (int i = 0; i < LATENTS_PER_FRAME; ++i) {
                const std::size_t dst =
                    static_cast<std::size_t>(fs.indices[static_cast<std::size_t>(i)]);
                const std::size_t src = static_cast<std::size_t>(f) * LATENTS_PER_FRAME +
                                        static_cast<std::size_t>(i);
                latents_full[dst] = std::clamp(slot_values[src], -10.0, 10.0);
                weights_full[dst] = slot_weights[src];
            }
        }
        // Anchored on p_start, not p0: the demod loop (and so the beacon
        // chip stream that frame_offset indexes) starts at p_start,
        // which is L_lo frames away from p0 whenever the blind lock is
        // not already at the buffer start. Using p0 put absolute frame 0
        // off by L_lo frames -- tens of seconds for a mid-stream lock.
        frame0_start = p_start - static_cast<std::int64_t>(*frame_offset) * FRAME_SAMPLES;
    }

    return BlindDemodResult{latents_full,
                            weights_full,
                            ba.freq_offset,
                            beacon_result,
                            beacon_result ? beacon_result->callsign : std::string(),
                            frame_offset,
                            n_f,
                            frame0_start,
                            estimate_snr_db(h_pilot, n_f)};
}

}  // namespace sstvae::modem
