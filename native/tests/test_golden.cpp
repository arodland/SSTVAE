// Check the C++ core against the committed golden corpus.
//
// The corpus is the Python reference's output at each module boundary
// (tools/gen_golden_vectors.py). This binary and `pytest` read the same
// bytes, so neither side gets to hold its own idea of the right answer.
//
// Tolerances are per-check and each one is justified where it is used.
// The rule: anything that is pure integer or pure sign arithmetic must
// match *exactly*, and only sums of transcendentals get a tolerance.

#include <cstdlib>
#include <string>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "dsp/dsp.hpp"
#include "golay/golay.hpp"
#include "ofdm/ofdm.hpp"
#include "testing/npy.hpp"

using namespace sstvae;
using sstvae::dsp::cdouble;
using sstvae::testing::load_c16;
using sstvae::testing::load_f8;
using sstvae::testing::load_i8;

namespace {

std::string golden_dir;

std::string g(const std::string& name) { return golden_dir + "/" + name + ".npy"; }

// Tolerance for a single phasor, and the reason it is not zero.
//
// **Both** implementations now reduce (n*f) mod FS in exact integer
// arithmetic before calling exp(), so the argument is under one turn on
// each side and identical between them. What is left is only that no
// standard requires exp/sin/cos to be correctly rounded, so two libms
// may differ in the last ulp -- about 2.2e-16 on a unit phasor.
//
// Measured C++ against Python: 9.6e-16 on MOD_MATRIX, 9.4e-16 on
// DEMOD_MATRIX, and exactly 0 on the pilot sequence. 1e-14 is ~10x that,
// which is margin for platforms whose libm rounds differently, not slack
// for the port.
//
// This was 2e-13 while sstvae/modem/ofdm.py still built its phasors on
// an unreduced argument reaching 262 rad: the tolerance then had to
// cover the *reference's* ~3e-14 error, which is a much weaker statement
// about the port. See docs/todo.md, item closed 2026-07-28.
constexpr double PHASOR_TOL = 1e-14;

// Sums of NC or M of those phasors, so the per-term error can
// accumulate across 24 or 160 terms; numpy additionally reaches its sums
// through BLAS, which blocks and vectorizes and therefore associates
// differently. Measured worst case 4.9e-15 (modulate_symbols); 1e-13
// gives ~20x. For scale, 1e-13 on a unit phasor is -260 dB.
constexpr double PHASOR_SUM_TOL = 1e-13;

void test_golay() {
    check::equal(golay::min_distance(), 8, "golay/min_distance");

    const std::vector<std::int64_t> codewords = load_i8(g("golay/all_codewords"));
    check::equal(codewords.size(), std::size_t{4096}, "golay/all_codewords size");
    bool all_exact = true;
    for (int m = 0; m < golay::N_MESSAGES; ++m)
        if (static_cast<std::int64_t>(golay::encode(m)) != codewords[static_cast<std::size_t>(m)])
            all_exact = false;
    // Integer arithmetic: exact or broken, no tolerance is meaningful.
    check::is_true(all_exact, "golay/encode matches every reference codeword");

    const std::vector<std::int64_t> msgs = load_i8(g("golay/bits_messages"));
    const std::vector<std::int64_t> bits = load_i8(g("golay/bits_expected"));
    bool bits_ok = true;
    for (std::size_t i = 0; i < msgs.size(); ++i) {
        const auto got = golay::codeword_bits(static_cast<int>(msgs[i]));
        for (int b = 0; b < golay::N_BITS; ++b)
            if (got[static_cast<std::size_t>(b)] !=
                bits[i * golay::N_BITS + static_cast<std::size_t>(b)])
                bits_ok = false;
    }
    check::is_true(bits_ok, "golay/codeword_bits matches reference");

    const testing::NpyFile soft_file = testing::read_npy(g("golay/soft_inputs"));
    const std::vector<double> soft = load_f8(g("golay/soft_inputs"));
    const std::vector<std::int64_t> expected = load_i8(g("golay/soft_expected"));
    const std::size_t n_cases = soft_file.rows();
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < n_cases; ++i) {
        const std::span<const double> row(soft.data() + i * golay::N_BITS,
                                          golay::N_BITS);
        if (golay::decode_soft(row) != expected[i]) ++mismatches;
    }
    // Includes deliberately noisy cases where the reference decoder is
    // *wrong*; the port has to be wrong in the same places, or it is
    // not the same decoder.
    check::equal(mismatches, std::size_t{0},
                 "golay/decode_soft agrees on all " + std::to_string(n_cases) +
                     " cases, including the noisy ones");
}

void test_ofdm_tables() {
    // Frequencies are exactly representable small integers; anything
    // other than equality here means a layout bug, not rounding.
    check::close(std::vector<double>(ofdm::carrier_freqs().begin(),
                                     ofdm::carrier_freqs().end()),
                 load_f8(g("ofdm/carrier_freqs")), 0.0, "ofdm/carrier_freqs exact");
    check::close(std::vector<double>(ofdm::baseband_freqs().begin(),
                                     ofdm::baseband_freqs().end()),
                 load_f8(g("ofdm/baseband_freqs")), 0.0, "ofdm/baseband_freqs exact");

    const auto mod = ofdm::mod_matrix();
    check::close(std::vector<ofdm::cdouble>(mod.begin(), mod.end()),
                 load_c16(g("ofdm/mod_matrix")), PHASOR_TOL, "ofdm/mod_matrix");
    const auto demod = ofdm::demod_matrix();
    check::close(std::vector<ofdm::cdouble>(demod.begin(), demod.end()),
                 load_c16(g("ofdm/demod_matrix")), PHASOR_TOL, "ofdm/demod_matrix");

    // Single calls to polar/exp of an identically-grouped argument: the
    // only possible disagreement is libm's last ulp, hence 1e-15 rather
    // than the looser sum tolerance.
    check::close(std::vector<ofdm::cdouble>(ofdm::pilot_sequence().begin(),
                                            ofdm::pilot_sequence().end()),
                 load_c16(g("ofdm/pilot_sequence")), 1e-15, "ofdm/pilot_sequence");

    check::close(ofdm::preamble_waveform(), load_f8(g("ofdm/preamble_waveform")),
                 PHASOR_SUM_TOL, "ofdm/preamble_waveform");
    check::close(ofdm::preamble_template(), load_c16(g("ofdm/preamble_template")),
                 PHASOR_SUM_TOL, "ofdm/preamble_template");
    check::close(ofdm::pilot_template(), load_c16(g("ofdm/pilot_template")),
                 PHASOR_SUM_TOL, "ofdm/pilot_template");
}

void test_ofdm_transforms() {
    const testing::NpyFile in_file = testing::read_npy(g("ofdm/modulate_input"));
    const std::vector<ofdm::cdouble> symbols = load_c16(g("ofdm/modulate_input"));
    const std::size_t n_sym = in_file.rows();
    check::close(ofdm::modulate_symbols(symbols, n_sym),
                 load_f8(g("ofdm/modulate_expected")), PHASOR_SUM_TOL,
                 "ofdm/modulate_symbols");

    const std::vector<ofdm::cdouble> z = load_c16(g("ofdm/demod_baseband"));
    const std::vector<std::int64_t> starts = load_i8(g("ofdm/demod_starts"));
    const std::vector<std::int64_t> backoffs = load_i8(g("ofdm/demod_backoffs"));
    const std::vector<ofdm::cdouble> expected = load_c16(g("ofdm/demod_expected"));
    std::vector<ofdm::cdouble> got;
    got.reserve(starts.size() * config::NC);
    for (std::size_t i = 0; i < starts.size(); ++i) {
        const auto row = ofdm::demod_window(z, static_cast<std::int64_t>(starts[i]),
                                            static_cast<std::int64_t>(backoffs[i]));
        got.insert(got.end(), row.begin(), row.end());
    }
    check::close(got, expected, PHASOR_SUM_TOL,
                 "ofdm/demod_window over every (start, backoff) pair");

    // The zero-padded tail. This is the case that decides what happens
    // at the end of a recording, so it gets its own vector rather than
    // being trusted to the loop above.
    const std::vector<std::int64_t> tail_start = load_i8(g("ofdm/demod_tail_start"));
    const auto tail = ofdm::demod_window(z, static_cast<std::int64_t>(tail_start[0]));
    check::close(std::vector<ofdm::cdouble>(tail.begin(), tail.end()),
                 load_c16(g("ofdm/demod_tail_expected")), PHASOR_SUM_TOL,
                 "ofdm/demod_window past the end of the signal");
}

void test_dsp() {
    // FIR designs. Tolerance because sinc and the Hamming window are
    // transcendental and the scale normalization sums 129/201 terms;
    // 1e-14 is the same one-ulp-of-exp() reasoning as PHASOR_TOL.
    check::close(dsp::firwin_lowpass(129, 850.0), load_f8(g("dsp/firwin_sync")),
                 1e-14, "dsp/firwin lowpass matches scipy");
    check::close(dsp::firwin_bandpass(201, config::TX_BANDPASS_LO,
                                      config::TX_BANDPASS_HI),
                 load_f8(g("dsp/firwin_tx")), 1e-14,
                 "dsp/firwin bandpass matches scipy");

    const std::vector<double> x = load_f8(g("dsp/signal_input"));

    // Both sides reduce the heterodyne exactly, and it is periodic in 16
    // samples, so this should be as close as two exp() calls can be.
    check::close(dsp::to_baseband(x), load_c16(g("dsp/to_baseband")), 1e-14,
                 "dsp/to_baseband");

    // The FFT is the one place where the two implementations run
    // genuinely different code: SciPy is on ducc0, this is pocketfft.
    // Same lineage, no guarantee of identical bits, and an FFT sums
    // 4096 terms -- hence the looser bound. Still ~1e10 tighter than
    // anything that could affect a decode.
    check::close(dsp::hilbert(x), load_c16(g("dsp/hilbert")), 1e-11,
                 "dsp/hilbert");
    const std::vector<double> x_odd = load_f8(g("dsp/signal_input_odd"));
    check::close(dsp::hilbert(x_odd), load_c16(g("dsp/hilbert_odd")), 1e-11,
                 "dsp/hilbert at odd length (the other mask branch)");

    check::close(dsp::sync_lowpass(dsp::to_baseband(x)),
                 load_c16(g("dsp/sync_lowpass")), 1e-13, "dsp/sync_lowpass");

    const std::vector<double> papr = load_f8(g("dsp/papr_db"));
    check::close(std::vector<double>{dsp::papr_db(x)}, papr, 1e-11, "dsp/papr_db");

    // tx_condition runs two clip-and-filter iterations over a hilbert
    // each, so the FFT difference compounds; it is also the function
    // whose output goes on air, so it gets its own check rather than
    // being trusted to its parts.
    check::close(dsp::tx_condition(x, config::CLIP_HEADROOM_DB),
                 load_f8(g("dsp/tx_condition")), 1e-10, "dsp/tx_condition");

    // Integer output: exact, and specifically checking that half-to-even
    // rounding was used. std::round would differ here.
    const std::vector<std::int64_t> want_i16 = load_i8(g("dsp/to_int16"));
    const std::vector<std::int16_t> got_i16 = dsp::to_int16(x);
    bool i16_ok = got_i16.size() == want_i16.size();
    if (i16_ok)
        for (std::size_t i = 0; i < got_i16.size(); ++i)
            if (static_cast<std::int64_t>(got_i16[i]) != want_i16[i]) i16_ok = false;
    check::is_true(i16_ok, "dsp/to_int16 matches np.round exactly");

    const std::vector<double> offsets = load_f8(g("dsp/freq_correct_offsets"));
    const std::vector<cdouble> fc_expected = load_c16(g("dsp/freq_correct"));
    const std::vector<cdouble> z = dsp::to_baseband(x);
    std::vector<cdouble> fc_got;
    fc_got.reserve(offsets.size() * z.size());
    for (double f : offsets) {
        const auto row = dsp::freq_correct(z, f);
        fc_got.insert(fc_got.end(), row.begin(), row.end());
    }
    check::close(fc_got, fc_expected, 1e-13,
                 "dsp/freq_correct across the acquisition range");
}

void test_config_header() {
    // config.hpp is generated, so this is not checking arithmetic -- it
    // is checking that the committed header was generated from the
    // config.py the golden vectors came from.
    check::equal(config::NC, 24, "config/NC");
    check::equal(config::M, config::FS / config::RS, "config/M");
    check::equal(static_cast<int>(config::MODES.size()), 3, "config/mode count");
    check::equal(config::MODES[2].n_frames, 660, "config/mode C frames");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <golden-dir>\n\nThe golden corpus is generated by "
                     "tools/gen_golden_vectors.py.\n",
                     argv[0]);
        return 2;
    }
    golden_dir = argv[1];

    try {
        test_config_header();
        test_dsp();
        test_golay();
        test_ofdm_tables();
        test_ofdm_transforms();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("golden vectors");
}
