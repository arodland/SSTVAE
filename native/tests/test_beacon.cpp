// beacon::decode's multi-repetition combining fallback -- ported from
// sstvae/modem/beacon.py, and tested here rather than only via golden
// vectors because it is genuinely new logic (a search, not a
// mechanical translation), with the two properties that matter most
// having no oracle in a golden corpus:
//
// - it can recover a superframe when no single repetition's own
//   Golay+CRC decode survives the noise, by combining evidence across
//   every repetition instead of trying them one at a time;
// - it must never confidently return a fake BeaconResult on long pure
//   noise, which the first version of this search did on every single
//   trial (see beacon.cpp's file-level comment on the combining
//   fallback for why, and how the final verification step avoids it).

#include "beacon/beacon.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "check.hpp"
#include "config.hpp"

namespace check = sstvae::check;
using namespace sstvae;

namespace {

// Deterministic pseudo-random values, the same construction
// test_modem_roundtrip.cpp uses: splitmix64 plus a sum of uniforms for
// an approximately-Gaussian shape. Not numpy's generator and not
// trying to be -- nothing here is compared against Python, only
// against this file's own expectations.
class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}

    double normal() {
        double acc = 0.0;
        for (int i = 0; i < 4; ++i) {
            state_ += 0x9E3779B97F4A7C15ULL;
            std::uint64_t z = state_;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            acc += static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5;
        }
        return acc * 1.7;  // ~unit variance, per the same construction elsewhere
    }

private:
    std::uint64_t state_;
};

void test_combining_decodes_where_no_single_repetition_can() {
    // Mode C's own frame count -- see sstvae/config.py's MODES["C"] --
    // duplicated as a literal rather than pulled in via the modem
    // headers, which this file otherwise has no reason to depend on.
    constexpr int kModeCFrames = 660;
    const std::vector<double> chips = beacon::chip_stream(0, kModeCFrames, "TEST");

    Rng rng(6);
    std::vector<double> noisy(chips.size());
    for (std::size_t i = 0; i < chips.size(); ++i) noisy[i] = chips[i] + rng.normal();

    // Confirm the noise is actually heavy enough that individual
    // repetitions mostly fail on their own -- otherwise this would not
    // be exercising combining at all, just re-testing decode_single_
    // repetition under a new name.
    const std::vector<std::int64_t> candidates =
        beacon::find_sync(noisy, /*threshold=*/0.3, /*max_candidates=*/30);
    int n_solo_ok = 0;
    for (std::int64_t off : candidates) {
        const std::size_t end = static_cast<std::size_t>(off) + beacon::SYNC_LEN + beacon::CODED_LEN;
        if (end > noisy.size()) continue;
        std::span<const double> coded(noisy.data() + off + beacon::SYNC_LEN, beacon::CODED_LEN);
        if (beacon::decode_single_repetition(off, coded)) ++n_solo_ok;
    }
    check::is_true(n_solo_ok <= 2,
                   "beacon/combining: noise scale should leave most individual "
                   "repetitions undecodable on their own (got " +
                       std::to_string(n_solo_ok) + " solo successes)");

    const auto result = beacon::decode(noisy);
    check::is_true(result.has_value(),
                   "beacon/combining: should recover the superframe from combined "
                   "evidence even though no single repetition can decode it alone");
    if (result) {
        check::equal(result->frame_index,
                     static_cast<int>(result->chip_offset / config::CHIPS_PER_FRAME),
                     "beacon/combining: frame_index matches chip_offset (chip_stream "
                     "starts at frame 0)");
        check::is_true(result->callsign == "TEST",
                       "beacon/combining: recovered callsign is 'TEST', got '" +
                           result->callsign + "'");
    }
}

void test_no_false_lock_on_long_pure_noise() {
    // Long enough to hold the same number of repetitions as the
    // combining test above -- a short buffer never reaches the
    // combining fallback at all (repetition_grid needs >= 3 positions),
    // so it would not exercise the bug this guards against.
    constexpr int kModeCFrames = 660;
    const std::size_t n_chips = static_cast<std::size_t>(kModeCFrames) * config::CHIPS_PER_FRAME;

    // ASan+UBSan's instrumentation makes decode()'s search markedly
    // slower -- measured 2.4 s/trial at this length under
    // SSTVAE_SANITIZE against ~0.05 s/trial without it, roughly 50x, so
    // 200 trials (the unsanitized run, ~10 s) becomes several minutes
    // and blows past both this file's own watchdog (180 s) and the
    // ctest TIMEOUT (240 s). The statistical claim -- 0 false locks in
    // 200 trials -- is still checked at full strength and full speed by
    // every unsanitized build (every local run, CI's `native` job, and
    // ThreadSanitizer, which doesn't set SSTVAE_SANITIZE); a sanitizer
    // run exists to catch a memory-safety bug in the same code path, not
    // to re-derive that statistic, and a fifth of the trials is still
    // plenty of distinct seeds for that. 20 trials measures at ~48 s
    // here -- comfortable margin under both deadlines, not "just enough".
#ifdef SSTVAE_SANITIZE_BUILD
    constexpr int kTrials = 20;
#else
    constexpr int kTrials = 200;
#endif
    int false_locks = 0;
    for (int trial = 0; trial < kTrials; ++trial) {
        Rng rng(10000 + static_cast<std::uint64_t>(trial));
        std::vector<double> junk(n_chips);
        for (double& v : junk) v = rng.normal();
        if (beacon::decode(junk)) ++false_locks;
    }
    check::equal(false_locks, 0,
                 "beacon/combining: no false locks on " + std::to_string(kTrials) +
                     " trials of long pure noise");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(180.0, "beacon");

    check::current_step.store("combining_decodes_where_no_single_repetition_can");
    test_combining_decodes_where_no_single_repetition_can();
    check::current_step.store("no_false_lock_on_long_pure_noise");
    test_no_false_lock_on_long_pure_noise();

    return check::report("beacon");
}
