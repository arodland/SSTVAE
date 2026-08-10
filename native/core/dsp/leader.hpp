// The VOX leader: audio sent ahead of a transmission purely to open a
// voice-operated transmitter's relay before the modem's own signal
// starts.
//
// Unlike the CW ID in morse.hpp, this is **not** something the receive
// side can ignore. It sits immediately in front of the preamble, inside
// the window `sync::acquire` searches, and `acquire` takes a hard argmax
// over that window -- so a leader that looks preamble-like does not
// merely waste airtime, it steals the lock and the transmission does not
// decode at all.
//
// **Which is exactly what a plain tone does.** The detector's metric is
// a lag-M autocorrelation normalized by the window's own energy, so for
// any steady sinusoid numerator and denominator are equal and it reads
// **1.000** -- the ceiling, above what a real preamble reaches in any
// noise. Measured (mode A, clean): a 500 ms 1500 Hz leader tone takes
// the lock and `demodulate` fails with "header decode failed". A tone is
// the obvious implementation of this feature and it is unusable.
//
// So the leader is a **swept** tone, which decorrelates at lag M, and
// the sweep is what has to be got right rather than the duration.
// docs/todo.md carries the general case: the receiver's vulnerability to
// a steady carrier is a real weakness that a transmitter-side chirp
// dodges rather than fixes.

#pragma once

#include <vector>

namespace sstvae::dsp {

// One sweep of the leader, low edge to high edge of the occupied band.
//
// **A fixed sweep period, repeated, rather than one sweep stretched over
// the requested duration** -- and that is the whole design, not a
// detail. What decorrelates the leader is the *rate* at which its
// frequency moves, so stretching one sweep to fill a longer leader makes
// it progressively more tone-like and progressively more dangerous.
// Measured, the leader's own peak detection metric against the 0.42
// threshold:
//
//     duration    one stretched sweep    repeated 0.25 s sweeps
//     0.3 s       0.359                  0.330
//     0.5 s       0.443  (over!)         0.330
//     1.0 s       0.556                  0.330
//     3.0 s       0.721                  0.330
//     10.0 s      0.969                  0.330
//
// The stretched form is already above threshold at the default 500 ms
// and is essentially a tone by 10 s; the repeated form is flat at 0.330
// at every duration, because the rate never changes. 0.330 is also below
// the 0.358 peak that 3000 s of AWGN produces through this detector
// (CLAUDE.md, PREAMBLE_REPEATS), so the leader is no more likely to
// cause a false lock than the silence it replaces.
//
// **Those numbers are the leader measured in front of a real
// transmission**, which is the only position worth measuring it in --
// standalone, the same 500 ms stretched sweep reads 0.104 rather than
// 0.443, because the energy floor and the filter state both depend on
// what surrounds it. `test_the_vox_leader_never_looks_like_a_preamble`
// measures it the same way and reproduces the whole column, which is
// also a cross-check of the C++ against the Python probe the design came
// from.
inline constexpr double VOX_SWEEP_S = 0.25;

// Silence between the end of the leader and the start of the
// transmission, so the relay has settled and no leader energy remains in
// the preamble's 480-sample correlation window.
inline constexpr double VOX_LEAD_GAP_S = 0.1;

// `seconds` of repeated up-sweeps across the carrier band, at
// `amplitude`, with 5 ms raised-cosine edges so keying it neither clicks
// nor splatters. The sweep is phase-continuous across the wrap from the
// high edge back to the low one.
//
// Returns empty for a non-positive duration, which is how the feature is
// switched off.
std::vector<double> vox_leader(double seconds, int sample_rate,
                               double amplitude = 1.0);

}  // namespace sstvae::dsp
