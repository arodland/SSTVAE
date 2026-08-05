// International Morse code tone generation, for the CW station ID.
//
// This has nothing to do with the SSTVAE waveform -- it is a plain
// on/off keyed sine tone, generated once per transmission and appended
// to the modulated audio so it goes out under the same PTT key-up.
// Nothing here is part of the on-air format `sstvae/modem/` defines;
// there is no receive side, because a human (or any other station) reads
// it by ear, the way CW IDs have always worked.

#pragma once

#include <string>
#include <vector>

namespace sstvae::dsp {

// Farnsworth-free ITU timing (the "PARIS" standard): a dot is
// 1.2/wpm seconds, a dash is 3 dots, the gap between elements of one
// character is 1 dot, between characters 3 dots, between words 7 dots.
//
// `text` is matched case-insensitively against A-Z, 0-9, '/' and space
// (word gap); any other character is silently dropped rather than
// rejected, since a callsign is operator-entered text and a stray
// character should not lose the rest of the ID. Tone bursts are
// raised-cosine ramped on and off (5 ms, or half the burst if shorter)
// to keep the keying free of clicks. Returns an empty vector for text
// with nothing recognized in it.
std::vector<double> generate_morse(const std::string& text, int sample_rate,
                                   double wpm = 18.0, double tone_hz = 1000.0,
                                   double amplitude = 1.0);

}  // namespace sstvae::dsp
