#include "dsp/morse.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>

namespace sstvae::dsp {

namespace {

constexpr double PI = 3.14159265358979323846;

// nullptr terminates the dot/dash string; nothing here needs it to be a
// std::string_view.
const char* code_for(char c) {
    switch (c) {
        case 'A': return ".-";
        case 'B': return "-...";
        case 'C': return "-.-.";
        case 'D': return "-..";
        case 'E': return ".";
        case 'F': return "..-.";
        case 'G': return "--.";
        case 'H': return "....";
        case 'I': return "..";
        case 'J': return ".---";
        case 'K': return "-.-";
        case 'L': return ".-..";
        case 'M': return "--";
        case 'N': return "-.";
        case 'O': return "---";
        case 'P': return ".--.";
        case 'Q': return "--.-";
        case 'R': return ".-.";
        case 'S': return "...";
        case 'T': return "-";
        case 'U': return "..-";
        case 'V': return "...-";
        case 'W': return ".--";
        case 'X': return "-..-";
        case 'Y': return "-.--";
        case 'Z': return "--..";
        case '0': return "-----";
        case '1': return ".----";
        case '2': return "..---";
        case '3': return "...--";
        case '4': return "....-";
        case '5': return ".....";
        case '6': return "-....";
        case '7': return "--...";
        case '8': return "---..";
        case '9': return "----.";
        case '/': return "-..-.";
        default: return nullptr;
    }
}

// One element of the keying timeline: `on` for a tone burst of
// `seconds`, off (silence) otherwise.
struct Segment {
    bool on;
    double seconds;
};

std::vector<Segment> build_timeline(const std::string& text, double dot_s) {
    std::vector<Segment> segs;
    bool need_word_gap = false;  // a space was seen and not yet emitted
    bool first_symbol = true;    // no gap before the very first element

    for (char raw : text) {
        if (raw == ' ') {
            need_word_gap = true;
            continue;
        }
        char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw)));
        const char* code = code_for(c);
        if (code == nullptr) continue;

        if (!first_symbol) {
            segs.push_back({false, (need_word_gap ? 7.0 : 3.0) * dot_s});
        }
        need_word_gap = false;
        first_symbol = false;

        for (const char* p = code; *p != '\0'; ++p) {
            if (p != code) segs.push_back({false, dot_s});
            segs.push_back({true, (*p == '-' ? 3.0 : 1.0) * dot_s});
        }
    }
    return segs;
}

}  // namespace

std::vector<double> generate_morse(const std::string& text, int sample_rate,
                                   double wpm, double tone_hz, double amplitude) {
    const double dot_s = 1.2 / wpm;  // PARIS-standard element length
    const std::vector<Segment> timeline = build_timeline(text, dot_s);
    if (timeline.empty()) return {};

    std::size_t total = 0;
    for (const Segment& s : timeline) {
        total += static_cast<std::size_t>(std::lround(s.seconds * sample_rate));
    }

    std::vector<double> out;
    out.reserve(total);

    const double phase_inc = tone_hz / sample_rate;  // cycles/sample
    double phase = 0.0;

    for (const Segment& seg : timeline) {
        const std::size_t n =
            static_cast<std::size_t>(std::lround(seg.seconds * sample_rate));
        if (!seg.on) {
            out.insert(out.end(), n, 0.0);
            phase = std::fmod(phase + phase_inc * n, 1.0);
            continue;
        }

        // Raised-cosine on/off ramp to keep the keying click-free: 5 ms,
        // or half the burst for anything shorter than 10 ms (only
        // reachable at extreme wpm settings this feature does not
        // expose, but the fallback keeps the function well-defined).
        const std::size_t ramp_n =
            std::min(n / 2, static_cast<std::size_t>(std::lround(0.005 * sample_rate)));

        for (std::size_t i = 0; i < n; ++i) {
            double env = 1.0;
            if (ramp_n > 0) {
                if (i < ramp_n) {
                    env = 0.5 - 0.5 * std::cos(PI * static_cast<double>(i) /
                                               static_cast<double>(ramp_n));
                } else if (i >= n - ramp_n) {
                    env = 0.5 - 0.5 * std::cos(PI * static_cast<double>(n - 1 - i) /
                                               static_cast<double>(ramp_n));
                }
            }
            out.push_back(amplitude * env * std::sin(2.0 * PI * phase));
            phase = std::fmod(phase + phase_inc, 1.0);
        }
    }

    return out;
}

}  // namespace sstvae::dsp
