// What live::run() reads audio from: live.py's StdinSource and WavSource
// (qrss/sources.hpp has them). One piece of 8 kHz mono audio and the unix
// time of its last sample, as live.py's sources yield (x, t_end).

#ifndef SSTVAE_QRSS_SOURCE_HPP
#define SSTVAE_QRSS_SOURCE_HPP

#include <optional>
#include <string>
#include <vector>

namespace sstvae::qrss::live {

struct Chunk {
    std::vector<double> x;
    double t_end = 0.0;
};

// next(timeout_s): the next chunk; std::nullopt when timeout_s passed with
// none (a live source: live.py's (None, None)) or when the source has
// ended, which eof() then says. Called from one thread at a time.
class Source {
public:
    virtual ~Source() = default;
    virtual std::optional<Chunk> next(double timeout_s) = 0;
    virtual bool eof() const = 0;
    virtual std::string name() const = 0;
};

}  // namespace sstvae::qrss::live

#endif
