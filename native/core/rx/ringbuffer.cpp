#include "rx/ringbuffer.hpp"

#include <algorithm>
#include <stdexcept>

namespace sstvae::rx {

RingBuffer::RingBuffer(double seconds, int fs) {
    const auto n = static_cast<std::size_t>(seconds * fs);
    if (n == 0) throw std::invalid_argument("RingBuffer: zero length");
    buf_.assign(n, 0.0);
}

void RingBuffer::write(std::span<const double> chunk) noexcept {
    if (chunk.empty()) return;
    const std::size_t n = buf_.size();
    const std::uint64_t total = total_.load(std::memory_order_relaxed);

    // An oversized chunk is written through rather than truncated: only
    // its last n samples survive either way, but writing through is what
    // keeps `total % n` pointing at the right place afterwards. It
    // cannot happen with any sane block size, and costs one extra pass
    // when it does.
    std::size_t skip = 0;
    if (chunk.size() > n) skip = chunk.size() - n;

    std::size_t pos = static_cast<std::size_t>((total + skip) % n);
    const double* src = chunk.data() + skip;
    std::size_t remaining = chunk.size() - skip;

    while (remaining > 0) {
        const std::size_t run = std::min(remaining, n - pos);
        std::copy_n(src, run, buf_.data() + pos);
        src += run;
        remaining -= run;
        pos += run;
        if (pos == n) pos = 0;
    }

    // Publish only after the samples are in place.
    total_.store(total + chunk.size(), std::memory_order_release);
}

std::vector<double> RingBuffer::snapshot(std::uint64_t* total_out) const {
    const std::uint64_t total = total_.load(std::memory_order_acquire);
    if (total_out) *total_out = total;

    const std::size_t n = buf_.size();
    if (total < n) {
        return std::vector<double>(buf_.begin(),
                                   buf_.begin() + static_cast<std::ptrdiff_t>(total));
    }
    const std::size_t pos = static_cast<std::size_t>(total % n);
    std::vector<double> out(n);
    // Oldest first: everything after the write cursor, then everything
    // before it. Copied outside any lock, on purpose.
    const auto split = std::copy(buf_.begin() + static_cast<std::ptrdiff_t>(pos),
                                 buf_.end(), out.begin());
    std::copy(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(pos), split);
    return out;
}

std::vector<double> RingBuffer::tail(std::size_t n) const {
    const std::size_t cap = buf_.size();
    const std::uint64_t total = total_.load(std::memory_order_acquire);
    n = std::min({n, cap, static_cast<std::size_t>(total)});
    if (n == 0) return {};

    const std::size_t end = static_cast<std::size_t>(total % cap);
    std::vector<double> out(n);
    if (end >= n) {
        std::copy(buf_.begin() + static_cast<std::ptrdiff_t>(end - n),
                  buf_.begin() + static_cast<std::ptrdiff_t>(end), out.begin());
    } else {
        const std::size_t head = n - end;  // wrapped part, at the buffer's tail
        const auto split = std::copy(buf_.end() - static_cast<std::ptrdiff_t>(head),
                                     buf_.end(), out.begin());
        std::copy(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(end), split);
    }
    return out;
}

void RingBuffer::clear() {
    // The counter stays where it is: absolute sample positions the
    // decode loop has already recorded must keep meaning the same
    // thing. Zeroing is enough to make the history read as silence.
    std::fill(buf_.begin(), buf_.end(), 0.0);
}

}  // namespace sstvae::rx
