// The rolling capture buffer shared by the decode loop and the display.
//
// Long enough to hold a whole mode-C transmission plus margin, so a
// mid-stream lock can still decode the frames that arrived *before*
// sync was acquired.
//
// **Nothing here may block the audio callback.** That is not a
// preference; it is the single most expensive lesson in this project.
// The reference once copied the whole buffer while holding the lock the
// writer needed -- 8 MB at the default 130 s -- so the receiver tore a
// hole in its own audio every poll interval, and the holes *grew* as
// the buffer filled and the copy slowed. Measured against a
// simultaneous clean capture: losses of 85 samples rising to 235, one
// per poll, 1718 over 50 s. Enough to put the picture 5 dB down and
// break the beacon, while still syncing and reporting every frame
// received. A microbenchmark of the old code showed writes blocked for
// 786 ms against a 0.43 ms snapshot.
//
// The C++ version goes further than the reference's fix. Python holds a
// mutex just long enough to publish two integers; here there is **one**
// published value and no mutex on the write path at all:
//
//   * There is exactly one writer, so where samples go depends only on
//     state that thread owns.
//   * `write_pos` is not stored -- it is always `total_written % n`, so
//     position and count cannot be observed disagreeing. Keeping that
//     invariant is why an oversized chunk is written through rather
//     than truncated to the last n samples and the position reset.
//   * Readers take an acquire load of `total_written` and copy outside
//     any lock, exactly as the reference does.
//
// The tradeoff is the reference's: once the buffer has wrapped, the
// writer may overwrite the oldest samples while a reader is copying
// them. That costs at most the few hundred samples produced during the
// copy, at the far end of a 130-second history, and the decoder
// reconstructs from weighted latents that tolerate it. Losing *live*
// audio tolerates nothing.

#ifndef SSTVAE_RX_RINGBUFFER_HPP
#define SSTVAE_RX_RINGBUFFER_HPP

#include <atomic>
#include <cstdint>
#include <span>
#include <vector>

#include "config.hpp"

namespace sstvae::rx {

class RingBuffer {
public:
    explicit RingBuffer(double seconds, int fs = config::FS);

    // Append captured audio. Called from the audio callback.
    // Wait-free: no allocation, no lock, no system call.
    void write(std::span<const double> chunk) noexcept;

    // Chronological copy of everything currently held (oldest first).
    // `total` receives the number of samples ever written, which the
    // display uses as an absolute clock.
    std::vector<double> snapshot(std::uint64_t* total = nullptr) const;

    // The most recent `n` samples, oldest first (shorter if less has
    // been captured). A waterfall wants a few thousand samples many
    // times a second; `snapshot` copies the whole buffer, which is fine
    // at the decode loop's 5 s poll and ruinous at 20 fps.
    std::vector<double> tail(std::size_t n) const;

    // Drop everything captured so far, keeping the sample counter
    // monotonic.
    //
    // Not currently on the resume-after-transmit path: both callers that
    // need to keep our own sidetone out of the decoder (the "start
    // receiving" button and ReceivePanel::resume_after_transmit) instead
    // discard the whole RingBuffer and construct a fresh one, which
    // starts decode_loop over from scratch -- so blind_acc and every
    // other loop-local accumulator get a clean slate too, not just the
    // audio. That leaves total_written() restarting at 0 rather than
    // staying monotonic through the gap, which is fine because nothing
    // survives the restart that could still be indexing against the old
    // count. This method stays as a lower-overhead primitive (no
    // reallocation, counter stays meaningful) for a caller that wants to
    // wipe the audio in place without restarting the loop -- but pairing
    // it with such a resume would need its own explicit reset of
    // blind_acc/blind_acc_pushed and the other decode_loop locals, since
    // none of those are reachable from here.
    void clear();

    std::size_t capacity() const { return buf_.size(); }
    std::uint64_t total_written() const {
        return total_.load(std::memory_order_acquire);
    }

private:
    std::vector<double> buf_;
    // The only published state. Release on store, acquire on load, so a
    // reader that sees a count also sees the samples it counts.
    std::atomic<std::uint64_t> total_{0};
};

}  // namespace sstvae::rx

#endif
