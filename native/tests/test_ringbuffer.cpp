// The capture ring buffer.
//
// Correctness only. The property that actually broke a receiver -- that
// `write` must never block, because it runs on the audio callback and a
// blocked callback means the host discards input -- is secured **by
// construction** rather than by a test: the write path takes no lock,
// makes no allocation and no system call, and the published state is a
// single atomic counter with the position derived from it. A timing
// assertion would only tell us how busy the machine was.
//
// What is tested here is everything that could silently corrupt the
// audio instead: ordering, the wrap seam, block-size independence, and
// that `clear` leaves the sample clock alone (the decode loop records
// absolute sample positions against it).

#include <algorithm>
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "rx/ringbuffer.hpp"

using namespace sstvae;

namespace {

std::vector<double> ramp(int from, int count) {
    std::vector<double> v(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) v[static_cast<std::size_t>(i)] = from + i;
    return v;
}

// A buffer of exactly `n` samples, bypassing the seconds/rate maths.
rx::RingBuffer make(std::size_t n) {
    return rx::RingBuffer(static_cast<double>(n), 1);
}

void test_partial_fill() {
    rx::RingBuffer rb = make(100);
    rb.write(ramp(0, 30));

    std::uint64_t total = 0;
    const std::vector<double> snap = rb.snapshot(&total);
    check::equal(total, std::uint64_t{30}, "ring/partial: total written");
    check::equal(snap.size(), std::size_t{30}, "ring/partial: snapshot is short");
    check::is_true(snap == ramp(0, 30), "ring/partial: chronological order");
}

void test_exact_fill_and_wrap() {
    rx::RingBuffer rb = make(100);
    rb.write(ramp(0, 100));
    check::is_true(rb.snapshot() == ramp(0, 100),
                   "ring/exact: a full buffer reads back in order");

    // Wrap by 30: the oldest 30 fall off the front.
    rb.write(ramp(100, 30));
    const std::vector<double> snap = rb.snapshot();
    check::equal(snap.size(), std::size_t{100}, "ring/wrap: still full");
    check::is_true(snap == ramp(30, 100),
                   "ring/wrap: oldest dropped, no gap at the seam");
}

void test_many_small_writes_match_one_big_one() {
    // The audio callback delivers small blocks; the contents must not
    // depend on how the same samples were split up.
    rx::RingBuffer a = make(100), b = make(100);
    const std::vector<double> all = ramp(0, 250);
    a.write(all);
    for (std::size_t i = 0; i < all.size(); i += 7) {
        const std::size_t run = std::min<std::size_t>(7, all.size() - i);
        b.write(std::span<const double>(all.data() + i, run));
    }
    check::is_true(a.snapshot() == b.snapshot(),
                   "ring/blocks: block size does not change the contents");
    check::equal(a.total_written(), b.total_written(), "ring/blocks: same count");
}

void test_odd_block_sizes_across_many_wraps() {
    // Several full laps with a block size coprime to the capacity, so
    // every possible alignment of a write against the seam occurs.
    rx::RingBuffer rb = make(64);
    const std::vector<double> all = ramp(0, 1000);
    for (std::size_t i = 0; i < all.size(); i += 13) {
        const std::size_t run = std::min<std::size_t>(13, all.size() - i);
        rb.write(std::span<const double>(all.data() + i, run));
    }
    check::is_true(rb.snapshot() == ramp(1000 - 64, 64),
                   "ring/laps: the last capacity samples, in order");
}

void test_oversized_write() {
    // Larger than the buffer: only the last n survive, and the write
    // position must still line up for the *next* write.
    rx::RingBuffer rb = make(100);
    rb.write(ramp(0, 250));
    check::is_true(rb.snapshot() == ramp(150, 100),
                   "ring/oversized: keeps the most recent n");

    rb.write(ramp(250, 10));
    check::is_true(rb.snapshot() == ramp(160, 100),
                   "ring/oversized: the following write lands contiguously");
}

void test_empty_write_is_a_no_op() {
    rx::RingBuffer rb = make(100);
    rb.write(ramp(0, 5));
    rb.write({});
    check::equal(rb.total_written(), std::uint64_t{5}, "ring/empty: count unchanged");
    check::is_true(rb.snapshot() == ramp(0, 5), "ring/empty: contents unchanged");
}

void test_tail() {
    rx::RingBuffer rb = make(100);
    rb.write(ramp(0, 30));
    check::is_true(rb.tail(10) == ramp(20, 10), "ring/tail: before wrapping");
    check::equal(rb.tail(500).size(), std::size_t{30},
                 "ring/tail: clamped to what exists");
    check::equal(rb.tail(0).size(), std::size_t{0}, "ring/tail: zero is empty");

    rb.write(ramp(30, 90));  // now wrapped
    check::is_true(rb.tail(10) == ramp(110, 10), "ring/tail: after wrapping");
    // A tail spanning the seam is where an off-by-one would show.
    check::is_true(rb.tail(60) == ramp(60, 60), "ring/tail: spanning the wrap seam");
    check::is_true(rb.tail(100) == rb.snapshot(),
                   "ring/tail: a full-length tail is the snapshot");
}

void test_tail_of_an_empty_buffer() {
    rx::RingBuffer rb = make(100);
    check::equal(rb.tail(10).size(), std::size_t{0}, "ring/tail: nothing captured yet");
    check::equal(rb.snapshot().size(), std::size_t{0}, "ring/snapshot: likewise");
}

void test_clear_keeps_the_clock() {
    rx::RingBuffer rb = make(100);
    rb.write(ramp(0, 250));
    const std::uint64_t before = rb.total_written();
    rb.clear();

    check::equal(rb.total_written(), before,
                 "ring/clear: the sample clock stays monotonic");
    const std::vector<double> snap = rb.snapshot();
    check::is_true(std::all_of(snap.begin(), snap.end(), [](double v) { return v == 0.0; }),
                   "ring/clear: history reads as silence");

    // And writing afterwards still lands where the clock says it should.
    rb.write(ramp(1000, 10));
    check::is_true(rb.tail(10) == ramp(1000, 10), "ring/clear: writes resume cleanly");
}

}  // namespace

int main() {
    try {
        test_partial_fill();
        test_exact_fill_and_wrap();
        test_many_small_writes_match_one_big_one();
        test_odd_block_sizes_across_many_wraps();
        test_oversized_write();
        test_empty_write_is_a_no_op();
        test_tail();
        test_tail_of_an_empty_buffer();
        test_clear_keeps_the_clock();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("ring buffer");
}
