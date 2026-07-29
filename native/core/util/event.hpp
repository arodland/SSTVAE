// A manual-reset event: Python's `threading.Event`, minus the reset.
//
// Shared by the receiver (where it is the decode loop's stop flag) and
// the transmitter (where it is both the cancel flag and the PTT
// watchdog's "the transmission finished normally" signal). A condition
// variable rather than a polled bool, because both users need `set()`
// to interrupt a wait that is already in progress: a receiver must not
// take a poll interval to shut down, and a watchdog that cannot be
// stood down early would keep a thread alive past every transmission.

#ifndef SSTVAE_UTIL_EVENT_HPP
#define SSTVAE_UTIL_EVENT_HPP

#include <condition_variable>
#include <mutex>

namespace sstvae::util {

class Event {
public:
    void set();
    // Back to unset. Only for reuse of a one-shot flag between runs (the
    // transmitter clears its cancel flag when a new transmission
    // starts); there is no race-free way to clear one that a waiter is
    // currently blocked on, and no caller needs that.
    void clear();
    bool is_set() const;
    // Sleep up to `seconds`. Returns true if the event is set (either
    // already, or because it was set during the wait).
    bool wait(double seconds);

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    bool set_ = false;
};

}  // namespace sstvae::util

#endif
