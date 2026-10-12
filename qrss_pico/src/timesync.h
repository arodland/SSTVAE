// UTC from SNTP, kept as a line through the free-running microsecond
// counter. Our own client rather than the core's, because the slot start
// must be known to well under a symbol (30 ms): it keeps the reply's
// fraction of a second and corrects by half the round trip, and
// successive syncs estimate how fast the Pico's crystal runs.
#pragma once

#include <stdint.h>

struct Clock {
    uint64_t mono0 = 0;        // time_us_64() at the last sync
    int64_t unix0_us = 0;      // UTC there
    int32_t rate_ppb = 0;      // how much faster UTC runs than the counter

    int64_t unix_us(uint64_t mono) const;
    uint64_t mono_at(int64_t unix_us) const;
};

namespace timesync {

bool synced();
Clock clock();
int64_t now_us();                // UTC now; meaningless unless synced()
uint32_t age_s();                // since the last good sync
int32_t last_step_us();          // how far the last sync moved the clock
uint32_t last_rtt_us();
// Query `server` (best of 4); true on success.
bool sync(const char* server);

}  // namespace timesync
