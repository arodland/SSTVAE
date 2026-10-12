// Scheduled sends, after native/gui/qrss_schedule.hpp with what only the
// desktop needs taken out: there is no encode to start minutes early, so
// no lead or chaining, and no rig to retune, so no frequency changes.
//
// An entry is one stored picture sent `count` times (0: until removed)
// from a quarter hour, then every `every_min` minutes (0: back to back,
// the next send starting the half hour after the last pass of the one
// before). A send is one pass per segment 0..mode, half an hour apart.
//
// Nothing here is state that changes as passes go by: which send is
// next is worked out from the clock, so a pass missed while powered off
// is simply in the past, and the schedule is written only when edited.
#pragma once

#include <stdint.h>

#include <string>
#include <vector>

namespace sched {

constexpr int64_t QUARTER_S = 900;
constexpr int64_t PASS_S = 1800;            // passes of a send are this far apart
constexpr int64_t HORIZON_S = 14 * 86400;   // how far ahead clashes are looked for

struct Entry {
    std::string id;
    uint32_t pid = 0;          // picture ID
    int mode = 0;              // 0..2 = A..C; the send is mode + 1 passes
    bool tiny = false;         // TINY test frame instead of FULL
    int64_t first_slot = 0;    // unix s, a quarter hour
    int count = 1;             // sends; 0 = until removed
    int every_min = 0;         // 0 = back to back
    bool enabled = true;
};

struct Pass {
    int entry = -1;            // index into the entries
    int send = 0;
    int segment = 0;
    int64_t qh = 0;            // the quarter hour it starts on
};

int64_t period_s(const Entry& e);
int64_t send_slot(const Entry& e, int k);
// Seconds from QH to the end of a pass's keying (frame end, no tail).
double pass_length_s(bool tiny);

// Empty when the entry can be scheduled; otherwise why not.
std::string problem(const Entry& e);

// The enabled entries' earliest pass starting at or after `not_before`;
// entry = -1 when there is none.
Pass next_pass(const std::vector<Entry>& entries, int64_t not_before);

// Empty when `e` overlaps no enabled entry in `others` (skipping one with
// e's own id) between now and HORIZON_S ahead; otherwise which and when.
std::string clash(const Entry& e, const std::vector<Entry>& others, int64_t now);

}  // namespace sched
