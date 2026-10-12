#include "schedule.h"

#include <stdio.h>

namespace sched {

int64_t period_s(const Entry& e)
{
    return e.every_min > 0 ? (int64_t)e.every_min * 60 : (e.mode + 1) * PASS_S;
}

int64_t send_slot(const Entry& e, int k)
{
    return e.first_slot + k * period_s(e);
}

double pass_length_s(bool tiny)
{
    // t0 = QH + 1 s, plus frame.PRESETS' keyed_end_pos * T; the native
    // test checks these against qrss_ce's keyed end.
    return 1.0 + (tiny ? 53.35 : 1782.66296875);
}

static double send_end(const Entry& e, int k)
{
    return (double)send_slot(e, k) + e.mode * PASS_S + pass_length_s(e.tiny);
}

std::string problem(const Entry& e)
{
    if (e.mode < 0 || e.mode > 2)
        return "mode must be A, B or C";
    if (e.first_slot % QUARTER_S)
        return "the first send must start on a quarter hour";
    if (e.count < 0)
        return "count must be 0 (until removed) or more";
    if (e.every_min < 0 || e.every_min % 15)
        return "the interval must be a whole number of quarter hours";
    if (e.every_min > 0 && e.every_min * 60.0 < e.mode * PASS_S + pass_length_s(e.tiny))
        return "the interval is shorter than one send";
    return "";
}

// The first send whose last pass starts at or after t.
static int first_send_from(const Entry& e, int64_t t)
{
    const int64_t from = t - e.first_slot - e.mode * PASS_S;
    if (from <= 0)
        return 0;
    const int64_t p = period_s(e);
    return (int)((from + p - 1) / p);
}

Pass next_pass(const std::vector<Entry>& entries, int64_t not_before)
{
    Pass best;
    for (size_t i = 0; i < entries.size(); i++) {
        const Entry& e = entries[i];
        if (!e.enabled || !problem(e).empty())
            continue;
        const int k = first_send_from(e, not_before);
        if (e.count > 0 && k >= e.count)
            continue;
        for (int s = 0; s <= e.mode; s++) {
            const int64_t qh = send_slot(e, k) + s * PASS_S;
            if (qh < not_before)
                continue;
            if (best.entry < 0 || qh < best.qh)
                best = Pass{(int)i, k, s, qh};
            break;
        }
    }
    return best;
}

std::string clash(const Entry& e, const std::vector<Entry>& others, int64_t now)
{
    const int64_t until = now + HORIZON_S;
    // ponytail: pairwise over every send in the horizon, a few hundred per
    // entry; fine for the handful of entries a beacon holds.
    for (int k = first_send_from(e, now); e.count == 0 || k < e.count; k++) {
        if (send_slot(e, k) > until)
            break;
        const double a0 = (double)send_slot(e, k), a1 = send_end(e, k);
        for (const Entry& o : others) {
            if (o.id == e.id || !o.enabled)
                continue;
            for (int j = first_send_from(o, now); o.count == 0 || j < o.count; j++) {
                if (send_slot(o, j) > until)
                    break;
                if (a0 < send_end(o, j) && (double)send_slot(o, j) < a1) {
                    char buf[96];
                    snprintf(buf, sizeof buf, "overlaps entry %s at unix %lld", o.id.c_str(),
                             (long long)send_slot(e, k));
                    return buf;
                }
            }
        }
    }
    return "";
}

}  // namespace sched
