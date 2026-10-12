#include "si5351_plan.h"

#include <math.h>

static const uint32_t C_MAX = 1048575;   // 2^20 - 1

bool si_plan(double xtal_hz, double f_out, SiPlan* p)
{
    if (!(f_out >= 8e3 && f_out <= 150e6) || !(xtal_hz > 0))
        return false;
    uint32_t r = 1, ms = 0;
    for (; r <= 128; r *= 2) {
        ms = (uint32_t)(900e6 / (f_out * r));
        ms -= ms % 2;
        if (ms <= 2048)
            break;
    }
    if (r > 128 || ms < 6)
        return false;
    const double ratio = f_out * ms * r / xtal_hz;    // f_vco / f_xtal
    if (ratio < 15.0 || ratio >= 90.0)
        return false;
    const double per_c = xtal_hz * 1000.0 / ((double)ms * r);   // milli-Hz * c per step
    uint32_t step = (uint32_t)ceil(per_c / C_MAX);
    if (step == 0)
        step = 1;
    uint32_t c = (uint32_t)llround(per_c / step);
    if (c > C_MAX)
        c = C_MAX;
    p->r_div = r;
    p->ms = ms;
    p->c = c;
    p->step_mhz = step;
    p->step_hz = xtal_hz / ((double)c * ms * r);
    p->base = llround(ratio * c);
    return true;
}

double si_freq(const SiPlan& p, double xtal_hz, int64_t steps)
{
    return xtal_hz * (double)(p.base + steps) / p.c / ((double)p.ms * p.r_div);
}

void si_pll_regs(uint32_t c, int64_t total, uint8_t out[8])
{
    const uint32_t a = (uint32_t)(total / c), b = (uint32_t)(total % c);
    const uint32_t f = (uint32_t)(((uint64_t)b * 128) / c);
    const uint32_t p1 = 128 * a + f - 512;
    const uint32_t p2 = (uint32_t)((uint64_t)b * 128 - (uint64_t)c * f);
    out[0] = (uint8_t)(c >> 8);
    out[1] = (uint8_t)c;
    out[2] = (uint8_t)((p1 >> 16) & 3);
    out[3] = (uint8_t)(p1 >> 8);
    out[4] = (uint8_t)p1;
    out[5] = (uint8_t)(((c >> 16) & 0xF) << 4 | ((p2 >> 16) & 0xF));
    out[6] = (uint8_t)(p2 >> 8);
    out[7] = (uint8_t)p2;
}

void si_ms_regs(uint32_t ms, uint32_t r_div, uint8_t out[8])
{
    uint32_t rbits = 0;
    while ((1u << rbits) < r_div)
        rbits++;
    const uint32_t p1 = 128 * ms - 512;              // P2 = 0, P3 = 1
    out[0] = 0;
    out[1] = 1;
    out[2] = (uint8_t)(rbits << 4 | ((p1 >> 16) & 3));
    out[3] = (uint8_t)(p1 >> 8);
    out[4] = (uint8_t)p1;
    out[5] = out[6] = out[7] = 0;
}

// (a * n) mod m for a < m < 2^63, by doubling: no 128-bit type on the M0+.
static uint64_t mulmod(uint64_t a, uint64_t n, uint64_t m)
{
    uint64_t r = 0;
    while (n) {
        if (n & 1) {
            r += a;
            if (r >= m)
                r -= m;
        }
        a += a;
        if (a >= m)
            a -= m;
        n >>= 1;
    }
    return r;
}

int64_t si_reached_add(int64_t reached, int64_t steps, uint32_t fu_num, uint32_t fu_den,
                       uint32_t step_mhz)
{
    // Same units and modulus as qrss_si5351_next.
    const uint64_t mod = (1000 * (uint64_t)fu_num) << 32;
    const uint64_t k = (((uint64_t)step_mhz * fu_den) << 32) % mod;
    const uint64_t d = mulmod(k, (uint64_t)(steps < 0 ? -steps : steps), mod);
    uint64_t r = (uint64_t)reached;
    if (steps < 0)
        r = r >= d ? r - d : r + (mod - d);
    else {
        r += d;
        if (r >= mod)
            r -= mod;
    }
    return (int64_t)r;
}
