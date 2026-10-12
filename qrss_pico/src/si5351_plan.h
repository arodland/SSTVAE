// Si5351 frequency plan and register arithmetic. No I2C here, so the
// host tests can check it.
//
// One output, integer MultiSynth M (even, 6..2048) and R divider, and
// PLL A at f_vco = f_xtal (a + b/c) as close under 900 MHz as M allows.
// The modulation steps the PLL numerator b by whole units, so one step
// is f_xtal / (c M R). c is chosen so that step is a whole number of
// milli-Hz (what qrss_si5351_next counts in), as small as c <= 2^20 - 1
// allows. That step grows with frequency (25 MHz crystal): 0.05 Hz at
// 1.8 MHz, 0.39 Hz at 14 MHz, 0.80 Hz at 28 MHz, 1.5 Hz at 50 MHz and
// 4.0 Hz at 144 MHz. The design's M6 measurement used 0.4 Hz.
#pragma once

#include <stdint.h>

struct SiPlan {
    uint32_t r_div = 1;       // 1, 2, 4 .. 128
    uint32_t ms = 0;          // MultiSynth integer divider, even
    uint32_t c = 0;           // PLL denominator
    uint32_t step_mhz = 0;    // one step of b at the output, milli-Hz
    int64_t base = 0;         // a*c + b at the carrier
    double step_hz = 0.0;     // the exact step (step_mhz rounds it)
};

// false when f_out is outside what this plan covers (8 kHz .. 150 MHz)
// or the crystal puts the PLL out of range. xtal_hz is the calibrated
// crystal frequency.
bool si_plan(double xtal_hz, double f_out, SiPlan* p);

// The output frequency at base + steps.
double si_freq(const SiPlan& p, double xtal_hz, int64_t steps);

// PLL A registers 26..33 for a*c + b = total.
void si_pll_regs(uint32_t c, int64_t total, uint8_t out[8]);

// MultiSynth registers (42 + 8n ..) for an integer divider and R.
void si_ms_regs(uint32_t ms, uint32_t r_div, uint8_t out[8]);

// qrss_si5351_next's `reached`, advanced by `steps` more steps than it
// was: what keeps the error feedback honest when an update is missed
// and the chip held the previous frequency through an interval.
int64_t si_reached_add(int64_t reached, int64_t steps, uint32_t fu_num, uint32_t fu_den,
                       uint32_t step_mhz);
