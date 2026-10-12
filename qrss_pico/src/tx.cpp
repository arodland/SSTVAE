#include "tx.h"

#include <Arduino.h>
#include <Wire.h>

#include <string.h>

extern "C" {
#include "qrss_ce.h"
}
#include "si5351_plan.h"

namespace tx {

static uint8_t buf[BUF_BYTES];
static Hw hw;
static volatile bool hw_given = false, pending = false, abort_req = false;
static bool hw_ready = false;
static Job job;
static Status st;

static TwoWire* bus = nullptr;
static SiPlan plan;
static uint8_t pll_cur[8];
static int oe_clk = 0;

uint8_t* buffer() { return buf; }
const Status& status() { return st; }
bool idle() { return !pending && (st.state == IDLE || st.state == DONE_ERROR); }
void abort() { abort_req = true; }

void start(const Hw& h)
{
    hw = h;
    __dmb();
    hw_given = true;
}

bool submit(const Job& j)
{
    if (!hw_given || !idle())
        return false;
    job = j;
    abort_req = false;
    __dmb();
    pending = true;
    return true;
}

// --- Si5351 ------------------------------------------------------------------

static bool wr(uint8_t reg, const uint8_t* d, size_t n)
{
    bus->beginTransmission(hw.addr);
    bus->write(reg);
    bus->write(d, n);
    const int code = bus->endTransmission();
    if (code) {
        st.i2c_code = code;
        st.i2c_reg = reg;
    }
    return code == 0;
}

static bool wr1(uint8_t reg, uint8_t v) { return wr(reg, &v, 1); }

// Register 0, or -1 when the chip does not answer. Bits: 7 SYS_INIT,
// 6 LOL_B, 5 LOL_A (PLL not locked), 4 LOS_CLKIN, 3 LOS_XTAL (no crystal
// signal), 1:0 REVID. PLL B is unused, so LOL_B is expected.
static int rd_status()
{
    bus->beginTransmission(hw.addr);
    bus->write((uint8_t)0);
    if (bus->endTransmission(false) != 0 || bus->requestFrom(hw.addr, 1) != 1)
        return -1;
    return bus->read();
}

static void oe(bool on) { wr1(3, on ? (uint8_t)~(1u << oe_clk) : 0xFF); }

static void tr(bool on)
{
    if (hw.tr_gpio >= 0)
        digitalWrite(hw.tr_gpio, on == hw.tr_active_high ? HIGH : LOW);
}

// Carrier + steps * plan.step_hz; writes only the PLL registers that change.
// ponytail: the bytes of one update land one by one, so the PLL may see a
// mixed P1/P2 for a few microseconds; measure before doing anything cleverer.
static bool set_steps(int64_t steps)
{
    uint8_t r[8];
    si_pll_regs(plan.c, plan.base + steps, r);
    int lo = 0, hi = 7;
    while (lo < 8 && r[lo] == pll_cur[lo])
        lo++;
    if (lo == 8)
        return true;
    while (r[hi] == pll_cur[hi])
        hi--;
    memcpy(pll_cur, r, 8);
    return wr((uint8_t)(26 + lo), r + lo, (size_t)(hi - lo + 1));
}

// Output `clk` from MultiSynth `clk` on PLL A at the carrier, output off.
// 0, or -100 no plan for that frequency, -101 the chip did not answer,
// -102 PLL A did not lock or the crystal has no signal (status in st.si_status).
static int synth_setup(const Job& j)
{
    if (!si_plan(j.xtal_hz, j.carrier_hz, &plan))
        return -100;
    oe_clk = j.clk;
    uint8_t ms[8];
    si_ms_regs(plan.ms, plan.r_div, ms);
    const uint8_t drive = (uint8_t)(j.drive_ma / 2 - 1);
    si_pll_regs(plan.c, plan.base, pll_cur);
    bool ok = wr1(3, 0xFF)
              && wr(26, pll_cur, 8)
              && wr((uint8_t)(42 + 8 * j.clk), ms, 8)
              // integer MultiSynth, PLL A, MultiSynth source, powered up
              && wr1((uint8_t)(16 + j.clk), (uint8_t)(0x40 | 0x0C | drive))
              && wr1(177, 0x20);                       // reset PLL A
    st.carrier_hz = si_freq(plan, j.xtal_hz, 0);
    st.step_hz = plan.step_hz;
    if (!ok)
        return -101;
    delay(20);                                         // lock takes well under 10 ms
    wr1(1, 0);                                         // clear the sticky bits, then look
    const int s0 = rd_status();
    st.si_status = s0;
    if (s0 < 0)
        return -101;
    return (s0 & 0xA8) ? -102 : 0;
}

static void hw_init()
{
    bus = ((hw.sda >> 1) & 1) ? &Wire1 : &Wire;
    bus->setSDA(hw.sda);
    bus->setSCL(hw.scl);
    bus->begin();
    bus->setClock(400000);
    if (hw.tr_gpio >= 0) {
        pinMode(hw.tr_gpio, OUTPUT);
        tr(false);
    }
    wr1(3, 0xFF);                                      // every output off
    for (uint8_t r = 16; r <= 23; r++)
        wr1(r, 0x80);                                  // and powered down
    wr1(149, 0);                                       // no spread spectrum
    wr1(183, 0xD2);   // crystal load 10 pF; bits 5:0 must be 010010 (AN619)
    hw_ready = true;
}

String si_status()
{
    if (!bus || !idle())
        return "the transmitter is busy or not started";
    const int s = rd_status();
    if (s < 0)
        return "Si5351 not answering";
    char out[96];
    snprintf(out, sizeof out, "Si5351 status 0x%02x:%s%s%s%s", s, (s & 0x80) ? " SYS_INIT" : "",
             (s & 0x20) ? " PLL_A_unlocked" : "", (s & 0x08) ? " no_crystal" : "",
             (s & 0xA8) ? "" : " ok");
    return out;
}

String i2c_scan()
{
    if (!bus || !idle())
        return "the transmitter is busy or not started";
    // ponytail: core 0 borrows core 1's bus, safe only because idle() means
    // core 1 is not touching it; a job submitted mid-scan could collide.
    String out = String(bus == &Wire1 ? "I2C1" : "I2C0") + " SDA " + hw.sda + " SCL " + hw.scl + ":";
    for (int a = 0x08; a <= 0x77; a++) {
        bus->beginTransmission(a);
        if (bus->endTransmission() == 0)
            out += " 0x" + String(a, HEX);
    }
    return out;
}

// --- jobs --------------------------------------------------------------------

static bool wait_until(uint64_t mono)
{
    while (time_us_64() < mono)
        if (abort_req)
            return false;
    return true;
}

static void finish(bool keyed_off_first, uint32_t tail_ms)
{
    if (keyed_off_first)
        oe(false);
    set_steps(0);
    const uint64_t t = time_us_64() + (uint64_t)tail_ms * 1000;
    while (time_us_64() < t) {
    }
    tr(false);
}

static void run_carrier(const Job& j)
{
    st.state = CARRIER;
    tr(true);
    delay(j.tr_lead_ms);
    oe(true);
    wait_until(time_us_64() + (uint64_t)j.carrier_s * 1000000);
    finish(true, j.tr_tail_ms);
    st.state = IDLE;
}

static void run_pass(const Job& j)
{
    static qrss_ce_t ce;
    const int rc = qrss_ce_init_frame(&ce, buf, j.len, (uint64_t)(j.qh / 900),
                                      qrss_frame_preset(j.tiny ? "tiny" : "full"));
    if (rc != QRSS_OK) {
        st.err = rc;
        st.state = DONE_ERROR;
        return;
    }
    const uint32_t fu = j.update_hz, step = plan.step_mhz;
    const int64_t Q22 = (int64_t)1 << 22, DEN = Q22 * QRSS_T_DEN;
    const uint32_t n = (uint32_t)(((ce.keyed_end_q22 + QRSS_SPAN * Q22) * QRSS_T_NUM * fu + DEN - 1) / DEN);
    const int64_t t_key = j.qh * 1000000 + 1000000 - 242500;      // t0 - 8T, 8T = 0.2425 s
    auto t_u = [&](uint32_t u) {
        return j.clock.mono_at(t_key + (int64_t)((uint64_t)u * 1000000 / fu));
    };
    st.n = n;
    st.u = 0;
    st.late = 0;
    st.state = WAITING;

    const int64_t lead_us = (int64_t)(j.lead_in_s * 1e6);
    bool on = false;
    if (!wait_until(j.clock.mono_at(t_key - lead_us - (int64_t)j.tr_lead_ms * 1000)))
        goto out;
    tr(true);
    if (lead_us > 0) {
        if (!wait_until(j.clock.mono_at(t_key - lead_us)))
            goto out;
        oe(true);
        on = true;
        st.state = KEYED;
    }
    {
        int64_t reached = qrss_si5351_start(&ce, fu, 1);
        int32_t f = qrss_si5351_next(&ce, 0, fu, 1, step, &reached);
        int32_t held = 0;
        for (uint32_t u = 0; u < n; u++) {
            st.u = u;
            if (time_us_64() >= t_u(u + 1)) {
                // Missed the whole interval: the chip held `held` through it.
                reached = si_reached_add(reached, (int64_t)held - f, fu, 1, step);
                st.late = st.late + 1;
            } else {
                if (!wait_until(t_u(u)))
                    goto out;
                if (!set_steps(f))
                    st.err = -101;
                held = f;
                const bool k = qrss_ce_keyed(&ce, u, fu, 1);
                if (k != on) {
                    oe(k);
                    on = k;
                }
                st.state = KEYED;
            }
            if (u + 1 < n)
                f = qrss_si5351_next(&ce, u + 1, fu, 1, step, &reached);
        }
        wait_until(t_u(n));
    }
out:
    finish(true, j.tr_tail_ms);
    st.state = st.err ? DONE_ERROR : IDLE;
}

void run()
{
    if (!hw_ready) {
        if (hw_given)
            hw_init();
        return;
    }
    if (!pending)
        return;
    const Job j = job;
    __dmb();
    st.qh = j.qh;
    st.i2c_code = 0;
    st.i2c_reg = -1;
    st.err = synth_setup(j);
    if (st.err) {
        st.state = DONE_ERROR;
        pending = false;
        return;
    }
    st.state = WAITING;     // before pending clears, so idle() never sees a gap
    pending = false;
    if (j.carrier_only)
        run_carrier(j);
    else
        run_pass(j);
}

}  // namespace tx
