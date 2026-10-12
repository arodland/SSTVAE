// Host tests: `pio test -e native`.
#include <unity.h>

#include <math.h>
#include <string.h>

#include <vector>

extern "C" {
#include "qrss_ce.h"
}
#include "schedule.h"
#include "si5351_plan.h"

using sched::Entry;

void setUp() {}
void tearDown() {}

// A FULL beacon file with zero latents and header bits and one keyed unit:
// valid as far as qrss_ce_init is concerned.
static std::vector<uint8_t> synthetic_bin()
{
    std::vector<uint8_t> f(366 + QRSS_MAX_LATENTS + 4, 0);
    memcpy(f.data(), "QRSB", 4);
    f[4] = 1;
    f[24] = QRSS_MAX_LATENTS & 0xFF;
    f[25] = (QRSS_MAX_LATENTS >> 8) & 0xFF;
    f[26] = (QRSS_MAX_LATENTS >> 16) & 0xFF;
    f[28] = QRSS_INT8_SCALE;
    f[32 + 1] = 0x80;                          // unit 8, the first of the call's room
    uint32_t crc = qrss_crc32(f.data(), (uint32_t)f.size() - 4);
    for (int i = 0; i < 4; i++)
        f[f.size() - 4 + i] = (uint8_t)(crc >> (8 * i));
    return f;
}

static void test_pass_lengths_match_qrss_ce()
{
    auto f = synthetic_bin();
    const char* names[2] = {"full", "tiny"};
    for (int i = 0; i < 2; i++) {
        qrss_ce_t s;
        TEST_ASSERT_EQUAL(QRSS_OK, qrss_ce_init_frame(&s, f.data(), (uint32_t)f.size(), 1,
                                                      qrss_frame_preset(names[i])));
        double end = 1.0 + (double)s.keyed_end_q22 / (1 << 22) * QRSS_T_NUM / QRSS_T_DEN;
        TEST_ASSERT_DOUBLE_WITHIN(1e-6, end, sched::pass_length_s(i == 1));
    }
}

static void test_reached_add_matches_next()
{
    auto f = synthetic_bin();
    qrss_ce_t s;
    TEST_ASSERT_EQUAL(QRSS_OK, qrss_ce_init(&s, f.data(), (uint32_t)f.size(), 1990584));
    const uint32_t fn = 990, fd = 1, step = 371;
    int64_t r0 = qrss_si5351_start(&s, fn, fd);
    for (uint32_t u = 0; u < 3000; u++) {
        int64_t r1 = r0;
        int32_t st = qrss_si5351_next(&s, u, fn, fd, step, &r1);
        TEST_ASSERT_EQUAL_INT64(r1, si_reached_add(r0, st, fn, fd, step));
        TEST_ASSERT_EQUAL_INT64(r0, si_reached_add(r1, -st, fn, fd, step));
        r0 = r1;
    }
}

static void test_plan_covers_the_bands()
{
    const double xtal = 25e6;
    const double bands[] = {137.5e3, 475.5e3, 1.8381e6, 3.5701e6, 5.3665e6, 7.0401e6,
                            10.1401e6, 14.0971e6, 18.1061e6, 21.0961e6, 24.9261e6,
                            28.1261e6, 50.2931e6, 144.4901e6};
    for (double f : bands) {
        SiPlan p;
        TEST_ASSERT_TRUE_MESSAGE(si_plan(xtal, f, &p), "no plan");
        TEST_ASSERT_TRUE(p.ms % 2 == 0 && p.ms >= 6 && p.ms <= 2048);
        TEST_ASSERT_TRUE(p.c >= 1 && p.c <= 1048575);
        double vco = f * p.ms * p.r_div;
        TEST_ASSERT_TRUE(vco >= 600e6 && vco <= 900e6);
        TEST_ASSERT_DOUBLE_WITHIN(p.step_hz / 2, f, si_freq(p, xtal, 0));
        TEST_ASSERT_DOUBLE_WITHIN(5e-7 * p.step_hz, p.step_mhz / 1000.0, p.step_hz);
        TEST_ASSERT_DOUBLE_WITHIN(1e-6, p.step_hz, si_freq(p, xtal, 1) - si_freq(p, xtal, 0));
        printf("%11.1f Hz: R %3u MS %4u c %7u step %.3f Hz\n", f, p.r_div, p.ms, p.c,
               p.step_hz);
    }
    SiPlan p;
    TEST_ASSERT_FALSE(si_plan(xtal, 200e6, &p));
    TEST_ASSERT_FALSE(si_plan(xtal, 5e3, &p));
}

static void test_pll_regs()
{
    // AN619's worked form: P1 = 128a + floor(128b/c) - 512, P2 = 128b - c floor(128b/c).
    uint8_t r[8];
    const uint32_t c = 1000000;
    const int64_t total = 32LL * c + 123457;        // a = 32, b = 123457
    si_pll_regs(c, total, r);
    uint32_t p1 = (uint32_t)(r[2] & 3) << 16 | r[3] << 8 | r[4];
    uint32_t p2 = (uint32_t)(r[5] & 0xF) << 16 | r[6] << 8 | r[7];
    uint32_t p3 = (uint32_t)(r[5] >> 4) << 16 | r[0] << 8 | r[1];
    TEST_ASSERT_EQUAL_UINT32(128 * 32 + 15 - 512, p1);   // floor(128*123457/1e6) = 15
    TEST_ASSERT_EQUAL_UINT32(128 * 123457 - 15 * c, p2);
    TEST_ASSERT_EQUAL_UINT32(c, p3);
    si_ms_regs(36, 4, r);
    TEST_ASSERT_EQUAL_HEX8(2 << 4, r[2] & 0xF0);
    TEST_ASSERT_EQUAL_UINT32(128 * 36 - 512, (uint32_t)r[3] << 8 | r[4]);
}

static Entry entry(const char* id, int mode, int64_t first, int count, int every)
{
    Entry e;
    e.id = id;
    e.mode = mode;
    e.first_slot = first;
    e.count = count;
    e.every_min = every;
    return e;
}

static void test_schedule()
{
    const int64_t t = 1800000000 - 1800000000 % 900;
    Entry b = entry("b", 1, t, 2, 0);                  // 2 x mode B, back to back
    TEST_ASSERT_EQUAL_STRING("", sched::problem(b).c_str());
    std::vector<Entry> v{b};
    int64_t expect[4] = {t, t + 1800, t + 3600, t + 5400};
    int seg[4] = {0, 1, 0, 1};
    int64_t now = t - 100;
    for (int i = 0; i < 4; i++) {
        sched::Pass p = sched::next_pass(v, now);
        TEST_ASSERT_EQUAL(0, p.entry);
        TEST_ASSERT_EQUAL_INT64(expect[i], p.qh);
        TEST_ASSERT_EQUAL(seg[i], p.segment);
        now = p.qh + 1;
    }
    TEST_ASSERT_EQUAL(-1, sched::next_pass(v, now).entry);   // count reached

    // A pass missed while powered off is just in the past.
    TEST_ASSERT_EQUAL_INT64(t + 1800, sched::next_pass(v, t + 60).qh);

    Entry hourly = entry("h", 0, t + 900, 0, 60);
    TEST_ASSERT_EQUAL_STRING("", sched::problem(hourly).c_str());
    TEST_ASSERT_TRUE(sched::clash(hourly, v, t) != "");      // inside b's first send
    hourly.first_slot = t + 7200;
    TEST_ASSERT_EQUAL_STRING("", sched::clash(hourly, v, t).c_str());
    TEST_ASSERT_EQUAL_INT64(t + 7200 + 3 * 3600, sched::next_pass({hourly}, t + 3 * 3600 + 7200 - 5).qh);

    TEST_ASSERT_TRUE(sched::problem(entry("x", 1, t, 1, 45)) != "");   // shorter than a send
    TEST_ASSERT_TRUE(sched::problem(entry("x", 0, t + 60, 1, 0)) != "");
    Entry tiny = entry("t", 0, t, 0, 15);
    tiny.tiny = true;
    TEST_ASSERT_EQUAL_STRING("", sched::problem(tiny).c_str());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_pass_lengths_match_qrss_ce);
    RUN_TEST(test_reached_add_matches_next);
    RUN_TEST(test_plan_covers_the_bands);
    RUN_TEST(test_pll_regs);
    RUN_TEST(test_schedule);
    return UNITY_END();
}
