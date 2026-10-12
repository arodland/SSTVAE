#include "timesync.h"

#include <WiFi.h>
#include <WiFiUdp.h>

int64_t Clock::unix_us(uint64_t mono) const
{
    const int64_t d = (int64_t)(mono - mono0);
    return unix0_us + d + d / 1000 * rate_ppb / 1000000;
}

uint64_t Clock::mono_at(int64_t t) const
{
    const int64_t d = t - unix0_us;
    return mono0 + (uint64_t)(d - d / 1000 * rate_ppb / 1000000);
}

namespace timesync {

static Clock clk;
static bool have = false;
static int32_t step_us = 0;
static uint32_t rtt_us = 0;

bool synced() { return have; }
Clock clock() { return clk; }
int64_t now_us() { return clk.unix_us(time_us_64()); }
uint32_t age_s() { return have ? (uint32_t)((time_us_64() - clk.mono0) / 1000000) : UINT32_MAX; }
int32_t last_step_us() { return step_us; }
uint32_t last_rtt_us() { return rtt_us; }

static int64_t ntp_us(const uint8_t* p)    // NTP timestamp -> unix microseconds
{
    const uint32_t s = (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3];
    const uint32_t f = (uint32_t)p[4] << 24 | p[5] << 16 | p[6] << 8 | p[7];
    return ((int64_t)s - 2208988800LL) * 1000000 + (int64_t)(((uint64_t)f * 1000000) >> 32);
}

bool sync(const char* server)
{
    IPAddress ip;
    if (WiFi.status() != WL_CONNECTED || !WiFi.hostByName(server, ip))
        return false;
    WiFiUDP udp;
    udp.begin(12300);
    bool got = false;
    uint64_t best_mono = 0;
    int64_t best_unix = 0;
    uint32_t best_delay = UINT32_MAX;
    for (int attempt = 0; attempt < 4; attempt++) {
        uint8_t pkt[48] = {0x23};                      // LI 0, version 4, client
        while (udp.parsePacket() > 0)
            udp.flush();
        const uint64_t t1 = time_us_64();
        udp.beginPacket(ip, 123);
        udp.write(pkt, sizeof pkt);
        udp.endPacket();
        uint64_t t4 = 0;
        while (time_us_64() - t1 < 1000000) {
            if (udp.parsePacket() >= 48) {
                t4 = time_us_64();
                udp.read(pkt, sizeof pkt);
                break;
            }
            delay(1);
        }
        if (!t4 || (pkt[0] & 7) != 4 || pkt[1] == 0 || pkt[1] > 15)
            continue;
        const int64_t t2 = ntp_us(pkt + 32), t3 = ntp_us(pkt + 40);
        const int64_t server_hold = t3 - t2;
        const int64_t rtt = (int64_t)(t4 - t1);
        if (server_hold < 0 || server_hold > rtt)
            continue;
        const uint32_t delay_us = (uint32_t)(rtt - server_hold);
        if (delay_us < best_delay) {
            best_delay = delay_us;
            best_mono = t4;
            best_unix = t3 + delay_us / 2;
        }
        got = true;
    }
    udp.stop();
    if (!got || best_delay > 500000)
        return false;
    Clock c = clk;
    if (have) {
        step_us = (int32_t)(best_unix - clk.unix_us(best_mono));
        const int64_t span = (int64_t)(best_mono - clk.mono0);
        // ponytail: one-step frequency estimate; a loop filter if syncs prove noisy.
        if (span > 600LL * 1000000) {
            int64_t r = c.rate_ppb + (int64_t)step_us * 1000 / (span / 1000000);
            c.rate_ppb = (int32_t)(r > 500000 ? 500000 : r < -500000 ? -500000 : r);
        }
    }
    c.mono0 = best_mono;
    c.unix0_us = best_unix;
    clk = c;
    rtt_us = best_delay;
    have = true;
    return true;
}

}  // namespace timesync
