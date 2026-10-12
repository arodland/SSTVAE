// The transmitter: core 1 runs one job at a time, a pass or a test
// carrier, and nothing else, so WiFi and the web server on core 0 never
// sit between two Si5351 updates.
//
// A pass keys from t0 - 8T (t0 = QH + 1 s), after `lead_in_s` of plain
// carrier and the T/R lead, and updates the PLL numerator `update_hz`
// times a second from qrss_si5351_next. An update that comes too late
// (a flash write on core 0 pauses this core) is skipped and the error
// feedback told the chip held the old frequency, so the phase recovers.
#pragma once

#include <Arduino.h>
#include <stdint.h>

#include "timesync.h"

namespace tx {

constexpr uint32_t BUF_BYTES = 366 + 50600 + 4;   // one full beacon file

enum State : uint8_t { IDLE, WAITING, KEYED, CARRIER, DONE_ERROR };

struct Hw {                     // fixed from boot
    int sda, scl, addr;
    int tr_gpio;
    bool tr_active_high;
};

struct Job {
    bool carrier_only = false;  // test carrier for `carrier_s` seconds, now
    uint32_t carrier_s = 0;
    uint32_t len = 0;           // beacon file bytes, in buffer()
    int64_t qh = 0;             // unix s of the slot's quarter hour
    bool tiny = false;
    double carrier_hz = 0, xtal_hz = 25e6;   // xtal_hz calibrated
    int clk = 0, drive_ma = 8;
    uint32_t update_hz = 990;
    double lead_in_s = 0;
    uint32_t tr_lead_ms = 50, tr_tail_ms = 50;
    Clock clock;                // frozen for the pass
};

struct Status {
    volatile State state = IDLE;
    volatile int64_t qh = 0;
    volatile uint32_t u = 0, n = 0, late = 0;
    volatile int err = 0;       // QRSS_E_*, -100 no plan, -101 Si5351 not answering,
                                // -102 PLL not locked / no crystal
    volatile double carrier_hz = 0, step_hz = 0;
    volatile int i2c_code = 0;  // Wire endTransmission() of the last failed write
    volatile int i2c_reg = -1;  // and the register it was writing
    volatile int si_status = -1; // Si5351 register 0 after the last setup
};

// Core 0, once: start core 1's loop with this hardware.
void start(const Hw& hw);
// Core 0: the buffer a job reads, only to be written while idle().
uint8_t* buffer();
bool idle();
// Core 0: hand over a job (false if busy).
bool submit(const Job& job);
void abort();
const Status& status();

// Core 0, while idle(): the I2C addresses that answer, as "0x60 ..."; and
// which bus and pins that is.
String i2c_scan();
// Core 0, while idle(): register 0 decoded.
String si_status();

// Core 1's loop body.
void run();

}  // namespace tx
