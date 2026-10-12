// Station settings, kept in /settings.json on the flash file system.
// The web page and the USB console change them through one call,
// `settings_merge`, so both validate the same way.
#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>

#include <vector>

struct WifiNet {
    String ssid, pass;
};

struct Settings {
    std::vector<WifiNet> wifi;          // tried in order of signal, up to 4
    String hostname = "qrss-beacon";    // also <hostname>.local
    String ntp = "pool.ntp.org";
    uint32_t sync_max_age_s = 3600;     // no transmitting on an older sync
    String web_user = "admin", web_pass = "qrsstvae";
    int sda = 4, scl = 5;               // the I2C bus follows from the pins
    int si_addr = 0x60;
    double xtal_hz = 25e6, xtal_ppm = 0.0;
    int clk = 0, drive_ma = 8;          // Si5351 output and its drive
    double base_hz = 0.0;               // RF carrier; 0 = not set, no transmitting
    double spread_hz = 0.0;             // each pass picks base +- spread at random
    int tr_gpio = -1;                   // -1: none
    bool tr_active_high = true;
    uint32_t tr_lead_ms = 50, tr_tail_ms = 50;
    double lead_in_s = 0.0;             // plain carrier before keying, 0..10
    uint32_t update_hz = 990;           // Si5351 updates per second
};

extern Settings settings;

// Load /settings.json (defaults for anything missing).
void settings_load();
// Apply the keys present in `in` to a copy, validate, then save and adopt
// it. Empty on success, else what is wrong (nothing changed). `reboot` is
// set when a changed key only takes effect after a restart.
String settings_merge(JsonVariantConst in, bool* reboot);
// The settings as JSON; passwords are left out (has_pass says whether set).
void settings_to_json(JsonObject out);
