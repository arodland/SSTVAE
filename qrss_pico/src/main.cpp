// QRSSTVAE waveform CE beacon: Pico W / Pico 2 W + Si5351A. See README.md.
//
// Core 0: WiFi, SNTP, the web server, the USB console and the scheduler.
// Core 1: the transmitter (tx.cpp) and nothing else.

#include <Arduino.h>
#include <LEAmDNS.h>
#include <LittleFS.h>
#include <WiFi.h>

#include <stdarg.h>

extern "C" {
#include "qrss_ce.h"
}

#include "app.h"
#include "settings.h"
#include "store.h"
#include "timesync.h"
#include "tx.h"

std::deque<String> log_lines;
static bool in_ap = false;
static int64_t refused_qh = -1;     // a pass already refused, so it is logged once

bool ap_mode() { return in_ap; }

void log_line(const char* fmt, ...)
{
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    char stamp[24] = "--:--:--";
    if (timesync::synced()) {
        const time_t t = (time_t)(timesync::now_us() / 1000000);
        strftime(stamp, sizeof stamp, "%m-%d %H:%M:%SZ", gmtime(&t));
    }
    String line = String(stamp) + " " + msg;
    Serial.println(line);
    log_lines.push_back(line);
    while (log_lines.size() > 40)
        log_lines.pop_front();
}

String not_ready()
{
    if (settings.base_hz <= 0)
        return "no carrier frequency set";
    if (!timesync::synced())
        return "clock never synced";
    if (timesync::age_s() > settings.sync_max_age_s)
        return "clock sync is " + String(timesync::age_s()) + " s old";
    return "";
}

static tx::Job job_base()
{
    tx::Job j;
    j.xtal_hz = settings.xtal_hz * (1.0 + settings.xtal_ppm * 1e-6);
    j.clk = settings.clk;
    j.drive_ma = settings.drive_ma;
    j.update_hz = settings.update_hz;
    j.tr_lead_ms = settings.tr_lead_ms;
    j.tr_tail_ms = settings.tr_tail_ms;
    j.carrier_hz = settings.base_hz;
    return j;
}

String start_carrier(uint32_t seconds)
{
    if (settings.base_hz <= 0)
        return "no carrier frequency set";
    tx::Job j = job_base();
    j.carrier_only = true;
    j.carrier_s = seconds > 120 ? 120 : seconds;
    if (!tx::submit(j))
        return "the transmitter is busy";
    log_line("test carrier, %u s", (unsigned)j.carrier_s);
    return "";
}

sched::Pass upcoming_pass()
{
    const int64_t now = timesync::synced() ? timesync::now_us() / 1000000 : 0;
    return sched::next_pass(store::entries, now + 1);
}

// Arm the next pass a few seconds before its keying starts. The gap between
// back-to-back FULL passes is about 6 s at the longest lead-in, so the job
// is handed over 3 s ahead: after the previous pass ends, before this one.
static void schedule_tick()
{
    if (!timesync::synced() || !tx::idle())
        return;
    const sched::Pass p = upcoming_pass();
    if (p.entry < 0 || p.qh == refused_qh)
        return;
    const int64_t key_us = p.qh * 1000000 + 1000000 - 242500 - (int64_t)(settings.lead_in_s * 1e6)
                           - (int64_t)settings.tr_lead_ms * 1000;
    const int64_t now = timesync::now_us();
    if (now < key_us - 3000000)
        return;
    const sched::Entry& e = store::entries[p.entry];
    refused_qh = p.qh;                     // whatever happens, this pass is decided
    String why = not_ready();
    if (!why.length() && now > key_us - 500000)
        why = "too late to start";
    tx::Job j = job_base();
    if (!why.length()) {
        j.len = store::load(e.pid, p.segment, tx::buffer(), tx::BUF_BYTES);
        if (!j.len)
            why = "picture file missing";
    }
    if (why.length()) {
        log_line("pass %08lx-%d at QH %lld refused: %s", (unsigned long)e.pid, p.segment,
                 (long long)p.qh, why.c_str());
        return;
    }
    j.qh = p.qh;
    j.tiny = e.tiny;
    j.lead_in_s = settings.lead_in_s;
    j.clock = timesync::clock();
    if (settings.spread_hz > 0)
        j.carrier_hz += ((double)rp2040.hwrand32() / 4294967296.0 * 2 - 1) * settings.spread_hz;
    if (tx::submit(j))
        log_line("pass %08lx-%d armed for QH %lld at %.1f Hz", (unsigned long)e.pid, p.segment,
                 (long long)p.qh, j.carrier_hz);
}

static void report_tx()
{
    static tx::State last = tx::IDLE;
    const tx::Status& s = tx::status();
    if (s.state == last)
        return;
    if (s.state == tx::DONE_ERROR)
        log_line("transmitter error %d (QRSS_E_* < 0; -100 no plan; -101 Si5351 not answering;"
                 " -102 PLL unlocked or no crystal); Si5351 status 0x%02x; last I2C failure:"
                 " code %d at register %d (2 address NACK, 3 data NACK, 4 other, 5 timeout)",
                 s.err, s.si_status & 0xFF, s.i2c_code, s.i2c_reg);
    else if (s.state == tx::IDLE && (last == tx::KEYED || last == tx::WAITING))
        log_line("pass done, %u of %u updates late", (unsigned)s.late, (unsigned)s.n);
    last = s.state;
}

static void sync_tick()
{
    static uint32_t next_ms = 0;
    if (in_ap || (int32_t)(millis() - next_ms) < 0)
        return;
    if (timesync::sync(settings.ntp.c_str())) {
        log_line("SNTP %s: stepped %ld us, delay %lu us, rate %+.3f ppm", settings.ntp.c_str(),
                 (long)timesync::last_step_us(), (unsigned long)timesync::last_rtt_us(),
                 timesync::clock().rate_ppb / 1000.0);
        next_ms = millis() + 600000;
    } else {
        next_ms = millis() + 60000;
    }
}

// Join the configured networks in order, by name. No scan first: a hidden
// SSID never appears in one, and WiFiMulti only tries what the scan lists.
// We poll for the link ourselves: arduino-pico's blocking begin() stops
// waiting once WiFi.mode() has been called (an ESP8266 compatibility
// hack), returning WL_IDLE_STATUS while the join is still under way.
static bool wifi_join()
{
    for (const WifiNet& n : settings.wifi) {
        n.pass.length() ? WiFi.beginNoBlock(n.ssid.c_str(), n.pass.c_str())
                        : WiFi.beginNoBlock(n.ssid.c_str());
        const uint32_t t0 = millis();
        while (!WiFi.connected() && millis() - t0 < 20000)
            delay(50);
        if (WiFi.status() == WL_CONNECTED)
            return true;
        log_line("WiFi %s: not joined (status %d)", n.ssid.c_str(), WiFi.status());
    }
    return false;
}

static void wifi_begin()
{
    WiFi.setHostname(settings.hostname.c_str());
    if (wifi_join()) {
        log_line("WiFi %s, http://%s/ or http://%s.local/", WiFi.SSID().c_str(),
                 WiFi.localIP().toString().c_str(), settings.hostname.c_str());
        MDNS.begin(settings.hostname.c_str());
        MDNS.addService("http", "tcp", 80);
        return;
    }
    // Bootstrap: an open access point serving the same (password-protected) page.
    in_ap = true;
    String id = rp2040.getChipID();
    String ssid = "qrss-" + id.substring(id.length() > 6 ? id.length() - 6 : 0);
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ssid);
    log_line("no WiFi network joined: access point %s, http://%s/", ssid.c_str(),
             WiFi.softAPIP().toString().c_str());
}

// --- USB console ---------------------------------------------------------------

static void console_line(String line)
{
    line.trim();
    if (line == "help" || line == "") {
        Serial.println("show | set {json} | wifi <ssid> [password] | scan | i2c | sync | carrier <s> | abort | bench | log | reboot");
    } else if (line == "show") {
        JsonDocument d;
        settings_to_json(d.to<JsonObject>());
        serializeJsonPretty(d, Serial);
        Serial.printf("\n%s, %s, IP %s\n", in_ap ? "access point" : "station",
                      not_ready().length() ? not_ready().c_str() : "ready",
                      (in_ap ? WiFi.softAPIP() : WiFi.localIP()).toString().c_str());
    } else if (line.startsWith("set ")) {
        JsonDocument d;
        bool reboot = false;
        if (deserializeJson(d, line.substring(4)) != DeserializationError::Ok) {
            Serial.println("not JSON");
            return;
        }
        String err = settings_merge(d.as<JsonVariantConst>(), &reboot);
        Serial.println(err.length() ? err : reboot ? "saved; reboot to apply" : "saved");
    } else if (line.startsWith("wifi ")) {
        String rest = line.substring(5);
        rest.trim();
        const int sp = rest.indexOf(' ');
        JsonDocument d;
        JsonObject n = d["wifi"].to<JsonArray>().add<JsonObject>();
        n["ssid"] = sp < 0 ? rest : rest.substring(0, sp);
        n["pass"] = sp < 0 ? "" : rest.substring(sp + 1);
        bool reboot;
        String err = settings_merge(d.as<JsonVariantConst>(), &reboot);
        Serial.println(err.length() ? err : "saved (replaces the network list); reboot to apply");
    } else if (line == "scan") {
        const int n = WiFi.scanNetworks();
        for (int i = 0; i < n; i++)
            Serial.printf("%4d dBm  ch %2d  %s\n", WiFi.RSSI(i), WiFi.channel(i), WiFi.SSID(i));
        Serial.printf("%d networks (2.4 GHz only; hidden ones do not appear)\n", n);
    } else if (line == "i2c") {
        Serial.println(tx::i2c_scan());
        Serial.println(tx::si_status());
    } else if (line == "sync") {
        Serial.println(timesync::sync(settings.ntp.c_str()) ? "synced" : "sync failed");
    } else if (line.startsWith("carrier ")) {
        String err = start_carrier((uint32_t)line.substring(8).toInt());
        Serial.println(err.length() ? err : "carrier on");
    } else if (line == "bench") {
        // What one Si5351 update costs (core 0, so roughly what core 1 sees).
        std::vector<PicInfo> pics = store::pictures();
        static qrss_ce_t ce;
        if (!tx::idle()) {
            Serial.println("the transmitter is busy");
            return;
        }
        uint8_t* buf = tx::buffer();
        const uint32_t len = pics.empty() ? 0 : store::load(pics[0].pid, pics[0].segment, buf, tx::BUF_BYTES);
        if (!len || qrss_ce_init(&ce, buf, len, 1990584) != QRSS_OK) {
            Serial.println("upload a beacon file first");
            return;
        }
        const uint32_t fu = settings.update_hz, n = 20000;
        int64_t reached = qrss_si5351_start(&ce, fu, 1);
        uint32_t worst = 0;
        const uint64_t t0 = time_us_64();
        for (uint32_t u = 0; u < n; u++) {
            const uint64_t a = time_us_64();
            qrss_si5351_next(&ce, u + 20 * fu, fu, 1, 385, &reached);   // from 20 s in: data
            const uint32_t d = (uint32_t)(time_us_64() - a);
            worst = d > worst ? d : worst;
        }
        Serial.printf("%u updates: mean %.1f us, worst %u us; the period at %u Hz is %u us\n",
                      (unsigned)n, (time_us_64() - t0) / (double)n, (unsigned)worst, (unsigned)fu,
                      (unsigned)(1000000 / fu));
    } else if (line == "abort") {
        tx::abort();
    } else if (line == "log") {
        for (const String& l : log_lines)
            Serial.println(l);
    } else if (line == "reboot") {
        rp2040.reboot();
    } else {
        Serial.println("unknown command; try help");
    }
}

static void console_tick()
{
    static String buf;
    while (Serial.available()) {
        const char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (buf.length())
                console_line(buf);
            buf = "";
        } else if (buf.length() < 512) {
            buf += c;
        }
    }
}

// --- the two cores ---------------------------------------------------------------

void setup()
{
    Serial.begin(115200);
    LittleFS.begin();
    settings_load();
    store::schedule_load();
    tx::start(tx::Hw{settings.sda, settings.scl, settings.si_addr, settings.tr_gpio,
                     settings.tr_active_high});
    wifi_begin();
    web_begin();
    if (settings.web_pass == Settings().web_pass)
        log_line("the web password is still the default: change it in Settings");
}

void loop()
{
    console_tick();
    static uint32_t retry_ms = 0;
    if (!in_ap && WiFi.status() != WL_CONNECTED && (int32_t)(millis() - retry_ms) >= 0) {
        // ponytail: blocks core 0 up to 20 s per network; core 1 keeps transmitting.
        log_line("WiFi lost, rejoining");
        wifi_join();
        retry_ms = millis() + 60000;
    }
    sync_tick();
    schedule_tick();
    report_tx();
    web_loop();
    MDNS.update();
}

void setup1() {}

void loop1()
{
    tx::run();
}
