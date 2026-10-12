#include "settings.h"

#include <LittleFS.h>

Settings settings;

static const char* PATH = "/settings.json";

// RP2040/RP2350 pin mux: GPIO n is I2C(n/2 % 2), SDA when n is even.
// 23-25 and 29 belong to the radio module on a Pico W / Pico 2 W.
static bool pin_free(int p)
{
    return p >= 0 && p <= 28 && p != 23 && p != 24 && p != 25;
}

static String check(const Settings& s)
{
    if (s.wifi.size() > 4)
        return "at most 4 WiFi networks";
    if (!pin_free(s.sda) || !pin_free(s.scl) || s.sda % 2 == 1 || s.scl % 2 == 0
        || ((s.sda >> 1) & 1) != ((s.scl >> 1) & 1))
        return "SDA and SCL must be the SDA and SCL pins of one I2C block (e.g. 4/5, 2/3, 26/27)";
    if (s.si_addr < 0x08 || s.si_addr > 0x77)
        return "bad I2C address";
    if (!(s.xtal_hz >= 10e6 && s.xtal_hz <= 40e6) || fabs(s.xtal_ppm) > 500)
        return "crystal must be 10-40 MHz and within +-500 ppm";
    if (s.clk < 0 || s.clk > 2)
        return "clk must be 0, 1 or 2";
    if (s.drive_ma != 2 && s.drive_ma != 4 && s.drive_ma != 6 && s.drive_ma != 8)
        return "drive must be 2, 4, 6 or 8 mA";
    if (s.base_hz != 0 && !(s.base_hz >= 8e3 && s.base_hz <= 150e6))
        return "carrier must be 8 kHz to 150 MHz";
    if (!(s.spread_hz >= 0 && s.spread_hz <= 1000))
        return "spread must be 0-1000 Hz";
    if (s.tr_gpio != -1 && (!pin_free(s.tr_gpio) || s.tr_gpio == s.sda || s.tr_gpio == s.scl))
        return "T/R GPIO must be a free pin, or -1";
    if (s.tr_lead_ms > 1000 || s.tr_tail_ms > 1000)
        return "T/R lead and tail are at most 1000 ms";
    if (!(s.lead_in_s >= 0 && s.lead_in_s <= 10))
        return "lead-in must be 0-10 s";
    if (s.update_hz < 50 || s.update_hz > 2000)
        return "update rate must be 50-2000 Hz";
    if (s.sync_max_age_s < 60)
        return "sync max age must be at least 60 s";
    if (s.web_pass.length() < 4)
        return "web password must be at least 4 characters";
    return "";
}

template <typename T>
static void take(JsonVariantConst in, const char* key, T& v)
{
    if (!in[key].isNull())
        v = in[key].as<T>();
}

static void apply(JsonVariantConst in, Settings& s)
{
    if (in["wifi"].is<JsonArrayConst>()) {
        std::vector<WifiNet> nets;
        for (JsonVariantConst n : in["wifi"].as<JsonArrayConst>()) {
            WifiNet w{n["ssid"].as<String>(), n["pass"].as<String>()};
            // An omitted password keeps the one already stored for that SSID.
            if (n["pass"].isNull())
                for (const WifiNet& old : s.wifi)
                    if (old.ssid == w.ssid)
                        w.pass = old.pass;
            if (w.ssid.length())
                nets.push_back(w);
        }
        s.wifi = nets;
    }
    take(in, "hostname", s.hostname);
    take(in, "ntp", s.ntp);
    take(in, "sync_max_age_s", s.sync_max_age_s);
    take(in, "web_user", s.web_user);
    take(in, "web_pass", s.web_pass);
    take(in, "sda", s.sda);
    take(in, "scl", s.scl);
    take(in, "si_addr", s.si_addr);
    take(in, "xtal_hz", s.xtal_hz);
    take(in, "xtal_ppm", s.xtal_ppm);
    take(in, "clk", s.clk);
    take(in, "drive_ma", s.drive_ma);
    take(in, "base_hz", s.base_hz);
    take(in, "spread_hz", s.spread_hz);
    take(in, "tr_gpio", s.tr_gpio);
    take(in, "tr_active_high", s.tr_active_high);
    take(in, "tr_lead_ms", s.tr_lead_ms);
    take(in, "tr_tail_ms", s.tr_tail_ms);
    take(in, "lead_in_s", s.lead_in_s);
    take(in, "update_hz", s.update_hz);
}

static void write_all(const Settings& s, JsonObject o, bool secrets)
{
    JsonArray w = o["wifi"].to<JsonArray>();
    for (const WifiNet& n : s.wifi) {
        JsonObject j = w.add<JsonObject>();
        j["ssid"] = n.ssid;
        if (secrets)
            j["pass"] = n.pass;
        else
            j["has_pass"] = n.pass.length() > 0;
    }
    o["hostname"] = s.hostname;
    o["ntp"] = s.ntp;
    o["sync_max_age_s"] = s.sync_max_age_s;
    o["web_user"] = s.web_user;
    if (secrets)
        o["web_pass"] = s.web_pass;
    o["sda"] = s.sda;
    o["scl"] = s.scl;
    o["si_addr"] = s.si_addr;
    o["xtal_hz"] = s.xtal_hz;
    o["xtal_ppm"] = s.xtal_ppm;
    o["clk"] = s.clk;
    o["drive_ma"] = s.drive_ma;
    o["base_hz"] = s.base_hz;
    o["spread_hz"] = s.spread_hz;
    o["tr_gpio"] = s.tr_gpio;
    o["tr_active_high"] = s.tr_active_high;
    o["tr_lead_ms"] = s.tr_lead_ms;
    o["tr_tail_ms"] = s.tr_tail_ms;
    o["lead_in_s"] = s.lead_in_s;
    o["update_hz"] = s.update_hz;
}

void settings_load()
{
    File f = LittleFS.open(PATH, "r");
    if (!f)
        return;
    JsonDocument doc;
    if (deserializeJson(doc, f) == DeserializationError::Ok) {
        Settings s;
        apply(doc.as<JsonVariantConst>(), s);
        if (check(s).length() == 0)
            settings = s;
        else
            Serial.printf("settings.json rejected: %s\n", check(s).c_str());
    }
    f.close();
}

String settings_merge(JsonVariantConst in, bool* reboot)
{
    Settings s = settings;
    apply(in, s);
    String err = check(s);
    if (err.length())
        return err;
    *reboot = s.sda != settings.sda || s.scl != settings.scl || s.si_addr != settings.si_addr
              || s.tr_gpio != settings.tr_gpio || s.tr_active_high != settings.tr_active_high
              || s.hostname != settings.hostname || !in["wifi"].isNull();
    JsonDocument doc;
    write_all(s, doc.to<JsonObject>(), true);
    File f = LittleFS.open("/settings.tmp", "w");
    if (!f || serializeJson(doc, f) == 0) {
        if (f)
            f.close();
        return "could not write the settings file";
    }
    f.close();
    LittleFS.remove(PATH);
    LittleFS.rename("/settings.tmp", PATH);
    settings = s;
    return "";
}

void settings_to_json(JsonObject out)
{
    write_all(settings, out, false);
}
