// The web page and its JSON API, all behind HTTP basic auth.
#include <ArduinoJson.h>
#include <LittleFS.h>
#include <WebServer.h>
#include <WiFi.h>

#include "app.h"
#include "page.h"
#include "settings.h"
#include "store.h"
#include "timesync.h"
#include "tx.h"

static WebServer server(80);
static String upload_err;
static PicInfo upload_info;

static bool authed()
{
    if (server.authenticate(settings.web_user.c_str(), settings.web_pass.c_str()))
        return true;
    server.requestAuthentication(BASIC_AUTH, "QRSS beacon");
    return false;
}

static void send_json(const JsonDocument& d, int code = 200)
{
    String out;
    serializeJson(d, out);
    server.send(code, "application/json", out);
}

static void send_result(const String& err, bool extra_warning = false)
{
    JsonDocument d;
    d["ok"] = err.length() == 0;
    if (err.length())
        d["error"] = err;
    // A flash write pauses core 1: say so whenever one happened mid-pass.
    if (extra_warning || tx::status().state == tx::KEYED)
        d["warning"] = "written during a pass: the transmitter paused briefly";
    send_json(d, err.length() ? 400 : 200);
}

static bool body(JsonDocument& d)
{
    if (deserializeJson(d, server.arg("plain")) == DeserializationError::Ok)
        return true;
    send_result("the request body is not JSON");
    return false;
}

static void status()
{
    JsonDocument d;
    const tx::Status& s = tx::status();
    static const char* names[] = {"idle", "waiting", "keyed", "carrier", "error"};
    d["now"] = timesync::synced() ? timesync::now_us() / 1000000 : 0;
    d["sync_age_s"] = timesync::synced() ? (long)timesync::age_s() : -1;
    d["sync_step_us"] = timesync::last_step_us();
    d["sync_delay_us"] = timesync::last_rtt_us();
    d["rate_ppm"] = timesync::clock().rate_ppb / 1000.0;
    d["not_ready"] = not_ready();
    JsonObject t = d["tx"].to<JsonObject>();
    t["state"] = names[s.state];
    t["qh"] = s.qh;
    t["u"] = s.u;
    t["n"] = s.n;
    t["late"] = s.late;
    t["err"] = s.err;
    t["carrier_hz"] = s.carrier_hz;
    t["step_hz"] = s.step_hz;
    const sched::Pass p = upcoming_pass();
    if (p.entry >= 0) {
        JsonObject n = d["next"].to<JsonObject>();
        n["id"] = store::entries[p.entry].id;
        n["qh"] = p.qh;
        n["segment"] = p.segment;
    }
    d["wifi"] = ap_mode() ? String("access point") : WiFi.SSID() + " " + String(WiFi.RSSI()) + " dBm";
    FSInfo fs;
    LittleFS.info(fs);
    d["fs_used"] = fs.usedBytes;
    d["fs_total"] = fs.totalBytes;
    JsonArray log = d["log"].to<JsonArray>();
    for (const String& l : log_lines)
        log.add(l);
    send_json(d);
}

static void pictures()
{
    JsonDocument d;
    JsonArray a = d.to<JsonArray>();
    for (const PicInfo& p : store::pictures()) {
        JsonObject o = a.add<JsonObject>();
        char pid[9];
        snprintf(pid, sizeof pid, "%08lx", (unsigned long)p.pid);
        o["file"] = p.file;
        o["pid"] = pid;
        o["mode"] = p.mode;
        o["segment"] = p.segment;
        o["callsign"] = p.callsign;
        o["grid"] = p.grid;
        o["ook"] = p.ook;
    }
    send_json(d);
}

static void upload_chunk()
{
    HTTPUpload& up = server.upload();
    if (!server.authenticate(settings.web_user.c_str(), settings.web_pass.c_str()))
        return;
    if (up.status == UPLOAD_FILE_START) {
        upload_err = "";
        store::upload_begin();
    } else if (up.status == UPLOAD_FILE_WRITE) {
        store::upload_add(up.buf, up.currentSize);
    } else if (up.status == UPLOAD_FILE_END) {
        upload_err = store::upload_finish(&upload_info);
        if (!upload_err.length())
            log_line("stored %s (%s)", upload_info.file.c_str(), upload_info.callsign.c_str());
    } else {
        upload_err = "upload aborted";
    }
}

static String entry_problem(const sched::Entry& e)
{
    String p = sched::problem(e).c_str();
    if (p.length())
        return p;
    if (!store::has_segments(e.pid, e.mode))
        return "the picture is missing a segment its mode sends";
    const int64_t now = timesync::synced() ? timesync::now_us() / 1000000 : 0;
    return sched::clash(e, store::entries, now).c_str();
}

static void schedule_put()
{
    JsonDocument d;
    if (!body(d))
        return;
    sched::Entry e = store::entry_from_json(d.as<JsonVariantConst>());
    // The mode is the picture's, not the client's to choose.
    for (const PicInfo& p : store::pictures())
        if (p.pid == e.pid)
            e.mode = p.mode;
    String err = entry_problem(e);
    if (err.length())
        return send_result(err);
    bool found = false;
    for (sched::Entry& old : store::entries)
        if (e.id.size() && old.id == e.id) {
            old = e;
            found = true;
        }
    if (!found) {
        char id[12];
        snprintf(id, sizeof id, "e%08lx", (unsigned long)rp2040.hwrand32());
        e.id = id;
        store::entries.push_back(e);
    }
    send_result(store::schedule_save() ? "" : "could not write the schedule");
}

static void schedule_get()
{
    JsonDocument d;
    JsonArray a = d.to<JsonArray>();
    for (const sched::Entry& e : store::entries)
        store::entry_to_json(e, a.add<JsonObject>());
    send_json(d);
}

static void schedule_delete()
{
    const String id = server.arg("id");
    for (size_t i = 0; i < store::entries.size(); i++)
        if (id == store::entries[i].id.c_str()) {
            store::entries.erase(store::entries.begin() + i);
            return send_result(store::schedule_save() ? "" : "could not write the schedule");
        }
    send_result("no such entry");
}

static void settings_get()
{
    JsonDocument d;
    settings_to_json(d.to<JsonObject>());
    send_json(d);
}

static void settings_post()
{
    JsonDocument d;
    if (!body(d))
        return;
    bool reboot = false;
    String err = settings_merge(d.as<JsonVariantConst>(), &reboot);
    if (err.length())
        return send_result(err);
    JsonDocument r;
    r["ok"] = true;
    r["reboot"] = reboot;
    if (tx::status().state == tx::KEYED)
        r["warning"] = "written during a pass: the transmitter paused briefly";
    send_json(r);
}

void web_begin()
{
    auto guard = [](void (*fn)()) {
        return [fn]() {
            if (authed())
                fn();
        };
    };
    server.on("/", HTTP_GET, guard([] { server.send_P(200, "text/html", PAGE); }));
    server.on("/api/status", HTTP_GET, guard(status));
    server.on("/api/pictures", HTTP_GET, guard(pictures));
    server.on("/api/pictures", HTTP_POST,
              guard([] {
                  if (upload_err.length())
                      return send_result(upload_err);
                  send_result("");
              }),
              upload_chunk);
    server.on("/api/pictures", HTTP_DELETE, guard([] {
        bool used = false;
        const String f = server.arg("file");
        for (const sched::Entry& e : store::entries) {
            char pre[10];
            snprintf(pre, sizeof pre, "%08lx-", (unsigned long)e.pid);
            used |= f.startsWith(pre);
        }
        if (used)
            return send_result("a schedule entry still sends this picture");
        send_result(store::remove_picture(f) ? "" : "no such file");
    }));
    server.on("/api/schedule", HTTP_GET, guard(schedule_get));
    server.on("/api/schedule", HTTP_POST, guard(schedule_put));
    server.on("/api/schedule", HTTP_DELETE, guard(schedule_delete));
    server.on("/api/settings", HTTP_GET, guard(settings_get));
    server.on("/api/settings", HTTP_POST, guard(settings_post));
    server.on("/api/carrier", HTTP_POST, guard([] {
        send_result(start_carrier((uint32_t)server.arg("seconds").toInt()));
    }));
    server.on("/api/abort", HTTP_POST, guard([] {
        tx::abort();
        send_result("");
    }));
    server.on("/api/reboot", HTTP_POST, guard([] {
        send_result("");
        delay(200);
        rp2040.reboot();
    }));
    server.begin();
}

void web_loop()
{
    server.handleClient();
}
