#include "store.h"

#include <LittleFS.h>

#include "tx.h"

namespace store {

std::vector<sched::Entry> entries;

static const char* TMP = "/pics/upload.tmp";

static uint32_t rd32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

// The fixed 56 bytes in front of the header bits: what we check and list.
static String parse_head(const uint8_t* h, uint32_t len, PicInfo* out)
{
    if (memcmp(h, "QRSB", 4) != 0)
        return "not a beacon file (make one with qrss_beacon.py)";
    if (h[4] != 1 || h[5] != 0 || (h[28] | h[29] << 8) != 20 || (h[30] & ~1) || h[31])
        return "unsupported beacon file version or waveform";
    const uint32_t n_lat = rd32(h + 24);
    if (n_lat != 50600)
        return "not a full segment (50,600 latents)";
    if (len && len != 366 + n_lat + 4)
        return "length does not match the latent count";
    if (h[6] > 2 || h[7] > h[6])
        return "bad mode or segment";
    bool keyed = false;
    for (int i = 32; i < 56; i++)
        keyed |= h[i] != 0;
    if (!keyed)
        return "the file keys no callsign";
    out->pid = rd32(h + 8);
    out->mode = h[6];
    out->segment = h[7];
    out->callsign = String((const char*)h + 14, 8);
    out->callsign.trim();
    out->grid = (uint16_t)(h[22] | h[23] << 8);
    out->ook = h[30] & 1;
    return "";
}

std::vector<PicInfo> pictures()
{
    std::vector<PicInfo> out;
    Dir d = LittleFS.openDir("/pics");
    while (d.next()) {
        if (!d.fileName().endsWith(".bin"))
            continue;
        File f = d.openFile("r");
        uint8_t h[56];
        PicInfo p;
        if (f.read(h, sizeof h) == sizeof h && parse_head(h, f.size(), &p).length() == 0) {
            p.file = d.fileName();
            out.push_back(p);
        }
        f.close();
    }
    return out;
}

static File up;
static uint32_t up_crc, up_len;
static uint8_t up_head[56], up_tail[4];

static uint32_t crc_update(uint32_t c, const uint8_t* p, size_t n)    // zlib's, unfinalised
{
    while (n--) {
        c ^= *p++;
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return c;
}

void upload_begin()
{
    LittleFS.mkdir("/pics");
    up = LittleFS.open(TMP, "w");
    up_crc = 0xFFFFFFFFu;
    up_len = 0;
}

void upload_add(const uint8_t* p, size_t n)
{
    if (!up)
        return;
    up.write(p, n);
    for (size_t i = 0; i < n; i++, up_len++) {
        if (up_len < sizeof up_head)
            up_head[up_len] = p[i];
        // Hold back the last 4 bytes seen: the CRC covers all but them.
        if (up_len >= 4)
            up_crc = crc_update(up_crc, &up_tail[up_len % 4], 1);
        up_tail[up_len % 4] = p[i];
    }
}

String upload_finish(PicInfo* info)
{
    if (!up)
        return "could not open a file for the upload";
    up.close();
    String err;
    if (up_len < 366 + 4 || up_len > tx::BUF_BYTES)
        err = "wrong size for a beacon file";
    if (!err.length()) {
        uint8_t t[4];                              // the stored CRC, in file order
        for (int i = 0; i < 4; i++)
            t[i] = up_tail[(up_len - 4 + i) % 4];
        if (rd32(t) != (up_crc ^ 0xFFFFFFFFu))
            err = "CRC mismatch: the file is damaged";
    }
    if (!err.length())
        err = parse_head(up_head, up_len, info);
    if (err.length()) {
        LittleFS.remove(TMP);
        return err;
    }
    char name[32];
    snprintf(name, sizeof name, "/pics/%08lx-%d.bin", (unsigned long)info->pid, info->segment);
    LittleFS.remove(name);
    if (!LittleFS.rename(TMP, name))
        return "could not store the file (flash full?)";
    info->file = name + 6;
    return "";
}

bool remove_picture(const String& file)
{
    if (file.indexOf('/') >= 0 || !file.endsWith(".bin"))
        return false;
    return LittleFS.remove("/pics/" + file);
}

uint32_t load(uint32_t pid, int seg, uint8_t* buf, uint32_t cap)
{
    char name[32];
    snprintf(name, sizeof name, "/pics/%08lx-%d.bin", (unsigned long)pid, seg);
    File f = LittleFS.open(name, "r");
    if (!f || f.size() > cap)
        return 0;
    const uint32_t n = f.read(buf, f.size());
    f.close();
    return n;
}

bool has_segments(uint32_t pid, int mode)
{
    for (int s = 0; s <= mode; s++) {
        char name[32];
        snprintf(name, sizeof name, "/pics/%08lx-%d.bin", (unsigned long)pid, s);
        if (!LittleFS.exists(name))
            return false;
    }
    return true;
}

void entry_to_json(const sched::Entry& e, JsonObject o)
{
    char pid[9];
    snprintf(pid, sizeof pid, "%08lx", (unsigned long)e.pid);
    o["id"] = e.id;
    o["pid"] = pid;
    o["mode"] = e.mode;
    o["tiny"] = e.tiny;
    o["first_slot"] = e.first_slot;
    o["count"] = e.count;
    o["every_min"] = e.every_min;
    o["enabled"] = e.enabled;
}

sched::Entry entry_from_json(JsonVariantConst v)
{
    sched::Entry e;
    e.id = v["id"] | "";
    e.pid = (uint32_t)strtoul(v["pid"] | "0", nullptr, 16);
    e.mode = v["mode"] | 0;
    e.tiny = v["tiny"] | false;
    e.first_slot = v["first_slot"] | (int64_t)0;
    e.count = v["count"] | 1;
    e.every_min = v["every_min"] | 0;
    e.enabled = v["enabled"] | true;
    return e;
}

void schedule_load()
{
    File f = LittleFS.open("/schedule.json", "r");
    if (!f)
        return;
    JsonDocument doc;
    if (deserializeJson(doc, f) == DeserializationError::Ok)
        for (JsonVariantConst v : doc.as<JsonArrayConst>())
            entries.push_back(entry_from_json(v));
    f.close();
}

bool schedule_save()
{
    JsonDocument doc;
    JsonArray a = doc.to<JsonArray>();
    for (const sched::Entry& e : entries)
        entry_to_json(e, a.add<JsonObject>());
    File f = LittleFS.open("/schedule.tmp", "w");
    if (!f)
        return false;
    const bool ok = serializeJson(doc, f) > 0;
    f.close();
    if (!ok)
        return false;
    LittleFS.remove("/schedule.json");
    return LittleFS.rename("/schedule.tmp", "/schedule.json");
}

}  // namespace store
