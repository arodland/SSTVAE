// Beacon files (`qrss_beacon.py` output, one segment each) in /pics, named
// <picture ID>-<segment>.bin, and the schedule in /schedule.json.
#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>

#include <vector>

#include "schedule.h"

struct PicInfo {
    String file;
    uint32_t pid;
    int mode, segment;
    String callsign;
    uint16_t grid;      // 32767 = none
    bool ook;
};

namespace store {

std::vector<PicInfo> pictures();
// Streamed upload: begin, add each chunk, then finish to validate (magic,
// format, length, CRC, a keyed callsign) and file it. finish returns empty
// or why the file was refused (and removes it).
void upload_begin();
void upload_add(const uint8_t* p, size_t n);
String upload_finish(PicInfo* info);
bool remove_picture(const String& file);
// Read segment `seg` of picture `pid` into buf; its length, or 0.
uint32_t load(uint32_t pid, int seg, uint8_t* buf, uint32_t cap);
bool has_segments(uint32_t pid, int mode);

extern std::vector<sched::Entry> entries;
void schedule_load();
bool schedule_save();
void entry_to_json(const sched::Entry& e, JsonObject o);
sched::Entry entry_from_json(JsonVariantConst v);

}  // namespace store
