// What main.cpp owns and web.cpp shows.
#pragma once

#include <Arduino.h>

#include <deque>

#include "schedule.h"

void log_line(const char* fmt, ...);
extern std::deque<String> log_lines;       // newest last

// Why the station may not transmit now ("" = it may).
String not_ready();
// The pass the scheduler will start next (entry = -1: none).
sched::Pass upcoming_pass();
// Start a test carrier at the base frequency; "" or why not.
String start_carrier(uint32_t seconds);
bool ap_mode();

void web_begin();
void web_loop();
