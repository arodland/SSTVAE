// The status log: bounding, sinks, formatting, and file rotation.
//
// The rotation test uses a tiny max_bytes so it exercises the shift
// chain (.log -> .1 -> .2) rather than needing a megabyte of lines.

#include "log/log.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "check.hpp"

namespace fs = std::filesystem;
namespace check = sstvae::check;
using sstvae::log::Entry;
using sstvae::log::FileWriter;
using sstvae::log::format_entry;
using sstvae::log::Severity;
using sstvae::log::StatusLog;

namespace {

std::string read_all(const fs::path& p) {
    std::ifstream in(p);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Portable and unique without getpid(); same idiom as test_checkpoint.
fs::path make_temp_dir() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path dir =
        fs::temp_directory_path() / ("sstvae-log-" + std::to_string(stamp));
    fs::create_directories(dir);
    return dir;
}

void test_append_and_snapshot() {
    StatusLog log(10);
    log.append("rig", Severity::Info, "Rig: 14.2300 MHz");
    log.append("tx", Severity::Error, "PTT OFF FAILED");
    const auto entries = log.snapshot();
    check::equal(entries.size(), std::size_t{2}, "two entries retained");
    check::equal(entries[0].source, std::string("rig"), "source kept");
    check::equal(entries[1].text, std::string("PTT OFF FAILED"), "text kept");
    check::is_true(entries[0].ms > 0, "wall-clock stamp set");
    check::is_true(entries[0].ms <= entries[1].ms, "entries in order");
}

void test_bounded() {
    StatusLog log(3);
    for (int i = 0; i < 10; ++i) {
        log.append("app", Severity::Info, "line " + std::to_string(i));
    }
    const auto entries = log.snapshot();
    check::equal(entries.size(), std::size_t{3}, "bounded to max_entries");
    check::equal(entries.front().text, std::string("line 7"), "oldest dropped");
    check::equal(entries.back().text, std::string("line 9"), "newest kept");
}

void test_sink_sees_every_entry() {
    StatusLog log(2);  // smaller than the number of appends
    std::vector<std::string> seen;
    log.add_sink([&](const Entry& e) { seen.push_back(e.text); });
    for (int i = 0; i < 5; ++i) {
        log.append("rx", Severity::Info, std::to_string(i));
    }
    check::equal(seen.size(), std::size_t{5},
                 "sink not bounded by the in-memory cap");
}

void test_format() {
    Entry e;
    e.ms = 1000;  // epoch + 1 s, any zone: the fields we assert are stable
    e.source = "rig";
    e.severity = Severity::Error;
    e.text = "no response";
    const std::string line = format_entry(e);
    check::is_true(line.find("rig") != std::string::npos, "source in line");
    check::is_true(line.find("ERROR") != std::string::npos, "severity in line");
    check::is_true(line.find("no response") != std::string::npos, "text in line");
    check::is_true(line.find("1970-") != std::string::npos ||
                       line.find("1969-") != std::string::npos,
                   "date rendered from the stamp");
}

void test_file_rotation(const fs::path& dir) {
    const fs::path path = dir / "sstvae.log";
    FileWriter writer(path, /*max_bytes=*/64, /*rotations=*/2);
    check::is_true(writer.error().empty(), "writer opens cleanly");

    Entry e;
    e.ms = 0;
    e.source = "app";
    e.severity = Severity::Info;
    for (int i = 0; i < 12; ++i) {
        e.text = "rotation filler line " + std::to_string(i);
        writer.write(e);
    }
    check::is_true(fs::exists(path), "live file exists");
    check::is_true(fs::exists(path.string() + ".1"), "first rotation exists");
    check::is_true(fs::exists(path.string() + ".2"), "second rotation exists");
    check::is_true(!fs::exists(path.string() + ".3"),
                   "nothing beyond the configured rotations");
    // The newest line survives in the live file or the newest rotation;
    // rotation must never *lose* it.
    const std::string joined =
        read_all(path) + read_all(path.string() + ".1");
    check::is_true(joined.find("line 11") != std::string::npos,
                   "newest line not lost to rotation");
    check::is_true(writer.error().empty(), "no error after rotating");
}

void test_missing_parent_is_created(const fs::path& dir) {
    // A fresh install has no config directory yet; the writer creates
    // it rather than silently producing no file for the first session.
    FileWriter writer(dir / "fresh-subdir" / "sstvae.log");
    check::is_true(writer.error().empty(),
                   "a missing parent directory is created");
    Entry e;
    e.source = "app";
    e.text = "first line";
    writer.write(e);
    check::is_true(fs::exists(dir / "fresh-subdir" / "sstvae.log"),
                   "and the file lands in it");
}

void test_file_failure_is_loud(const fs::path& dir) {
    // A parent that cannot be a directory, because it is a file: the
    // open fails, error() says so, and write() is a no-op rather than
    // a crash.
    std::ofstream(dir / "blocker").put('\n');
    FileWriter writer(dir / "blocker" / "sstvae.log");
    check::is_true(!writer.error().empty(), "open failure reported");
    Entry e;
    e.source = "app";
    e.text = "dropped";
    writer.write(e);  // must not throw
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(60.0, "log");

    check::current_step.store("append_and_snapshot");
    test_append_and_snapshot();
    check::current_step.store("bounded");
    test_bounded();
    check::current_step.store("sink");
    test_sink_sees_every_entry();
    check::current_step.store("format");
    test_format();

    const fs::path dir = make_temp_dir();
    check::current_step.store("rotation");
    test_file_rotation(dir);
    check::current_step.store("missing_parent");
    test_missing_parent_is_created(dir);
    check::current_step.store("failure_is_loud");
    test_file_failure_is_loud(dir);
    std::error_code ec;
    fs::remove_all(dir, ec);

    return check::report("log");
}
