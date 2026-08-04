// The application's status log: a bounded, timestamped record of what
// happened, feeding both the GUI's log pane and an optional file.
//
// This exists because the app previously had *no* record of anything:
// every status was a one-line label overwritten by the next write, so
// "PTT OFF FAILED -- unkey it manually" was provably destroyed by the
// "Sent" that followed it, and a dismissed dialog was unrecoverable.
// The engines already emit everything worth keeping; this is where it
// lands.
//
// Qt-free on purpose, like the engines: the GUI adapts entries to
// signals, and a CLI could reuse the same sink. Thread-safe -- entries
// arrive from the decode thread, the transmit thread, the rig worker
// and the GUI thread alike.

#ifndef SSTVAE_LOG_LOG_HPP
#define SSTVAE_LOG_LOG_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace sstvae::log {

enum class Severity { Info, Warning, Error };

const char* severity_name(Severity s);  // "info" / "warn" / "ERROR"

struct Entry {
    // Wall-clock milliseconds since the epoch. Wall clock rather than
    // steady: these lines are read next to a radio's own clock and a
    // logbook, so they must agree with the wall.
    std::int64_t ms = 0;
    std::string source;  // "rig", "rx", "tx", "opt", "app"
    Severity severity = Severity::Info;
    std::string text;
};

// "2026-08-01 14:02:31  rig    ERROR  text" -- the one line format,
// shared by the file and the pane's Copy button so a pasted bug report
// and the on-disk log read the same.
std::string format_entry(const Entry& entry);

class StatusLog {
public:
    // How many entries to retain in memory. The file is not bounded by
    // this; it has its own rotation.
    explicit StatusLog(std::size_t max_entries = 2000);

    void append(std::string source, Severity severity, std::string text);

    // Everything currently retained, oldest first.
    std::vector<Entry> snapshot() const;

    // Called for every append, under the log's lock -- so a sink must
    // be fast and must never call back into the log. The two intended
    // sinks are: a queued Qt signal emission (non-blocking by design)
    // and FileWriter::write.
    void add_sink(std::function<void(const Entry&)> sink);

private:
    mutable std::mutex mutex_;
    std::size_t max_entries_;
    std::deque<Entry> entries_;
    std::vector<std::function<void(const Entry&)>> sinks_;
};

// Appends formatted entries to a file, rotating at `max_bytes`:
// sstvae.log -> sstvae.log.1 -> ... -> sstvae.log.<rotations>, oldest
// deleted. Failures are recorded, not thrown -- a read-only directory
// must not take the in-memory log down with it, but it must not be
// silent either: the GUI shows `error()` in the pane header.
class FileWriter {
public:
    explicit FileWriter(std::filesystem::path path,
                        std::uint64_t max_bytes = 1u << 20,
                        int rotations = 3);

    void write(const Entry& entry);

    const std::filesystem::path& path() const { return path_; }
    // Empty while healthy; the first failure's description afterwards.
    std::string error() const;

private:
    void rotate();

    std::filesystem::path path_;
    std::uint64_t max_bytes_;
    int rotations_;

    mutable std::mutex mutex_;
    std::ofstream out_;
    std::uint64_t written_ = 0;
    std::string error_;
};

}  // namespace sstvae::log

#endif
