#include "log/log.hpp"

#include <cstddef>
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <system_error>
#include <utility>

namespace sstvae::log {

namespace fs = std::filesystem;

const char* severity_name(Severity s) {
    switch (s) {
        case Severity::Info: return "info";
        case Severity::Warning: return "warn";
        case Severity::Error: return "ERROR";
    }
    return "info";
}

std::string format_entry(const Entry& entry) {
    const std::time_t seconds = static_cast<std::time_t>(entry.ms / 1000);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &seconds);
#else
    localtime_r(&seconds, &tm);
#endif
    // **The year is clamped, and that is the point.** GCC cannot know
    // `tm_year` is sane, so at any buffer size it warns that the year
    // could truncate the rest of the stamp -- and growing the buffer
    // only moves the warning to whatever formats the stamp next. Saying
    // out loud that a four-digit year is all this format supports fixes
    // it at the source; a clock returning year 70000 gets a wrong stamp
    // rather than a silently cut one.
    char stamp[32];
    const int year = std::clamp(tm.tm_year + 1900, 0, 9999);
    std::snprintf(stamp, sizeof stamp, "%04d-%02d-%02d %02d:%02d:%02d", year,
                  tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

    // Fixed-width source and severity columns so a screenful of lines
    // reads as a table. Built as a string rather than a second fixed
    // buffer: the source is caller-supplied and has no length anyone
    // here can promise.
    auto column = [](std::string text, std::size_t width) {
        if (text.size() < width) text.resize(width, ' ');
        return text;
    };
    return std::string(stamp) + "  " + column(entry.source, 4) + " " +
           column(severity_name(entry.severity), 5) + " " + entry.text;
}

StatusLog::StatusLog(std::size_t max_entries) : max_entries_(max_entries) {}

void StatusLog::append(std::string source, Severity severity, std::string text) {
    Entry entry;
    entry.ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
    entry.source = std::move(source);
    entry.severity = severity;
    entry.text = std::move(text);

    const std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back(entry);
    while (entries_.size() > max_entries_) entries_.pop_front();
    for (const auto& sink : sinks_) sink(entry);
}

std::vector<Entry> StatusLog::snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return {entries_.begin(), entries_.end()};
}

void StatusLog::add_sink(std::function<void(const Entry&)> sink) {
    const std::lock_guard<std::mutex> lock(mutex_);
    sinks_.push_back(std::move(sink));
}

FileWriter::FileWriter(fs::path path, std::uint64_t max_bytes, int rotations)
    : path_(std::move(path)), max_bytes_(max_bytes), rotations_(rotations) {
    std::error_code ec;
    // On a fresh install nothing has created the config directory yet
    // -- settings::save() does, but the first session may never call
    // it, and the first session is exactly the one whose log is worth
    // having (first-run download failures live there).
    if (path_.has_parent_path()) fs::create_directories(path_.parent_path(), ec);
    const std::uint64_t existing = fs::file_size(path_, ec);
    written_ = ec ? 0 : existing;
    out_.open(path_, std::ios::app);
    if (!out_) error_ = "cannot open " + path_.string() + " for append";
}

void FileWriter::write(const Entry& entry) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!out_.is_open()) return;
    if (written_ >= max_bytes_) rotate();
    const std::string line = format_entry(entry);
    out_ << line << '\n';
    out_.flush();
    if (!out_) {
        if (error_.empty()) error_ = "write failed on " + path_.string();
        out_.close();
        return;
    }
    written_ += line.size() + 1;
}

std::string FileWriter::error() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

void FileWriter::rotate() {
    // Called with mutex_ held. sstvae.log.<rotations> falls off the
    // end; everything else shifts up by one; the live file restarts.
    out_.close();
    std::error_code ec;
    fs::remove(path_.string() + "." + std::to_string(rotations_), ec);
    for (int i = rotations_ - 1; i >= 1; --i) {
        fs::rename(path_.string() + "." + std::to_string(i),
                   path_.string() + "." + std::to_string(i + 1), ec);
    }
    fs::rename(path_, path_.string() + ".1", ec);
    out_.open(path_, std::ios::trunc);
    written_ = 0;
    if (!out_ && error_.empty()) {
        error_ = "cannot reopen " + path_.string() + " after rotation";
    }
}

}  // namespace sstvae::log
