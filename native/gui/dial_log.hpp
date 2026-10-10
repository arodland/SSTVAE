// The radio's dial frequency over time, for the QRSS listener:
// `dial_changes.jsonl` in the listener's state directory, one line per
// change, {"t": unix seconds, "dial_hz": integer Hz, "why": ...}.
//
// The listener treats a little either side of each change as unheard and
// receives each dial separately, so a line is written for every change
// the app commands, every change seen on the rig readout (a knob turned
// on the radio), and the dial read when rig control connects -- but never
// for the same dial as the line before, and never when the dial is
// unknown. Appended a whole line per write; lines older than 48 h are
// trimmed, keeping the newest line before the cutoff, which says what the
// dial was then.

#ifndef SSTVAE_GUI_DIAL_LOG_HPP
#define SSTVAE_GUI_DIAL_LOG_HPP

#include <QString>

#include <optional>

namespace sstvae::gui {

class DialLog {
public:
    static constexpr double KEEP_S = 48 * 3600.0;

    // The file `dir`/dial_changes.jsonl; empty `dir` writes nothing.
    explicit DialLog(QString dir = {});

    void set_dir(const QString& dir);
    QString path() const;

    // Writes a line unless `hz` is the dial last written. True if it
    // wrote one.
    bool record(double t, double hz, const char* why);
    std::optional<long long> last_hz() const { return last_hz_; }

    // Drops lines older than KEEP_S before `now`, keeping the newest of
    // those. Done on opening and now and then after.
    void trim(double now);

private:
    void read_last();

    QString dir_;
    std::optional<long long> last_hz_;
    int writes_since_trim_ = 0;
};

}  // namespace sstvae::gui

#endif
