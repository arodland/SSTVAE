// Scheduled QRSS sends: a picture sent again and again on chosen quarter
// hours, while the app runs.
//
// An entry is one picture, one QRSS mode and one carrier, sent `count`
// times (0: until it is removed) starting at a quarter hour and then
// every `every_min` minutes -- 0 meaning back to back, the next send
// starting the half hour after the last pass of the one before. So
// "2 x A" is mode A, two sends, back to back (two passes, half an hour
// apart, both carrying part 1 of the picture), and "alternating hours"
// is every 120 minutes. Every send of an entry is the same picture in
// the same mode, so it has the same picture ID and receivers add the
// passes together.
//
// Back-to-back sends are handed to the transmitter as one run
// (`qrss_tx::Request::passes`), because the transmitter has about 2 s
// between one pass's audio and the next and a fresh send needs minutes
// to encode. Sends with a gap between them are separate runs, and the
// app goes back to receiving in the gap. Two entries may not overlap:
// the transmitter can send one thing at a time, and starting a send
// takes `START_LEAD_S`. The one exception is a send that starts the half
// hour after another send's last pass (`follows_on`): that one joins the
// run on the air (`qrss_tx::Chain`), is encoded while the pass before it
// plays, and needs no lead of its own.
//
// The model (`Entry` and the free functions) is plain data and
// arithmetic, so `test_qrss_schedule.cpp` checks it on its own;
// `QrssScheduler` adds the file, the clock and the transmitter.

#ifndef SSTVAE_GUI_QRSS_SCHEDULE_HPP
#define SSTVAE_GUI_QRSS_SCHEDULE_HPP

#include <QObject>
#include <QString>

#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "images/types.hpp"
#include "qrss_tx.hpp"

class QTimer;

namespace sstvae::gui::qrss_schedule {

inline constexpr double QUARTER_S = 900.0;
// A scheduled send is started (encoded, its first pass made) this long
// before its first pass's audio.
inline constexpr double START_LEAD_S = 240.0;
// A send can be added, or moved, up to this long before its quarter
// hour; the scheduler starts it at once.
inline constexpr double ADD_BY_S = 30.0;
// Not started by this long before its audio (which begins 11 s before
// the quarter hour): the send is missed. Encoding the picture and making
// the first pass took about 8 s on 4 shared cloud CPUs (2026-10-12), so
// this leaves room; ADD_BY_S must stay above it plus those 11 s.
inline constexpr double MIN_LEAD_S = 15.0;
// A send that follows straight on is handed to the run on the air no
// later than this long before that run's last pass keys.
inline constexpr double CHAIN_LEAD_S = 60.0;
// A back-to-back run is at most this many passes: a day on the air.
inline constexpr int MAX_RUN_PASSES = 48;
// The repeat intervals the window offers, in minutes; 0 is back to back.
inline constexpr std::array<int, 11> EVERY_CHOICES_MIN = {0,   60,  90,  120, 180, 240,
                                                          360, 480, 720, 1440, 2880};

struct Entry {
    std::string id;            // also the picture file's name
    std::string picture;       // PNG, already 640 x 480
    std::string label;         // what the operator calls it
    std::string send_id;       // the Send it was made from (gui/qrss_sends.hpp), if any
    std::string mode = "A";    // QRSS A, B or C
    std::string waveform = qrss_tx::WAVEFORM_CE;   // CE or L
    double freq_hz = qrss_tx::FREQ_DEFAULT_HZ;
    double first_slot = 0.0;   // unix seconds, a quarter hour: the first send's first pass
    int count = 1;             // sends; 0 = until removed
    int every_min = 0;         // 0 = back to back
    bool enabled = true;
    int next = 0;              // the first send neither started nor missed
};

// Minutes from one send's start to the next's.
int period_min(const Entry& e);
// The quarter hour of send k's first pass.
double send_slot(const Entry& e, int k);
// Every send started or missed.
bool finished(const Entry& e);
// Send k's passes: one per part of the picture, half an hour apart.
std::vector<qrss_tx::Pass> passes_of(const Entry& e, int k);
// When send k needs the transmitter: from START_LEAD_S before its first
// pass's audio to the end of its last pass's audio.
std::pair<double, double> busy(const Entry& e, int k);

// Whether send k of `e` starts the half hour after the last pass of a
// send of an entry in `all` (another entry, or an earlier send of `e`
// itself), early enough at `now` to join that send's run. Such a send
// needs the transmitter only from its own audio.
bool follows_on(const Entry& e, int k, const std::vector<Entry>& all, double now);

// Empty when the entry can be scheduled; otherwise why not.
std::string problem(const Entry& e);
// "2 x mode A, back to back" / "mode B every 2 hours, until removed".
std::string describe(const Entry& e);
// "15:00Z" for a quarter hour, with the day when it is not `today`'s.
std::string when(double slot, double today);

// The earliest quarter hour a send added now could start on.
double earliest_slot(double now);

struct Upcoming {
    std::size_t entry = 0;     // index into the entries
    int send = 0;
    double slot = 0.0;
};
// The enabled entries' sends from `now` on, in time order, at most `max`
// of them and none more than `horizon_s` away.
std::vector<Upcoming> upcoming(const std::vector<Entry>& entries, double now,
                               double horizon_s, std::size_t max);

struct Clash {
    std::string other;         // the other entry's label
    double slot = 0.0;         // this entry's send that clashes
    double other_slot = 0.0;   // and the other's
};
// The first send of `e` that would need the transmitter while an enabled
// entry in `others` (not `e` itself) does, within a week of `now`.
std::optional<Clash> clash(const Entry& e, const std::vector<Entry>& others, double now);
// The first enabled scheduled send that needs the transmitter in
// [from, to], for a send started by hand.
std::optional<Upcoming> clash_with(const std::vector<Entry>& entries, double from, double to,
                                   double now);

// --- frequency changes ------------------------------------------------
//
// A scheduled change of the radio's dial frequency, at a moment between
// transmissions. Five seconds later the antenna tuner may follow
// (Settings > Rig control), so a change keeps CHANGE_CLEAR_S clear of
// any send's span before it, and after it the tune's delay plus the
// longest a tune takes (TUNE_MAX_S: Jeff's tuner, rarely, 45 s).

inline constexpr double TUNE_DELAY_S = 5.0;
inline constexpr double CHANGE_CLEAR_S = 30.0;
inline constexpr double TUNE_MAX_S = 45.0;
// Not done within this long of its time (the transmitter was busy, the
// app was not running): missed.
inline constexpr double CHANGE_LATE_S = 600.0;

struct FreqChange {
    std::string id;
    double at = 0.0;        // unix seconds
    double dial_hz = 0.0;   // the radio's dial frequency
    bool done = false;      // made, or missed
};

// Empty when the change can be scheduled; otherwise why not.
std::string problem(const FreqChange& c);
// The span a change needs clear of transmissions.
std::pair<double, double> busy(const FreqChange& c);
// The first enabled send in `entries` that a change at `c` would cut
// into, as for `clash_with`.
std::optional<Upcoming> clash(const FreqChange& c, const std::vector<Entry>& entries);
// The first pending change that `e`'s sends would cut into (within a
// week of `now`), else nothing.
std::optional<FreqChange> clash(const Entry& e, const std::vector<FreqChange>& changes,
                                double now);
// "10.141000 MHz".
std::string mhz(double hz);

std::string to_json(const std::vector<Entry>& entries,
                    const std::vector<FreqChange>& changes = {});
// Unreadable entries are dropped, not fatal.
std::vector<Entry> from_json(const std::string& text);
std::vector<FreqChange> changes_from_json(const std::string& text);

}  // namespace sstvae::gui::qrss_schedule

namespace sstvae::gui {

// Keeps the schedule in a file and starts each send when it is due.
class QrssScheduler : public QObject {
    Q_OBJECT

public:
    enum class Start {
        Started,    // the run is under way
        Busy,       // the transmitter is in use; ask again next tick
        Failed,     // cannot be sent (already reported); count it as missed
    };
    // Starts a run of one entry's passes.
    using Starter = std::function<Start(const qrss_schedule::Entry& entry,
                                        const std::vector<qrss_tx::Pass>& passes,
                                        std::string* why)>;
    using Clock = std::function<double()>;
    // Adds an entry's passes to the end of the run on the air; false
    // (with why) when the run will not take them.
    using Chainer = std::function<bool(const qrss_schedule::Entry& entry,
                                       const std::vector<qrss_tx::Pass>& passes,
                                       std::string* why)>;
    // Makes a frequency change: Started when the radio was told, Busy
    // when it must wait (the transmitter is in use), Failed otherwise.
    using Retuner = std::function<Start(const qrss_schedule::FreqChange& change,
                                        std::string* why)>;

    // `file` is the schedule (JSON); pictures go in a directory beside
    // it. Empty `clock` is the system clock. `tick_ms` 0 leaves the
    // timer off, for tests that call `tick` themselves.
    QrssScheduler(std::filesystem::path file, Starter starter, Clock clock = {},
                  int tick_ms = 5000, QObject* parent = nullptr);
    ~QrssScheduler() override;

    const std::vector<qrss_schedule::Entry>& entries() const { return entries_; }
    double now() const { return clock_(); }
    std::filesystem::path picture_dir() const;

    // Adds an entry for `picture`; its `picture` path and `id` are filled
    // in here. Empty on success, else why not (nothing is added).
    std::string add(qrss_schedule::Entry entry, const images::Picture& picture);
    void remove(const std::string& id);
    // Changes an entry -- its mode, its repeats, when it starts -- through
    // `edit`, refusing (and changing nothing) when the result could not
    // be scheduled: an impossible repeat, too soon to start, or a clash.
    // An entry that has already sent some of its sends is first cut down
    // to the ones still to come, so the change moves only those. Empty on
    // success, else why not.
    std::string change(const std::string& id,
                       const std::function<void(qrss_schedule::Entry&)>& edit);
    // Ends an entry after its send `k`'s predecessor: send k and the ones
    // after it are dropped. Removes the entry when nothing would be left.
    void truncate(const std::string& id, int k);
    // Empty on success; resuming an entry that would clash is refused.
    std::string set_enabled(const std::string& id, bool on);

    // Frequency changes, in time order. Made ones are kept a day, for
    // the timeline.
    const std::vector<qrss_schedule::FreqChange>& changes() const { return changes_; }
    void set_retuner(Retuner retuner) { retuner_ = std::move(retuner); }
    void set_chainer(Chainer chainer) { chainer_ = std::move(chainer); }
    // Adds a change; its `id` is filled in here. Empty on success, else
    // why not (too soon, or it would cut into a send).
    std::string add_change(qrss_schedule::FreqChange change);
    // Moves or retunes a pending change, with the same checks.
    std::string edit_change(const std::string& id,
                            const std::function<void(qrss_schedule::FreqChange&)>& edit);
    void remove_change(const std::string& id);

    // The entry whose run is on the air (or getting ready), else empty.
    const std::string& active() const { return active_; }
    // Whether `id` has a send in the run on the air: the entry that
    // started it, or one that followed on.
    bool on_air(const std::string& id) const;
    // The transmitter finished whatever it was doing.
    void run_finished();

public slots:
    // Skips missed sends and starts one that is due. On a timer.
    void tick();

signals:
    void changed();
    // A line for the app's log; `severity` is a `log::Severity` (0 info,
    // 1 warning, 2 error).
    void logged(int severity, const QString& message);

private:
    void load();
    void save();
    qrss_schedule::Entry* find(const std::string& id);
    // The first send of `e` that can still start at `now`.
    int first_startable(const qrss_schedule::Entry& e, double now) const;

    std::filesystem::path file_;
    Starter starter_;
    Clock clock_;
    QTimer* timer_ = nullptr;
    std::vector<qrss_schedule::Entry> entries_;
    std::vector<qrss_schedule::FreqChange> changes_;
    Retuner retuner_;
    std::string change_refused_;   // the change whose wait was logged
    // Why a send could not go where asked because of a change, or empty.
    std::string change_clash_text(const qrss_schedule::Entry& e, double now) const;
    void tick_changes(double now, bool& dirty);
    std::string active_;
    std::vector<std::string> on_air_;   // active_ and the entries that followed on
    double run_end_ = 0.0;              // the run's last pass, its quarter hour
    Chainer chainer_;
    // Hands sends that follow straight on to the run on the air.
    void chain(double now, bool& dirty);
    // The last refusal logged, "<id>/<send>", so a busy transmitter is
    // logged once per send rather than once per tick.
    std::string refused_;
    std::string refused_why_;
};

}  // namespace sstvae::gui

#endif
