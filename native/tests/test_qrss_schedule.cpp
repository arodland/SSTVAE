// Scheduled QRSS sends (gui/qrss_schedule.*): which quarter hours a
// schedule asks for, what may not overlap, and the scheduler's clock --
// due, started, waiting on a busy transmitter, missed -- against a fake
// clock and a fake transmitter. Changing and cutting short what is
// scheduled. The Sends kept on disk. Then the window, offscreen: a Send
// chosen enables the form, Add schedules it, the timeline shows it and
// its right-click menu changes it; and the Editor, which frames a file
// that is not 640 x 480 first and saves to the Sends.

#include <QApplication>
#include <QComboBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QFile>
#include <QGroupBox>
#include <QLineEdit>
#include <QMenu>
#include <QLabel>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QListWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QTimer>
#include <QTreeWidget>

#include <cstdint>
#include <filesystem>
#include <variant>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "crop_dialog.hpp"
#include "images/images.hpp"
#include "images/types.hpp"
#include "qrss_schedule.hpp"
#include "overlay/model.hpp"
#include "overlay_editor.hpp"
#include "qrss_editor.hpp"
#include "qrss_schedule_window.hpp"
#include "qrss_sends.hpp"
#include "qrss_timeline.hpp"
#include "waterfall.hpp"
#include "dial_log.hpp"

using namespace sstvae;
using namespace sstvae::gui;
namespace qs = sstvae::gui::qrss_schedule;

namespace {

constexpr double Q = 1791594000.0;    // 2026-10-10T01:00Z, a quarter hour
constexpr double H = 3600.0;

qs::Entry entry(const std::string& mode, double first, int count, int every_min,
                const std::string& label = "test") {
    qs::Entry e;
    e.id = label;
    e.label = label;
    e.picture = "/nowhere/" + label + ".png";
    e.mode = mode;
    e.first_slot = first;
    e.count = count;
    e.every_min = every_min;
    return e;
}

void test_the_sends_of_a_schedule() {
    // "2 x A": mode A twice, back to back: part 1 of the picture, twice.
    const qs::Entry aa = entry("A", Q, 2, 0);
    check::is_true(qs::problem(aa).empty(), "schedule/2xA: a valid schedule");
    check::equal(qs::period_min(aa), 30, "schedule/2xA: back to back is half an hour for A");
    std::vector<qrss_tx::Pass> p = qs::passes_of(aa, 1);
    check::is_true(p.size() == 1 && p[0].slot == Q + 1800.0 && p[0].segment == 0,
                   "schedule/2xA: the second send is the next half hour, part 1 again");
    check::equal(qs::describe(aa), std::string("2 x mode A, back to back"),
                 "schedule/2xA: described as Jeff wrote it");

    // Mode B every other hour, until removed.
    const qs::Entry b = entry("B", Q, 0, 120);
    check::is_true(qs::problem(b).empty(), "schedule/B: alternating hours is valid");
    p = qs::passes_of(b, 2);
    check::is_true(p.size() == 2 && p[0].slot == Q + 4 * H && p[1].slot == Q + 4 * H + 1800.0 &&
                       p[1].segment == 1,
                   "schedule/B: the third send, both of its parts");
    check::equal(qs::describe(b),
                 std::string("mode B every 2 hours (alternating hours), until removed"),
                 "schedule/B: described");
    const auto [from, to] = qs::busy(b, 0);
    check::is_true(from == Q - 11.0 - qs::START_LEAD_S && to == Q + 1800.0 + 1786.7,
                   "schedule/B: busy from the start lead to the end of the second pass");
    check::equal(qs::describe(entry("C", Q, 1, 0)), std::string("mode C, once"),
                 "schedule/C: a single send");
    check::equal(qs::describe(entry("A", Q, 5, 90)),
                 std::string("mode A every 90 minutes, 5 times"), "schedule/A: a count");
}

void test_what_cannot_be_scheduled() {
    check::is_true(!qs::problem(entry("A", Q, 0, 0)).empty(),
                   "schedule/problem: back to back until removed never leaves the air");
    check::is_true(!qs::problem(entry("A", Q, 49, 0)).empty(),
                   "schedule/problem: more than a day back to back");
    check::is_true(qs::problem(entry("A", Q, 48, 0)).empty(), "schedule/problem: a day is fine");
    check::is_true(!qs::problem(entry("C", Q, 17, 0)).empty(),
                   "schedule/problem: 17 x C is 51 passes");
    check::is_true(!qs::problem(entry("B", Q, 3, 30)).empty(),
                   "schedule/problem: an hour-long send every half hour overlaps itself");
    check::is_true(qs::problem(entry("A", Q, 3, 45)).empty(),
                   "schedule/problem: A every 45 minutes is fine");
    check::is_true(!qs::problem(entry("A", Q, 3, 50)).empty(),
                   "schedule/problem: not a whole number of quarter hours");
    check::is_true(!qs::problem(entry("A", Q + 60.0, 1, 0)).empty(),
                   "schedule/problem: off the quarter hour");
    qs::Entry far = entry("A", Q, 1, 0);
    far.freq_hz = 2800.0;
    check::is_true(!qs::problem(far).empty(), "schedule/problem: carrier out of the band");
    check::is_true(qs::earliest_slot(Q - qs::ADD_BY_S) == Q &&
                       qs::earliest_slot(Q - qs::ADD_BY_S + 1.0) == Q + 900.0,
                   "schedule/earliest: up to 30 s before the quarter hour");
}

void test_clashes() {
    const std::vector<qs::Entry> one{entry("A", Q, 1, 0, "first")};
    // Straight after: it joins the run on the air, so no lead of its own.
    check::is_true(!qs::clash(entry("B", Q + 1800.0, 1, 0, "next"), one, Q - H).has_value(),
                   "schedule/clash: another picture may follow straight on");
    check::is_true(!qs::clash(entry("A", Q - 1800.0, 1, 0, "before"), one, Q - H).has_value(),
                   "schedule/clash: or come straight before");
    check::is_true(qs::follows_on(entry("B", Q + 1800.0, 1, 0, "next"), 0, one, Q - H),
                   "schedule/clash: and is known to follow on");
    check::is_true(qs::clash(entry("A", Q + 1800.0, 1, 0, "next"), one, Q - 11.0 - 30.0)
                       .has_value(),
                   "schedule/clash: but not once the last pass is about to key");
    check::is_true(qs::clash(entry("A", Q + 900.0, 1, 0, "inside"), one, Q - H).has_value(),
                   "schedule/clash: a send inside another is still refused");
    check::is_true(!qs::clash(entry("A", Q + 2700.0, 1, 0, "later"), one, Q - H).has_value(),
                   "schedule/clash: a quarter hour's gap is enough");
    // A every 2 h from Q against A every 3 h from Q + 1 h: they meet at
    // Q + 4 h, the second send of the latter.
    const std::vector<qs::Entry> even{entry("A", Q, 0, 120, "even")};
    const auto c = qs::clash(entry("A", Q + H, 0, 180, "thirds"), even, Q - H);
    check::is_true(c && c->slot == Q + 4 * H && c->other_slot == Q + 4 * H &&
                       c->other == "even",
                   "schedule/clash: two repeating schedules meet where they meet");
    std::vector<qs::Entry> paused = even;
    paused[0].enabled = false;
    check::is_true(!qs::clash(entry("A", Q + H, 0, 180), paused, Q - H),
                   "schedule/clash: a paused entry is no obstacle");
    // A send by hand that runs over a scheduled one.
    const auto w = qs::clash_with(even, Q + 1.5 * H, Q + 2.5 * H, Q);
    check::is_true(w && w->send == 1 && w->slot == Q + 2 * H,
                   "schedule/clash: a send by hand over the 03:00Z send");
    check::is_true(!qs::clash_with(even, Q + 0.5 * H, Q + 1.5 * H, Q),
                   "schedule/clash: but not in the gap");
}

void test_upcoming_and_the_file() {
    std::vector<qs::Entry> list{entry("B", Q + H, 3, 180, "b"), entry("A", Q, 2, 0, "aa")};
    list[1].next = 1;
    const std::vector<qs::Upcoming> u = qs::upcoming(list, Q - H, 7 * 86400.0, 4);
    check::is_true(u.size() == 4 && u[0].entry == 1 && u[0].slot == Q + 1800.0 &&
                       u[1].entry == 0 && u[1].slot == Q + H && u[2].slot == Q + 4 * H &&
                       u[3].slot == Q + 7 * H,
                   "schedule/upcoming: what is left of each, in time order");
    list[0].enabled = false;
    list[0].freq_hz = 987.0;
    list[0].waveform = "L";
    const std::vector<qs::Entry> back = qs::from_json(qs::to_json(list));
    check::is_true(back.size() == 2 && back[0].id == "b" && !back[0].enabled &&
                       back[0].freq_hz == 987.0 && back[0].every_min == 180 &&
                       back[0].count == 3 && back[1].next == 1 && back[1].first_slot == Q &&
                       back[1].mode == "A" && back[0].waveform == "L" && back[1].waveform == "CE",
                   "schedule/file: every field survives the round trip");
    check::equal(qs::describe(back[0]), std::string("waveform L mode B every 3 hours, 3 times"),
                 "schedule/file: and an L send says so");
    {
        // The same entry as a schedule from before L: no waveform field.
        std::vector<qs::Entry> one{list[1]};
        one[0].waveform = "L";
        QJsonObject root =
            QJsonDocument::fromJson(QByteArray::fromStdString(qs::to_json(one))).object();
        QJsonArray list_json = root.value(QStringLiteral("entries")).toArray();
        QJsonObject only = list_json.at(0).toObject();
        only.remove(QStringLiteral("waveform"));
        list_json[0] = only;
        root[QStringLiteral("entries")] = list_json;
        const QString old = QString::fromUtf8(QJsonDocument(root).toJson());
        const auto back_old = qs::from_json(old.toStdString());
        check::is_true(!old.contains(QStringLiteral("waveform")) && back_old.size() == 1 &&
                           back_old[0].waveform == "CE",
                       "schedule/file: a schedule from before L is CE");
    }
    check::is_true(qs::from_json("not json").empty(), "schedule/file: a broken file is empty");
}

struct Fake {
    double now = Q - H;
    bool chain_ok = true;
    std::vector<std::vector<qrss_tx::Pass>> chained;
    QrssScheduler::Start answer = QrssScheduler::Start::Started;
    std::string why;
    std::vector<std::vector<qrss_tx::Pass>> runs;
    std::vector<std::pair<int, std::string>> log;
};

QrssScheduler* make(Fake& fake, const std::filesystem::path& file) {
    auto* s = new QrssScheduler(
        file,
        [&fake](const qs::Entry&, const std::vector<qrss_tx::Pass>& passes, std::string* why) {
            if (fake.answer == QrssScheduler::Start::Started) fake.runs.push_back(passes);
            *why = fake.why;
            return fake.answer;
        },
        [&fake] { return fake.now; }, 0);
    s->set_chainer([&fake](const qs::Entry&, const std::vector<qrss_tx::Pass>& passes,
                           std::string* why) {
        if (!fake.chain_ok) {
            *why = "the send on the air has already keyed its last pass";
            return false;
        }
        fake.chained.push_back(passes);
        return true;
    });
    QObject::connect(s, &QrssScheduler::logged, [&fake](int sev, const QString& m) {
        fake.log.push_back({sev, m.toStdString()});
    });
    return s;
}

bool logged(const Fake& f, const std::string& part) {
    for (const auto& [sev, m] : f.log) {
        if (m.find(part) != std::string::npos) return true;
    }
    return false;
}

void test_the_scheduler_starts_what_is_due(const std::filesystem::path& dir) {
    Fake fake;
    const std::filesystem::path file = dir / "a" / "qrss_schedule.json";
    std::unique_ptr<QrssScheduler> s(make(fake, file));
    const images::Picture pic(64, 48);

    qs::Entry aa = entry("A", Q, 2, 0, "two A");
    aa.id.clear();
    aa.picture.clear();
    check::is_true(s->add(aa, pic).empty(), "scheduler/add: 2 x A at 01:00Z");
    check::is_true(s->entries().size() == 1 &&
                       std::filesystem::exists(s->entries()[0].picture) &&
                       std::filesystem::exists(file),
                   "scheduler/add: the picture and the schedule are on disk");
    check::is_true(!s->add(entry("A", Q + 1800.0, 1, 0, "clash"), pic).empty(),
                   "scheduler/add: an overlapping send is refused");
    fake.now = Q + 900.0 - qs::ADD_BY_S + 1.0;
    check::is_true(!s->add(entry("A", Q + 900.0, 1, 0, "soon"), pic).empty(),
                   "scheduler/add: one under 30 s before its quarter hour is refused");
    fake.now = Q - H;

    s->tick();
    check::is_true(fake.runs.empty(), "scheduler/tick: nothing due an hour early");
    fake.now = Q - 11.0 - qs::START_LEAD_S - 1.0;
    s->tick();
    check::is_true(fake.runs.empty(), "scheduler/tick: or a second before its lead");
    fake.now += 2.0;
    s->tick();
    check::is_true(fake.runs.size() == 1 && fake.runs[0].size() == 2 &&
                       fake.runs[0][0].slot == Q && fake.runs[0][1].slot == Q + 1800.0 &&
                       fake.runs[0][1].segment == 0,
                   "scheduler/tick: both sends handed over as one run of two passes");
    check::is_true(qs::finished(s->entries()[0]) && s->active() == s->entries()[0].id,
                   "scheduler/tick: the entry is done and on the air");
    s->tick();
    check::equal(fake.runs.size(), std::size_t{1}, "scheduler/tick: and started once");
    s->run_finished();
    check::is_true(s->active().empty(), "scheduler/tick: until the transmitter finishes");

    // A busy transmitter: waited for once in the log, then missed.
    fake.log.clear();
    check::is_true(s->add(entry("B", Q + 3 * H, 2, 180, "b"), pic).empty(),
                   "scheduler/busy: B at 04:00Z and 07:00Z");
    fake.answer = QrssScheduler::Start::Busy;
    fake.why = "the transmitter is busy";
    fake.now = Q + 3 * H - 11.0 - 200.0;
    s->tick();
    s->tick();
    int waiting = 0;
    for (const auto& [sev, m] : fake.log) waiting += m.find("is waiting") != std::string::npos;
    check::equal(waiting, 1, "scheduler/busy: logged once, not every tick");
    fake.now = Q + 3 * H - 11.0 - qs::MIN_LEAD_S + 1.0;
    s->tick();
    check::is_true(s->entries()[1].next == 1 && logged(fake, "missed 1 send") &&
                       logged(fake, "the transmitter is busy"),
                   "scheduler/busy: too late to start is missed, with the reason");

    // A failure (no callsign, say) costs that send and no more.
    fake.answer = QrssScheduler::Start::Failed;
    fake.why = "no callsign";
    fake.now = Q + 6 * H - 11.0 - 200.0;
    s->tick();
    check::is_true(s->entries()[1].next == 2 && logged(fake, "was not sent: no callsign"),
                   "scheduler/failed: reported and passed over");

    // Another listener of the same file sees it all.
    Fake other;
    std::unique_ptr<QrssScheduler> again(make(other, file));
    check::is_true(again->entries().size() == 2 && again->entries()[1].next == 2 &&
                       qs::finished(again->entries()[0]),
                   "scheduler/file: progress survives a restart");
    const std::string png = again->entries()[0].picture;
    again->remove(again->entries()[0].id);
    check::is_true(again->entries().size() == 1 && !std::filesystem::exists(png),
                   "scheduler/remove: the entry and its picture go");
}

// Two different Sends back to back: the second joins the first's run.
void test_a_late_send_starts_at_once(const std::filesystem::path& dir) {
    Fake fake;
    std::unique_ptr<QrssScheduler> s(make(fake, dir / "late" / "qrss_schedule.json"));
    fake.now = Q - qs::ADD_BY_S;
    check::is_true(s->add(entry("A", Q, 1, 0, "late"), images::Picture(64, 48)).empty(),
                   "scheduler/late: a send 30 s before its quarter hour is accepted");
    check::is_true(fake.runs.size() == 1 && fake.runs[0][0].slot == Q,
                   "scheduler/late: and handed to the transmitter straight away");

    // Removing it stops that run rather than leaving it to key a picture
    // that is no longer on the schedule (Jeff, 2026-10-12).
    int cancels = 0;
    s->set_canceller([&cancels] { ++cancels; });
    s->remove(s->entries()[0].id);
    check::is_true(cancels == 1 && logged(fake, "stopped its send"),
                   "scheduler/remove: removing the entry on the air stops its run");
    s->run_finished();
    fake.now = Q - 20.0;
    s->remove("no such id");
    check::equal(cancels, 1, "scheduler/remove: nothing else stops a run");
}

void test_sends_follow_straight_on(const std::filesystem::path& dir) {
    Fake fake;
    std::unique_ptr<QrssScheduler> s(make(fake, dir / "follow" / "qrss_schedule.json"));
    const images::Picture pic(64, 48);
    check::is_true(s->add(entry("A", Q, 1, 0, "one"), pic).empty(), "follow/add: A at 01:00Z");
    check::is_true(s->add(entry("B", Q + 1800.0, 1, 0, "two"), pic).empty(),
                   "follow/add: B straight after, at 01:30Z");
    check::is_true(s->add(entry("A", Q + 5400.0, 1, 0, "three"), pic).empty(),
                   "follow/add: and A straight after that, at 02:30Z");
    fake.now = Q - 11.0 - qs::START_LEAD_S + 1.0;
    s->tick();
    check::is_true(fake.runs.size() == 1 && fake.runs[0].size() == 1 && fake.runs[0][0].slot == Q,
                   "follow/tick: the first starts its own run");
    check::is_true(fake.chained.size() == 2 && fake.chained[0].size() == 2 &&
                       fake.chained[0][0].slot == Q + 1800.0 &&
                       fake.chained[0][1].segment == 1 && fake.chained[1].size() == 1 &&
                       fake.chained[1][0].slot == Q + 5400.0,
                   "follow/tick: both others join it, in order");
    const std::string one = s->entries()[0].id, two = s->entries()[1].id,
                      three = s->entries()[2].id;
    check::is_true(s->on_air(one) && s->on_air(two) && s->on_air(three) && s->active() == one,
                   "follow/tick: all three are on the air");
    check::is_true(!s->change(two, [](qs::Entry& e) { e.freq_hz = 900.0; }).empty(),
                   "follow/tick: one that joined is not changed under the run");
    fake.now = Q + 1800.0 - 11.0 - qs::START_LEAD_S + 1.0;
    s->tick();
    check::equal(fake.runs.size(), std::size_t{1}, "follow/tick: nothing starts a run of its own");
    s->run_finished();
    check::is_true(!s->on_air(two), "follow/done: off the air with the run");

    // The run will not take it: logged once, then missed like any send
    // the transmitter was too busy for.
    Fake late;
    std::unique_ptr<QrssScheduler> t(make(late, dir / "late" / "qrss_schedule.json"));
    check::is_true(t->add(entry("A", Q, 1, 0, "one"), pic).empty() &&
                       t->add(entry("A", Q + 1800.0, 1, 0, "two"), pic).empty(),
                   "follow/refused: two As back to back");
    late.chain_ok = false;
    late.now = Q - 11.0 - qs::START_LEAD_S + 1.0;
    t->tick();
    t->tick();
    check::is_true(late.chained.empty() && logged(late, "could not follow on"),
                   "follow/refused: reported");
    int reports = 0;
    for (const auto& [sev, m] : late.log) reports += m.find("could not follow on") != std::string::npos;
    check::equal(reports, 1, "follow/refused: once");
}

void test_missed_and_paused(const std::filesystem::path& dir) {
    Fake fake;
    const std::filesystem::path file = dir / "b" / "qrss_schedule.json";
    std::unique_ptr<QrssScheduler> s(make(fake, file));
    const images::Picture pic(64, 48);
    check::is_true(s->add(entry("A", Q, 0, 60, "hourly"), pic).empty(),
                   "scheduler/missed: hourly from 01:00Z");
    // The app was not running from 00:00 to 03:20: three sends missed.
    fake.now = Q + 2 * H + 1200.0;
    s->tick();
    check::is_true(s->entries()[0].next == 3 && logged(fake, "missed 3 send(s)") &&
                       logged(fake, "the app was not running"),
                   "scheduler/missed: logged as one line, the next is 04:00Z");
    check::is_true(fake.runs.empty(), "scheduler/missed: and nothing started late");
    fake.log.clear();
    s->set_enabled(s->entries()[0].id, false);
    fake.now = Q + 6 * H;
    s->tick();
    check::is_true(s->entries()[0].next == 3 && !logged(fake, "missed"),
                   "scheduler/paused: a paused entry neither sends nor misses");
    check::is_true(s->set_enabled(s->entries()[0].id, true).empty(),
                   "scheduler/paused: resumed");
    check::is_true(s->entries()[0].next == 7 && !logged(fake, "missed") &&
                       logged(fake, "next send 08:00Z"),
                   "scheduler/paused: what fell due while paused is skipped quietly");
}

// A solid red picture file of the given size.
std::string red_png(const std::filesystem::path& dir, int w, int h) {
    images::Picture p(w, h);
    for (std::size_t i = 0; i < p.rgb.size(); i += 3) p.rgb[i] = 255;
    const std::string path = (dir / ("red-" + std::to_string(w) + "x" + std::to_string(h) +
                                     ".png")).string();
    images::save_png(p, path);
    return path;
}

// Answers the framing dialog, if one opens, the next time events are
// processed: zooms all the way out (the whole picture, padded) and
// presses OK. Counts how many opened.
void answer_framing(int* opened) {
    QTimer::singleShot(0, [opened] {
        auto* dialog = qobject_cast<CropDialog*>(QApplication::activeModalWidget());
        if (!dialog) return;
        ++*opened;
        if (auto* view = dialog->findChild<CropView*>()) {
            images::Framing all;
            all.zoom = view->min_zoom();
            view->set_framing(all);
        }
        dialog->accept();
    });
}

// A 640 x 480 picture of one colour.
images::Picture solid(int r, int g, int b) {
    images::Picture p(images::IMG_W, images::IMG_H);
    for (std::size_t i = 0; i < p.rgb.size(); i += 3) {
        p.rgb[i] = static_cast<std::uint8_t>(r);
        p.rgb[i + 1] = static_cast<std::uint8_t>(g);
        p.rgb[i + 2] = static_cast<std::uint8_t>(b);
    }
    return p;
}

overlay::Doc caption(const std::string& text) {
    overlay::Doc doc;
    overlay::TextItem t;
    t.text = text;
    doc.items.emplace_back(t);
    return doc;
}

void test_the_sends_store(const std::filesystem::path& dir) {
    const std::filesystem::path where = dir / "sends";
    QrssSends sends(where);
    check::is_true(sends.sends().empty(), "sends: a new store is empty");
    int changed = 0;
    QObject::connect(&sends, &QrssSends::changed, [&changed] { ++changed; });
    const std::string a = sends.add("Sunset", solid(200, 0, 0), caption("AG7EW"), solid(255, 0, 0));
    const std::string b = sends.add("CQ card", solid(0, 0, 200), {}, solid(0, 0, 255));
    check::is_true(!a.empty() && !b.empty() && a != b, "sends: two added, each with its own id");
    check::equal(changed, 2, "sends: and each says so");
    check::is_true(sends.sends().size() == 2, "sends: both listed");
    const QrssSend* sa = sends.find(a);
    check::is_true(sa != nullptr && std::filesystem::exists(sa->picture) &&
                       std::filesystem::exists(sa->base) &&
                       std::filesystem::exists(where / (a + ".json")),
                   "sends: the picture, its base and its composition are on disk");

    // Read back by a store that never saw them added: what survives a
    // restart of the app.
    QrssSends again(where);
    check::is_true(again.sends().size() == 2 && again.find(a) != nullptr &&
                       again.find(a)->label == "Sunset",
                   "sends: read back from disk");
    const overlay::Doc doc = again.load_doc(a);
    const auto* t = doc.items.empty() ? nullptr : std::get_if<overlay::TextItem>(&doc.items[0]);
    check::is_true(t != nullptr && t->text == "AG7EW", "sends: the composition can be edited again");
    const auto base = again.load_base(a);
    check::is_true(base && base->width == images::IMG_W && base->rgb[0] == 200,
                   "sends: on the picture it was made on");
    check::is_true(again.update(a, "Sunset 2", solid(0, 200, 0), caption("CN85"), solid(0, 255, 0)),
                   "sends: updated");
    check::is_true(again.find(a)->label == "Sunset 2" &&
                       images::load(again.find(a)->picture.string()).rgb[1] == 255,
                   "sends: the update replaces the picture and the name");
    check::is_true(again.rename(b, "CQ"), "sends: renamed");
    again.remove(a);
    check::is_true(again.find(a) == nullptr && !std::filesystem::exists(where / (a + ".png")) &&
                       !std::filesystem::exists(where / (a + ".json")),
                   "sends: removed, files and all");
    QrssSends third(where);
    check::is_true(third.sends().size() == 1 && third.sends()[0].label == "CQ",
                   "sends: and the rename and removal stick");
}

void test_change_and_truncate(const std::filesystem::path& dir) {
    Fake fake;
    std::unique_ptr<QrssScheduler> s(make(fake, dir / "ch" / "qrss_schedule.json"));
    const images::Picture pic(64, 48);
    qs::Entry e = entry("B", Q, 3, 120, "three B");
    e.id.clear();
    e.picture.clear();
    check::is_true(s->add(e, pic).empty(), "change: 3 x B every 2 hours from 01:00Z");
    qs::Entry other = entry("A", Q + 5.25 * H, 1, 0, "later");
    other.id.clear();
    other.picture.clear();
    check::is_true(s->add(other, pic).empty(), "change: and one A at 06:15Z");
    const std::string id = s->entries()[0].id;

    check::is_true(s->change(id, [](qs::Entry& x) { x.freq_hz = 1600; }).empty() &&
                       s->entries()[0].freq_hz == 1600,
                   "change: a new carrier");
    check::is_true(!s->change(id, [](qs::Entry& x) { x.mode = "C"; }).empty() &&
                       s->entries()[0].mode == "B",
                   "change: mode C would run into the 06:15Z send: refused, unchanged");
    check::is_true(!s->change(id, [](qs::Entry& x) { x.first_slot = Q - 2 * H; }).empty() &&
                       s->entries()[0].first_slot == Q,
                   "change: moving into the past is refused");
    check::is_true(!s->change(id, [](qs::Entry& x) { x.every_min = 30; }).empty(),
                   "change: B every 30 minutes is refused");
    check::is_true(!s->change("nobody", [](qs::Entry&) {}).empty(),
                   "change: an entry that is not there");

    // The first send goes on the air; it may not be changed, and once it
    // is over only the sends still to come change.
    fake.now = Q - 200;
    s->tick();
    check::is_true(s->active() == id && s->entries()[0].next == 1, "change: the first send started");
    check::is_true(!s->change(id, [](qs::Entry& x) { x.mode = "A"; }).empty(),
                   "change: not while it is on the air");
    s->run_finished();
    fake.now = Q + H;
    check::is_true(s->change(id, [](qs::Entry& x) { x.mode = "A"; }).empty(),
                   "change: after it, the rest become mode A");
    const qs::Entry& r = s->entries()[0];
    check::is_true(r.next == 0 && r.count == 2 && r.first_slot == Q + 2 * H && r.mode == "A",
                   "change: rebased on the two sends still to come");

    s->truncate(id, 1);
    check::is_true(s->entries()[0].count == 1, "truncate: stops after the first remaining send");
    s->truncate(id, 0);
    check::is_true(s->entries().size() == 1 && s->entries()[0].label == "later",
                   "truncate: from the next send on removes the entry");
}

void test_the_window(const std::filesystem::path& dir) {
    Fake fake;
    std::unique_ptr<QrssScheduler> s(make(fake, dir / "c" / "qrss_schedule.json"));
    QrssSends sends(dir / "c" / "qrss_sends");
    const std::string red = sends.add("Red", solid(200, 0, 0), {}, solid(255, 0, 0));
    const std::string blue = sends.add("Blue", solid(0, 0, 200), {}, solid(0, 0, 255));
    QrssScheduleWindow w(s.get(), &sends,
                         [] { return QrssScheduleWindow::Defaults{"B", 1234.0}; }, {});
    w.resize(1000, 700);
    w.show();
    QApplication::processEvents();
    auto* list = w.findChild<QListWidget*>(QStringLiteral("schedule_sends"));
    auto* form = w.findChild<QGroupBox*>(QStringLiteral("schedule_form"));
    auto* mode = w.findChild<QComboBox*>(QStringLiteral("schedule_mode"));
    auto* freq = w.findChild<QSpinBox*>(QStringLiteral("schedule_freq"));
    auto* count = w.findChild<QSpinBox*>(QStringLiteral("schedule_count"));
    auto* every = w.findChild<QComboBox*>(QStringLiteral("schedule_every"));
    auto* summary = w.findChild<QLabel*>(QStringLiteral("schedule_summary"));
    auto* add = w.findChild<QPushButton*>(QStringLiteral("schedule_add"));
    const bool all = list && form && mode && freq && count && every && summary && add;
    check::is_true(all, "window: every control is there");
    if (!all) return;
    check::is_true(w.findChild<QListWidget*>(QStringLiteral("schedule_coming")) == nullptr &&
                       w.findChild<QTreeWidget*>(QStringLiteral("schedule_table")) == nullptr,
                   "window: the timeline replaces the coming-up and scheduled lists");
    check::equal(list->count(), 2, "window: both Sends listed");
    check::is_true(!form->isEnabled() && !add->isEnabled(),
                   "window: nothing to schedule until a Send is chosen");

    w.select_send(blue);
    QApplication::processEvents();
    check::is_true(w.selected_send() == blue && form->isEnabled(),
                   "window: choosing a Send enables the form");
    check::is_true(mode->currentData().toString() == QStringLiteral("B") && freq->value() == 1234,
                   "window: mode and carrier start as the Transmit pane has them");
    check::is_true(w.chosen_slot() == qs::earliest_slot(fake.now),
                   "window: the first send starts at the earliest quarter hour");
    check::is_true(summary->text().contains(QStringLiteral("2 x mode B, back to back")) &&
                       add->isEnabled(),
                   "window: the default is two sends back to back");
    // The waveform: CE unless chosen, L saying what it needs.
    auto* waveform = w.findChild<QComboBox*>(QStringLiteral("schedule_waveform"));
    auto* note = w.findChild<QLabel*>(QStringLiteral("schedule_waveform_note"));
    check::is_true(waveform && note && waveform->currentData().toString() == QStringLiteral("CE"),
                   "window: CE unless L is chosen");
    if (waveform && note) {
        waveform->setCurrentIndex(waveform->findData(QStringLiteral("L")));
        check::is_true(note->text().contains(QStringLiteral("ALC")) &&
                           note->text().contains(QStringLiteral("current code")) &&
                           summary->text().contains(QStringLiteral("waveform L mode B")),
                       "window: choosing L says what it needs");
        waveform->setCurrentIndex(waveform->findData(QStringLiteral("CE")));
    }
    mode->setCurrentIndex(mode->findData(QStringLiteral("A")));
    count->setValue(0);
    every->setCurrentIndex(every->findData(120));
    check::is_true(summary->text().contains(QStringLiteral("(alternating hours), until removed")),
                   "window: every other hour, until removed");
    every->setCurrentIndex(every->findData(0));
    check::is_true(!add->isEnabled() && summary->text().startsWith(QStringLiteral("Can't add")),
                   "window: back to back endless is refused, and says why");

    // A slot clicked on the timeline is where the form starts.
    w.set_chosen_slot(Q);
    count->setValue(2);
    add->click();
    QApplication::processEvents();
    check::is_true(s->entries().size() == 1 && s->entries()[0].mode == "A" &&
                       s->entries()[0].count == 2 && s->entries()[0].freq_hz == 1234.0 &&
                       s->entries()[0].send_id == blue && s->entries()[0].label == "Blue" &&
                       s->entries()[0].first_slot == Q,
                   "window: Add schedules the chosen Send as the form says");
    if (s->entries().size() != 1) return;
    const std::string id = s->entries()[0].id;
    {
        // What the app finds after a restart: the schedule, its Send and
        // its picture, from disk.
        Fake later;
        std::unique_ptr<QrssScheduler> again(make(later, dir / "c" / "qrss_schedule.json"));
        check::is_true(again->entries().size() == 1 && again->entries()[0].id == id &&
                           again->entries()[0].send_id == blue &&
                           std::filesystem::exists(again->entries()[0].picture),
                       "window: the schedule and its link to the Send survive a restart");
    }
    check::is_true(images::load(s->entries()[0].picture).rgb[2] == 255,
                   "window: with its own copy of the Send's picture");

    // The Send on the schedule is tinted in the list; the other is not.
    bool blue_marked = false;
    bool red_marked = false;
    for (int i = 0; i < list->count(); ++i) {
        const std::string sid = list->item(i)->data(Qt::UserRole).toString().toStdString();
        const bool marked = list->item(i)->data(Qt::UserRole + 1).toBool() &&
                            list->item(i)->text().contains(QStringLiteral("scheduled"));
        (sid == blue ? blue_marked : red_marked) = marked;
    }
    check::is_true(blue_marked && !red_marked, "window: the scheduled Send is marked in the list");

    // Its two sends are blocks on the timeline, a half hour each.
    QrssTimeline* t = w.timeline();
    const auto& blocks = t->blocks();
    check::equal(blocks.size(), std::size_t{2}, "timeline: one block per send");
    if (blocks.size() != 2) return;
    check::is_true(blocks[0].slot == Q && blocks[0].end == Q + 1800.0 &&
                       blocks[1].slot == Q + 1800.0 && blocks[0].send_id == blue,
                   "timeline: back to back, half an hour each, from its Send");
    check::is_true(t->block_caption(blocks[0]) == QStringLiteral("A 1/2") &&
                       t->block_caption(blocks[1]) == QStringLiteral("A 2/2"),
                   "timeline: captioned with the mode and which repeat");
    const QRect r0 = t->block_rect(blocks[0]);
    check::is_true(r0.width() > 1.5 * QrssTimeline::CELL_W && t->block_at(r0.center()) == 0 &&
                       t->slot_at(r0.left() + 2) == Q,
                   "timeline: a block spans its two quarter hours, where its slot is");
    QrssTimeline::Block endless = blocks[0];
    endless.count = 0;
    endless.send = 4;
    check::is_true(t->block_caption(endless) == QStringLiteral("A #5"),
                   "timeline: an until-removed series counts up");

    // Clicking a block selects its entry and its Send.
    w.select_send(red);
    emit t->blockClicked(QString::fromStdString(id), 1);
    QApplication::processEvents();
    check::is_true(w.selected_send() == blue && w.selected_entry() == id,
                   "timeline: a click selects the entry and its Send");

    // The right-click menu.
    check::is_true(QrssScheduleWindow::repeat_choices("A").front() == 30 &&
                       QrssScheduleWindow::repeat_choices("B").front() == 60 &&
                       QrssScheduleWindow::repeat_choices("C").front() == 90,
                   "menu: repeat no sooner than the mode's own length");
    std::unique_ptr<QMenu> menu(w.menu_for(id, 1));
    auto* change = menu->findChild<QMenu*>(QStringLiteral("menu_change_mode"));
    auto* repeat = menu->findChild<QMenu*>(QStringLiteral("menu_repeat_in"));
    check::is_true(menu->findChild<QAction*>(QStringLiteral("menu_remove")) && change && repeat &&
                       menu->findChild<QAction*>(QStringLiteral("menu_remove_from")),
                   "menu: Remove, Remove from here, Change Mode, Repeat in");
    if (!(change && repeat)) return;
    check::equal(change->actions().size(), qsizetype{3}, "menu: three modes");
    QAction* to_b = nullptr;
    for (QAction* a : change->actions()) {
        if (a->text().startsWith(QStringLiteral("Mode B"))) to_b = a;
    }
    check::is_true(to_b != nullptr && !to_b->isChecked(), "menu: mode B on offer");
    if (to_b) to_b->trigger();
    check::is_true(s->entries()[0].mode == "B" && s->entries()[0].count == 2,
                   "menu: Change Mode makes it 2 x B");
    menu.reset(w.menu_for(id, 0));
    repeat = menu->findChild<QMenu*>(QStringLiteral("menu_repeat_in"));
    bool short_offered = false;
    QAction* in_3h = nullptr;
    for (QAction* a : repeat->actions()) {
        if (a->data().toInt() < 60) short_offered = true;
        if (a->data().toInt() == 180) in_3h = a;
    }
    check::is_true(!short_offered && in_3h != nullptr,
                   "menu: a mode B send repeats in an hour at the soonest");
    check::is_true(menu->findChild<QAction*>(QStringLiteral("menu_remove_from")) == nullptr,
                   "menu: from the first send, Remove is all there is");
    if (in_3h) in_3h->trigger();
    check::is_true(s->entries()[0].every_min == 180 && s->entries()[0].count == 2,
                   "menu: Repeat in 180 minutes");
    check::is_true(w.change_mode(id, "C").empty() && s->entries()[0].mode == "C",
                   "menu: C fits every 3 hours");
    check::is_true(!w.repeat_every(id, 60).empty() && s->entries()[0].every_min == 180,
                   "menu: but not every hour, and says why");

    // Pause and Resume, which the scheduled list used to have.
    {
        menu.reset(w.menu_for(id, 0));
        QAction* pause = menu->findChild<QAction*>(QStringLiteral("menu_pause"));
        check::is_true(pause && pause->text() == QStringLiteral("Pause"), "menu: Pause on offer");
        if (pause) pause->trigger();
        check::is_true(!s->entries()[0].enabled, "menu: Pause pauses it");
        menu.reset(w.menu_for(id, 0));
        pause = menu->findChild<QAction*>(QStringLiteral("menu_pause"));
        check::is_true(pause && pause->text() == QStringLiteral("Resume"),
                       "menu: and then Resume");
        if (pause) pause->trigger();
        check::is_true(s->entries()[0].enabled, "menu: Resume resumes it");
    }

    // Change Offset moves the carrier of every send, within range.
    menu.reset(w.menu_for(id, 0));
    check::is_true(menu->findChild<QAction*>(QStringLiteral("menu_change_offset")) != nullptr,
                   "menu: Change Offset on offer");
    check::is_true(w.change_offset(id, 1200).empty() && s->entries()[0].freq_hz == 1200.0,
                   "menu: Change Offset moves the carrier");
    check::is_true(!w.change_offset(id, 5000).empty() && s->entries()[0].freq_hz == 1200.0,
                   "menu: but not outside the passband, and says why");

    // Change Waveform: CE to L and back, the picture unchanged.
    menu.reset(w.menu_for(id, 0));
    check::is_true(menu->findChild<QMenu*>(QStringLiteral("menu_change_waveform")) != nullptr,
                   "menu: Change Waveform on offer");
    check::is_true(w.change_waveform(id, "L").empty() && s->entries()[0].waveform == "L",
                   "menu: Change Waveform to L");
    check::is_true(w.change_waveform(id, "CE").empty() && s->entries()[0].waveform == "CE",
                   "menu: and back to CE");

    // A single send asked to repeat becomes two.
    w.select_send(red);
    w.set_chosen_slot(Q + 12 * H);
    mode->setCurrentIndex(mode->findData(QStringLiteral("A")));
    count->setValue(1);
    add->click();
    QApplication::processEvents();
    check::equal(s->entries().size(), std::size_t{2}, "window: a second entry, one A");
    if (s->entries().size() != 2) return;
    const std::string single = s->entries()[1].id;
    check::is_true(w.repeat_every(single, 45).empty() && s->entries()[1].count == 2 &&
                       s->entries()[1].every_min == 45,
                   "menu: Repeat in on a single send makes it two");

    // Dragging a send moves the whole series.
    check::is_true(w.move_send(id, 1, Q + 6 * H).empty() &&
                       s->entries()[0].first_slot == Q + 3 * H,
                   "drag: the second send to 07:00Z moves the first to 04:00Z");
    check::is_true(!w.move_send(id, 0, Q + 12 * H).empty() &&
                       s->entries()[0].first_slot == Q + 3 * H,
                   "drag: onto another send is refused, unchanged");
    check::is_true(w.remove_from(id, 1).empty() && s->entries()[0].count == 1,
                   "menu: Remove from the second send keeps the first");
    check::is_true(w.remove_entry(id).empty() && s->entries().size() == 1 &&
                       s->entries()[0].id == single,
                   "menu: Remove removes it");
    // A Send dragged from the list onto the timeline is scheduled on the
    // quarter hour it is dropped in, as the form says.
    {
        QModelIndex blue_index;
        for (int i = 0; i < list->count(); ++i) {
            if (list->item(i)->data(Qt::UserRole).toString().toStdString() == blue) {
                blue_index = list->model()->index(i, 0);
            }
        }
        std::unique_ptr<QMimeData> mime(list->model()->mimeData({blue_index}));
        check::is_true(mime && mime->data(QString::fromLatin1(QrssTimeline::SEND_MIME)) ==
                                   QByteArray::fromStdString(blue),
                       "drop: a Send drags as its id");
        mode->setCurrentIndex(mode->findData(QStringLiteral("A")));
        count->setValue(1);
        const QPoint at(t->x_of(Q + 20 * H) + QrssTimeline::CELL_W / 2, t->height() / 2);
        if (mime) {
            QDragEnterEvent enter(at, Qt::CopyAction, mime.get(), Qt::LeftButton, Qt::NoModifier);
            QApplication::sendEvent(t, &enter);
            check::is_true(enter.isAccepted(), "drop: the timeline takes a Send");
            QDropEvent drop(QPointF(at), Qt::CopyAction, mime.get(), Qt::LeftButton,
                            Qt::NoModifier);
            QApplication::sendEvent(t, &drop);
        }
        QApplication::processEvents();
        const qs::Entry* dropped = nullptr;
        for (const qs::Entry& e : s->entries()) {
            if (e.send_id == blue) dropped = &e;
        }
        check::is_true(dropped != nullptr && dropped->first_slot == Q + 20 * H &&
                           dropped->count == 1 && dropped->mode == "A",
                       "drop: scheduled on the quarter hour it lands in");
        const std::size_t n = s->entries().size();
        check::is_true(!w.drop_send(blue, Q + 20 * H).empty() && s->entries().size() == n,
                       "drop: onto a send already there is refused, and says why");
    }

    w.delete_send(blue);
    QApplication::processEvents();
    check::equal(list->count(), 1, "window: a deleted Send leaves the list");
}

void test_the_editor(const std::filesystem::path& dir) {
    QrssSends sends(dir / "e" / "qrss_sends");
    QrssEditor::Context context;
    context.default_text = "AG7EW";
    QrssEditor e(&sends, context);
    e.show();
    QApplication::processEvents();
    auto* framing = e.findChild<QPushButton*>(QStringLiteral("editor_framing"));
    auto* save_new = e.findChild<QPushButton*>(QStringLiteral("editor_save_new"));
    auto* name = e.findChild<QLineEdit*>(QStringLiteral("editor_name"));
    check::is_true(framing && save_new && name && e.findChild<OverlayEditor*>(),
                   "editor: the canvas and its controls are there");
    if (!(framing && save_new && name)) return;

    e.start_blank();
    check::is_true(!framing->isEnabled() && !save_new->isVisible(),
                   "editor: a blank picture, nothing to frame, a new Send");
    int opened = 0;
    answer_framing(&opened);
    check::is_true(e.load_picture(QString::fromStdString(red_png(dir, 640, 480))),
                   "editor: a 640 x 480 picture loads");
    QApplication::processEvents();
    check::equal(opened, 0, "editor: and goes straight in");
    check::is_true(framing->isEnabled(), "editor: but can still be framed");
    check::is_true(name->text() == QStringLiteral("red-640x480"), "editor: named after its file");

    answer_framing(&opened);
    e.load_picture(QString::fromStdString(red_png(dir, 800, 450)));
    QApplication::processEvents();
    check::equal(opened, 1, "editor: 16:9 opens the framing dialog");
    answer_framing(&opened);
    framing->click();
    QApplication::processEvents();
    check::equal(opened, 2, "editor: Framing... opens it again");

    e.editor()->add_text("CQ");
    name->setText(QStringLiteral("Red card"));
    const std::string id = e.save();
    check::is_true(!id.empty() && sends.find(id) != nullptr && sends.find(id)->label == "Red card",
                   "editor: Save puts it in the Sends");
    if (id.empty()) return;
    const images::Picture sent = images::load(sends.find(id)->picture.string());
    auto is_red = [&sent](int x, int y) {
        return sent.rgb[(static_cast<std::size_t>(y) * sent.width + x) * 3] > 200;
    };
    check::is_true(sent.width == 640 && sent.height == 480 && !is_red(320, 10) &&
                       is_red(5, 300) && is_red(634, 300),
                   "editor: the Send is the framing chosen, whole and padded");
    const images::Picture base = *sends.load_base(id);
    check::is_true(base.rgb != sent.rgb, "editor: the text is in the picture, not the base");

    check::is_true(e.open_send(id) && save_new->isVisible() && e.send_id() == id,
                   "editor: a Send opens again for changes");
    name->setText(QStringLiteral("Red card 2"));
    check::is_true(e.save() == id && sends.sends().size() == 1 &&
                       sends.find(id)->label == "Red card 2",
                   "editor: Save changes replaces it");
    const std::string copy = e.save(true);
    check::is_true(!copy.empty() && copy != id && sends.sends().size() == 2,
                   "editor: Save as new Send keeps the old one");
}

void test_a_change_needs_no_sends(const std::filesystem::path& dir) {
    // Jeff, 2026-10-12: frequency changes run with no sends scheduled.
    Fake fake;
    std::unique_ptr<QrssScheduler> s(make(fake, dir / "lone" / "qrss_schedule.json"));
    std::vector<double> retuned;
    s->set_retuner([&](const qs::FreqChange& c, std::string*) {
        retuned.push_back(c.dial_hz);
        return QrssScheduler::Start::Started;
    });
    qs::FreqChange c;
    c.dial_hz = 5'352'000.0;
    c.at = Q;
    fake.now = Q - H;
    check::is_true(s->add_change(c).empty() && s->entries().empty(),
                   "lone change: scheduled with no sends");
    fake.now = Q + 1.0;
    s->tick();
    check::is_true(retuned.size() == 1 && retuned[0] == 5'352'000.0 && s->changes()[0].done,
                   "lone change: made at its time");
}

void test_frequency_changes(const std::filesystem::path& dir) {
    Fake fake;
    std::unique_ptr<QrssScheduler> s(make(fake, dir / "f" / "qrss_schedule.json"));
    std::vector<double> retuned;
    QrssScheduler::Start retune_answer = QrssScheduler::Start::Started;
    s->set_retuner([&](const qs::FreqChange& c, std::string* why) {
        if (retune_answer == QrssScheduler::Start::Started) retuned.push_back(c.dial_hz);
        *why = "the transmitter is busy";
        return retune_answer;
    });
    const images::Picture pic(64, 48);
    // 2 x A from 01:00Z: on the air (with its lead) ~00:56 to ~01:59:32.
    check::is_true(s->add(entry("A", Q, 2, 0, "aa"), pic).empty(), "change: a send to avoid");

    qs::FreqChange c;
    c.dial_hz = 10'141'000.0;
    c.at = Q + 600.0;
    check::is_true(!s->add_change(c).empty(), "change: not in the middle of a send");
    c.at = Q + 2 * 1800.0 - 11.0 + 1782.7 - 1800.0 + 10.0;   // 10 s after its audio ends
    check::is_true(!s->add_change(c).empty(), "change: nor within 30 s of its end");
    c.at = Q - H + 30.0;
    check::is_true(!s->add_change(c).empty(), "change: nor with less than a minute's notice");
    check::is_true(qs::busy(c).first == c.at - 30.0 && qs::busy(c).second == c.at + 50.0,
                   "change: 30 s clear before, 5 s plus a 45 s tune after");
    c.at = Q + H + 120.0;
    check::is_true(s->add_change(c).empty() && s->changes().size() == 1 &&
                       !s->changes()[0].id.empty(),
                   "change: clear of the send, it is scheduled");
    check::is_true(!s->add(entry("A", Q + H, 1, 0, "over"), pic).empty(),
                   "change: and a send over it is refused");
    check::is_true(!s->change(s->entries()[0].id, [](qs::Entry& e) { e.count = 3; }).empty(),
                   "change: as is lengthening one into it");

    // Saved beside the sends, and read back.
    {
        Fake other;
        std::unique_ptr<QrssScheduler> again(make(other, dir / "f" / "qrss_schedule.json"));
        check::is_true(again->changes().size() == 1 &&
                           again->changes()[0].dial_hz == 10'141'000.0 &&
                           again->changes()[0].at == Q + H + 120.0,
                       "change: kept in the schedule file");
    }

    // Moving it: into a send is refused, elsewhere is fine.
    const std::string id = s->changes()[0].id;
    check::is_true(!s->edit_change(id, [](qs::FreqChange& x) { x.at = Q + 900.0; }).empty() &&
                       s->changes()[0].at == Q + H + 120.0,
                   "change: rescheduling into a send is refused, and changes nothing");
    check::is_true(s->edit_change(id, [](qs::FreqChange& x) { x.at = Q + 2 * H; }).empty() &&
                       s->changes()[0].at == Q + 2 * H,
                   "change: rescheduled");

    // Made at its time, once; a busy transmitter makes it wait.
    fake.now = Q + 2 * H - 1.0;
    s->tick();
    check::is_true(retuned.empty(), "change: not before its time");
    retune_answer = QrssScheduler::Start::Busy;
    fake.now = Q + 2 * H + 1.0;
    s->tick();
    s->tick();
    check::is_true(retuned.empty() && !s->changes()[0].done, "change: waits while busy");
    retune_answer = QrssScheduler::Start::Started;
    s->tick();
    s->tick();
    check::is_true(retuned.size() == 1 && retuned[0] == 10'141'000.0 && s->changes()[0].done,
                   "change: then made, once");
    check::is_true(!s->edit_change(id, [](qs::FreqChange& x) { x.at = Q + 3 * H; }).empty(),
                   "change: a made one cannot be moved");

    // Too late: missed, not made.
    qs::FreqChange late;
    late.dial_hz = 7'040'000.0;
    late.at = Q + 3 * H;
    check::is_true(s->add_change(late).empty(), "change: another");
    fake.now = Q + 3 * H + qs::CHANGE_LATE_S + 1.0;
    s->tick();
    check::is_true(retuned.size() == 1 && s->changes().back().done &&
                       logged(fake, "missed the frequency change"),
                   "change: one long past is missed");
    s->remove_change(s->changes().back().id);
    check::equal(s->changes().size(), std::size_t{1}, "change: removed");

    // The timeline's green marks, and their wider grip.
    QrssTimeline t;
    t.set_view(Q, 16, Q);
    t.set_marks({{"m", Q + 2 * 900.0, 10e6, false}});
    const int x = t.x_of(Q + 2 * 900.0);
    check::is_true(t.mark_at(QPoint(x + QrssTimeline::MARK_GRAB_PX, 40)) == 0 &&
                       t.mark_at(QPoint(x - QrssTimeline::MARK_GRAB_PX, 40)) == 0 &&
                       t.mark_at(QPoint(x + QrssTimeline::MARK_GRAB_PX + 2, 40)) < 0,
                   "timeline: a mark is grabbed a few pixels either side");

    // Dragging a mark asks to move it, to the minute; a made one stays put.
    {
        t.resize(t.sizeHint());
        std::vector<double> moved;
        QObject::connect(&t, &QrssTimeline::markMoveRequested,
                         [&moved](const QString&, double at) { moved.push_back(at); });
        const auto drag = [&t](int from, int to) {
            const auto ev = [&t](QEvent::Type type, int px, Qt::MouseButtons held) {
                QMouseEvent e(type, QPointF(px, 60), QPointF(t.mapToGlobal(QPoint(px, 60))),
                              Qt::LeftButton, held, Qt::NoModifier);
                QApplication::sendEvent(&t, &e);
            };
            ev(QEvent::MouseButtonPress, from, Qt::LeftButton);
            ev(QEvent::MouseMove, (from + to) / 2, Qt::LeftButton);
            ev(QEvent::MouseMove, to, Qt::LeftButton);
            ev(QEvent::MouseButtonRelease, to, Qt::NoButton);
        };
        drag(x, x + QrssTimeline::CELL_W);   // a quarter hour later
        check::is_true(moved.size() == 1 && std::abs(moved[0] - (Q + 3 * 900.0)) < 1.0 &&
                           std::fmod(moved[0], 60.0) == 0.0,
                       "timeline: a dragged change moves, to the minute");
        t.set_marks({{"m", Q + 2 * 900.0, 10e6, true}});
        drag(x, x + QrssTimeline::CELL_W);
        check::equal(moved.size(), std::size_t{1}, "timeline: a made change does not move");
    }

    // "<-30m" labels for the spectrum views.
    check::is_true(dial_mark_label(45) == QStringLiteral("<-45s") &&
                       dial_mark_label(1800) == QStringLiteral("<-30m") &&
                       dial_mark_label(7500) == QStringLiteral("<-2h05m"),
                   "marks: how long ago, in the spectrum views");
}

QList<QByteArray> dial_lines(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    QList<QByteArray> out;
    for (const QByteArray& l : f.readAll().split('\n')) {
        if (!l.isEmpty()) out.push_back(l);
    }
    return out;
}

void test_the_dial_log(const std::filesystem::path& dir) {
    const QString d = QString::fromStdString((dir / "dial").string());
    DialLog log(d);
    check::is_true(log.record(1000.0, 10'141'000.0, "start"), "dial log: the first dial");
    check::is_true(!log.record(1001.0, 10'141'000.4, "radio"), "dial log: the same dial is no change");
    check::is_true(log.record(1002.0, 7'040'000.0, "menu"), "dial log: a change");
    const QList<QByteArray> lines = dial_lines(log.path());
    check::is_true(lines.size() == 2 && lines[1].contains("\"dial_hz\":7040000") &&
                       lines[1].contains("\"t\":1002") && lines[1].contains("\"why\":\"menu\""),
                   "dial log: one JSON line a change, integer Hz");
    DialLog again(d);
    check::is_true(again.last_hz() == 7'040'000LL && !again.record(1003.0, 7'040'000.0, "start"),
                   "dial log: reopened, it knows the last dial");
    check::is_true(!DialLog().record(1.0, 1e7, "menu"), "dial log: no directory, no file");

    // Trimmed to 48 h, keeping the newest line before the cutoff.
    const double now = 1000.0 + DialLog::KEEP_S + 100.0;
    log.record(1050.0, 14'000'000.0, "radio");
    log.record(now - 10.0, 3'500'000.0, "radio");
    log.trim(now);
    const QList<QByteArray> kept = dial_lines(log.path());
    check::is_true(kept.size() == 2 && kept[0].contains("14000000") && kept[1].contains("3500000"),
                   "dial log: old lines trimmed, the dial at the cutoff kept");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    QTemporaryDir tmp;
    const std::filesystem::path dir(tmp.path().toStdString());
    test_the_sends_of_a_schedule();
    test_what_cannot_be_scheduled();
    test_clashes();
    test_upcoming_and_the_file();
    test_the_scheduler_starts_what_is_due(dir);
    test_a_late_send_starts_at_once(dir);
    test_sends_follow_straight_on(dir);
    test_missed_and_paused(dir);
    test_the_sends_store(dir);
    test_change_and_truncate(dir);
    test_the_window(dir);
    test_the_editor(dir);
    test_a_change_needs_no_sends(dir);
    test_frequency_changes(dir);
    test_the_dial_log(dir);
    return check::report("qrss schedule");
}
