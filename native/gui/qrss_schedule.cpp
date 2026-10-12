#include "qrss_schedule.hpp"

#include <QDateTime>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QTimeZone>
#include <QTimer>
#include <QUuid>

#include <algorithm>
#include <cmath>
#include <exception>
#include <system_error>
#include <utility>

#include "images/images.hpp"

namespace sstvae::gui::qrss_schedule {

namespace {

constexpr double WEEK_S = 7 * 86400.0;

// Send k of `e` needs the transmitter over [slot + a, slot + b], where
// slot = first_slot + k * P.
double busy_from(const Entry&) { return qrss_tx::audio_start(0.0) - START_LEAD_S; }
double busy_to(const Entry& e) {
    return qrss_tx::audio_end((qrss_tx::passes_for(e.mode) - 1) * qrss_tx::PASS_SPACING_S);
}

// The sends k of `o` whose busy span overlaps (from, to), clamped to the
// ones still to come: empty when lo > hi.
std::pair<int, int> overlapping(const Entry& o, double from, double to) {
    const double p = period_min(o) * 60.0;
    int lo = static_cast<int>(std::floor((from - o.first_slot - busy_to(o)) / p)) + 1;
    int hi = static_cast<int>(std::ceil((to - o.first_slot - busy_from(o)) / p)) - 1;
    lo = std::max(lo, o.next);
    if (o.count > 0) hi = std::min(hi, o.count - 1);
    return {lo, hi};
}

QDateTime utc(double t) {
    return QDateTime::fromSecsSinceEpoch(static_cast<qint64>(std::llround(t)), QTimeZone::utc());
}

std::string interval(int minutes) {
    if (minutes % 60 != 0) return "every " + std::to_string(minutes) + " minutes";
    const int h = minutes / 60;
    if (h == 1) return "every hour";
    if (h == 2) return "every 2 hours (alternating hours)";
    if (h % 24 == 0) return h == 24 ? "every day" : "every " + std::to_string(h / 24) + " days";
    return "every " + std::to_string(h) + " hours";
}

}  // namespace

int period_min(const Entry& e) {
    return e.every_min > 0 ? e.every_min : 30 * std::max(1, qrss_tx::passes_for(e.mode));
}

double send_slot(const Entry& e, int k) { return e.first_slot + k * period_min(e) * 60.0; }

bool finished(const Entry& e) { return e.count > 0 && e.next >= e.count; }

std::vector<qrss_tx::Pass> passes_of(const Entry& e, int k) {
    std::vector<qrss_tx::Pass> out;
    const double slot = send_slot(e, k);
    for (int j = 0; j < qrss_tx::passes_for(e.mode); ++j) {
        out.push_back({slot + j * qrss_tx::PASS_SPACING_S, j});
    }
    return out;
}

std::pair<double, double> busy(const Entry& e, int k) {
    const double slot = send_slot(e, k);
    return {slot + busy_from(e), slot + busy_to(e)};
}

bool follows_on(const Entry& e, int k, const std::vector<Entry>& all, double now) {
    if (!problem(e).empty()) return false;
    const double slot = send_slot(e, k);
    for (const Entry& o : all) {
        if (!o.enabled || !problem(o).empty()) continue;
        if (o.id == e.id && o.every_min == 0) continue;   // already one run
        const double len = qrss_tx::passes_for(o.mode) * qrss_tx::PASS_SPACING_S;
        const double p = period_min(o) * 60.0;
        const double x = (slot - len - o.first_slot) / p;
        if (x < 0.0 || x != std::floor(x)) continue;
        const int j = static_cast<int>(x);
        if (o.id == e.id && j >= k) continue;
        if (o.count > 0 && j >= o.count) continue;
        const double last = slot - qrss_tx::PASS_SPACING_S;
        if (now < qrss_tx::audio_start(last) - CHAIN_LEAD_S) return true;
    }
    return false;
}

std::string problem(const Entry& e) {
    const int n = qrss_tx::passes_for(e.mode);
    if (n == 0) return "unknown QRSS mode '" + e.mode + "'";
    if (!qrss_tx::known_waveform(e.waveform)) return "unknown QRSS waveform '" + e.waveform + "'";
    if (!(e.freq_hz >= qrss_tx::FREQ_MIN_HZ && e.freq_hz <= qrss_tx::FREQ_MAX_HZ)) {
        return "the QRSS carrier must be between 300 and 2700 Hz";
    }
    if (e.first_slot <= 0.0 || std::fmod(e.first_slot, QUARTER_S) != 0.0) {
        return "a send starts on a quarter hour";
    }
    if (e.count < 0) return "the number of sends cannot be negative";
    if (e.every_min == 0) {
        if (e.count == 0) {
            return "back-to-back sends never leave the air: choose how many, or a gap "
                   "between them";
        }
        if (e.count * n > MAX_RUN_PASSES) {
            return "back to back, at most " + std::to_string(MAX_RUN_PASSES / n) +
                   " sends of mode " + e.mode + " (a day on the air)";
        }
        return {};
    }
    if (e.every_min < 0 || e.every_min % 15 != 0) {
        return "sends repeat a whole number of quarter hours apart";
    }
    if (e.every_min < 30 * n) {
        return "a mode " + e.mode + " send lasts " + std::to_string(30 * n) +
               " minutes, so it cannot repeat more often than that";
    }
    return {};
}

std::string describe(const Entry& e) {
    // CE was the only waveform for long enough that it goes unsaid.
    const std::string mode =
        (e.waveform == qrss_tx::WAVEFORM_CE ? std::string() : "waveform " + e.waveform + " ") +
        "mode " + e.mode;
    if (e.count == 1) return mode + ", once";
    if (e.every_min == 0) return std::to_string(e.count) + " x " + mode + ", back to back";
    const std::string times = e.count == 0 ? "until removed" : std::to_string(e.count) + " times";
    return mode + " " + interval(e.every_min) + ", " + times;
}

std::string when(double slot, double today) {
    const QDateTime t = utc(slot);
    const bool same_day = t.date() == utc(today).date();
    return t.toString(same_day ? QStringLiteral("HH:mm") : QStringLiteral("ddd d MMM HH:mm"))
               .toStdString() +
           "Z";
}

double earliest_slot(double now) {
    static_assert(ADD_BY_S > MIN_LEAD_S + qrss_tx::AUDIO_LEAD_S - qrss_tx::T0_OFFSET_S,
                  "a send added at the last moment would already be missed");
    return std::ceil((now + ADD_BY_S) / QUARTER_S) * QUARTER_S;
}

std::vector<Upcoming> upcoming(const std::vector<Entry>& entries, double now,
                               double horizon_s, std::size_t max) {
    std::vector<Upcoming> out;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const Entry& e = entries[i];
        if (!e.enabled || !problem(e).empty()) continue;
        std::size_t taken = 0;
        for (int k = e.next; (e.count == 0 || k < e.count) && taken < max; ++k, ++taken) {
            const double slot = send_slot(e, k);
            if (slot > now + horizon_s) break;
            out.push_back({i, k, slot});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const Upcoming& a, const Upcoming& b) { return a.slot < b.slot; });
    if (out.size() > max) out.resize(max);
    return out;
}

std::optional<Clash> clash(const Entry& e, const std::vector<Entry>& others, double now) {
    if (!problem(e).empty()) return std::nullopt;
    const double end = std::max(now, e.first_slot) + WEEK_S;
    // Everything that will be on the schedule, `e` as it would be.
    std::vector<Entry> all;
    for (const Entry& o : others) {
        if (o.id != e.id) all.push_back(o);
    }
    all.push_back(e);
    // A send that follows straight on joins the run before it, so it
    // needs the transmitter only from its own audio (and PTT).
    auto span = [&](const Entry& x, int k) {
        auto s = busy(x, k);
        if (follows_on(x, k, all, now)) s.first = qrss_tx::audio_start(send_slot(x, k)) - 1.0;
        return s;
    };
    std::optional<Clash> best;
    for (const Entry& o : others) {
        if (o.id == e.id || !o.enabled || finished(o) || !problem(o).empty()) continue;
        for (int k = e.next; e.count == 0 || k < e.count; ++k) {
            if (best && send_slot(e, k) >= best->slot) break;
            if (send_slot(e, k) > end) break;
            const auto [from, to] = span(e, k);
            // `overlapping` allows every send its full lead, so it finds
            // every candidate; each is then checked with its real span.
            const auto [lo, hi] = overlapping(o, from, to);
            bool hit = false;
            for (int j = lo; j <= hi; ++j) {
                const auto [of, ot] = span(o, j);
                if (of < to && from < ot) {
                    best = Clash{o.label, send_slot(e, k), send_slot(o, j)};
                    hit = true;
                    break;
                }
            }
            if (hit) break;
        }
    }
    return best;
}

std::optional<Upcoming> clash_with(const std::vector<Entry>& entries, double from, double to,
                                   double) {
    std::optional<Upcoming> best;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const Entry& o = entries[i];
        if (!o.enabled || finished(o) || !problem(o).empty()) continue;
        const auto [lo, hi] = overlapping(o, from, to);
        if (lo <= hi && (!best || send_slot(o, lo) < best->slot)) {
            best = Upcoming{i, lo, send_slot(o, lo)};
        }
    }
    return best;
}

std::string problem(const FreqChange& c) {
    if (!(c.dial_hz > 0.0 && c.dial_hz < 1e10)) return "the frequency must be above 0 Hz";
    if (!(c.at > 0.0)) return "a frequency change needs a time";
    return {};
}

std::pair<double, double> busy(const FreqChange& c) {
    return {c.at - CHANGE_CLEAR_S, c.at + TUNE_DELAY_S + std::max(CHANGE_CLEAR_S, TUNE_MAX_S)};
}

std::optional<Upcoming> clash(const FreqChange& c, const std::vector<Entry>& entries) {
    const auto [from, to] = busy(c);
    return clash_with(entries, from, to, 0.0);
}

std::optional<FreqChange> clash(const Entry& e, const std::vector<FreqChange>& changes,
                                double now) {
    if (!e.enabled || finished(e) || !problem(e).empty()) return std::nullopt;
    const double end = std::max(now, e.first_slot) + WEEK_S;
    std::optional<FreqChange> best;
    for (const FreqChange& c : changes) {
        if (c.done || c.at > end) continue;
        const auto [from, to] = busy(c);
        const auto [lo, hi] = overlapping(e, from, to);
        if (lo <= hi && (!best || c.at < best->at)) best = c;
    }
    return best;
}

std::string mhz(double hz) {
    return QString::number(hz / 1e6, 'f', 6).toStdString() + " MHz";
}

std::string to_json(const std::vector<Entry>& entries, const std::vector<FreqChange>& changes) {
    QJsonArray list;
    for (const Entry& e : entries) {
        QJsonObject o;
        o[QStringLiteral("id")] = QString::fromStdString(e.id);
        o[QStringLiteral("picture")] = QString::fromStdString(e.picture);
        o[QStringLiteral("label")] = QString::fromStdString(e.label);
        if (!e.send_id.empty()) o[QStringLiteral("send_id")] = QString::fromStdString(e.send_id);
        o[QStringLiteral("mode")] = QString::fromStdString(e.mode);
        o[QStringLiteral("waveform")] = QString::fromStdString(e.waveform);
        o[QStringLiteral("freq_hz")] = e.freq_hz;
        o[QStringLiteral("first_slot")] = e.first_slot;
        o[QStringLiteral("count")] = e.count;
        o[QStringLiteral("every_min")] = e.every_min;
        o[QStringLiteral("enabled")] = e.enabled;
        o[QStringLiteral("next")] = e.next;
        list.append(o);
    }
    QJsonObject root;
    root[QStringLiteral("version")] = 1;
    root[QStringLiteral("entries")] = list;
    QJsonArray moves;
    for (const FreqChange& c : changes) {
        QJsonObject o;
        o[QStringLiteral("id")] = QString::fromStdString(c.id);
        o[QStringLiteral("at")] = c.at;
        o[QStringLiteral("dial_hz")] = c.dial_hz;
        o[QStringLiteral("done")] = c.done;
        moves.append(o);
    }
    if (!moves.isEmpty()) root[QStringLiteral("freq_changes")] = moves;
    return QJsonDocument(root).toJson(QJsonDocument::Indented).toStdString();
}

std::vector<Entry> from_json(const std::string& text) {
    std::vector<Entry> out;
    const QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(text));
    for (const QJsonValue& v : doc.object().value(QStringLiteral("entries")).toArray()) {
        const QJsonObject o = v.toObject();
        Entry e;
        e.id = o.value(QStringLiteral("id")).toString().toStdString();
        e.picture = o.value(QStringLiteral("picture")).toString().toStdString();
        e.label = o.value(QStringLiteral("label")).toString().toStdString();
        e.send_id = o.value(QStringLiteral("send_id")).toString().toStdString();
        e.mode = o.value(QStringLiteral("mode")).toString(QStringLiteral("A")).toStdString();
        e.freq_hz = o.value(QStringLiteral("freq_hz")).toDouble(qrss_tx::FREQ_DEFAULT_HZ);
        // Schedules saved before L existed are CE.
        e.waveform = o.value(QStringLiteral("waveform"))
                         .toString(QString::fromLatin1(qrss_tx::WAVEFORM_CE))
                         .toStdString();
        e.first_slot = o.value(QStringLiteral("first_slot")).toDouble();
        e.count = o.value(QStringLiteral("count")).toInt(1);
        e.every_min = o.value(QStringLiteral("every_min")).toInt();
        e.enabled = o.value(QStringLiteral("enabled")).toBool(true);
        e.next = std::max(0, o.value(QStringLiteral("next")).toInt());
        if (e.id.empty() || e.picture.empty() || !problem(e).empty()) continue;
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<FreqChange> changes_from_json(const std::string& text) {
    std::vector<FreqChange> out;
    const QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(text));
    for (const QJsonValue& v : doc.object().value(QStringLiteral("freq_changes")).toArray()) {
        const QJsonObject o = v.toObject();
        FreqChange c;
        c.id = o.value(QStringLiteral("id")).toString().toStdString();
        c.at = o.value(QStringLiteral("at")).toDouble();
        c.dial_hz = o.value(QStringLiteral("dial_hz")).toDouble();
        c.done = o.value(QStringLiteral("done")).toBool();
        if (c.id.empty() || !problem(c).empty()) continue;
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
              [](const FreqChange& a, const FreqChange& b) { return a.at < b.at; });
    return out;
}

}  // namespace sstvae::gui::qrss_schedule

namespace sstvae::gui {

namespace qs = qrss_schedule;

QrssScheduler::QrssScheduler(std::filesystem::path file, Starter starter, Clock clock,
                             int tick_ms, QObject* parent)
    : QObject(parent), file_(std::move(file)), starter_(std::move(starter)),
      clock_(std::move(clock)) {
    if (!clock_) {
        clock_ = [] { return static_cast<double>(QDateTime::currentMSecsSinceEpoch()) / 1000.0; };
    }
    load();
    if (tick_ms > 0) {
        timer_ = new QTimer(this);
        timer_->setInterval(tick_ms);
        connect(timer_, &QTimer::timeout, this, &QrssScheduler::tick);
        timer_->start();
        // The first look is as soon as the event loop runs, so a send due
        // when the app starts is not a tick late.
        QTimer::singleShot(0, this, &QrssScheduler::tick);
    }
}

QrssScheduler::~QrssScheduler() = default;

std::filesystem::path QrssScheduler::picture_dir() const {
    return file_.parent_path() / "qrss_schedule";
}

void QrssScheduler::load() {
    QFile f(QString::fromStdString(file_.string()));
    if (!f.open(QIODevice::ReadOnly)) return;
    const std::string text = f.readAll().toStdString();
    entries_ = qs::from_json(text);
    changes_ = qs::changes_from_json(text);
}

void QrssScheduler::save() {
    std::error_code ec;
    std::filesystem::create_directories(file_.parent_path(), ec);
    QSaveFile f(QString::fromStdString(file_.string()));
    const std::string text = qs::to_json(entries_, changes_);
    if (!f.open(QIODevice::WriteOnly) ||
        f.write(text.data(), static_cast<qint64>(text.size())) !=
            static_cast<qint64>(text.size()) ||
        !f.commit()) {
        emit logged(2, tr("QRSS schedule: could not save %1")
                              .arg(QString::fromStdString(file_.string())));
    }
}

qs::Entry* QrssScheduler::find(const std::string& id) {
    for (qs::Entry& e : entries_) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

std::string QrssScheduler::add(qs::Entry entry, const images::Picture& picture) {
    const double now = clock_();
    entry.next = 0;
    entry.enabled = true;
    if (std::string why = qs::problem(entry); !why.empty()) return why;
    const double earliest = qs::earliest_slot(now);
    if (entry.first_slot < earliest) {
        return "that is too soon: a send added now can start at " + qs::when(earliest, now) +
               " at the earliest";
    }
    if (const auto c = qs::clash(entry, entries_, now)) {
        return "its " + qs::when(c->slot, now) + " send would overlap the " +
               qs::when(c->other_slot, now) + " send of \"" + c->other +
               "\"; the transmitter sends one thing at a time";
    }
    if (std::string why = change_clash_text(entry, now); !why.empty()) return why;
    do {
        entry.id = QUuid::createUuid().toString(QUuid::Id128).left(12).toStdString();
    } while (find(entry.id) != nullptr);
    std::error_code ec;
    std::filesystem::create_directories(picture_dir(), ec);
    const std::filesystem::path png = picture_dir() / (entry.id + ".png");
    try {
        images::save_png(picture, png.string());
    } catch (const std::exception& e) {
        return std::string("could not save the picture: ") + e.what();
    }
    entry.picture = png.string();
    if (entry.label.empty()) entry.label = "picture " + entry.id.substr(0, 4);
    entries_.push_back(entry);
    save();
    emit logged(0, tr("QRSS schedule: added \"%1\", %2 at %3 Hz, first send %4")
                           .arg(QString::fromStdString(entry.label),
                                QString::fromStdString(qs::describe(entry)))
                           .arg(entry.freq_hz, 0, 'f', 0)
                           .arg(QString::fromStdString(qs::when(entry.first_slot, now))));
    emit changed();
    tick();  // a send added close to its time starts now, not at the next tick
    return {};
}

void QrssScheduler::remove(const std::string& id) {
    const auto it = std::find_if(entries_.begin(), entries_.end(),
                                 [&](const qs::Entry& e) { return e.id == id; });
    if (it == entries_.end()) return;
    std::error_code ec;
    const std::filesystem::path png(it->picture);
    if (png.parent_path() == picture_dir()) std::filesystem::remove(png, ec);
    const QString label = QString::fromStdString(it->label);
    entries_.erase(it);
    save();
    // Removing the entry whose run is on the air (or getting ready, which
    // starts minutes before the first pass keys) stops that run: left
    // going, it would key a picture that is no longer on the schedule,
    // and the slot it holds would look free to the next one added. A
    // picture that followed on is already part of another entry's run.
    const bool stop = id == active_ && canceller_;
    emit logged(0, stop ? tr("QRSS schedule: removed \"%1\" and stopped its send").arg(label)
                   : on_air(id)
                       ? tr("QRSS schedule: removed \"%1\"; the send on the air carries "
                            "on (Cancel stops it)")
                             .arg(label)
                       : tr("QRSS schedule: removed \"%1\"").arg(label));
    if (stop) canceller_();
    emit changed();
}

std::string QrssScheduler::change(const std::string& id,
                                  const std::function<void(qs::Entry&)>& edit) {
    qs::Entry* e = find(id);
    if (e == nullptr) return "it is no longer on the schedule";
    if (on_air(id)) return "it is on the air now; Cancel on the Transmit pane stops it";
    if (qs::finished(*e)) return "every send of it is past";
    const double now = clock_();
    qs::Entry trial = *e;
    // The sends already made stay made: what changes is the rest.
    if (trial.next > 0) {
        trial.first_slot = qs::send_slot(trial, trial.next);
        if (trial.count > 0) trial.count -= trial.next;
        trial.next = 0;
    }
    const double before = qs::send_slot(trial, 0);
    edit(trial);
    if (std::string why = qs::problem(trial); !why.empty()) return why;
    const double first = qs::send_slot(trial, trial.next);
    if (first != before && first < qs::earliest_slot(now)) {
        return "that is too soon: the earliest a send can start now is " +
               qs::when(qs::earliest_slot(now), now);
    }
    if (trial.enabled) {
        if (const auto c = qs::clash(trial, entries_, now)) {
            return "its " + qs::when(c->slot, now) + " send would overlap the " +
                   qs::when(c->other_slot, now) + " send of \"" + c->other +
                   "\"; the transmitter sends one thing at a time";
        }
        if (std::string why = change_clash_text(trial, now); !why.empty()) return why;
    }
    *e = trial;
    save();
    emit logged(0, tr("QRSS schedule: changed \"%1\" to %2, next send %3")
                           .arg(QString::fromStdString(e->label),
                                QString::fromStdString(qs::describe(*e)),
                                QString::fromStdString(qs::when(qs::send_slot(*e, e->next), now))));
    emit changed();
    tick();  // moved close to its time: start it now, not at the next tick
    return {};
}

void QrssScheduler::truncate(const std::string& id, int k) {
    qs::Entry* e = find(id);
    if (e == nullptr) return;
    if (k <= e->next) {
        remove(id);
        return;
    }
    if (e->count > 0 && k >= e->count) return;
    e->count = k;
    save();
    emit logged(0, tr("QRSS schedule: \"%1\" now ends after send %2")
                           .arg(QString::fromStdString(e->label))
                           .arg(k));
    emit changed();
}

std::string QrssScheduler::set_enabled(const std::string& id, bool on) {
    qs::Entry* e = find(id);
    if (e == nullptr || e->enabled == on) return {};
    if (on) {
        qs::Entry trial = *e;
        trial.enabled = true;
        if (const auto c = qs::clash(trial, entries_, clock_())) {
            return "its " + qs::when(c->slot, clock_()) + " send would overlap the " +
                   qs::when(c->other_slot, clock_()) + " send of \"" + c->other + "\"";
        }
        if (std::string why = change_clash_text(trial, clock_()); !why.empty()) return why;
    }
    e->enabled = on;
    QString note;
    if (on) {
        // What fell due while it was paused was not missed: it was not
        // wanted.
        const double now = clock_();
        e->next = std::max(e->next, first_startable(*e, now));
        note = qs::finished(*e) ? tr("; every send is past")
                                : tr("; next send %1").arg(QString::fromStdString(
                                      qs::when(qs::send_slot(*e, e->next), now)));
    }
    save();
    emit logged(0, (on ? tr("QRSS schedule: resumed \"%1\"%2") : tr("QRSS schedule: paused \"%1\"%2"))
                       .arg(QString::fromStdString(e->label), note));
    emit changed();
    if (on) tick();
    return {};
}

int QrssScheduler::first_startable(const qs::Entry& e, double now) const {
    const double p = qs::period_min(e) * 60.0;
    const double x = (now + qs::MIN_LEAD_S - qrss_tx::audio_start(e.first_slot)) / p;
    int k = x < 0.0 ? 0 : static_cast<int>(std::floor(x)) + 1;
    if (e.count > 0) k = std::min(k, e.count);
    return k;
}

void QrssScheduler::run_finished() {
    if (active_.empty()) return;
    active_.clear();
    on_air_.clear();
    run_end_ = 0.0;
    emit changed();
}

bool QrssScheduler::on_air(const std::string& id) const {
    return std::find(on_air_.begin(), on_air_.end(), id) != on_air_.end();
}

void QrssScheduler::chain(double now, bool& dirty) {
    if (active_.empty() || !chainer_ || run_end_ <= 0.0) return;
    for (;;) {
        qs::Entry* follow = nullptr;
        for (qs::Entry& e : entries_) {
            if (!e.enabled || qs::finished(e)) continue;
            if (qs::send_slot(e, e.next) == run_end_ + qrss_tx::PASS_SPACING_S) {
                follow = &e;
                break;
            }
        }
        if (follow == nullptr) return;
        qs::Entry& e = *follow;
        const int n = qrss_tx::passes_for(e.mode);
        const int m = e.every_min == 0 ? std::min(e.count - e.next, qs::MAX_RUN_PASSES / n) : 1;
        std::vector<qrss_tx::Pass> passes;
        for (int k = e.next; k < e.next + m; ++k) {
            for (const qrss_tx::Pass& p : qs::passes_of(e, k)) passes.push_back(p);
        }
        const std::string key = e.id + "/" + std::to_string(e.next);
        const QString label = QString::fromStdString(e.label);
        const QString at = QString::fromStdString(qs::when(qs::send_slot(e, e.next), now));
        std::string why;
        if (!chainer_(e, passes, &why)) {
            if (refused_ != key) {
                emit logged(1, tr("QRSS schedule: the %1 send of \"%2\" could not follow on: %3")
                                       .arg(at, label, QString::fromStdString(why)));
                refused_ = key;
                refused_why_ = why;
            }
            return;
        }
        emit logged(0, tr("QRSS schedule: \"%1\" (%2) follows straight on at %3")
                               .arg(label, QString::fromStdString(qs::describe(e)), at));
        e.next += m;
        if (!on_air(e.id)) on_air_.push_back(e.id);
        run_end_ = passes.back().slot;
        refused_.clear();
        dirty = true;
    }
}

std::string QrssScheduler::change_clash_text(const qs::Entry& e, double now) const {
    const auto c = qs::clash(e, changes_, now);
    if (!c) return {};
    return "it would be on the air at the " + qs::when(c->at, now) + " frequency change (to " +
           qs::mhz(c->dial_hz) + "); move or remove the change first";
}

namespace {

std::string change_problem(const qs::FreqChange& c, const std::vector<qs::Entry>& entries,
                           double now) {
    if (std::string why = qs::problem(c); !why.empty()) return why;
    if (c.at < now + 60.0) return "that is too soon: a frequency change needs a minute's notice";
    if (const auto u = qs::clash(c, entries)) {
        return "it would cut into the " + qs::when(u->slot, now) + " send of \"" +
               entries[u->entry].label +
               "\"; a change keeps 30 s clear of a send on each side (and 5 s for the tuner)";
    }
    return {};
}

}  // namespace

std::string QrssScheduler::add_change(qs::FreqChange change) {
    const double now = clock_();
    change.done = false;
    if (std::string why = change_problem(change, entries_, now); !why.empty()) return why;
    const auto taken = [this](const std::string& id) {
        return std::any_of(changes_.begin(), changes_.end(),
                           [&](const qs::FreqChange& c) { return c.id == id; });
    };
    do {
        change.id = QUuid::createUuid().toString(QUuid::Id128).left(12).toStdString();
    } while (taken(change.id));
    changes_.push_back(change);
    std::stable_sort(changes_.begin(), changes_.end(),
                     [](const qs::FreqChange& a, const qs::FreqChange& b) { return a.at < b.at; });
    save();
    emit logged(0, tr("QRSS schedule: frequency change to %1 at %2")
                           .arg(QString::fromStdString(qs::mhz(change.dial_hz)),
                                QString::fromStdString(qs::when(change.at, now))));
    emit changed();
    return {};
}

std::string QrssScheduler::edit_change(const std::string& id,
                                       const std::function<void(qs::FreqChange&)>& edit) {
    const auto it = std::find_if(changes_.begin(), changes_.end(),
                                 [&](const qs::FreqChange& c) { return c.id == id; });
    if (it == changes_.end()) return "it is no longer on the schedule";
    if (it->done) return "it has already been made";
    const double now = clock_();
    qs::FreqChange trial = *it;
    edit(trial);
    trial.id = id;
    if (std::string why = change_problem(trial, entries_, now); !why.empty()) return why;
    *it = trial;
    std::stable_sort(changes_.begin(), changes_.end(),
                     [](const qs::FreqChange& a, const qs::FreqChange& b) { return a.at < b.at; });
    save();
    emit logged(0, tr("QRSS schedule: frequency change now to %1 at %2")
                           .arg(QString::fromStdString(qs::mhz(trial.dial_hz)),
                                QString::fromStdString(qs::when(trial.at, now))));
    emit changed();
    return {};
}

void QrssScheduler::remove_change(const std::string& id) {
    const auto it = std::find_if(changes_.begin(), changes_.end(),
                                 [&](const qs::FreqChange& c) { return c.id == id; });
    if (it == changes_.end()) return;
    const QString what = tr("%1 at %2").arg(QString::fromStdString(qs::mhz(it->dial_hz)),
                                            QString::fromStdString(qs::when(it->at, clock_())));
    changes_.erase(it);
    save();
    emit logged(0, tr("QRSS schedule: removed the frequency change to %1").arg(what));
    emit changed();
}

void QrssScheduler::tick_changes(double now, bool& dirty) {
    // Made ones are kept a day, for the timeline.
    const auto old = std::remove_if(changes_.begin(), changes_.end(), [now](const qs::FreqChange& c) {
        return c.done && c.at < now - 86400.0;
    });
    if (old != changes_.end()) {
        changes_.erase(old, changes_.end());
        dirty = true;
    }
    for (qs::FreqChange& c : changes_) {
        if (c.done || now < c.at) continue;
        const QString what = tr("%1 at %2").arg(QString::fromStdString(qs::mhz(c.dial_hz)),
                                                QString::fromStdString(qs::when(c.at, now)));
        if (now > c.at + qs::CHANGE_LATE_S) {
            emit logged(1, tr("QRSS schedule: missed the frequency change to %1 (%2)")
                                  .arg(what, change_refused_ == c.id
                                                 ? tr("the transmitter stayed busy")
                                                 : tr("the app was not running then")));
            c.done = true;
            dirty = true;
            continue;
        }
        if (!retuner_) continue;
        std::string why;
        const Start result = retuner_(c, &why);
        if (result == Start::Started) {
            emit logged(0, tr("QRSS schedule: changing frequency to %1").arg(what));
            c.done = true;
            dirty = true;
        } else if (result == Start::Busy) {
            if (change_refused_ != c.id) {
                emit logged(0, tr("QRSS schedule: the frequency change to %1 is waiting: %2")
                                       .arg(what, QString::fromStdString(why)));
                change_refused_ = c.id;
            }
        } else {
            emit logged(2, tr("QRSS schedule: the frequency change to %1 was not made: %2")
                                  .arg(what, QString::fromStdString(why)));
            c.done = true;
            dirty = true;
        }
        break;   // one radio command a tick
    }
}

void QrssScheduler::tick() {
    const double now = clock_();
    bool dirty = false;
    tick_changes(now, dirty);
    // Sends that follow straight on join the run on the air before
    // anything below can count them missed or try to start them alone.
    chain(now, dirty);
    // Sends whose audio is too close (or past) to start: missed.
    for (qs::Entry& e : entries_) {
        if (!e.enabled || qs::finished(e)) continue;
        const int k = first_startable(e, now);
        if (k <= e.next) continue;
        const int missed = k - e.next;
        const std::string last = qs::when(qs::send_slot(e, k - 1), now);
        const std::string key_prefix = e.id + "/";
        std::string why = "the app was not running then";
        if (refused_.rfind(key_prefix, 0) == 0) why = refused_why_;
        emit logged(1, tr("QRSS schedule: \"%1\" missed %2 send(s), the last at %3 (%4)")
                              .arg(QString::fromStdString(e.label))
                              .arg(missed)
                              .arg(QString::fromStdString(last), QString::fromStdString(why)));
        e.next = k;
        dirty = true;
    }
    // The earliest send that is due.
    qs::Entry* due = nullptr;
    for (qs::Entry& e : entries_) {
        if (!e.enabled || qs::finished(e)) continue;
        if (now < qrss_tx::audio_start(qs::send_slot(e, e.next)) - qs::START_LEAD_S) continue;
        if (due == nullptr || qs::send_slot(e, e.next) < qs::send_slot(*due, due->next)) due = &e;
    }
    if (due != nullptr) {
        qs::Entry& e = *due;
        const int n = qrss_tx::passes_for(e.mode);
        const int m = e.every_min == 0 ? std::min(e.count - e.next, qs::MAX_RUN_PASSES / n) : 1;
        std::vector<qrss_tx::Pass> passes;
        for (int k = e.next; k < e.next + m; ++k) {
            for (const qrss_tx::Pass& p : qs::passes_of(e, k)) passes.push_back(p);
        }
        const std::string key = e.id + "/" + std::to_string(e.next);
        std::string why;
        const Start result = starter_(e, passes, &why);
        const QString label = QString::fromStdString(e.label);
        const QString at = QString::fromStdString(qs::when(qs::send_slot(e, e.next), now));
        if (result == Start::Started) {
            const QString which =
                e.count == 1 ? QString()
                : m == 1     ? tr(", send %1%2").arg(e.next + 1).arg(
                               e.count > 0 ? tr(" of %1").arg(e.count) : QString())
                             : tr(", sends %1-%2 of %3").arg(e.next + 1).arg(e.next + m).arg(e.count);
            emit logged(0, tr("QRSS schedule: starting \"%1\" (%2) at %3%4")
                                   .arg(label, QString::fromStdString(qs::describe(e)), at, which));
            e.next += m;
            active_ = e.id;
            on_air_ = {e.id};
            run_end_ = passes.back().slot;
            refused_.clear();
            dirty = true;
        } else if (result == Start::Busy) {
            if (refused_ != key) {
                emit logged(0, tr("QRSS schedule: the %1 send of \"%2\" is waiting: %3")
                                       .arg(at, label, QString::fromStdString(why)));
                refused_ = key;
                refused_why_ = why;
            }
        } else {
            emit logged(2, tr("QRSS schedule: the %1 send of \"%2\" was not sent: %3")
                                  .arg(at, label, QString::fromStdString(why)));
            e.next += 1;
            dirty = true;
        }
    }
    chain(now, dirty);
    if (dirty) {
        save();
        emit changed();
    }
}

}  // namespace sstvae::gui
