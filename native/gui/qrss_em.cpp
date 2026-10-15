#include "qrss_em.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QPalette>
#include <QProgressBar>
#include <QPushButton>
#include <QStringList>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "info_button.hpp"
// The search's headers name a member `slots`, which is a Qt keyword macro.
#pragma push_macro("slots")
#undef slots
#include "qrss/codec/render.hpp"
#include "qrss/em.hpp"
#include "qrss/em_api.hpp"
#include "qrss/png.hpp"
#include "qrss/store.hpp"
#pragma pop_macro("slots")

namespace sstvae::gui::qrss_em {

namespace {

namespace em_api = qrss::em_api;

bool finished_status(const QString& status) {
    return status != QLatin1String("receiving") && status != QLatin1String("finishing");
}

// Seconds to receive one pass again, by frame (em_api.hpp's measurements,
// the middle of each range).
double refine_pass_s(const QString& frame) {
    const QString f = ce_frame(frame);
    if (f == QLatin1String("tiny")) return 0.1;
    if (f == QLatin1String("short")) return 0.75;
    if (f == QLatin1String("medium")) return 2.0;
    return 4.0;
}

// Minutes to search a full 48 h store, by frame.
int search_minutes(const QString& frame) {
    const QString f = ce_frame(frame);
    if (f == QLatin1String("tiny")) return 1;
    if (f == QLatin1String("short")) return 3;
    if (f == QLatin1String("medium")) return 8;
    return 20;
}

QString about(double seconds) {
    if (seconds < 90) {
        return QObject::tr("about %1 seconds").arg(std::max(5, static_cast<int>(std::lround(seconds / 5) * 5)));
    }
    const int minutes = static_cast<int>(std::lround(seconds / 60));
    return minutes == 1 ? QObject::tr("about a minute") : QObject::tr("about %1 minutes").arg(minutes);
}

// "1 pass", "3 passes".
QString count(std::int64_t n, const QString& one, const QString& many) {
    return QStringLiteral("%1 %2").arg(n).arg(n == 1 ? one : many);
}

QString signed_db(double db) {
    return (db >= 0 ? QStringLiteral("+") : QStringLiteral("−")) + QString::number(std::abs(db), 'f', 2) +
           QObject::tr(" dB");
}

// One decoder for the app's redraws, kept once loaded (seconds to load).
std::shared_ptr<const qrss::codec::LoadedDecoder> decoder_for(std::int64_t codec_id) {
    static std::mutex mu;
    static std::map<std::int64_t, std::shared_ptr<const qrss::codec::LoadedDecoder>> loaded;
    std::lock_guard lock(mu);
    auto& slot = loaded[codec_id];
    if (!slot) {
        slot = std::make_shared<const qrss::codec::LoadedDecoder>(
            qrss::codec::load("fp32", std::nullopt, codec_id));
    }
    return slot;
}

}  // namespace

QString ce_frame(const QString& frame) {
    QString f = frame.isEmpty() ? QStringLiteral("full") : frame;
    if (f.startsWith(QLatin1String("l-"))) f = f.mid(2);
    return f;
}

// --- controller -----------------------------------------------------------------------------

Controller::Controller(QString state_dir, QObject* parent)
    : QObject(parent),
      state_dir_(std::move(state_dir)),
      store_root_(QString::fromStdString(qrss::store::default_root())) {}

Controller::~Controller() {
    for (const auto& job : jobs_) job->cancel->store(true);
    for (const auto& job : jobs_) {
        if (job->thread.joinable()) job->thread.join();
    }
}

QString Controller::passband_dir() const { return QDir(store_root_).filePath(QStringLiteral("passband")); }

QString Controller::unavailable(Kind kind, const Target& t) const {
    if (!t.valid()) return tr("No header decoded yet, so there is no picture to work on.");
    if (!finished_status(t.status)) return tr("Wait for this pass to finish.");
    if (kind == Kind::Improve) {
        if (t.waveform == QLatin1String("L") && t.passes <= 1) {
            return tr("Not for this picture: its pass is waveform L, and only CE passes can be "
                      "received again.");
        }
        return {};
    }
    if (!QFileInfo(passband_dir()).isDir()) {
        return tr("Not available: no recording has been kept yet. The listener keeps the last 48 "
                  "hours of audio while it runs.");
    }
    return {};
}

const Controller::State* Controller::state(const QString& key) const {
    auto it = states_.constFind(key);
    return it == states_.constEnd() ? nullptr : &it.value();
}

bool Controller::start(Kind kind, const Target& t) {
    if (!unavailable(kind, t).isEmpty()) return false;
    const QString key = t.key();
    State& st = states_[key];
    targets_[key] = t;
    if (st.running) {
        if (*st.running == kind || std::find(st.queued.begin(), st.queued.end(), kind) != st.queued.end()) {
            return false;
        }
        st.queued.push_back(kind);
        emit changed(key);
        return true;
    }
    launch(key, kind, t);
    return true;
}

void Controller::cancel(const QString& key) {
    auto st = states_.find(key);
    if (st != states_.end()) st->queued.clear();
    if (auto job = jobs_.value(key)) job->cancel->store(true);
    emit changed(key);
}

void Controller::launch(const QString& key, Kind kind, const Target& t) {
    State& st = states_[key];
    st.running = kind;
    st.done = 0;
    st.total = 0;
    st.what = kind == Kind::Improve ? tr("Improving") : tr("Searching the recording");
    auto job = std::make_shared<Job>();
    job->kind = kind;
    job->target = t;
    job->cancel = std::make_shared<std::atomic<bool>>(false);
    jobs_.insert(key, job);

    const Runner runner = runner_ ? runner_ : [this](Kind k, const Target& tt, const qrss::em::Hooks& h) {
        return run_default(k, tt, h);
    };
    const auto cancel = job->cancel;
    job->thread = std::thread([this, key, kind, t, runner, cancel] {
        qrss::em::Hooks hooks;
        hooks.cancel = cancel.get();
        hooks.progress = [this, key, kind](std::int64_t done, std::int64_t total, const std::string& what) {
            const QString w = QString::fromStdString(what);
            QMetaObject::invokeMethod(this, [this, key, kind, done, total, w] {
                auto st = states_.find(key);
                if (st == states_.end() || st->running != kind) return;
                st->done = done;
                st->total = total;
                if (kind == Kind::Improve) {
                    // "round <r> pass <uid>": round r of EM_ROUNDS, pass p of the round's.
                    const int per = std::max<std::int64_t>(1, total / qrss::em::EM_ROUNDS);
                    const int round = w.startsWith(QLatin1String("round "))
                                          ? w.section(QLatin1Char(' '), 1, 1).toInt()
                                          : static_cast<int>((done + per - 1) / per);
                    st->what = w.startsWith(QLatin1String("converged"))
                                   ? tr("Improving · finishing")
                                   : tr("Improving · round %1 of %2 · pass %3 of %4")
                                         .arg(std::max(1, round))
                                         .arg(qrss::em::EM_ROUNDS)
                                         .arg((done - 1) % per + 1)
                                         .arg(per);
                } else {
                    st->what = tr("Searching the recording · slot %1 of %2").arg(done).arg(total);
                }
                emit changed(key);
            }, Qt::QueuedConnection);
        };
        Outcome out;
        try {
            out = runner(kind, t, hooks);
        } catch (const std::exception& e) {
            out.ok = false;
            out.line = (kind == Kind::Improve ? tr("Improve failed: %1") : tr("Search failed: %1"))
                           .arg(QString::fromLocal8Bit(e.what()));
        }
        QMetaObject::invokeMethod(this, [this, key, kind, out] { finished(key, kind, out); },
                                  Qt::QueuedConnection);
    });
}

void Controller::finished(const QString& key, Kind kind, const Outcome& out) {
    if (auto job = jobs_.take(key); job && job->thread.joinable()) job->thread.join();
    State& st = states_[key];
    st.running.reset();
    st.done = st.total = 0;
    st.what.clear();
    (kind == Kind::Improve ? st.improve_line : st.find_line) = out.line;
    (kind == Kind::Improve ? st.improve_ok : st.find_ok) = out.ok;
    if (!out.image.isEmpty()) st.image = out.image;
    if (out.mean_w_db) st.mean_w_db = out.mean_w_db;
    if (!st.queued.empty()) {
        const Kind next = st.queued.front();
        st.queued.pop_front();
        Target t = targets_.value(key);
        if (unavailable(next, t).isEmpty()) launch(key, next, t);
    }
    emit changed(key);
}

Outcome Controller::run_default(Kind kind, const Target& t, const qrss::em::Hooks& hooks) const {
    // Read on the worker thread: what the GUI may change meanwhile is copied.
    namespace store = qrss::store;
    store::Store st(store_root_.toStdString());
    const store::Key key{t.callsign.toStdString(), t.picture_id.toLongLong(nullptr, 16)};
    Outcome out;
    bool changed = false;
    if (kind == Kind::Improve) {
        const em_api::RefineResult r = em_api::refine(st, key, {}, hooks);
        const double before = r.history.empty() ? r.mean_w_db : r.history.front();
        const QString gain = signed_db(r.mean_w_db - before);
        const int n = static_cast<int>(r.passes.size());
        if (n == 0) {
            out.line = r.cancelled ? tr("Improve stopped before any pass")
                                   : tr("Nothing to improve: no CE pass of this picture kept its audio");
        } else if (r.cancelled) {
            out.line = tr("Improve stopped after %1: %2 kept").arg(count(n, tr("pass"), tr("passes")), gain);
        } else {
            out.line = tr("Improved: %1 over %2").arg(gain, count(r.rounds_run, tr("round"), tr("rounds")));
        }
        changed = n > 0;
    } else {
        const em_api::RetroResult r =
            em_api::retro(st, key, em_api::Passband::at(passband_dir().toStdString()),
                          ce_frame(t.frame).toStdString(), {}, hooks);
        int mine = 0, others = 0;
        for (const em_api::RetroEntry& f : r.finds) {
            if (f.rule.empty()) continue;   // a member already
            (f.key && *f.key == key ? mine : others)++;
        }
        const QString where =
            r.cancelled ? tr("Search stopped at slot %1 of %2").arg(r.slots_done).arg(r.slots_total)
                        : tr("Searched %1 h").arg(QString::number(r.slots_total / 2.0, 'g', 3));
        out.line = mine == 0 ? where + (r.cancelled ? tr(": nothing new") : tr(": no missed passes"))
                             : where + tr(": found %1").arg(count(mine, tr("missed pass"), tr("missed passes")));
        if (others > 0) out.line += tr("; %1 more for other pictures").arg(others);
        changed = mine > 0;
    }
    const store::Accumulator acc = st.accumulator(key);
    out.mean_w_db = qrss::em::acc_mean_w_db(acc);
    if (changed) {
        try {
            const auto dec = decoder_for(acc.codec_id);
            const std::vector<double> S(acc.S.begin(), acc.S.end());
            const std::vector<double> W(acc.W.begin(), acc.W.end());
            const images::Picture pic = qrss::codec::render(dec->decoder, S, W, acc.mode);
            const QString dir = QDir(state_dir_).filePath(QStringLiteral("em"));
            QDir().mkpath(dir);
            // A new name each time: the tile and the viewer read it by name.
            static std::atomic<int> serial{0};
            const QString path = QDir(dir).filePath(
                QStringLiteral("%1-%2.png").arg(t.key()).arg(serial.fetch_add(1)));
            const QString tmp = path + QStringLiteral(".part");
            qrss::png::write_png(pic, tmp.toStdString());
            if (!QFile::rename(tmp, path)) {
                QFile::remove(tmp);
                throw std::runtime_error("could not write " + path.toStdString());
            }
            // Earlier redraws of this picture are no longer shown.
            for (const QFileInfo& old : QDir(dir).entryInfoList({t.key() + QStringLiteral("-*.png")})) {
                if (old.absoluteFilePath() != QFileInfo(path).absoluteFilePath()) QFile::remove(old.absoluteFilePath());
            }
            out.image = path;
        } catch (const std::exception& e) {
            out.line += tr("; could not draw the picture: %1").arg(QString::fromLocal8Bit(e.what()));
        }
    }
    return out;
}

// --- the bar --------------------------------------------------------------------------------

EmBar::EmBar(Controller* controller, QWidget* parent) : QWidget(parent), controller_(controller) {
    setObjectName(QStringLiteral("em_bar"));
    QFont small = font();
    small.setPointSizeF(std::max(6.5, small.pointSizeF() * 0.82));
    auto* box = new QVBoxLayout(this);
    box->setContentsMargins(0, 2, 0, 0);
    box->setSpacing(2);

    actions_ = new QWidget(this);
    auto* row = new QHBoxLayout(actions_);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(2);
    const auto button = [&](const QString& text, const char* name) {
        auto* b = new QPushButton(text, actions_);
        b->setObjectName(QString::fromLatin1(name));
        b->setFont(small);
        b->setAutoDefault(false);
        return b;
    };
    const auto info = [&](Kind kind, const char* name) {
        auto* i = new InfoButton([this, kind] { return card(kind); }, actions_);
        i->setObjectName(QString::fromLatin1(name));
        i->set_glyph(QStringLiteral("?"));
        return i;
    };
    improve_ = button(tr("Improve"), "em_improve");
    improve_info_ = info(Kind::Improve, "em_improve_info");
    find_ = button(tr("Find missed passes"), "em_find");
    find_info_ = info(Kind::Find, "em_find_info");
    row->addWidget(improve_);
    row->addWidget(improve_info_);
    row->addSpacing(6);
    row->addWidget(find_);
    row->addWidget(find_info_);
    row->addStretch(1);
    box->addWidget(actions_);

    working_ = new QWidget(this);
    auto* wbox = new QVBoxLayout(working_);
    wbox->setContentsMargins(0, 0, 0, 0);
    wbox->setSpacing(2);
    auto* wrow = new QHBoxLayout;
    work_ = new QLabel(working_);
    work_->setObjectName(QStringLiteral("em_progress"));
    work_->setFont(small);
    work_->setWordWrap(true);
    cancel_ = button(tr("Cancel"), "em_cancel");
    cancel_->setParent(working_);
    wrow->addWidget(work_, 1);
    wrow->addWidget(cancel_);
    wbox->addLayout(wrow);
    bar_ = new QProgressBar(working_);
    bar_->setTextVisible(false);
    bar_->setFixedHeight(4);
    wbox->addWidget(bar_);
    box->addWidget(working_);

    result_ = new QLabel(this);
    result_->setObjectName(QStringLiteral("em_result"));
    result_->setFont(small);
    result_->setWordWrap(true);
    result_->setTextFormat(Qt::RichText);
    box->addWidget(result_);

    connect(improve_, &QPushButton::clicked, this, [this] {
        improve_info_->hide_card();
        controller_->start(Kind::Improve, target_);
    });
    connect(find_, &QPushButton::clicked, this, [this] {
        find_info_->hide_card();
        controller_->start(Kind::Find, target_);
    });
    connect(cancel_, &QPushButton::clicked, this, [this] { controller_->cancel(target_.key()); });
    connect(controller_, &Controller::changed, this, [this](const QString& key) {
        if (target_.valid() && key == target_.key()) refresh();
    });
    setVisible(false);
}

void EmBar::set_target(const Target& t) {
    target_ = t;
    refresh();
}

QString EmBar::card(Kind kind) const {
    const QPalette& pal = palette();
    const QString why = controller_->unavailable(kind, target_);
    if (kind == Kind::Improve) {
        const QString here =
            why.isEmpty() ? tr("Here: %1, %2.")
                                .arg(count(target_.passes, tr("pass"), tr("passes")),
                                     about(target_.passes * qrss::em::EM_ROUNDS * refine_pass_s(target_.frame)))
                          : why;
        return info::info_html(
            pal, tr("Improve"),
            tr("Receives each pass of this picture again, using the other passes as a guide to where "
               "the signal is. Better tracking gives the decoder cleaner numbers, so the picture can "
               "sharpen a little."),
            here, why.isEmpty(), tr("Use it when"),
            {tr("A picture you want to keep"), tr("After Find missed passes adds passes"),
             tr("A pass that faded or drifted")},
            tr("Keep in mind"),
            {tr("Usually a small gain, 0 to 0.2 dB"), tr("About a minute for 4 full passes"),
             tr("Only CE passes; L passes stay as they are")},
            info::footnote(pal, tr("The weight shown on the tile can rise by about twice the real gain, "
                                   "because it is the receiver's own estimate.")));
    }
    const QString here =
        why.isEmpty() ? tr("Here: the last 48 hours of recording, about %1 minutes for %2 frames.")
                            .arg(search_minutes(target_.frame))
                            .arg(ce_frame(target_.frame))
                      : why;
    return info::info_html(
        pal, tr("Find missed passes"),
        tr("Searches the last 48 hours of recorded audio for passes of this picture that were too "
           "weak to notice when they arrived. Now that the picture's header is known, the app can "
           "look for exactly this signal, which finds passes the live listener missed. Any it finds "
           "are added to the picture."),
        here, why.isEmpty(), tr("Use it when"),
        {tr("You started listening late"), tr("The listener was stopped for a while"),
         tr("The sender repeats this picture")},
        tr("Keep in mind"),
        {tr("Takes minutes: about 3 for short frames, about 20 for full ones"),
         tr("Looks only for CE passes"), tr("Only the last 48 hours")},
        info::footnote(pal, tr("Press it before Improve: new passes give Improve more to work with.")));
}

void EmBar::refresh() {
    setVisible(target_.valid());
    if (!target_.valid()) return;
    const Controller::State* st = controller_->state(target_.key());
    const bool running = st && st->running;
    actions_->setVisible(!running);
    working_->setVisible(running);
    // Improve re-receives CE passes only, so an L tile has no Improve at
    // all (Jeff, 2026-10-15); Find stays, since it looks for CE resends.
    const bool l_tile = target_.waveform == QLatin1String("L");
    improve_->setVisible(!l_tile);
    improve_info_->setVisible(!l_tile);
    improve_->setEnabled(controller_->unavailable(Kind::Improve, target_).isEmpty());
    find_->setEnabled(controller_->unavailable(Kind::Find, target_).isEmpty());
    if (running) {
        QString what = st->what;
        if (!st->queued.empty()) {
            what += st->queued.front() == Kind::Improve ? tr(" · Improve next") : tr(" · Find next");
        }
        work_->setText(what);
        bar_->setRange(0, st->total > 0 ? 1000 : 0);
        bar_->setValue(st->total > 0 ? static_cast<int>(1000 * st->done / st->total) : 0);
    }
    QStringList lines;
    const auto add = [&](const QString& line, bool ok) {
        if (line.isEmpty()) return;
        lines << QStringLiteral("<span style=\"color:%1;\">%2</span>")
                     .arg(ok ? info::good_colour(palette()) : info::cost_colour(palette()), line.toHtmlEscaped());
    };
    if (st) {
        add(st->find_line, st->find_ok);
        add(st->improve_line, st->improve_ok);
    }
    result_->setText(lines.join(QStringLiteral("<br>")));
    result_->setVisible(!lines.isEmpty());
    improve_info_->refresh();
    find_info_->refresh();
    if (st && !st->image.isEmpty() && st->image != shown_image_) {
        shown_image_ = st->image;
        emit pictureChanged(shown_image_, st->mean_w_db);
    }
}

}  // namespace sstvae::gui::qrss_em
