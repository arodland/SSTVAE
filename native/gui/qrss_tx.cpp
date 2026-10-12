#include "qrss_tx.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QTemporaryDir>
#include <QTimeZone>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <deque>
#include <future>
#include <utility>

#include "audio/wavio.hpp"
#include "images/images.hpp"

namespace sstvae::gui::qrss_tx {

int passes_for(const std::string& mode) {
    if (mode == "A") return 1;
    if (mode == "B") return 2;
    if (mode == "C") return 3;
    return 0;
}

int minutes_for(const std::string& mode) { return 30 * passes_for(mode); }

bool known_waveform(const std::string& waveform) {
    return waveform == WAVEFORM_CE || waveform == WAVEFORM_L;
}

std::vector<double> plan(double now, int passes, double prep_s) {
    std::vector<double> out;
    if (passes <= 0) return out;
    const double audio_start = T0_OFFSET_S - AUDIO_LEAD_S;   // -11 s from the quarter hour
    double slot = std::ceil((now + prep_s - audio_start) / 900.0) * 900.0;
    for (int k = 0; k < passes; ++k) out.push_back(slot + k * PASS_SPACING_S);
    return out;
}

double audio_start(double slot) { return slot + T0_OFFSET_S - AUDIO_LEAD_S; }

double audio_end(double slot) { return slot + T0_OFFSET_S + PASS_S + AUDIO_TAIL_S; }

QString slot_iso(double slot) {
    return QDateTime::fromSecsSinceEpoch(static_cast<qint64>(std::llround(slot)),
                                         QTimeZone::utc())
        .toString(QStringLiteral("yyyy-MM-ddTHH:mmZ"));
}

std::optional<Tools> find_tools() {
    QString dir = QString::fromLocal8Bit(qgetenv("SSTVAE_QRSS_DIR"));
    if (dir.isEmpty()) {
        QDir d(QCoreApplication::applicationDirPath());
        for (int up = 0; up <= 4; ++up) {
            if (QFileInfo::exists(d.filePath(QStringLiteral("qrss_transmit.py")))) {
                dir = d.absolutePath();
                break;
            }
            if (!d.cdUp()) break;
        }
    }
    if (dir.isEmpty()) return std::nullopt;
    QString python = QString::fromLocal8Bit(qgetenv("SSTVAE_QRSS_PYTHON"));
    if (python.isEmpty()) {
        python = QStringLiteral("python3");
        for (const char* venv : {".venv/bin/python", ".venv/Scripts/python.exe"}) {
            const QString p = QDir(dir).filePath(QString::fromLatin1(venv));
            if (QFileInfo::exists(p)) {
                python = p;
                break;
            }
        }
    }
    return Tools{python, dir};
}

namespace {

bool is_locator(const std::string& g) {
    if (g.size() < 4) return false;
    auto up = [](char c) { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); };
    return up(g[0]) >= 'A' && up(g[0]) <= 'R' && up(g[1]) >= 'A' && up(g[1]) <= 'R' &&
           std::isdigit(static_cast<unsigned char>(g[2])) &&
           std::isdigit(static_cast<unsigned char>(g[3]));
}

QString script(const Tools& tools, const char* name) {
    return QDir(tools.dir).filePath(QString::fromLatin1(name));
}

bool default_runner(const QStringList& argv, QString* output, tx::TxEngine& engine) {
    QProcess proc;
    proc.setProcessChannelMode(QProcess::MergedChannels);
    proc.start(argv.front(), argv.mid(1));
    if (!proc.waitForStarted(30000)) {
        if (output) *output = QStringLiteral("could not start %1").arg(argv.front());
        return false;
    }
    while (!proc.waitForFinished(250)) {
        if (engine.cancelled()) {
            proc.kill();
            proc.waitForFinished(5000);
            if (output) *output = QStringLiteral("cancelled");
            return false;
        }
    }
    if (output) *output = QString::fromLocal8Bit(proc.readAll()).trimmed();
    return proc.exitStatus() == QProcess::NormalExit && proc.exitCode() == 0;
}

std::string hhmmss(double t) {
    return QDateTime::fromSecsSinceEpoch(static_cast<qint64>(std::llround(t)), QTimeZone::utc())
               .toString(QStringLiteral("HH:mm:ss"))
               .toStdString() +
           "Z";
}

std::string hhmm(double t) {
    return QDateTime::fromSecsSinceEpoch(static_cast<qint64>(std::llround(t)), QTimeZone::utc())
               .toString(QStringLiteral("HH:mm"))
               .toStdString() +
           "Z";
}

}  // namespace

QStringList encode_args(const Tools& tools, const QString& png, const QString& qrsp,
                        const std::string& mode) {
    return {tools.python, script(tools, "qrss_encode.py"), png, qrsp, QStringLiteral("--mode"),
            QString::fromStdString(mode)};
}

QStringList transmit_args(const Tools& tools, const QString& qrsp, const QString& wav,
                          double slot, int segment, const Request& request) {
    QStringList a{tools.python,
                  script(tools, "qrss_transmit.py"),
                  qrsp,
                  wav,
                  QStringLiteral("--slot"),
                  slot_iso(slot),
                  QStringLiteral("--callsign"),
                  QString::fromStdString(request.callsign).toUpper(),
                  QStringLiteral("--segment"),
                  QString::number(segment),
                  QStringLiteral("--freq"),
                  QString::number(request.freq_hz, 'f', 1),
                  QStringLiteral("--float")};
    // CE is qrss_transmit.py's default, so a CE command line is as it was.
    if (request.waveform != WAVEFORM_CE) {
        a << QStringLiteral("--waveform") << QString::fromStdString(request.waveform);
    }
    if (is_locator(request.grid)) {
        a << QStringLiteral("--grid") << QString::fromStdString(request.grid.substr(0, 4)).toUpper();
    }
    return a;
}

namespace {

// What a picture's own passes must satisfy, in a request or a part.
std::string passes_problem(const std::string& mode, const std::string& waveform,
                           double freq_hz, const std::vector<Pass>& passes) {
    if (passes_for(mode) == 0) return "unknown QRSS mode '" + mode + "'";
    if (!known_waveform(waveform)) return "unknown QRSS waveform '" + waveform + "'";
    if (!(freq_hz >= FREQ_MIN_HZ && freq_hz <= FREQ_MAX_HZ)) {
        return "the QRSS carrier must be between 300 and 2700 Hz";
    }
    const int n = passes_for(mode);
    for (std::size_t k = 0; k < passes.size(); ++k) {
        const Pass& p = passes[k];
        if (p.segment < 0 || p.segment >= n) {
            return "mode " + mode + " has no pass " + std::to_string(p.segment + 1);
        }
        if (std::fmod(p.slot, 900.0) != 0.0) return "QRSS passes start on a quarter hour";
        if (k > 0 && p.slot - passes[k - 1].slot < PASS_SPACING_S) {
            return "QRSS passes last half an hour, so they must be at least that far apart";
        }
    }
    return {};
}

}  // namespace

std::string problem(const Request& request) {
    if (passes_for(request.mode) == 0) return "unknown QRSS mode '" + request.mode + "'";
    const std::string& c = request.callsign;
    if (c.empty()) {
        return "QRSS sends your callsign in every pass (a legal requirement): set it in "
               "Settings first.";
    }
    const bool ok = c.size() <= 8 && std::all_of(c.begin(), c.end(), [](char ch) {
        return std::isalnum(static_cast<unsigned char>(ch)) || ch == '/';
    });
    if (!ok) return "QRSS callsigns are 1-8 of A-Z, 0-9 and /; '" + c + "' is not.";
    return passes_problem(request.mode, request.waveform, request.freq_hz, request.passes);
}

std::string problem(const Part& part, double after) {
    if (part.passes.empty()) return "a following picture needs its passes";
    if (std::string why = passes_problem(part.mode, part.waveform, part.freq_hz, part.passes); !why.empty()) {
        return why;
    }
    if (part.passes.front().slot - after < PASS_SPACING_S) {
        return "a following picture starts at least half an hour after the last pass";
    }
    return {};
}

bool Chain::append(Part part) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return false;
    parts_.push_back(std::move(part));
    return true;
}

std::optional<Part> Chain::take_or_close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (parts_.empty()) {
        closed_ = true;
        return std::nullopt;
    }
    Part p = std::move(parts_.front());
    parts_.pop_front();
    return p;
}

void Chain::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    parts_.clear();
}

bool run(tx::TxEngine& engine, const Request& request, const Tools& tools,
         const std::function<void(const std::string&)>& on_error, Clock clock, Runner runner) {
    if (!clock) {
        clock = [] { return static_cast<double>(QDateTime::currentMSecsSinceEpoch()) / 1000.0; };
    }
    if (!runner) {
        runner = [&engine](const QStringList& argv, QString* out) {
            return default_runner(argv, out, engine);
        };
    }
    // However the run ends, nothing more may join it.
    struct CloseChain {
        Chain* chain;
        ~CloseChain() {
            if (chain) chain->close();
        }
    } close_chain{request.chain.get()};
    auto fail = [&](const std::string& msg) {
        on_error(msg);
        engine.report(tx::TxPhase::Failed, msg);
        return false;
    };
    auto cancelled = [&] {
        engine.report(tx::TxPhase::Cancelled, "cancelled");
        return false;
    };
    const std::string why = problem(request);
    if (!why.empty()) return fail(why);

    QTemporaryDir tmp;
    if (!tmp.isValid()) return fail("QRSS: no temporary directory");

    // Each picture in the run: its own mode, carrier and stored picture.
    // The first is the request's; any others joined through the chain.
    struct Picture {
        Request request;    // its picture, mode and carrier; the rest the run's
        QString qrsp;
        bool encoded = false;
    };
    std::deque<Picture> pictures;
    pictures.push_back({request, {}, false});
    pictures.front().request.chain.reset();
    // Encode picture i; empty on success, else the tool's output.
    auto encode = [&](std::size_t i) -> std::string {
        Picture& p = pictures[i];
        if (p.encoded) return {};
        const QString png = tmp.filePath(QStringLiteral("picture%1.png").arg(i));
        p.qrsp = tmp.filePath(QStringLiteral("picture%1.qrsp").arg(i));
        images::save_png(p.request.picture, png.toStdString());
        QString out;
        const bool ok = runner(encode_args(tools, png, p.qrsp, p.request.mode), &out);
        QFile::remove(png);
        if (!ok) return out.isEmpty() ? std::string("failed") : out.toStdString();
        p.encoded = true;
        return {};
    };

    engine.report(tx::TxPhase::Encoding, "QRSS: encoding the picture (not keyed)");
    if (std::string out = encode(0); !out.empty()) {
        if (engine.cancelled()) return cancelled();
        return fail("QRSS encode failed: " + out);
    }

    // Passes can follow each other with about 2 s between one's audio and
    // the next (a pass's audio lasts ~1798 s, passes are 1800 s apart),
    // so only the first pass's audio is made before keying, and each
    // later one is made and read in while the pass before it plays. A
    // following picture is encoded then too.
    struct Slot {
        std::size_t picture = 0;
        Pass pass;
    };
    std::vector<Slot> passes;
    if (request.passes.empty()) {
        const int segments = passes_for(request.mode);
        const std::vector<double> quarter_hours = plan(clock(), segments, request.prep_s);
        for (int k = 0; k < segments; ++k) {
            passes.push_back({0, {quarter_hours[static_cast<std::size_t>(k)], k}});
        }
    } else {
        for (const Pass& p : request.passes) passes.push_back({0, p});
    }
    auto what = [&](std::size_t k) {
        const Slot& s = passes[k];
        const Request& r = pictures[s.picture].request;
        const int segments = passes_for(r.mode);
        std::string w = "QRSS pass " + std::to_string(k + 1) + " of " +
                        std::to_string(passes.size());
        if (pictures.size() > 1) w += " (picture " + std::to_string(s.picture + 1) + ")";
        if (segments > 1) {
            w += " (part " + std::to_string(s.pass.segment + 1) + " of " +
                 std::to_string(segments) + ")";
        }
        return w + " at " + hhmm(s.pass.slot) + ", " + r.waveform + " at " +
               QString::number(r.freq_hz, 'f', 0).toStdString() + " Hz";
    };
    // Make pass k's audio and read it in; empty with `*error` set if not.
    // Encodes its picture first when that has not been done.
    auto make = [&](std::size_t k, std::string* error) {
        const Slot& s = passes[k];
        if (std::string out = encode(s.picture); !out.empty()) {
            *error = "encoding the next picture: " + out;
            return std::vector<double>();
        }
        const Picture& p = pictures[s.picture];
        const QString wav = tmp.filePath(QStringLiteral("pass%1.wav").arg(k));
        QString output;
        if (!runner(transmit_args(tools, p.qrsp, wav, s.pass.slot, s.pass.segment, p.request),
                    &output)) {
            *error = output.toStdString();
            return std::vector<double>();
        }
        std::vector<double> w =
            tx::condition_for_output(audio::read_wav(wav.toStdString()), request.tx.level);
        QFile::remove(wav);
        return w;
    };
    // Until the first key, every message says when that is: a run
    // getting ready must not read as one on the air.
    const std::string keys_at =
        "; keys at " + hhmmss(audio_start(passes[0].pass.slot) - request.tx.ptt_lead_s);
    if (!engine.wait(audio_start(passes[0].pass.slot) - request.prep_s - clock(),
                     tx::TxPhase::Waiting, "waiting to make " + what(0) + keys_at)) {
        return cancelled();
    }
    engine.report(tx::TxPhase::Modulating, "making " + what(0) + keys_at);
    std::string error;
    std::vector<double> wave = make(0, &error);
    if (wave.empty()) {
        if (engine.cancelled()) return cancelled();
        return fail("QRSS pass audio failed: " + error);
    }
    for (std::size_t k = 0; k < passes.size(); ++k) {
        const double lead = audio_start(passes[k].pass.slot) - request.tx.ptt_lead_s - clock();
        if (lead < -1.0) {
            return fail("QRSS: " + what(k) + " was due to start " + std::to_string(-lead) +
                        " s ago; not sending it late");
        }
        if (!engine.wait(lead, tx::TxPhase::Waiting,
                         "starting " + what(k) + (k == 0 ? keys_at : std::string()))) {
            return cancelled();
        }
        // Keying the last pass: a picture waiting to follow joins now, or
        // the run closes to more.
        while (k + 1 == passes.size() && request.chain) {
            std::optional<Part> part = request.chain->take_or_close();
            if (!part) break;
            if (std::string bad = problem(*part, passes.back().pass.slot); !bad.empty()) {
                on_error("QRSS: the picture to follow was not added: " + bad);
                continue;
            }
            Request r = request;
            r.chain.reset();
            r.passes.clear();
            r.picture = std::move(part->picture);
            r.mode = part->mode;
            r.waveform = part->waveform;
            r.freq_hz = part->freq_hz;
            pictures.push_back({std::move(r), {}, false});
            for (const Pass& p : part->passes) passes.push_back({pictures.size() - 1, p});
        }
        std::string next_error;
        std::future<std::vector<double>> next;
        if (k + 1 < passes.size()) next = std::async(std::launch::async, make, k + 1, &next_error);
        const bool sent = engine.transmit_wave(wave, request.tx);
        if (next.valid()) wave = next.get();
        if (!sent) return false;
        if (k + 1 < passes.size() && wave.empty()) {
            if (engine.cancelled()) return cancelled();
            return fail("QRSS audio for " + what(k + 1) + " failed: " + next_error);
        }
    }
    engine.report(tx::TxPhase::Done, "QRSS sent");
    return true;
}

}  // namespace sstvae::gui::qrss_tx
