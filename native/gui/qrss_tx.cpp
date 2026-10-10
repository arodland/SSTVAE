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

std::vector<double> plan(double now, int passes, double prep_s) {
    std::vector<double> out;
    if (passes <= 0) return out;
    const double audio_start = T0_OFFSET_S - AUDIO_LEAD_S;   // -11 s from the quarter hour
    double slot = std::ceil((now + prep_s - audio_start) / 900.0) * 900.0;
    for (int k = 0; k < passes; ++k) out.push_back(slot + k * PASS_SPACING_S);
    return out;
}

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
    if (is_locator(request.grid)) {
        a << QStringLiteral("--grid") << QString::fromStdString(request.grid.substr(0, 4)).toUpper();
    }
    return a;
}

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
    if (!(request.freq_hz >= FREQ_MIN_HZ && request.freq_hz <= FREQ_MAX_HZ)) {
        return "the QRSS carrier must be between 300 and 2700 Hz";
    }
    return {};
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
    auto fail = [&](const std::string& msg) {
        on_error(msg);
        engine.report(tx::TxPhase::Failed, msg);
        return false;
    };
    const std::string why = problem(request);
    if (!why.empty()) return fail(why);

    QTemporaryDir tmp;
    if (!tmp.isValid()) return fail("QRSS: no temporary directory");
    const QString png = tmp.filePath(QStringLiteral("picture.png"));
    const QString qrsp = tmp.filePath(QStringLiteral("picture.qrsp"));
    images::save_png(request.picture, png.toStdString());

    engine.report(tx::TxPhase::Encoding, "QRSS: encoding the picture");
    QString out;
    if (!runner(encode_args(tools, png, qrsp, request.mode), &out)) {
        if (engine.cancelled()) {
            engine.report(tx::TxPhase::Cancelled, "cancelled");
            return false;
        }
        return fail("QRSS encode failed: " + out.toStdString());
    }

    const int n = passes_for(request.mode);
    const std::vector<double> pass_slots = plan(clock(), n, request.prep_s);
    for (int k = 0; k < n; ++k) {
        const double slot = pass_slots[static_cast<std::size_t>(k)];
        const double audio_start = slot + T0_OFFSET_S - AUDIO_LEAD_S;
        const std::string what = "QRSS pass " + std::to_string(k + 1) + " of " +
                                 std::to_string(n) + " at " + hhmm(slot) + ", " +
                                 QString::number(request.freq_hz, 'f', 0).toStdString() + " Hz";
        if (!engine.wait(audio_start - request.prep_s - clock(), tx::TxPhase::Waiting,
                         "waiting for " + what)) {
            engine.report(tx::TxPhase::Cancelled, "cancelled");
            return false;
        }
        engine.report(tx::TxPhase::Modulating, "making " + what);
        const QString wav = tmp.filePath(QStringLiteral("pass%1.wav").arg(k));
        if (!runner(transmit_args(tools, qrsp, wav, slot, k, request), &out)) {
            if (engine.cancelled()) {
                engine.report(tx::TxPhase::Cancelled, "cancelled");
                return false;
            }
            return fail("QRSS pass audio failed: " + out.toStdString());
        }
        std::vector<double> wave =
            tx::condition_for_output(audio::read_wav(wav.toStdString()), request.tx.level);
        QFile::remove(wav);
        const double lead = audio_start - request.tx.ptt_lead_s - clock();
        if (lead < -1.0) {
            return fail("QRSS: the pass audio was ready " + std::to_string(-lead) +
                        " s after its start; not sending it late");
        }
        if (!engine.wait(lead, tx::TxPhase::Waiting, "starting " + what)) {
            engine.report(tx::TxPhase::Cancelled, "cancelled");
            return false;
        }
        if (!engine.transmit_wave(wave, request.tx)) return false;
    }
    engine.report(tx::TxPhase::Done, "QRSS sent");
    return true;
}

}  // namespace sstvae::gui::qrss_tx
