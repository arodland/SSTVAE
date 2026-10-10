// QRSS sending from the app (gui/qrss_tx.*): which quarter hours the
// passes take, what the two scripts are asked for, and the sequence --
// encode, make the pass, key it on its start -- against a stub runner,
// a stub player and a clock that starts just before a quarter hour.

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QStringList>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "audio/wavio.hpp"
#include "check.hpp"
#include "qrss_tx.hpp"

using namespace sstvae;
using namespace sstvae::gui;

namespace {

constexpr double Q = 1791594000.0;    // 2026-10-10T01:00Z, a quarter hour

void test_plan() {
    // The first pass whose audio (11 s before its quarter hour) starts
    // at least the prep margin from now.
    std::vector<double> p = qrss_tx::plan(Q - 11.0 - 60.0, 1);
    check::equal(p.size(), std::size_t{1}, "qrss_tx/plan: one pass for mode A");
    check::is_true(p[0] == Q, "qrss_tx/plan: exactly the margin: this quarter hour");
    p = qrss_tx::plan(Q - 11.0 - 59.0, 3);
    check::is_true(p[0] == Q + 900.0, "qrss_tx/plan: a second short: the next one");
    check::is_true(p[1] == Q + 900.0 + 1800.0, "qrss_tx/plan: passes a half hour apart");
    check::is_true(p[2] == Q + 900.0 + 3600.0, "qrss_tx/plan: mode C has three");
    check::equal(qrss_tx::slot_iso(Q).toStdString(), std::string("2026-10-10T01:00Z"),
                 "qrss_tx/plan: slot names");
    check::equal(qrss_tx::minutes_for("B"), 60, "qrss_tx/plan: mode B is an hour");
    check::equal(qrss_tx::passes_for("SSTVAE"), 0, "qrss_tx/plan: not a QRSS mode");
}

void test_problems_and_args() {
    qrss_tx::Request r;
    check::is_true(!qrss_tx::problem(r).empty(), "qrss_tx/problem: no callsign is refused");
    r.callsign = "ag7ew";
    check::is_true(qrss_tx::problem(r).empty(), "qrss_tx/problem: a callsign is enough");
    r.callsign = "AG7EW-1";
    check::is_true(!qrss_tx::problem(r).empty(), "qrss_tx/problem: '-' is not sendable");
    r.callsign = "ag7ew";
    r.freq_hz = 2800.0;
    check::is_true(!qrss_tx::problem(r).empty(), "qrss_tx/problem: carrier outside the band");
    r.freq_hz = 1234.0;

    const qrss_tx::Tools tools{QStringLiteral("py"), QStringLiteral("/repo")};
    r.grid = "CN85qm";
    QStringList a = qrss_tx::transmit_args(tools, "p.qrsp", "o.wav", Q, 1, r);
    const QString line = a.join(QLatin1Char(' '));
    check::equal(line.toStdString(),
                 std::string("py /repo/qrss_transmit.py p.qrsp o.wav --slot 2026-10-10T01:00Z "
                             "--callsign AG7EW --segment 1 --freq 1234.0 --float --grid CN85"),
                 "qrss_tx/args: one pass of the picture, as the script wants it");
    r.grid = "nowhere";
    a = qrss_tx::transmit_args(tools, "p.qrsp", "o.wav", Q, 0, r);
    check::is_true(!a.contains(QStringLiteral("--grid")),
                   "qrss_tx/args: a grid that is not a locator is left out");
    check::equal(qrss_tx::encode_args(tools, "a.png", "p.qrsp", "C")
                     .join(QLatin1Char(' '))
                     .toStdString(),
                 std::string("py /repo/qrss_encode.py a.png p.qrsp --mode C"),
                 "qrss_tx/args: the encode");
}

struct Rig {
    std::vector<int> ptt;              // 1 key, 0 unkey
    std::vector<double> played_at;     // clock when the player started
    std::size_t played_n = 0;
};

void test_run_keys_the_pass_on_its_start() {
    Rig rig;
    QElapsedTimer since;
    since.start();
    const double prep = 0.3;
    const double start = Q - 11.0 - prep;   // the pass's audio is due prep s from now
    auto clock = [&] { return start + since.elapsed() / 1000.0; };

    tx::TxEngine engine(
        [&](bool on) { rig.ptt.push_back(on ? 1 : 0); },
        [&](const std::string&, std::span<const double> wave, int,
            const std::function<void(double)>&, const std::function<bool()>&,
            const std::function<void(const std::string&)>&) {
            rig.played_at.push_back(clock());
            rig.played_n = wave.size();
            return true;
        },
        [](const images::ImageArray&) { return std::vector<double>(); });

    std::vector<QStringList> calls;
    auto runner = [&](const QStringList& argv, QString*) {
        calls.push_back(argv);
        if (argv.at(1).endsWith(QStringLiteral("qrss_transmit.py"))) {
            audio::write_wav_float(argv.at(3).toStdString(), std::vector<double>(800, 0.25));
        }
        return true;
    };
    qrss_tx::Request r;
    r.picture = images::Picture(32, 24);
    r.callsign = "AG7EW";
    r.freq_hz = 1100.0;
    r.prep_s = prep;
    r.tx.ptt_lead_s = 0.0;
    r.tx.ptt_tail_s = 0.0;
    r.tx.level = 0.5;
    std::vector<std::string> errors;
    const bool ok = qrss_tx::run(
        engine, r, {QStringLiteral("py"), QStringLiteral("/repo")},
        [&](const std::string& e) { errors.push_back(e); }, clock, runner);
    check::is_true(ok && errors.empty(), "qrss_tx/run: mode A sent");
    check::equal(calls.size(), std::size_t{2}, "qrss_tx/run: one encode, one pass");
    check::is_true(calls.size() == 2 && calls[1].contains(QStringLiteral("2026-10-10T01:00Z")),
                   "qrss_tx/run: the pass is for its own quarter hour");
    check::is_true(rig.ptt == std::vector<int>{1, 0}, "qrss_tx/run: keyed once, unkeyed once");
    check::equal(rig.played_n, std::size_t{800}, "qrss_tx/run: the pass's audio, whole");
    check::is_true(rig.played_at.size() == 1 && rig.played_at[0] >= Q - 11.0 - 0.02 &&
                       rig.played_at[0] < Q - 11.0 + 0.5,
                   "qrss_tx/run: and not before its start");
}

void test_cancel_while_waiting() {
    tx::TxEngine engine([](bool) {}, {}, {});
    // The pass is fifteen minutes away.
    auto clock = [] { return Q - 11.0 - 900.0; };
    auto runner = [](const QStringList&, QString*) { return true; };
    qrss_tx::Request r;
    r.picture = images::Picture(8, 8);
    r.callsign = "AG7EW";
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.cancel();
    });
    QElapsedTimer t;
    t.start();
    const bool ok = qrss_tx::run(engine, r, {QStringLiteral("py"), QStringLiteral("/r")},
                                 [](const std::string&) {}, clock, runner);
    canceller.join();
    check::is_true(!ok, "qrss_tx/cancel: a cancelled send reports so");
    check::is_true(t.elapsed() < 5000, "qrss_tx/cancel: and stops waiting at once");
    check::equal(std::string(tx::phase_name(engine.state().phase)), std::string("cancelled"),
                 "qrss_tx/cancel: phase");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    QCoreApplication app(argc, argv);
    test_plan();
    test_problems_and_args();
    test_run_keys_the_pass_on_its_start();
    test_cancel_while_waiting();
    return check::report("qrss tx");
}
