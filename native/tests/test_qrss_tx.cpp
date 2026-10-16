// QRSS sending from the app (gui/qrss_tx.*): which quarter hours the
// passes take, what the C++ transmitter and the two scripts are asked
// for, and the sequence -- encode, make the pass, key it on its start --
// against a stub maker or runner, a stub player and a clock that starts
// just before a quarter hour.

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QStringList>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
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
    r.waveform = "L";
    check::is_true(qrss_tx::transmit_args(tools, "p.qrsp", "o.wav", Q, 1, r)
                       .join(QLatin1Char(' '))
                       .endsWith(QStringLiteral("--waveform L --grid CN85")),
                   "qrss_tx/args: waveform L is asked for; CE, the default, is left out");
    r.waveform = "X";
    check::is_true(!qrss_tx::problem(r).empty(), "qrss_tx/problem: an unknown waveform");
    r.waveform = "CE";
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

void test_pass_request() {
    qrss_tx::Request r;
    r.callsign = "ag7ew";
    r.waveform = "L";
    r.freq_hz = 1234.56;
    r.grid = "cn85qm";
    const qrss::api::PassRequest p = qrss_tx::pass_request(r, {Q + 1800.0, 1});
    check::is_true(p.q == qrss::api::slot_of(Q + 1800.0) && p.segment == 1,
                   "qrss_tx/pass_request: the pass's slot and group");
    check::is_true(p.waveform == qrss::api::Waveform::L, "qrss_tx/pass_request: the waveform");
    check::is_true(p.freq_hz == 1234.6,
                   "qrss_tx/pass_request: the carrier to 0.1 Hz, as the script was given it");
    check::is_true(p.callsign == std::optional<std::string>("AG7EW") &&
                       p.grid == std::optional<std::string>("CN85"),
                   "qrss_tx/pass_request: callsign and locator in capitals");
    check::is_true(p.frame == "full" && p.lead_in_s == 0.0 && !p.ook,
                   "qrss_tx/pass_request: a whole pass, no extras");
    r.grid = "nowhere";
    check::is_true(!qrss_tx::pass_request(r, {Q, 0}).grid,
                   "qrss_tx/pass_request: a grid that is not a locator is left out");

    // The C++ transmitter is the default; the scripts only when asked.
    ::unsetenv("SSTVAE_QRSS_BACKEND");
    check::is_true(qrss_tx::backend() == qrss_tx::Backend::Native,
                   "qrss_tx/backend: C++ unless asked otherwise");
    ::setenv("SSTVAE_QRSS_BACKEND", "python", 1);
    check::is_true(qrss_tx::backend() == qrss_tx::Backend::Python,
                   "qrss_tx/backend: SSTVAE_QRSS_BACKEND=python runs the scripts");
    ::unsetenv("SSTVAE_QRSS_BACKEND");

    std::string error;
    check::is_true(qrss_tx::native_maker()->make(0, r, {Q, 0}, &error).empty() && !error.empty(),
                   "qrss_tx/native: a picture not encoded makes no audio, and says so");
}

// run() through a Maker, as the app sends with the C++ transmitter: one
// encode per picture, one make per pass, the output level applied.
void test_run_through_a_maker() {
    struct Stub final : qrss_tx::Maker {
        std::vector<std::string> calls;
        std::string encode(std::size_t i, const qrss_tx::Request& p) override {
            calls.push_back("encode " + std::to_string(i) + " " + p.mode);
            return {};
        }
        std::vector<double> make(std::size_t i, const qrss_tx::Request&, const qrss_tx::Pass& pass,
                                 std::string*) override {
            calls.push_back("make " + std::to_string(i) + " " + std::to_string(pass.segment));
            return std::vector<double>(600, 0.25);
        }
    } maker;
    QElapsedTimer since;
    since.start();
    const double prep = 0.2;
    const double start = Q - 11.0 - prep - 0.5;
    double jump = 0.0;
    std::mutex mu;
    auto clock = [&] {
        std::lock_guard<std::mutex> lock(mu);
        return start + since.elapsed() / 1000.0 + jump;
    };
    double peak = 0.0;
    int played = 0;
    tx::TxEngine engine(
        [](bool) {},
        [&](const std::string&, std::span<const double> wave, int,
            const std::function<void(double)>&, const std::function<bool()>&,
            const std::function<void(const std::string&)>&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
            std::lock_guard<std::mutex> lock(mu);
            for (double v : wave) peak = std::max(peak, std::abs(v));
            ++played;
            jump += (Q - 11.0 + 1800.0 * played - 2.0) - (start + since.elapsed() / 1000.0 + jump);
            return true;
        },
        [](const images::ImageArray&) { return std::vector<double>(); });
    qrss_tx::Request r;
    r.picture = images::Picture(16, 12);
    r.callsign = "AG7EW";
    r.mode = "B";
    r.prep_s = prep;
    r.tx.ptt_lead_s = 0.0;
    r.tx.ptt_tail_s = 0.0;
    r.tx.level = 0.5;
    std::vector<std::string> errors;
    const bool ok = qrss_tx::run(engine, r, maker,
                                 [&](const std::string& e) { errors.push_back(e); }, clock);
    check::is_true(ok && errors.empty(), "qrss_tx/maker: mode B sent");
    check::is_true(maker.calls == std::vector<std::string>{"encode 0 B", "make 0 0", "make 0 1"},
                   "qrss_tx/maker: encoded once, then each pass made");
    check::is_true(played == 2 && std::abs(peak - 0.5) < 1e-9,
                   "qrss_tx/maker: both passes played, at the output level");
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
    // The pass's audio is due prep s from now, with 0.5 s spare. The
    // spare is what lets the run plan this quarter hour: with none, a
    // millisecond spent before `plan` read the clock moved the pass to
    // the next quarter hour, fifteen (simulated and real) minutes away,
    // and the test hung whenever the machine was busy.
    const double start = Q - 11.0 - prep - 0.5;
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

// A pass's audio ends ~2 s before the next one's starts: the player
// jumps the clock to that moment, so the next pass's audio has to have
// been made while this one played. `passes` empty sends the mode's own
// passes; otherwise those (a scheduled send's repeats).
void check_back_to_back(const std::string& name, const std::string& mode,
                        std::vector<qrss_tx::Pass> passes, std::vector<int> segments) {
    const int n = static_cast<int>(segments.size());
    std::mutex mu;
    std::vector<std::string> events;
    QElapsedTimer since;
    since.start();
    const double prep = 0.2;
    double jump = 0.0;
    int played = 0;
    // 1 s spare before the first pass's making is due, so a busy
    // machine still plans this quarter hour (see above).
    const double start = Q - 11.0 - prep - 1.0;
    auto clock = [&] {
        std::lock_guard<std::mutex> lock(mu);
        return start + since.elapsed() / 1000.0 + jump;
    };
    tx::TxEngine engine(
        [](bool) {},
        [&](const std::string&, std::span<const double> wave, int,
            const std::function<void(double)>&, const std::function<bool()>&,
            const std::function<void(const std::string&)>&) {
            // Long enough for the next pass to be made "during" this one.
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            std::lock_guard<std::mutex> lock(mu);
            events.push_back("play " + std::to_string(wave.size()));
            ++played;
            jump += (Q - 11.0 + 1800.0 * played - 2.0) -
                    (start + since.elapsed() / 1000.0 + jump);
            return true;
        },
        [](const images::ImageArray&) { return std::vector<double>(); });
    int made = 0;
    auto runner = [&](const QStringList& argv, QString*) {
        if (argv.at(1).endsWith(QStringLiteral("qrss_transmit.py"))) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const int seg = argv.at(argv.indexOf(QStringLiteral("--segment")) + 1).toInt();
            std::lock_guard<std::mutex> lock(mu);
            // Pass k's audio is 700 + 10 k + segment samples long, so a
            // play names both which pass it is and which part it carries.
            audio::write_wav_float(argv.at(3).toStdString(),
                                   std::vector<double>(700 + 10 * made + seg, 0.25));
            events.push_back("make " + std::to_string(made));
            ++made;
        }
        return true;
    };
    qrss_tx::Request r;
    r.picture = images::Picture(16, 12);
    r.callsign = "AG7EW";
    r.mode = mode;
    r.prep_s = prep;
    r.passes = std::move(passes);
    r.tx.ptt_lead_s = 0.0;
    r.tx.ptt_tail_s = 0.0;
    std::vector<std::string> errors;
    const bool ok = qrss_tx::run(engine, r, {QStringLiteral("py"), QStringLiteral("/repo")},
                                 [&](const std::string& e) { errors.push_back(e); }, clock,
                                 runner);
    check::is_true(ok && errors.empty(), name + "every pass sent");
    auto at = [&](const std::string& e) {
        return std::find(events.begin(), events.end(), e) - events.begin();
    };
    bool order = static_cast<int>(events.size()) == 2 * n && events.front() == "make 0";
    for (int k = 0; k < n && order; ++k) {
        const std::string play = "play " + std::to_string(700 + 10 * k + segments[k]);
        order = at("make " + std::to_string(k)) < at(play);
        if (k > 0) {
            const std::string before =
                "play " + std::to_string(700 + 10 * (k - 1) + segments[k - 1]);
            // In order, and made while the pass before it played: there
            // is no time to make it after.
            order = order && at(before) < at(play) && at("make " + std::to_string(k)) < at(before);
        }
        // Never more than one pass ahead: two passes' audio on disk.
        if (k >= 2) {
            order = order && at("make " + std::to_string(k)) >
                                 at("play " + std::to_string(700 + 10 * (k - 2) +
                                                             segments[k - 2]));
        }
    }
    check::is_true(order, name + "each pass made while the one before plays, in order");
}

void test_passes_back_to_back() {
    check_back_to_back("qrss_tx/mode B: ", "B", {}, {0, 1});
    check_back_to_back("qrss_tx/mode C: ", "C", {}, {0, 1, 2});
    // A scheduled "2 x A": the same one-pass picture, twice in a row.
    check_back_to_back("qrss_tx/2 x A: ", "A", {{Q, 0}, {Q + 1800.0, 0}}, {0, 0});
    // Mode B twice: parts 1, 2, 1, 2.
    check_back_to_back("qrss_tx/2 x B: ", "B",
                       {{Q, 0}, {Q + 1800.0, 1}, {Q + 3600.0, 0}, {Q + 5400.0, 1}},
                       {0, 1, 0, 1});
}

void test_pass_lists_are_checked() {
    qrss_tx::Request r;
    r.callsign = "AG7EW";
    r.mode = "A";
    r.passes = {{Q, 0}, {Q + 1800.0, 0}};
    check::is_true(qrss_tx::problem(r).empty(), "qrss_tx/passes: two A passes, back to back");
    r.passes = {{Q, 0}, {Q + 900.0, 0}};
    check::is_true(!qrss_tx::problem(r).empty(), "qrss_tx/passes: overlapping passes refused");
    r.passes = {{Q, 1}};
    check::is_true(!qrss_tx::problem(r).empty(), "qrss_tx/passes: mode A has no part 2");
    r.passes = {{Q + 60.0, 0}};
    check::is_true(!qrss_tx::problem(r).empty(), "qrss_tx/passes: off the quarter hour");
    check::is_true(qrss_tx::audio_start(Q) == Q - 11.0 && qrss_tx::audio_end(Q) == Q + 4.0 + qrss::api::PASS_S,
                   "qrss_tx/passes: a pass's audio runs from 11 s before its quarter hour");
}

// A different picture following straight on: it joins the run, is
// encoded and made while the pass before it plays, and goes out in its
// own mode on its own carrier.
void test_a_following_picture_joins_the_run() {
    std::mutex mu;
    std::vector<std::string> events;
    QElapsedTimer since;
    since.start();
    const double prep = 0.2;
    double jump = 0.0;
    int played = 0;
    const double start = Q - 11.0 - prep - 1.0;
    auto clock = [&] {
        std::lock_guard<std::mutex> lock(mu);
        return start + since.elapsed() / 1000.0 + jump;
    };
    auto chain = std::make_shared<qrss_tx::Chain>();
    tx::TxEngine engine(
        [](bool) {},
        [&](const std::string&, std::span<const double> wave, int,
            const std::function<void(double)>&, const std::function<bool()>&,
            const std::function<void(const std::string&)>&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            std::lock_guard<std::mutex> lock(mu);
            events.push_back("play " + std::to_string(wave.size()));
            ++played;
            jump += (Q - 11.0 + 1800.0 * played - 2.0) -
                    (start + since.elapsed() / 1000.0 + jump);
            return true;
        },
        [](const images::ImageArray&) { return std::vector<double>(); });
    int made = 0;
    auto runner = [&](const QStringList& argv, QString*) {
        std::lock_guard<std::mutex> lock(mu);
        if (argv.at(1).endsWith(QStringLiteral("qrss_encode.py"))) {
            events.push_back("encode " + argv.at(5).toStdString());
            return true;
        }
        const QString freq = argv.at(argv.indexOf(QStringLiteral("--freq")) + 1);
        const int seg = argv.at(argv.indexOf(QStringLiteral("--segment")) + 1).toInt();
        audio::write_wav_float(argv.at(3).toStdString(),
                               std::vector<double>(700 + 10 * made + seg, 0.25));
        events.push_back("make " + std::to_string(made) + " " + freq.toStdString() + " " +
                         QFileInfo(argv.at(2)).fileName().toStdString());
        ++made;
        return true;
    };
    qrss_tx::Request r;
    r.picture = images::Picture(16, 12);
    r.callsign = "AG7EW";
    r.mode = "A";
    r.prep_s = prep;
    r.passes = {{Q, 0}};
    r.chain = chain;
    r.tx.ptt_lead_s = 0.0;
    r.tx.ptt_tail_s = 0.0;
    qrss_tx::Part part;
    part.picture = images::Picture(16, 12);
    part.mode = "B";
    part.freq_hz = 900.0;
    part.passes = {{Q + 1800.0, 0}, {Q + 3600.0, 1}};
    check::is_true(chain->append(part), "qrss_tx/chain: a following picture is taken");
    std::vector<std::string> errors;
    const bool ok = qrss_tx::run(engine, r, {QStringLiteral("py"), QStringLiteral("/repo")},
                                 [&](const std::string& e) { errors.push_back(e); }, clock,
                                 runner);
    check::is_true(ok && errors.empty(), "qrss_tx/chain: every pass of both pictures sent");
    const std::vector<std::string> want = {
        "encode A",
        "make 0 1500.0 picture0.qrsp",
        "encode B",                       // while pass 1 plays
        "make 1 900.0 picture1.qrsp",
        "play 700",
        "make 2 900.0 picture1.qrsp",
        "play 710",
        "play 721",
    };
    check::is_true(events == want, "qrss_tx/chain: encoded and made while the pass before plays");
    if (events != want) {
        for (const auto& e : events) std::fprintf(stderr, "  %s\n", e.c_str());
    }
    check::is_true(!chain->append(part), "qrss_tx/chain: a finished run takes no more");

    qrss_tx::Part early = part;
    early.passes = {{Q + 900.0, 0}, {Q + 2700.0, 1}};
    check::is_true(!qrss_tx::problem(early, Q).empty(),
                   "qrss_tx/chain: a part may not start before the last pass ends");
    check::is_true(qrss_tx::problem(part, Q).empty(), "qrss_tx/chain: straight on is fine");
}

// A pass taken off the schedule while the run plays is not sent, and
// its audio is not made (Clear schedule on "3 x A", 2026-10-16).
void test_a_dropped_pass_is_skipped() {
    struct Stub final : qrss_tx::Maker {
        int made = 0;
        std::string encode(std::size_t, const qrss_tx::Request&) override { return {}; }
        std::vector<double> make(std::size_t, const qrss_tx::Request&, const qrss_tx::Pass&,
                                 std::string*) override {
            ++made;
            return std::vector<double>(600, 0.25);
        }
    } maker;
    QElapsedTimer since;
    since.start();
    const double prep = 0.2;
    const double start = Q - 11.0 - prep - 0.5;
    double jump = 0.0;
    std::mutex mu;
    auto clock = [&] {
        std::lock_guard<std::mutex> lock(mu);
        return start + since.elapsed() / 1000.0 + jump;
    };
    auto chain = std::make_shared<qrss_tx::Chain>();
    int played = 0;
    tx::TxEngine engine(
        [](bool) {},
        [&](const std::string&, std::span<const double>, int, const std::function<void(double)>&,
            const std::function<bool()>&, const std::function<void(const std::string&)>&) {
            if (played == 0) chain->drop({Q + 3600.0});
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
            std::lock_guard<std::mutex> lock(mu);
            ++played;
            jump += (Q - 11.0 + 1800.0 * played - 2.0) - (start + since.elapsed() / 1000.0 + jump);
            return true;
        },
        [](const images::ImageArray&) { return std::vector<double>(); });
    qrss_tx::Request r;
    r.picture = images::Picture(16, 12);
    r.callsign = "AG7EW";
    r.mode = "A";
    r.prep_s = prep;
    r.passes = {{Q, 0}, {Q + 1800.0, 0}, {Q + 3600.0, 0}};
    r.chain = chain;
    r.tx.ptt_lead_s = 0.0;
    r.tx.ptt_tail_s = 0.0;
    std::vector<std::string> errors;
    const bool ok = qrss_tx::run(engine, r, maker,
                                 [&](const std::string& e) { errors.push_back(e); }, clock);
    check::is_true(ok && errors.empty() && played == 2 && maker.made == 2,
                   "qrss_tx/drop: two passes sent, the third neither made nor sent");
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
    // About 20 s when well; a pass planned on the wrong quarter hour
    // waits fifteen minutes, and this names the step that does.
    check::Watchdog watchdog(90.0, "qrss tx");
    check::current_step = "plan";
    test_plan();
    check::current_step = "problems and args";
    test_problems_and_args();
    check::current_step = "pass request";
    test_pass_request();
    check::current_step = "run through a maker";
    test_run_through_a_maker();
    check::current_step = "run keys the pass on its start";
    test_run_keys_the_pass_on_its_start();
    check::current_step = "passes back to back";
    test_passes_back_to_back();
    check::current_step = "pass lists are checked";
    test_pass_lists_are_checked();
    check::current_step = "a following picture joins the run";
    test_a_following_picture_joins_the_run();
    test_a_dropped_pass_is_skipped();
    check::current_step = "cancel while waiting";
    test_cancel_while_waiting();
    return check::report("qrss tx");
}
