// native/core/qrss/em_api (the app's EM and retro calls) without Python.
//
// What is checked is the API's own: that refine is em::em_refine (the same
// history and the same stored passes as a direct call on a copy of the
// store), its progress (done rising by one per pass, total never falling,
// the last call complete), its typed result, and its cancel (from the
// progress hook, mid-round and at a round's end, and before the call: a
// result, never an exception, with every file of the store still loading
// and exactly the passes reported re-received); that retro finds a weak
// TINY pass the passband store holds (made by the C++ transmitter, at
// -16 dB against this file's noise), attaches it by correlation, reports
// it typed and finds nothing on a second run, with its own progress and
// cancel; the refusals, each with the CLI's text and in the CLI's order;
// and both calls on the store and passband store of a running
// listen_api::Session (Session::store() / passband()), with the session's
// listener writing the same passband directory meanwhile.
//
// The numbers EM and retro compute are their ports' (test_em_refine.cpp,
// test_em_search.cpp, and the parity files against Python); this file
// pins none of them.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "em_api_fixture.hpp"
#include "em_refine_pins.hpp"
#include "qrss/api.hpp"
#include "qrss/ce.hpp"
#include "qrss/config.hpp"
#include "qrss/em.hpp"
#include "qrss/em_api.hpp"
#include "qrss/em_refine.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/listen_api.hpp"
#include "qrss/passband.hpp"
#include "qrss/precoder.hpp"
#include "qrss/rational.hpp"
#include "qrss/receiver.hpp"
#include "qrss/rx_api.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace em = sstvae::qrss::em;
namespace ea = sstvae::qrss::em_api;
namespace rc = sstvae::qrss::receiver;
namespace fr = sstvae::qrss::frame;
namespace hd = sstvae::qrss::header;
namespace qs = sstvae::qrss::store;
namespace pb = sstvae::qrss::passband;
namespace ls = sstvae::qrss::listen;
namespace types = sstvae::qrss::types;
namespace cfg = sstvae::qrss::config;
namespace api = sstvae::qrss::api;
namespace fx = sstvae::em_fixture;
namespace P = em_refine_pins;
namespace fs = std::filesystem;
using cd = std::complex<double>;
using qrss::Rational;

namespace {

struct Step {
    explicit Step(const char* name) { check::current_step = name; }
};

template <typename E, typename F>
void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("threw another type: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

fs::path fresh_dir(const std::string& name) {
    const fs::path p = fs::temp_directory_path() /
                       ("sstvae_test_em_api_" + name + "_" +
                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

struct Cleanup {
    fs::path dir;
    ~Cleanup() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// --- fixture A: three TINY passes with captures (test_em_refine.cpp's) -----------------------------

std::vector<double> hnoise(std::int64_t n, std::uint64_t seed) {
    constexpr std::uint64_t M32 = 0xFFFFFFFFull;
    std::vector<double> out(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        std::uint64_t h = (static_cast<std::uint64_t>(i) * 0x9E3779B1ull + seed) & M32;
        h ^= h >> 15;
        h = (h * 0x85EBCA77ull) & M32;
        h ^= h >> 13;
        h = (h * 0xC2B2AE3Dull) & M32;
        h ^= h >> 16;
        out[static_cast<std::size_t>(i)] = static_cast<double>(h) / 4294967296.0 - 0.5;
    }
    return out;
}

std::vector<cd> noise_of(std::int64_t n, double scale, std::uint64_t k) {
    const std::uint64_t s = 8 * k;
    const auto u1 = hnoise(n, s + 1), u2 = hnoise(n, s + 2), u3 = hnoise(n, s + 3), u4 = hnoise(n, s + 4);
    const auto u5 = hnoise(n, s + 5), u6 = hnoise(n, s + 6), u7 = hnoise(n, s + 7), u8 = hnoise(n, s + 8);
    std::vector<cd> out(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double re = ((u1[i] + u2[i]) + u3[i]) + u4[i];
        const double im = ((u5[i] + u6[i]) + u7[i]) + u8[i];
        out[i] = cd(re * scale, im * scale);
    }
    return out;
}

hd::HeaderFields test_header() {
    hd::HeaderFields h;
    h.callsign = "K1ABC";
    h.grid = "FN42";
    h.picture_id = 0x12345678;
    h.mode = 0;
    h.segment = 0;
    h.codec_id = 0xD1D8;
    return h;
}

const qs::Key KEY{"K1ABC", 0x12345678};

std::vector<double> air(const fr::FrameSpec& spec) {
    std::vector<double> a;
    for (std::int64_t k = 0; k < spec.n_data(); ++k) a.push_back((k * 5 + 1) % 3 == 0 ? 1.0 : -1.0);
    return a;
}

rc::Prepared cap(const fr::FrameSpec& spec, std::int64_t n, std::int64_t k) {
    const std::int64_t q = P::Q + 7 * k;
    const std::vector<std::uint8_t> bits = hd::encode(test_header());
    const fr::HdrBits hb = fr::HdrBits::of(bits);
    const std::vector<double> data = qrss::precoder::precode(air(spec), static_cast<std::uint64_t>(q));
    const auto sym = fr::assemble(spec, spec.n_hdr() ? &hb : nullptr, data);
    const auto phi = qrss::ce::phase_grid(250, -3000, n, sym, spec);
    const auto noise = noise_of(n, P::NOISE, static_cast<std::uint64_t>(k));
    rc::Prepared p;
    for (std::int64_t i = 0; i < n; ++i) {
        const double di = static_cast<double>(i);
        const double sq = (di / 250) * (di / 250);
        const double ph = phi[static_cast<std::size_t>(i)] + 2 * std::numbers::pi * (0.02 * di / 250 + 0.0005 * sq);
        const cd z = cd(std::cos(ph), std::sin(ph)) + noise[static_cast<std::size_t>(i)];
        p.fe.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    }
    p.q = q;
    p.t0_index = 3002.5;
    p.fs = cfg::CH_FS;
    return p;
}

types::Detection det_of() {
    types::Detection d;
    d.f_hz = 1500.0;
    d.path = types::FreqPath{{-10.0, 0.0, 1800.0}, {1500.0, 1500.0, 1500.0}, {1.0, 1.0, 1.0}};
    d.z_ref = std::numeric_limits<double>::quiet_NaN();
    d.method = "tbd";
    d.waveform = 0;
    return d;
}

std::vector<types::PassResult> tiny_passes() {
    const auto& TINY = fr::get("tiny");
    std::vector<types::PassResult> out;
    for (std::int64_t k = 0; k < 3; ++k) {
        const rc::Prepared prep = cap(TINY, P::N_TINY, k);
        rc::PassArgs a;
        a.f_mix_hz = Rational(0);
        rc::PassOutcome o = rc::receive_pass(&prep, TINY, det_of(), a);
        if (!o.pass) throw std::runtime_error("receive_pass refused the TINY fixture");
        out.push_back(std::move(*o.pass));
    }
    return out;
}

// A store at `dir` holding the passes, attached to KEY in order.
void fill(const fs::path& dir, const std::vector<types::PassResult>& passes) {
    qs::Store st(dir.string());
    for (const types::PassResult& p : passes) {
        st.add_pass(p);
        st.attach(KEY, p.uid, 0, "test", 0, 0xD1D8);
    }
}

// Every file of the store loads, and the accumulator holds the members.
void store_loads(const fs::path& dir, const qs::Key& key, const std::string& what) {
    try {
        qs::Store st(dir.string());
        for (const std::string& u : st.pass_uids()) (void)st.load_pass(u);
        const qs::Accumulator acc = st.accumulator(key);
        check::is_true(!acc.uids().empty(), what + ": the accumulator holds its members");
        check::is_true(std::isfinite(em::acc_mean_w_db(acc)), what + ": its mean W is finite");
    } catch (const std::exception& e) {
        check::fail(what, std::string("the store does not load: ") + e.what());
    }
}

struct Progress {
    std::vector<std::int64_t> done, total;
    std::vector<std::string> what;
    em::Hooks hooks(std::vector<std::string>* log = nullptr) {
        em::Hooks h;
        h.progress = [this](std::int64_t d, std::int64_t t, const std::string& w) {
            done.push_back(d);
            total.push_back(t);
            what.push_back(w);
        };
        if (log) h.log = [log](const std::string& l) { log->push_back(l); };
        return h;
    }
    // done rises by one per call and never passes total, which never falls;
    // except refine's last call when EM converges early, (done, done,
    // "converged after round r"): done repeated, total lowered to it
    void monotone(const std::string& tag) const {
        for (std::size_t i = 0; i < done.size(); ++i) {
            check::is_true(done[i] <= total[i], tag + ": done <= total at call " + std::to_string(i));
            if (i == 0) continue;
            const bool converged = i + 1 == done.size() && what[i].rfind("converged after round ", 0) == 0;
            if (converged) {
                check::is_true(done[i] == done[i - 1] && total[i] == done[i],
                               tag + ": the converged call closes at done");
                continue;
            }
            check::is_true(done[i] == done[i - 1] + 1, tag + ": done rises by one at call " + std::to_string(i));
            check::is_true(total[i] >= total[i - 1], tag + ": total never falls at call " + std::to_string(i));
        }
    }
};

// --- refine ------------------------------------------------------------------------------------------

void test_refine(const std::vector<types::PassResult>& passes) {
    Step step("refine: against em_refine");
    const fs::path root = fresh_dir("refine");
    Cleanup cleanup{root};
    fill(root / "a", passes);
    fs::copy(root / "a", root / "b", fs::copy_options::recursive);

    // the API, two rounds that never converge (tol -inf)
    ea::RefineOptions o;
    o.rounds = 2;
    o.tol_db = -std::numeric_limits<double>::infinity();
    Progress pr;
    std::vector<std::string> log;
    const auto t = std::chrono::steady_clock::now();
    qs::Store a((root / "a").string());
    const ea::RefineResult r = ea::refine(a, KEY, o, pr.hooks(&log));
    std::printf("refine: two rounds over three TINY passes in %.2f s\n", seconds_since(t));

    // em_refine itself on the copy
    qs::Store b((root / "b").string());
    em::EmRefineArgs args;
    args.rounds = 2;
    args.tol_db = o.tol_db;
    const std::vector<double> direct = em::em_refine(b, KEY, args);

    check::is_true(!r.cancelled, "refine: not cancelled");
    check::equal(r.history.size(), std::size_t{3}, "refine: history before + 2 rounds");
    check::is_true(r.history == direct, "refine: history is em_refine's, to the bit");
    check::equal(r.rounds_run, std::int64_t{2}, "refine: rounds_run");
    check::is_true(r.mean_w_db == r.history.back(), "refine: mean_w_db is the last of history");
    check::equal(r.passes.size(), std::size_t{6}, "refine: six passes re-received");
    check::equal(log.size(), std::size_t{6}, "refine: six log lines");
    for (std::size_t i = 0; i < r.passes.size(); ++i) {
        const ea::RefinedPass& p = r.passes[i];
        const std::int64_t round = static_cast<std::int64_t>(i / 3) + 1;
        check::equal(p.round, round, "refine: pass " + std::to_string(i) + "'s round");
        check::equal(p.uid, passes[i % 3].uid, "refine: pass " + std::to_string(i) + " in member order");
        const std::string want = "round " + std::to_string(round) + " pass " + p.uid + ": mean W " +
                                 fr::py_format(p.mean_w_db_before, "%+.2f") + " -> " +
                                 fr::py_format(p.mean_w_db_after, "%+.2f") + " dB";
        check::equal(log[i], want, "refine: log line " + std::to_string(i) + " says the pass's numbers");
    }
    // the passes stored are em_refine's
    for (const types::PassResult& p : passes) {
        const types::PassResult pa = a.load_pass(p.uid), pb_ = b.load_pass(p.uid);
        check::is_true(pa.z == pb_.z && pa.w == pb_.w && pa.em_round == pb_.em_round,
                       "refine: pass " + p.uid + " stored as em_refine stores it");
        check::equal(pa.em_round, std::int64_t{2}, "refine: pass " + p.uid + " re-received in round 2");
    }
    // progress: one call per pass, done 1..6 of 6, complete
    pr.monotone("refine");
    check::equal(pr.done.size(), std::size_t{6}, "refine: six progress calls");
    check::is_true(!pr.done.empty() && pr.done.back() == 6 && pr.total.back() == 6, "refine: progress complete");
    check::equal(pr.what.empty() ? std::string() : pr.what.front(), "round 1 pass " + passes[0].uid,
                 "refine: progress says what");

    // converging: the "converged after round r" call closes the progress
    Step step2("refine: convergence");
    fs::remove_all(root / "c");
    fill(root / "c", passes);
    qs::Store c((root / "c").string());
    ea::RefineOptions oc;
    oc.rounds = 3;
    oc.tol_db = 1e9;   // the first round's gain is below it
    Progress pc;
    const ea::RefineResult rc_ = ea::refine(c, KEY, oc, pc.hooks());
    check::equal(rc_.history.size(), std::size_t{2}, "refine: stops after one round");
    pc.monotone("refine converging");
    check::is_true(!pc.what.empty() && pc.what.back() == "converged after round 1", "refine: the converged call");
    check::is_true(!pc.done.empty() && pc.done.back() == pc.total.back(), "refine: progress closed at done");

    // rounds 0 and below: nothing runs, no progress
    Progress p0;
    ea::RefineOptions o0;
    o0.rounds = 0;
    const ea::RefineResult r0 = ea::refine(c, KEY, o0, p0.hooks());
    check::is_true(r0.history.size() == 1 && r0.passes.empty() && p0.done.empty() && !r0.cancelled,
                   "refine: rounds 0 runs nothing");
    o0.rounds = -5;
    check::equal(ea::refine(c, KEY, o0).history.size(), std::size_t{1}, "refine: negative rounds run nothing");
    // a huge round count is clamped (the progress total cannot overflow) and still converges
    Progress ph;
    ea::RefineOptions big;
    big.rounds = std::numeric_limits<std::int64_t>::max();
    big.tol_db = 1e9;
    const ea::RefineResult rb = ea::refine(c, KEY, big, ph.hooks());
    check::equal(rb.history.size(), std::size_t{2}, "refine: a huge round count stops at convergence");
    check::is_true(!ph.total.empty() && ph.total.front() == ea::MAX_ROUNDS * 3,
                   "refine: the total counts MAX_ROUNDS rounds");
}

void test_refine_cancel(const std::vector<types::PassResult>& passes) {
    Step step("refine: cancel");
    const fs::path root = fresh_dir("cancel");
    Cleanup cleanup{root};

    // mid-round: cancel set by the second progress call
    fill(root / "a", passes);
    {
        qs::Store st((root / "a").string());
        const double before = em::acc_mean_w_db(st.accumulator(KEY));
        std::atomic<bool> stop{false};
        Progress pr;
        em::Hooks h = pr.hooks();
        auto inner = h.progress;
        h.progress = [&](std::int64_t d, std::int64_t t, const std::string& w) {
            inner(d, t, w);
            if (d == 2) stop = true;
        };
        h.cancel = &stop;
        ea::RefineOptions o;
        o.rounds = 3;
        o.tol_db = -std::numeric_limits<double>::infinity();
        ea::RefineResult r;
        try {
            r = ea::refine(st, KEY, o, h);
        } catch (const std::exception& e) {
            check::fail("refine cancel mid-round", std::string("threw: ") + e.what());
        }
        check::is_true(r.cancelled, "refine cancel mid-round: cancelled");
        check::equal(r.passes.size(), std::size_t{2}, "refine cancel mid-round: two passes stored");
        check::is_true(r.history.size() == 1 && r.history[0] == before, "refine cancel mid-round: history is the before");
        check::equal(r.rounds_run, std::int64_t{0}, "refine cancel mid-round: no round ran to its end");
        check::is_true(r.mean_w_db == em::acc_mean_w_db(st.accumulator(KEY)),
                       "refine cancel mid-round: mean_w_db is the store's now");
        for (std::size_t i = 0; i < passes.size(); ++i)
            check::equal(st.load_pass(passes[i].uid).em_round, std::int64_t{i < 2 ? 1 : 0},
                         "refine cancel mid-round: pass " + std::to_string(i) + "'s em_round");
        store_loads(root / "a", KEY, "refine cancel mid-round");
    }

    // at a round's end: the cancel comes with the third pass, the round is whole
    fill(root / "b", passes);
    {
        qs::Store st((root / "b").string());
        std::atomic<bool> stop{false};
        em::Hooks h;
        h.progress = [&](std::int64_t d, std::int64_t, const std::string&) {
            if (d == 3) stop = true;
        };
        h.cancel = &stop;
        ea::RefineOptions o;
        o.rounds = 3;
        o.tol_db = -std::numeric_limits<double>::infinity();
        const ea::RefineResult r = ea::refine(st, KEY, o, h);
        check::is_true(r.cancelled && r.passes.size() == 3, "refine cancel at a round's end: three passes stored");
        check::is_true(r.history.size() == 1, "refine cancel at a round's end: round 1's value only in mean_w_db");
        check::is_true(r.mean_w_db == em::acc_mean_w_db(st.accumulator(KEY)),
                       "refine cancel at a round's end: mean_w_db is after round 1");
        store_loads(root / "b", KEY, "refine cancel at a round's end");

        // and the second round's start is recorded once it begins
        std::atomic<bool> stop2{false};
        em::Hooks h2;
        h2.progress = [&](std::int64_t d, std::int64_t, const std::string&) {
            if (d == 4) stop2 = true;
        };
        h2.cancel = &stop2;
        const ea::RefineResult r2 = ea::refine(st, KEY, o, h2);
        check::is_true(r2.cancelled && r2.passes.size() == 4 && r2.history.size() == 2 && r2.rounds_run == 1,
                       "refine cancel in round 2: round 1 in history");
    }

    // before the call: nothing re-received, nothing written
    fill(root / "c", passes);
    {
        qs::Store st((root / "c").string());
        std::atomic<bool> stop{true};
        em::Hooks h;
        h.cancel = &stop;
        const ea::RefineResult r = ea::refine(st, KEY, {}, h);
        check::is_true(r.cancelled && r.passes.empty() && r.history.size() == 1, "refine cancel before: nothing ran");
        for (const types::PassResult& p : passes)
            check::equal(st.load_pass(p.uid).em_round, std::int64_t{0}, "refine cancel before: " + p.uid + " untouched");
    }
}

// --- retro --------------------------------------------------------------------------------------------

const std::int64_t Q_RETRO = P::Q + 100;
constexpr double WEAK_DB = -16.0;

struct RetroStore {
    qrss::picture::StoredPicture sp;
    qs::Key key;
    std::string member;
};

// root/store: the picture's accumulator, one genie FULL pass at +5 dB (no
// capture); root/store/passband: slot Q_RETRO holding a TINY pass of it at
// WEAK_DB (the C++ transmitter's audio, this file's noise), written as the
// listener writes the front end.
RetroStore retro_store(const fs::path& root) {
    RetroStore r;
    r.sp = fx::picture0();
    r.key = qs::Key{"K1ABC", r.sp.picture_id};
    qs::Store st((root / "store").string());
    const types::PassResult m = fx::genie_pass(r.sp, fr::get("full"), std::pow(10.0, 0.5), 1, Q_RETRO - 7);
    st.add_pass(m);
    st.attach(r.key, m.uid, 0, "header", 0, 0xD1D8);
    r.member = m.uid;

    api::PassRequest req;
    req.q = Q_RETRO;
    req.callsign = "K1ABC";
    req.frame = "tiny";
    const std::vector<float> audio = api::pass_audio(api::Source(r.sp), req);
    const std::size_t n = audio.size() + static_cast<std::size_t>(30 * cfg::FS);
    std::vector<double> rec = fx::gauss(7, n);
    fx::add_pass(rec, audio, 0, 1.0, WEAK_DB);
    const qrss::frontend::Capture c = api::capture_from_audio(std::span<const double>(rec), Q_RETRO);
    pb::PassbandStore pbs((root / "store" / "passband").string());
    pbs.write_capture(c, false);
    return r;
}

void test_retro() {
    Step step("retro: a weak TINY pass");
    const fs::path root = fresh_dir("retro");
    Cleanup cleanup{root};
    const RetroStore rs = retro_store(root);
    qs::Store st((root / "store").string());

    Progress pr;
    std::vector<std::string> log;
    const auto t = std::chrono::steady_clock::now();
    const ea::RetroResult r = ea::retro(st, rs.key, ea::Passband{}, "tiny", {}, pr.hooks(&log));
    std::printf("retro: %lld slot(s), %lld searched, in %.2f s\n", static_cast<long long>(r.slots_total),
                static_cast<long long>(r.slots_searched), seconds_since(t));
    for (const std::string& l : log) std::printf("  %s\n", l.c_str());
    check::is_true(!r.cancelled, "retro: not cancelled");
    check::equal(r.finds.size(), std::size_t{1}, "retro: one find");
    if (r.finds.size() == 1) {
        const ea::RetroEntry& f = r.finds[0];
        check::equal(f.q, Q_RETRO, "retro: the slot");
        check::is_true(std::abs(f.f_hz - 1500.0) < 0.5, "retro: at 1500 Hz");
        check::is_true(f.z > cfg::Z_ACCEPT, "retro: Z over the gate");
        check::equal(f.segment, std::int64_t{0}, "retro: segment 0");
        check::is_true(f.key && *f.key == rs.key, "retro: attached to the picture");
        check::equal(f.rule, std::string("corr"), "retro: by correlation");
        check::is_true(st.has_pass(f.uid), "retro: the pass is stored");
        const std::vector<std::string> members = st.accumulator(rs.key).uids();
        check::is_true(std::find(members.begin(), members.end(), f.uid) != members.end(), "retro: and a member");
        check::is_true(std::isfinite(f.mean_w_db), "retro: its mean W");
        const std::string want = em::retro_find_line(f.q, f.z, f.f_hz, f.segment, f.key, f.rule);
        check::is_true(std::find(log.begin(), log.end(), want) != log.end(), "retro: the log line says the find");
    }
    check::is_true(r.slots_total >= 1 && r.slots_done == r.slots_total, "retro: every slot examined");
    check::is_true(r.slots_searched >= 1, "retro: a slot searched");
    pr.monotone("retro");
    check::is_true(!pr.done.empty() && pr.done.back() == pr.total.back() && pr.total.back() == r.slots_total,
                   "retro: progress complete");
    check::is_true(!pr.what.empty() && pr.what.front().rfind("slot q=", 0) == 0, "retro: progress says the slot");

    // again: the slot is held now, nothing found, nothing searched
    const ea::RetroResult again = ea::retro(st, rs.key, ea::Passband::at((root / "store" / "passband").string()),
                                            "tiny");
    check::is_true(again.finds.empty() && again.slots_searched == 0, "retro again: the slot is held");

    // only the slots asked for: one the store does not hold
    ea::RetroOptions only;
    only.slots = std::vector<std::int64_t>{Q_RETRO + 50};
    const ea::RetroResult none = ea::retro(st, rs.key, ea::Passband{}, "tiny", only);
    check::is_true(none.finds.empty() && none.slots_total == 1 && none.slots_done == 1 && none.slots_searched == 0,
                   "retro: a slot not on disk is skipped");
}

void test_retro_cancel() {
    Step step("retro: cancel");
    const fs::path root = fresh_dir("retro_cancel");
    Cleanup cleanup{root};
    const RetroStore rs = retro_store(root);
    qs::Store st((root / "store").string());
    // before the call
    std::atomic<bool> stop{true};
    em::Hooks h;
    h.cancel = &stop;
    const ea::RetroResult r = ea::retro(st, rs.key, ea::Passband{}, "tiny", {}, h);
    check::is_true(r.cancelled && r.finds.empty() && r.slots_done == 0 && r.slots_total >= 1,
                   "retro cancel before: nothing examined, the total known");
    // from the first progress call: the rest is not examined
    ea::RetroOptions o;
    o.slots = std::vector<std::int64_t>{Q_RETRO + 50, Q_RETRO, Q_RETRO + 51};
    std::atomic<bool> stop2{false};
    em::Hooks h2;
    h2.progress = [&](std::int64_t, std::int64_t, const std::string&) { stop2 = true; };
    h2.cancel = &stop2;
    const ea::RetroResult r2 = ea::retro(st, rs.key, ea::Passband{}, "tiny", o, h2);
    check::is_true(r2.cancelled && r2.slots_done == 1 && r2.slots_total == 3 && r2.finds.empty(),
                   "retro cancel from progress: one slot examined of three");
    check::equal(st.accumulator(rs.key).uids().size(), std::size_t{1}, "retro cancel: the accumulator unchanged");
    store_loads(root / "store", rs.key, "retro cancel");
}

// --- refusals --------------------------------------------------------------------------------------------

void test_refusals(const std::vector<types::PassResult>& passes) {
    Step step("refusals");
    const fs::path root = fresh_dir("refusals");
    Cleanup cleanup{root};
    fill(root / "st", passes);
    qs::Store st((root / "st").string());
    const qs::Key missing{"K1ABC/P", 1};
    const std::string no_pic = "no picture K1ABC/P 00000001 in " + st.root();
    throws<ea::NoPicture>([&] { ea::refine(st, missing); }, no_pic, "refine: no picture");
    throws<ea::NoPicture>([&] { ea::check_picture(st, missing); }, no_pic, "check_picture");
    // the picture first, then the directory, then the frame
    throws<ea::NoPicture>([&] { ea::retro(st, missing, ea::Passband::at("/nonexistent"), "l-short"); }, no_pic,
                          "retro: the picture first");
    const std::string dflt = st.root() + "/passband";
    check::equal(ea::default_passband_dir(st), dflt, "default_passband_dir is STORE/passband");
    throws<ea::NoPassbandStore>([&] { ea::retro(st, KEY, ea::Passband{}, "bogus"); },
                                "--retro: no passband store at " + dflt, "retro: no STORE/passband, before the frame");
    throws<ea::NoPassbandStore>([&] { ea::retro(st, KEY, ea::Passband::at(""), "tiny"); },
                                "--retro: no passband store at " + dflt, "retro: \"\" is the default");
    const std::string file = (root / "a_file").string();
    std::ofstream(file) << "x";
    throws<ea::NoPassbandStore>([&] { ea::retro(st, KEY, ea::Passband::at(file), "tiny"); },
                                "--retro: no passband store at " + file, "retro: a file is not a passband store");
    throws<ea::NoPassbandStore>([&] { ea::retro(st, KEY, ea::Passband::at(file + "/x"), "tiny"); },
                                "--retro: no passband store at " + file + "/x", "retro: ENOTDIR reads as missing");
    const std::string pbdir = (root / "pb").string();
    fs::create_directories(pbdir);
    throws<ea::NotCeFrame>([&] { ea::retro(st, KEY, ea::Passband::at(pbdir), "l-short"); },
                           "--frame 'l-short': --retro searches for CE passes only", "retro: an L frame");
    throws<ea::NotCeFrame>([&] { ea::retro(st, KEY, ea::Passband::at(pbdir), "l-bogus"); },
                           "--frame 'l-bogus': --retro searches for CE passes only", "retro: any l- name");
    throws<ea::UnknownFrame>([&] { ea::retro(st, KEY, ea::Passband::at(pbdir), "bogus"); },
                             "--frame 'bogus': not one of full, medium, short, tiny", "retro: an unknown frame");
    throws<ea::UnknownFrame>([&] { ea::retro_frame("it's"); },
                             "--frame \"it's\": not one of full, medium, short, tiny", "retro_frame: Python's repr");
    check::equal(ea::retro_frame("full-v1").name(), std::string("full-v1"), "retro_frame: a version 1 frame");
    check::equal(ea::retro_frame("full-spreadonly").name(), std::string("full-spreadonly"),
                 "retro_frame: a prototype");
    // the estimator, up front
    ea::RefineOptions bad;
    bad.estimator = "bogus";
    throws<std::invalid_argument>([&] { ea::refine(st, KEY, bad); },
                                  "estimator must be 'joint' or 'plain', not 'bogus'", "refine: a bad estimator");
    ea::RetroOptions rbad;
    rbad.estimator = "Joint";
    throws<std::invalid_argument>([&] { ea::retro(st, KEY, ea::Passband::at(pbdir), "tiny", rbad); },
                                  "estimator must be 'joint' or 'plain', not 'Joint'", "retro: a bad estimator");
    // an open passband store skips the directory check
    auto open = std::make_shared<const pb::PassbandStore>(pbdir);
    const ea::RetroResult r = ea::retro(st, KEY, ea::Passband::of(open), "tiny");
    check::is_true(r.finds.empty() && r.slots_total == 0, "retro: an empty passband store, nothing to examine");
    // nothing was refused part-way: the store still holds its passes as they were
    for (const types::PassResult& p : passes)
        check::equal(st.load_pass(p.uid).em_round, std::int64_t{0}, "refusals: " + p.uid + " untouched");
}

// --- sharing the store with a running listener session --------------------------------------------------

struct Clock {
    std::shared_ptr<std::atomic<double>> t = std::make_shared<std::atomic<double>>(1.0e9);
    qrss::live::Clock fn() const {
        auto p = t;
        return [p] { return p->load(); };
    }
};

void test_session_sharing(const std::vector<types::PassResult>& passes) {
    Step step("session: shared store");
    const fs::path root = fresh_dir("session");
    Cleanup cleanup{root};
    const RetroStore rs = retro_store(root);
    {   // the TINY fixture's picture too, for refine
        qs::Store st((root / "store").string());
        for (const types::PassResult& p : passes) {
            st.add_pass(p);
            st.attach(KEY, p.uid, 0, "test", 0, 0xD1D8);
        }
    }
    Clock clk;
    ls::Options o;
    o.frame = "tiny";
    o.just_ce = true;
    o.render = false;
    o.recover_hours = 0.0;
    o.store_dir = (root / "store").string();
    o.state_dir = (root / "state").string();
    o.clock = clk.fn();
    std::unique_ptr<ls::Session> s = ls::Session::start(o);
    check::is_true(s->store() != nullptr && s->passband() != nullptr, "session: holds the store and the passband");
    if (!s->store() || !s->passband()) return;
    check::equal(s->store()->root(), *s->store_root(), "session: store() is the root it reports");

    // the listener writes the passband directory while retro and refine run
    std::atomic<bool> busy{true};
    ea::RetroResult rr;
    ea::RefineResult fr_;
    std::exception_ptr err;
    std::thread worker([&] {
        try {
            rr = ea::retro(*s->store(), rs.key, ea::Passband::of(s->passband()), "tiny");
            ea::RefineOptions ro;
            ro.rounds = 1;
            fr_ = ea::refine(*s->store(), KEY, ro);
        } catch (...) {
            err = std::current_exception();
        }
        busy = false;
    });
    std::int64_t chunks = 0;
    const std::vector<double> g = fx::gauss(11, 2000);
    std::vector<float> chunk(2000);
    for (std::size_t i = 0; i < chunk.size(); ++i) chunk[i] = static_cast<float>(0.01 * g[i]);
    while (busy && chunks < 4000) {
        clk.t->store(clk.t->load() + 0.25);
        s->push(chunk);
        (void)s->tiles();
        ++chunks;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    worker.join();
    if (err) {
        try {
            std::rethrow_exception(err);
        } catch (const std::exception& e) {
            check::fail("session: em_api beside the listener", e.what());
        }
    }
    std::printf("session: %lld chunks pushed while retro and refine ran\n", static_cast<long long>(chunks));
    check::equal(rr.finds.size(), std::size_t{1}, "session: retro finds the weak pass through the session's stores");
    check::is_true(!fr_.cancelled && fr_.passes.size() == 3, "session: refine re-receives the TINY passes");
    // a second Store on the root sees what the session's store wrote
    qs::Store other((root / "store").string());
    check::equal(other.accumulator(rs.key).uids().size(), std::size_t{2}, "session: a separate Store sees the find");
    for (const types::PassResult& p : passes)
        check::equal(other.load_pass(p.uid).em_round, std::int64_t{1}, "session: a separate Store sees " + p.uid);
    // and runs refine itself while the session holds the root
    ea::RefineOptions ro;
    ro.rounds = 1;
    const ea::RefineResult again = ea::refine(other, KEY, ro);
    check::is_true(!again.cancelled && again.passes.size() == 3, "session: refine on a separate Store");
    s->stop();
    check::is_true(s->ended(), "session: ended");
    check::is_true(s->error() == nullptr, "session: no error");
    store_loads(root / "store", KEY, "session");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(600.0, "qrss_em_api");
    try {
        check::current_step = "fixture";
        const std::vector<types::PassResult> passes = tiny_passes();
        test_refusals(passes);
        test_refine(passes);
        test_refine_cancel(passes);
        test_retro();
        test_retro_cancel();
        test_session_sharing(passes);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL in %s: %s\n", check::current_step.load(), e.what());
        return 1;
    }
    return check::report("qrss_em_api");
}
