// native/core/qrss/store against sstvae/qrss/store.py, without Python: the
// scripted session of gen_store_pins.py (passes stored, associated by
// every rule, updated, merged, detached, clustered, logged, expired)
// replayed through the C++ store, with each op's result and, at each tree
// checkpoint, the SHA-256 of every file the store holds pinned from the
// reference -- so the round trip through files is byte for byte store.py's;
// the reference's refusals; expiry exactly at its cutoff; the clock,
// replace and cache-size seams; threads sharing one Store, and two Stores
// sharing one root. tests/test_qrss_native_store.py checks the
// same against Python directly, both ways.
//
//   PYTHONPATH=. python native/tests/qrss/gen_store_pins.py   # store_pins.hpp

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "qrss/associate.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/store.hpp"
#include "qrss_check.hpp"
#include "store_pins.hpp"
#include "store_session.hpp"
#include "util/fileio.hpp"

using namespace sstvae;
namespace qs = sstvae::qrss::store;
namespace qa = sstvae::qrss::associate;
namespace ss = store_session;
using qrss::pyjson::Value;

namespace fs = std::filesystem;

static std::string fresh_root(const std::string& name) {
    const fs::path p = fs::temp_directory_path() / ("sstvae_test_store_" + name + "_" +
                                                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(p);
    return (p / "qrss").string();
}

struct Cleanup {
    std::string root;
    ~Cleanup() {
        std::error_code ec;
        fs::remove_all(fs::path(root).parent_path(), ec);
    }
};

template <typename E, typename F>
static std::optional<std::string> thrown(F f) {
    try {
        f();
    } catch (const E& e) {
        return std::string(e.what());
    }
    return std::nullopt;
}

// --- the session ------------------------------------------------------------------------

static void test_session() {
    check::current_step = "session";
    const std::string root = fresh_root("session");
    Cleanup cleanup{root};
    auto passes = ss::make_passes();
    for (const store_pins::Pass& s : store_pins::PASSES)
        check::equal(passes.at(s.name).pass.uid, std::string(s.uid), std::string("uid of pass ") + s.name);
    double now = store_pins::T0;
    qs::Store st(root, qs::Store::Seams{[&] { return now; }, {}, {}});
    std::optional<qs::Accumulator> early;   // a copy handed out early: later calls must not reach it
    std::vector<float> early_S;
    int k = 0;
    for (const store_pins::Op& op : store_pins::OPS) {
        now = op.clock;
        std::string got;
        try {
            got = ss::run_op(op, st, passes, root);
        } catch (const std::exception& e) {
            got = std::string("threw: ") + e.what();
        }
        check::equal(got, std::string(op.want),
                     "op " + std::to_string(k) + " " + op.kind + " " + op.arg + " " + op.extra);
        if (k == 1) {
            early = st.accumulator(ss::key_a());
            early_S = early->S;
        }
        ++k;
    }
    // Accumulators are values: the copy taken after op 1 still has its two members and its sums
    check::equal(early->members.size(), std::size_t(2), "an early copy keeps its members");
    check::is_true(early->S == early_S, "an early copy keeps its S");
    check::is_true(!fs::exists(fs::path(root) / "acc" / "K1ABC_a1b2c3d4.json"), "expired at the end");
}

// --- the reference's refusals and plain values --------------------------------------------

static void test_errors_and_values() {
    check::current_step = "errors";
    const std::string root = fresh_root("errors");
    Cleanup cleanup{root};
    auto passes = ss::make_passes();
    qs::Store st(root, qs::Store::Seams{[] { return store_pins::T0; }, {}, {}});
    qa::associate(passes.at("a0").pass, st);
    const qs::Key key = ss::key_a();
    const std::string a0 = passes.at("a0").pass.uid;
    std::map<std::string, std::pair<std::string, std::string>> got;   // call -> (type, args[0])
    auto run = [&](const std::string& name, auto f) {
        try {
            f();
            got[name] = {"none", ""};
        } catch (const qrss::pyjson::KeyError& e) {
            got[name] = {"KeyError", e.key()};
        } catch (const std::invalid_argument& e) {
            got[name] = {"ValueError", e.what()};
        } catch (const std::out_of_range& e) {
            got[name] = {"IndexError", e.what()};
        }
    };
    run("canonical_index(0, 0)", [&] { qs::canonical_index(0, 0); });
    run("canonical_index(0, 50601)", [&] { qs::canonical_index(0, 50601); });
    run("canonical_index(3, 10)", [&] { qs::canonical_index(3, 10); });
    run("loo(nope)", [&] { st.accumulator(key).loo("nope", st); });
    run("pass_path(a/b)", [&] { st.pass_path("a/b"); });
    run("pass_path(.x)", [&] { st.pass_path(".x"); });
    run("accumulator(NOPE, 1)", [&] { st.accumulator(qs::Key{"NOPE", 1}); });
    run("attach(nope)", [&] { st.attach(key, "nope", 0, "x"); });
    run("attach(a0, 3)", [&] { st.attach(key, a0, 3, "x"); });
    run("has_data(3)", [&] { st.accumulator(key).has_data(3); });
    for (const store_pins::Error& e : store_pins::ERRORS) {
        const auto it = got.find(e.call);
        if (it == got.end()) {
            check::fail(e.call, "not run");
            continue;
        }
        check::equal(it->second.first, std::string(e.type), std::string(e.call) + ": exception type");
        check::equal(it->second.second, std::string(e.text), std::string(e.call) + ": message");
    }
    check::equal(got.size(), std::size(store_pins::ERRORS), "every refusal pinned");

    std::map<std::string, std::string> vals;
    auto r = [](double d) { return qrss::pyjson::dumps(Value::number(d)); };
    vals["key_name(A/B, -5)"] = qs::key_name({"A/B", -5});
    vals["key_name(K1ABC, 0xA1B2C3D4)"] = qs::key_name(key);
    vals["slot_time(Q0)"] = r(qs::slot_time(store_pins::Q0));
    vals["mean_w_db([])"] = r(qs::mean_w_db(std::vector<double>{}));
    vals["mean_w_db([0, 0])"] = r(qs::mean_w_db(std::vector<double>{0.0, 0.0}));
    vals["mean_w_db([10, 10, 10])"] = r(qs::mean_w_db(std::vector<double>{10.0, 10.0, 10.0}));
    vals["mean_w_db([nan])"] = r(qs::mean_w_db(std::vector<double>{std::nan("")}));
    vals["mean_w_db([-1, 0.5])"] = r(qs::mean_w_db(std::vector<double>{-1.0, 0.5}));
    std::vector<double> w10(10, 0.0);
    w10[0] = 1000.0;
    vals["mean_w_db([1000, 0, 0, 0, 0, 0, 0, 0, 0, 0])"] = r(qs::mean_w_db(w10));
    for (const store_pins::Text& t : store_pins::VALUES) check::equal(vals.at(t.call), std::string(t.want), t.call);
    // the float32 form agrees with the float64 one on the same values
    check::equal(qs::mean_w_db(std::vector<float>{10.0f, 10.0f, 10.0f}), 10.0, "mean_w_db of float32");
}

// --- the seams ---------------------------------------------------------------------------

static void test_clock_seam() {
    check::current_step = "clock";
    const std::string root = fresh_root("clock");
    Cleanup cleanup{root};
    int calls = 0;
    double t = 1700000000.0;
    qs::Store st(root, qs::Store::Seams{[&] {
                     ++calls;
                     return t;
                 },
                                        {}, {}});
    const qs::Key key{"N0CALL", 7};
    const qs::Accumulator a = st.create(key, 1, 0xD1D8);
    check::equal(calls, 1, "create asks the clock once");
    check::equal(a.created, t, "created is the clock's time");
    check::equal(a.updated, t, "updated is the clock's time");
    t += 60.0;
    qs::Accumulator b = st.accumulator(key);
    st.commit(b);
    check::equal(calls, 2, "commit asks the clock once");
    check::equal(st.accumulator(key).updated, t, "commit stamps updated");
    check::equal(st.accumulator(key).created, t - 60.0, "and keeps created");
    Value ev = Value::object();
    ev.set("event", Value::string("tick"));
    st.log(ev);
    check::equal(calls, 3, "log asks the clock once");
    std::ifstream f(fs::path(root) / "log.jsonl");
    std::string line;
    std::getline(f, line);
    check::equal(line, std::string("{\"time\": 1700000060.0, \"event\": \"tick\"}"), "log line");
    check::equal(st.now(), t, "now() is the clock");
    check::equal(calls, 4, "now() asks it");
    // expire without `now` asks the clock; nothing is old at the clock's own time
    const qs::Expired e = st.expire();
    check::equal(calls, 5, "expire() asks the clock");
    check::is_true(e.acc.empty() && e.prov.empty() && e.passes.empty(), "nothing expired at now");
    // past 7 days of the clock, it goes
    t += 8 * qs::DAY_S;
    const qs::Expired e2 = st.expire();
    check::equal(e2.acc.size(), std::size_t(1), "expired 8 days on");

    // no clock: the system clock (time.time())
    const std::string root2 = fresh_root("clock2");
    Cleanup cleanup2{root2};
    qs::Store sys(root2);
    const double wall = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    check::is_true(std::abs(sys.now() - wall) < 5.0, "the default clock is the system clock");
}

// expire() drops what is older than the cutoff: an accumulator, a
// cluster or an orphan pass exactly at it stays, one a ulp older goes.
static void test_expire_cutoff() {
    check::current_step = "expire cutoff";
    const std::string root = fresh_root("expire");
    Cleanup cleanup{root};
    double t = 1760000000.0;
    qs::Store st(root, qs::Store::Seams{[&] { return t; }, {}, {}});
    auto passes = ss::make_passes();
    for (const char* n : {"a0", "b0", "b0b"}) st.add_pass(passes.at(n).pass);
    const qs::Key key = ss::key_a();
    st.attach(key, passes.at("a0").pass.uid, 0, "header", 2);   // updated t
    const double acc_t = t;
    t += 50.0;
    const std::string cluster = st.prov_add(passes.at("b0b").pass.uid);   // updated t + 50
    const double week = 7 * qs::DAY_S;
    auto none = [](const qs::Expired& e) { return e.acc.empty() && e.prov.empty() && e.passes.empty(); };
    auto past = [](double x) { return std::nextafter(x, 1e300); };
    check::is_true(none(st.expire(acc_t + week)), "an accumulator at the cutoff stays");
    qs::Expired e = st.expire(past(acc_t + week));
    check::is_true(e.acc.size() == 1 && e.prov.empty(), "a ulp older it goes");
    check::is_true(e.passes == std::vector<std::string>{passes.at("a0").pass.uid}, "and its old pass with it");
    check::is_true(none(st.expire(t + week)), "a cluster at the cutoff stays");
    e = st.expire(past(t + week));
    check::is_true(e.acc.empty() && e.prov.size() == 1 && e.prov[0].as_string() == cluster && e.passes.empty(),
                   "a ulp older it goes, its newer pass kept");
    for (const char* n : {"b0", "b0b"}) {
        const double slot = qs::slot_time(passes.at(n).pass.q);
        check::is_true(none(st.expire(slot + week)), std::string("orphan ") + n + " at the cutoff stays");
        e = st.expire(past(slot + week));
        check::is_true(e.passes == std::vector<std::string>{passes.at(n).pass.uid},
                       std::string("orphan ") + n + " a ulp older goes");
    }
    check::is_true(st.pass_uids().empty() && st.open_keys().empty() && st.prov_clusters().empty(), "all gone");
}

static void test_replace_seam() {
    check::current_step = "replace";
    const std::string root = fresh_root("replace");
    Cleanup cleanup{root};
    std::vector<std::pair<std::string, std::string>> moves;
    int fail_at = -1;
    qs::Store st(root, qs::Store::Seams{[] { return store_pins::T0; },
                                        [&](const std::string& from, const std::string& to) {
                                            moves.emplace_back(from, to);
                                            if (static_cast<int>(moves.size()) == fail_at)
                                                throw qs::FileError("cannot replace", from, 13);
                                            std::error_code ec;
                                            if (!util::replace_file(from, to, ec)) throw std::system_error(ec);
                                        },
                                        {}});
    auto passes = ss::make_passes();
    const auto& a0 = passes.at("a0").pass;
    st.add_pass(a0);
    check::equal(moves.size(), std::size_t(1), "a pass is one replace");
    const qs::Key key = ss::key_a();
    st.create(key, 2, 0xD1D8);
    check::equal(moves.size(), std::size_t(3), "an accumulator is two (arrays, then index)");
    for (const auto& [from, to] : moves) {
        const fs::path f(from), t(to);
        check::equal(f.parent_path().string(), t.parent_path().string(), "the temporary file is a sibling");
        const std::string name = f.filename().string(), want = "." + t.filename().string() + ".";
        check::equal(name.substr(0, want.size()), want, "named .<name>.<pid>.tmp");
        check::equal(name.substr(name.size() - 4), std::string(".tmp"), "ending .tmp");
        check::is_true(!fs::exists(f) && fs::exists(t), "moved into place");
    }
    check::equal(fs::path(moves[1].second).extension().string(), std::string(".npz"), "arrays first");
    check::equal(fs::path(moves[2].second).extension().string(), std::string(".json"), "the index last");

    auto leftovers = [&] {
        int n = 0;
        for (const auto& d : {"acc", "passes", "prov"})
            for (const auto& e : fs::directory_iterator(fs::path(root) / d))
                if (e.path().filename().string()[0] == '.') ++n;
        return n;
    };
    // the arrays' replace fails: nothing changes on disk, the temporary file is gone
    const auto before = ss::tree(root);
    fail_at = static_cast<int>(moves.size()) + 1;
    const auto what = thrown<qs::FileError>([&] { st.attach(key, a0.uid, 0, "header"); });
    check::is_true(what.has_value(), "a failed replace propagates");
    check::equal(leftovers(), 0, "and leaves no temporary file");
    check::is_true(ss::tree(root) == before, "and changes no file");
    check::equal(st.accumulator(key).members.size(), std::size_t(0), "the accumulator as it was");
    // the index's replace fails after the arrays went in (store.py's order)
    fail_at = static_cast<int>(moves.size()) + 2;
    check::is_true(thrown<qs::FileError>([&] { st.attach(key, a0.uid, 0, "header"); }).has_value(),
                   "a failed index replace propagates");
    check::equal(leftovers(), 0, "no temporary file");
    check::equal(st.accumulator(key).members.size(), std::size_t(0), "the index as it was");
    // and then it works
    fail_at = -1;
    check::equal(st.attach(key, a0.uid, 0, "header").members.size(), std::size_t(1), "attached");
    check::equal(st.accumulator(key).members.size(), std::size_t(1), "on disk");
}

static void test_zw_cache_seam() {
    check::current_step = "zw_cache";
    const std::string root = fresh_root("zw");
    Cleanup cleanup{root};
    std::size_t size = 2;
    qs::Store st(root, qs::Store::Seams{[] { return store_pins::T0; }, {}, [&] { return size; }});
    auto passes = ss::make_passes();
    std::vector<std::string> uids;
    for (const char* n : {"a0", "a0b", "b0", "b0b"}) {
        st.add_pass(passes.at(n).pass);
        uids.push_back(passes.at(n).pass.uid);
    }
    for (const auto& u : uids) st.pass_zw(u);
    check::is_true(st.zw_cached() == std::vector<std::string>{uids[2], uids[3]}, "the last two, oldest first");
    st.pass_zw(uids[2]);
    check::is_true(st.zw_cached() == std::vector<std::string>{uids[3], uids[2]}, "a hit moves to the end");
    size = 3;   // asked at each read
    st.pass_zw(uids[0]);
    check::is_true(st.zw_cached() == std::vector<std::string>{uids[3], uids[2], uids[0]}, "now three");
    const qs::ZWPtr zw = st.pass_zw(uids[0]);
    check::equal(zw->z.size(), std::size_t(store_pins::N), "z read");
    check::equal(zw->z[5], static_cast<double>(passes.at("a0").pass.z[5]), "z as float64");
    // a pass rewritten is forgotten
    st.add_pass(passes.at("a0e").pass);   // a0's uid, other weights
    check::is_true(st.zw_cached() == std::vector<std::string>{uids[3], uids[2]}, "add_pass forgets it");
    check::equal(st.pass_zw(uids[0])->w[0], 100.0, "and reads the new file");
    check::equal(zw->w[0], 10.0, "while a (z, w) handed out before keeps its values");
}

// --- threads -------------------------------------------------------------------------

// One thread replays the session's associations (up to the second tree),
// one reads everything it can, one appends to the log; afterwards every
// file but the log is the reference's, and the log is whole lines.
static void test_threads() {
    check::current_step = "threads";
    const std::string root = fresh_root("threads");
    Cleanup cleanup{root};
    auto passes = ss::make_passes();
    std::atomic<double> now{store_pins::T0};
    qs::Store st(root, qs::Store::Seams{[&] { return now.load(); }, {}, {}});
    std::size_t second_tree = 0, trees = 0;
    for (std::size_t i = 0; i < std::size(store_pins::OPS); ++i)
        if (std::string(store_pins::OPS[i].kind) == "tree" && ++trees == 2) second_tree = i;

    std::atomic<bool> done{false};
    std::vector<std::string> errors;
    std::mutex errors_mu;
    auto record = [&](const std::string& what) {
        const std::lock_guard<std::mutex> lock(errors_mu);
        errors.push_back(what);
    };
    std::vector<std::pair<std::size_t, std::string>> results;
    std::thread writer([&] {
        try {
            for (std::size_t i = 0; i < second_tree; ++i) {
                const store_pins::Op& op = store_pins::OPS[i];
                if (std::string(op.kind) == "tree") continue;
                now = op.clock;
                results.emplace_back(i, ss::run_op(op, st, passes, root));
            }
        } catch (const std::exception& e) {
            record(std::string("writer: ") + e.what());
        }
        done = true;
    });
    std::atomic<long> reads{0};
    std::thread reader([&] {
        try {
            std::vector<std::string> uids;
            for (const auto& [name, b] : passes) uids.push_back(b.pass.uid);
            while (!done) {
                for (const qs::Key& k : st.open_keys()) {
                    const qs::Accumulator a = st.accumulator(k);
                    if (a.S.size() != qs::PLANE) record("reader: torn accumulator");
                    for (const auto& u : a.uids()) st.pass_info(u);
                }
                for (const auto& u : uids)
                    if (st.has_pass(u)) st.pass_zw(u);
                st.prov_clusters();
                st.due(store_pins::Q0 + 4);
                st.pass_uids();
                ++reads;
            }
        } catch (const std::exception& e) {
            record(std::string("reader: ") + e.what());
        }
    });
    constexpr int N_LOG = 300;
    std::thread logger([&] {
        try {
            for (int i = 0; i < N_LOG; ++i) {
                Value ev = Value::object();
                ev.set("event", Value::string("hammer"));
                ev.set("i", Value::integer(i));
                st.log(ev);
            }
        } catch (const std::exception& e) {
            record(std::string("logger: ") + e.what());
        }
    });
    writer.join();
    reader.join();
    logger.join();
    for (const auto& e : errors) check::fail("threads", e);
    check::is_true(reads > 0, "the reader ran");
    for (const auto& [i, got] : results)
        check::equal(got, std::string(store_pins::OPS[i].want), "threaded op " + std::to_string(i));

    // every file but the log as the reference wrote it
    const Value want = qrss::pyjson::loads(store_pins::OPS[second_tree].want);
    std::vector<std::pair<std::string, std::string>> want_files;
    for (const Value& e : want.items())
        if (e.items()[0].as_string() != "log.jsonl")
            want_files.emplace_back(e.items()[0].as_string(), e.items()[1].as_string());
    std::vector<std::pair<std::string, std::string>> got_files;
    for (const auto& f : ss::tree(root))
        if (f.first != "log.jsonl") got_files.push_back(f);
    check::is_true(got_files == want_files, "the files are the reference's");
    // the log: whole lines, every hammer line once
    std::ifstream f(fs::path(root) / "log.jsonl");
    std::string line;
    int hammer = 0, lines = 0;
    while (std::getline(f, line)) {
        ++lines;
        try {
            const Value v = qrss::pyjson::loads(line);
            if (v.at("event").as_string() == "hammer") ++hammer;
        } catch (const std::exception& e) {
            check::fail("log line " + std::to_string(lines), e.what());
        }
    }
    check::equal(hammer, N_LOG, "every log line written whole");
}

// Two Stores open on one directory (the second spelled through "..") in
// two threads: they share the root's mutex, so neither trips over the
// other's ".<name>.<pid>.tmp" nor loses the other's read-modify-write of the
// accumulator they both attach to; one Store on another root is not held up.
static void test_two_stores_one_root() {
    check::current_step = "two stores";
    const std::string root = fresh_root("two");
    Cleanup cleanup{root};
    auto passes = ss::make_passes();
    const qs::Store::Seams seams{[] { return store_pins::T0; }, {}, {}};
    qs::Store a(root, seams);
    qs::Store b(root + "/../" + fs::path(root).filename().string(), seams);
    check::is_true(a.root() != b.root(), "two spellings of one root");
    const qs::Key key = ss::key_a();
    const std::vector<std::vector<std::string>> mine = {{"a0", "a0f", "a0g"}, {"b0", "b0b", "x0"}};
    for (const auto& names : mine)
        for (const auto& n : names) a.add_pass(passes.at(n).pass);
    a.create(key, 2, 0xD1D8);
    std::vector<std::string> errors;
    std::mutex errors_mu;
    auto work = [&](qs::Store& st, const std::vector<std::string>& names) {
        try {
            for (int round = 0; round < 12; ++round) {
                for (const auto& n : names) st.attach(key, passes.at(n).pass.uid, 0, "corr", 2);
                for (const auto& n : names) st.detach(key, passes.at(n).pass.uid);
            }
            for (const auto& n : names) st.attach(key, passes.at(n).pass.uid, 0, "corr", 2);
        } catch (const std::exception& e) {
            const std::lock_guard<std::mutex> lock(errors_mu);
            errors.push_back(e.what());
        }
    };
    std::thread ta([&] { work(a, mine[0]); });
    std::thread tb([&] { work(b, mine[1]); });
    ta.join();
    tb.join();
    for (const auto& e : errors) check::fail("two stores", e);
    qs::Store c(root, seams);
    qs::Accumulator acc = c.accumulator(key);
    check::equal(acc.members.size(), std::size_t(6), "every pass of both threads attached");
    const std::vector<float> S0 = acc.S, W0 = acc.W;
    acc.rebuild(c);
    check::is_true(acc.S == S0 && acc.W == W0, "and the arrays are its members summed");
    int leftovers = 0;
    for (const auto& d : {"acc", "passes", "prov"})
        for (const auto& e : fs::directory_iterator(fs::path(root) / d))
            if (e.path().filename().string()[0] == '.') ++leftovers;
    check::equal(leftovers, 0, "no temporary file left");

    // another root's Store is not serialised behind this one's transaction
    const std::string other = fresh_root("two_other");
    Cleanup cleanup_other{other};
    qs::Store o(other, seams);
    std::atomic<bool> held{false}, done{false};
    std::thread holder([&] {
        const auto lk = a.hold();
        held = true;
        while (!done) std::this_thread::yield();
    });
    while (!held) std::this_thread::yield();
    o.create(qs::Key{"N0CALL", 1}, 0, 0);   // would wait forever on a shared mutex
    check::is_true(o.has_accumulator(qs::Key{"N0CALL", 1}), "another root works while this one is held");
    done = true;
    holder.join();
}

// --- the reference's fixes of 2026-10-14 (tests/test_qrss_multipass.py's twins) -----------

// A pass cut to its first n latents (a SHORT-sized frame), uid recomputed.
static qrss::types::PassResult prefix_of(qrss::types::PassResult p, std::size_t n) {
    p.z.resize(n);
    p.w.resize(n);
    p.uid = qrss::types::pass_uid(p.q, p.f_hz, p.z);
    return p;
}

static void test_reference_fixes() {
    check::current_step = "reference fixes";
    const std::string root = fresh_root("fixes");
    Cleanup cleanup{root};
    auto passes = ss::make_passes();
    std::vector<std::string> moved;
    std::size_t cap = qs::ZW_CACHE;
    qs::Store st(root, qs::Store::Seams{[] { return store_pins::T0; },
                                        [&](const std::string& from, const std::string& to) {
                                            moved.push_back(fs::path(from).filename().string());
                                            std::error_code ec;
                                            if (!util::replace_file(from, to, ec)) throw std::system_error(ec);
                                        },
                                        [&] { return cap; }});
    const auto& a0 = passes.at("a0").pass;
    st.add_pass(a0);
    // the temporary sibling carries the thread: .<name>.<pid>.<thread id>.tmp
    check::equal(moved.at(0), "." + a0.uid + ".npz." + std::to_string(qrss::rawfile::process_id()) + "." +
                                  std::to_string(qrss::rawfile::thread_ident()) + ".tmp",
                 "temporary name");

    // rule 1 gates on the median W over the pass's own prefix: SHORT-sized
    // passes of picture 1 under picture 0's header are a collision
    const qs::Key key = ss::key_a();
    const auto mine = prefix_of(a0, 4000), stranger = prefix_of(passes.at("x0").pass, 4000);
    st.add_pass(mine);
    st.attach(key, mine.uid, 0, "header", 2, 0xD1D8);
    const qs::Accumulator acc = st.accumulator(key);
    check::equal(acc.median_w(0), 0.0, "the whole segment's median W is 0");
    check::equal(acc.median_w(0, 4000), 10.0, "its prefix's is 10");
    const std::vector<double> sz(stranger.z.begin(), stranger.z.end()), sw(stranger.w.begin(), stranger.w.end());
    const qa::Validation v = qa::validate_header_match(st, key, 2, 0, sz, sw);
    check::is_true(!v.ok, "a SHORT-sized impostor is a collision");
    check::equal(v.details.at("M").as_int(), std::int64_t(4000), "checked over its 4000 latents");
    const std::vector<double> mz(mine.z.begin(), mine.z.end()), mw(mine.w.begin(), mine.w.end());
    check::is_true(qa::validate_header_match(st, key, 2, 0, mz, mw).ok, "the picture's own pass is not");

    // ZW_CACHE = 0: the pass returned, nothing cached
    cap = 0;
    check::equal(st.pass_zw(a0.uid)->z.size(), a0.z.size(), "returned at size 0");
    check::is_true(st.zw_cached().empty(), "and nothing cached");
    cap = qs::ZW_CACHE;

    // a pass rewritten through another Store is read again
    check::equal(st.pass_zw(a0.uid)->w[0], 10.0, "cached");
    check::equal(st.pass_info(a0.uid).em_round, std::int64_t(0), "cached info");
    {
        qs::Store other(root, qs::Store::Seams{[] { return store_pins::T0; }, {}, {}});
        other.add_pass(passes.at("a0e").pass);   // a0's uid, w = 100, EM round 1
    }
    const fs::path ppath = fs::path(root) / "passes" / (a0.uid + ".npz");
    fs::last_write_time(ppath, fs::last_write_time(ppath) + std::chrono::seconds(1));
    check::equal(st.pass_zw(a0.uid)->w[0], 100.0, "the new file's (z, w)");
    check::equal(st.pass_info(a0.uid).em_round, std::int64_t(1), "and its info");

    // a new cluster never writes over one of the same name
    const auto& b0 = passes.at("b0").pass;
    const auto& b0b = passes.at("b0b").pass;
    st.add_pass(b0);
    st.add_pass(b0b);
    const std::string c = st.prov_add(b0.uid);
    check::equal(c, "c_" + b0.uid, "c_<uid>");
    st.prov_add(b0b.uid, c);
    st.prov_remove(b0.uid);
    check::equal(st.prov_add(b0.uid), "c_" + b0.uid + "_2", "c_<uid>_2 when c_<uid> holds another");
    check::equal(st.prov_cluster_of(b0b.uid)->at("cluster").as_string(), c, "which keeps it");
    check::equal(st.prov_clusters().size(), std::size_t(2), "two clusters");

    // due: an accumulator whose later member's pass is gone gives no rows
    const auto& b1 = passes.at("b1").pass;
    st.add_pass(b1);
    st.attach(qs::Key{"DUE", 7}, b0.uid, 0, "corr", 2);
    st.attach(qs::Key{"DUE", 7}, b1.uid, 1, "corr", 2);
    check::equal(st.due(b0.q + 4).size(), std::size_t(1), "segment 2 due from the first member");
    fs::remove(fs::path(root) / "passes" / (b1.uid + ".npz"));
    check::equal(st.due(b0.q + 4).size(), std::size_t(0), "none once the second member's pass is gone");

#ifndef _WIN32
    // an empty $HOME counts as unset
    const char* old_home = std::getenv("HOME");
    const std::string saved_home = old_home ? old_home : "";
    const char* old_q = std::getenv("QRSSTVAE_HOME");
    const std::string saved_q = old_q ? old_q : "";
    ::unsetenv("QRSSTVAE_HOME");
    ::setenv("HOME", "", 1);
    const std::string r = qs::default_root();
    check::is_true(r != "/.local/share/qrsstvae", "not under the filesystem root: " + r);
    check::equal(r.substr(r.size() - std::string("/.local/share/qrsstvae").size()), std::string("/.local/share/qrsstvae"),
                 "the password database's home");
    ::setenv("HOME", "/h", 1);
    check::equal(qs::default_root(), std::string("/h/.local/share/qrsstvae"), "a set $HOME");
    if (old_home) ::setenv("HOME", saved_home.c_str(), 1); else ::unsetenv("HOME");
    if (old_q) ::setenv("QRSSTVAE_HOME", saved_q.c_str(), 1);
#endif
}

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(600.0, "qrss_core_store");
    try {
        test_session();
        test_errors_and_values();
        test_clock_seam();
        test_expire_cutoff();
        test_replace_seam();
        test_zw_cache_seam();
        test_reference_fixes();
        test_threads();
        test_two_stores_one_root();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s (in %s)\n", e.what(), check::current_step.load());
        return 1;
    }
    return check::report("qrss_core_store");
}
