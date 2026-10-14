// native/core/qrss/associate against sstvae/qrss/associate.py, without
// Python, on inputs pinned by gen_store_pins.py: the statistic (corr_stat,
// expected_t, corr_accept) on formula vectors -- prefixes, non-finite
// values, nothing kept, past one pairwise block -- bit for bit;
// header_agrees either side of its threshold; the rule each association
// of the scripted session takes ("header", "soft-header", "corr", "prov",
// "" for a member already or a twin left out, a collision kept
// provisional), with rule 2's search replaced through the seam as the
// listener's tests monkeypatch it; and the refusals. Store-level effects
// of the same session (files, log) are test_store.cpp's.
// tests/test_qrss_native_associate.py checks the same against Python.
//
//   PYTHONPATH=. python native/tests/qrss/gen_store_pins.py   # store_pins.hpp

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "qrss/associate.hpp"
#include "qrss/header.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/store.hpp"
#include "qrss_check.hpp"
#include "store_pins.hpp"
#include "store_session.hpp"

using namespace sstvae;
namespace qs = sstvae::qrss::store;
namespace qa = sstvae::qrss::associate;
namespace ss = store_session;
using qrss::pyjson::Value;

namespace fs = std::filesystem;

static std::string fresh_root(const std::string& name) {
    const fs::path p = fs::temp_directory_path() / ("sstvae_test_associate_" + name + "_" +
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

static double u(std::uint64_t x, int shift) { return static_cast<double>(ss::h32(x) >> shift); }

// gen_store_pins.stats_case
static void test_statistics() {
    check::current_step = "statistics";
    int k = 0;
    for (const store_pins::Stat& c : store_pins::STATS) {
        const std::string family = c.family, edits = c.edits;
        const std::size_t n = static_cast<std::size_t>(std::max(c.n_z, c.n_m));
        std::vector<double> z(n), w(n), m(n), W(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t kk = static_cast<std::uint64_t>(k);
            z[i] = (u(i + 7 * kk + 11, 24) - 127.5) / 32.0;
            w[i] = u(i + 1000 + 7 * kk, 28) / 4.0;
            m[i] = (u(i + 2000 + 7 * kk, 24) - 127.5) / 32.0;
            W[i] = u(i + 3000 + 7 * kk, 28) / 2.0;
            if (family == "same") m[i] = m[i] * (edits == "weak" ? 8.0 : 0.5) + z[i];
            if (edits == "nonfinite") {
                if (i % 97 == 0) z[i] = std::numeric_limits<double>::quiet_NaN();
                if (i % 89 == 0) m[i] = HUGE_VAL;
                if (i % 83 == 0) W[i] = std::numeric_limits<double>::quiet_NaN();
                if (i % 79 == 0) w[i] = -1.0;
            }
            if (edits == "w0") w[i] = 0.0;
        }
        z.resize(static_cast<std::size_t>(c.n_z));
        w.resize(static_cast<std::size_t>(c.n_z));
        m.resize(static_cast<std::size_t>(c.n_m));
        W.resize(static_cast<std::size_t>(c.n_m));
        const qa::Corr r = qa::corr_stat(z, w, m, W);
        const double te = qa::expected_t(w, W);
        const Value got = Value::array({Value::number(r.t), Value::integer(r.M),
                                        Value::number(te), Value::number(qa::expected_t(w, W, 0.0)),
                                        Value::boolean(qa::corr_accept(r.t, te))});
        check::equal(qrss::pyjson::dumps(got), std::string(c.want), "stats case " + std::to_string(k));
        ++k;
    }
    // corr_accept's two gates
    check::is_true(!qa::corr_accept(6.0, 0.0), "t must exceed 6");
    check::is_true(qa::corr_accept(6.5, 13.0), "and reach half its expectation");
    check::is_true(!qa::corr_accept(6.5, 13.0000001), "not less");
    check::is_true(!qa::corr_accept(std::nan(""), 0.0), "NaN is not accepted");
    // a pass shorter in w than in z is numpy's broadcast refusal
    const std::vector<double> four(4, 1.0), three(3, 1.0);
    bool threw = false;
    try {
        qa::corr_stat(four, three, four, four);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "w shorter than the common prefix refused");
}

// gen_store_pins.agree_llr
static void test_header_agrees() {
    check::current_step = "header_agrees";
    int k = 0;
    for (const store_pins::Agree& c : store_pins::AGREE) {
        const auto bits = qrss::header::encode(ss::hdr_of(c.true_pic, c.true_seg));
        std::vector<double> llr(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            const double noise = (u(i + 9000 + 13 * static_cast<std::uint64_t>(k), 24) - 127.5) / 64.0;
            llr[i] = c.a * (1.0 - 2.0 * bits[i]) + c.b * noise;
            if ((ss::h32(i + 5000 + static_cast<std::uint64_t>(k)) >> 28) < static_cast<std::uint32_t>(c.zeros))
                llr[i] = 0.0;
        }
        check::equal(qa::header_agrees(ss::hdr_of(c.pic, c.seg), std::span<const double>(llr)), c.want,
                     "agree case " + std::to_string(k));
        ++k;
    }
    const auto h = ss::hdr_of(0, 0);
    check::is_true(qa::header_agrees(h, std::nullopt), "no LLRs kept: agrees");
    const std::vector<double> short_llr(10, 5.0);
    check::is_true(qa::header_agrees(h, std::span<const double>(short_llr)), "LLRs of another length: agrees");
}

// The session's associations: each rule's string, as the reference's.
static void test_rules() {
    check::current_step = "rules";
    const std::string root = fresh_root("rules");
    Cleanup cleanup{root};
    auto passes = ss::make_passes();
    double now = store_pins::T0;
    qs::Store st(root, qs::Store::Seams{[&] { return now; }, {}, {}});
    std::vector<std::string> rules;
    for (const store_pins::Op& op : store_pins::OPS) {
        const std::string kind = op.kind;
        if (kind == "tree") continue;   // test_store's
        now = op.clock;
        check::equal(ss::run_op(op, st, passes, root), std::string(op.want),
                     std::string("session op ") + op.kind + " " + op.arg);
        if (kind == "associate") rules.push_back(qrss::pyjson::loads(op.want).items()[1].as_string());
        if (std::string(op.arg) == "xm") break;
    }
    // every rule (and the empty string) was reached
    for (const char* r : {"header", "soft-header", "corr", "prov", ""}) {
        bool seen = false;
        for (const auto& s : rules) seen = seen || s == r;
        check::is_true(seen, std::string("rule \"") + r + "\" reached");
    }
}

// Rule 2's search replaced (the binding passes a monkeypatched
// associate._soft_header through here): nothing found, so b0b goes to
// rule 4 instead -- the reference with _soft_header patched to return None.
static void test_soft_header_seam() {
    check::current_step = "seam";
    const std::string root = fresh_root("seam");
    Cleanup cleanup{root};
    auto passes = ss::make_passes();
    double now = store_pins::T0;
    qs::Store st(root, qs::Store::Seams{[&] { return now; }, {}, {}});
    int calls = 0;
    const qa::SoftHeaderFn none = [&](qs::Store&, const std::string&, std::span<const double>) {
        ++calls;
        return std::optional<qa::SoftHeader>();
    };
    std::size_t j = 0;
    for (const store_pins::Op& op : store_pins::OPS) {
        if (std::string(op.kind) != "associate") continue;
        now = op.clock;
        const std::string got = ss::run_op(op, st, passes, root, none);
        check::is_true(j < std::size(store_pins::SEAM), "seam: more associations than pinned");
        if (j < std::size(store_pins::SEAM))
            check::equal(got, std::string(store_pins::SEAM[j]), std::string("seam ") + op.arg);
        ++j;
        if (std::string(op.arg) == "xm") break;
    }
    check::equal(j, std::size(store_pins::SEAM), "seam: every pinned association");
    check::equal(calls, store_pins::SEAM_CALLS, "seam: rule 2 asked as often");
}

static void test_refusals() {
    check::current_step = "refusals";
    const std::string root = fresh_root("refusals");
    Cleanup cleanup{root};
    qs::Store st(root, qs::Store::Seams{[] { return store_pins::T0; }, {}, {}});
    try {
        qa::cluster_sums(st, {});
        check::fail("cluster_sums of no uids", "did not throw");
    } catch (const std::invalid_argument& e) {
        check::equal(std::string(e.what()), std::string("max() iterable argument is empty"), "cluster_sums of none");
    }
    // a pass not in the store is stored first
    auto passes = ss::make_passes();
    const auto& a0 = passes.at("a0").pass;
    check::is_true(!st.has_pass(a0.uid), "not stored yet");
    const qa::Result r = qa::associate(a0, st);
    check::is_true(st.has_pass(a0.uid) && r.rule == "header", "stored, then attached by its header");
    check::equal(qa::rule(st, *r.key, a0.uid, "corr"), std::string("corr"), "rule() of a member");
    check::equal(qa::rule(st, *r.key, "nope", "corr"), std::string(""), "rule() of a non-member");
    // nothing open: rule 3 finds nothing
    const std::vector<double> z(100, 1.0), w(100, 1.0);
    check::is_true(!qa::best_accumulator(st, z, w, {*r.key}).has_value(), "an excluded key is not searched");
}

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(600.0, "qrss_core_associate");
    try {
        test_statistics();
        test_header_agrees();
        test_rules();
        test_soft_header_seam();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s (in %s)\n", e.what(), check::current_step.load());
        return 1;
    }
    return check::report("qrss_core_associate");
}
