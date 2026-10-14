// native/core/qrss/em (the shared pieces of sstvae/qrss/em.py) without
// Python, on inputs pinned by gen_em_pins.py (em_pins.hpp; the formulas are
// in that script): the tail statistics (_gauss_z, template_z) to 1e-12
// relative with their refusals; lmmse_latents and prior_from bit for bit
// (a SHA-256 of their outputs, NaN written as numpy's np.nan); the
// store-backed pieces -- em_prior, acc_mean_w_db, known_header and
// known_keying -- on formula passes stored and attached one accumulator per
// scenario, against the reference's results; pass_spec / is_l / ce_only;
// _flat; and the hooks (check_cancel, log_line, report_progress), which
// are the C++'s own. tests/test_qrss_native_em.py compares the same
// functions against Python directly (and _psi_frame / _path_of on real
// tracks, which are not pinned here). The special functions under the
// statistics are test_special.cpp's.
//
//   PYTHONPATH=. python native/tests/qrss/gen_em_pins.py    # em_pins.hpp

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "em_pins.hpp"
#include "infra_testgen.hpp"
#include "qrss/config.hpp"
#include "qrss/em.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"
#include "qrss_check.hpp"
#include "util/fileio.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
namespace em = sstvae::qrss::em;
namespace qs = sstvae::qrss::store;
namespace types = sstvae::qrss::types;
namespace fs = std::filesystem;

namespace {

double worst_rel = 0.0;

// --- helpers -------------------------------------------------------------------------------

std::vector<double> units(std::uint64_t seed, std::size_t n) {
    const auto z = qrss_testgen::splitmix(seed, n);
    std::vector<double> u(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = qrss_testgen::unit(z[i]);
    return u;
}

// gen_em_pins.sha(canon_nan(a), canon_nan(b))
std::string sha_canon(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<std::uint8_t> out;
    out.reserve((a.size() + b.size()) * sizeof(double));
    for (const auto* v : {&a, &b})
        for (double x : *v) {
            if (std::isnan(x)) x = std::numeric_limits<double>::quiet_NaN();
            const auto* p = reinterpret_cast<const std::uint8_t*>(&x);
            out.insert(out.end(), p, p + sizeof(double));
        }
    return util::to_hex(util::sha256(std::span<const std::uint8_t>(out)));
}

std::string hex(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%a", x);
    return buf;
}

// A pinned value against the C++'s: NaN and infinities and zero exactly,
// anything else to 1e-12 relative (the worst one printed at the end).
void near(double got, double want, const std::string& what) {
    if (std::isnan(want) || std::isinf(want) || want == 0.0) {
        const bool ok = std::isnan(want) ? std::isnan(got) : got == want;
        check::is_true(ok, what + ": got " + hex(got) + ", want " + hex(want));
        return;
    }
    const double rel = std::abs(got - want) / std::abs(want);
    if (rel > worst_rel) worst_rel = rel;
    check::is_true(rel <= 1e-12, what + ": got " + hex(got) + ", want " + hex(want));
}

std::uint32_t h32(std::uint64_t in) {
    std::uint32_t x = static_cast<std::uint32_t>(in);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

// A fresh directory under the temporary one; the store's root (UTF-8, as the
// Store takes it: util::path_text, never path::string(), which goes through
// the ANSI code page on Windows) is its "qrss".
fs::path fresh_dir(const std::string& name) {
    const fs::path p = fs::temp_directory_path() /
                       ("sstvae_test_em_" + name + "_" +
                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(p);
    return p;
}

struct Cleanup {
    fs::path dir;
    ~Cleanup() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (const char c : s) {
        if (c == sep) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    for (std::size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size()))
        s.replace(at, from.size(), to);
    return s;
}

// --- the formula passes (gen_em_pins.make_pass) ---------------------------------------------

qrss::header::HeaderFields hdr_of(int pic, int segment) {
    const em_pins::Pic& p = em_pins::PICS[pic];
    qrss::header::HeaderFields h;
    h.callsign = p.callsign;
    if (p.grid) h.grid = std::string(p.grid);
    h.picture_id = p.picture_id;
    h.mode = p.mode;
    h.segment = segment;
    h.codec_id = p.codec_id;
    return h;
}

const em_pins::Pass& pass_spec_named(const std::string& name) {
    for (const em_pins::Pass& p : em_pins::PASSES)
        if (name == p.name) return p;
    throw std::runtime_error("no pinned pass " + name);
}

types::PassResult make_pass(const em_pins::Pass& s) {
    const std::size_t n = static_cast<std::size_t>(s.n);
    types::PassResult p;
    p.z.resize(n);
    p.w.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = (static_cast<double>(h32(i + 1000003u * static_cast<std::uint64_t>(s.pic) + 77u) >> 27) -
                          15.5) / 8.0;
        const double noise =
            (static_cast<double>(h32(i + 1000003u * static_cast<std::uint64_t>(s.seed + 7)) >> 28) - 7.5) / 8.0;
        const double w = (s.erase_from >= 0 && i >= static_cast<std::size_t>(s.erase_from)) ? 0.0 : s.w;
        p.z[i] = w > 0 ? static_cast<float>(a + noise) : 0.0f;
        p.w[i] = static_cast<float>(w);
    }
    const std::string llr = s.llr;
    if (llr == "zero") {
        p.hdr_llr.assign(qrss::config::N_HDR_BITS, 0.0f);
    } else if (llr == "short") {
        p.hdr_llr.assign(10, 1.0f);
    } else {
        const auto bits = qrss::header::encode(hdr_of(s.hdr_pic, s.hdr_seg));
        p.hdr_llr.resize(bits.size());
        for (std::size_t j = 0; j < bits.size(); ++j) {
            const double sign = 1.0 - 2.0 * bits[j];
            double v;
            if (llr == "strong") {
                v = 4.0 * sign;
            } else {
                const double noise =
                    (static_cast<double>(h32(j + 9000u + 13u * static_cast<std::uint64_t>(s.seed)) >> 24) - 127.5) /
                    64.0;
                v = em_pins::WEAK_A * sign + noise;
            }
            p.hdr_llr[j] = static_cast<float>(v);
        }
    }
    if (s.decoded) p.header = hdr_of(s.hdr_pic, s.hdr_seg);
    if (s.cw_keying[0] != '\0') {
        types::CwIdResult cw;
        cw.text = "K1ABC";
        cw.z_match = s.cw_z;
        cw.keying = s.cw_keying;
        if (s.cw_agrees >= 0) cw.agrees = s.cw_agrees == 1;
        p.cw = cw;
    }
    const std::string base = s.n == 1024 ? "tiny" : "short";
    p.frame = s.waveform == qrss::config::WAVEFORM_L ? "l-" + base : base;
    p.uid = types::pass_uid(s.q, s.f_hz, p.z);
    p.q = s.q;
    p.waveform = s.waveform;
    p.f_hz = s.f_hz;
    p.timing = {0.0, 0.0, 0.0, 10.0, math::Matrix(2, 2, {1.0, 0.0, 0.0, 1.0})};
    p.report = {s.f_hz, 0.0, 0.0, 0.0, 0.0, -20.0, 10.0, 1.0, true, false};
    p.ch_fs = 250;
    p.ch_t0_index = 0.0;
    p.f_mix_hz = qrss::Rational(1500);
    p.psi.assign(4, 1.0f);
    p.estimator = "joint";
    p.em_round = 0;
    return p;
}

qs::Key key_of(int k) {
    if (k >= 0) return {em_pins::PICS[k].callsign, em_pins::PICS[k].picture_id};
    return {"K1ABC", 5000 + (-1 - k)};
}

std::string header_text(const std::optional<qrss::header::HeaderFields>& h) {
    if (!h) return "None";
    std::ostringstream o;
    o << h->callsign << '|' << (h->grid ? *h->grid : std::string("<none>")) << '|' << h->picture_id << '|' << h->mode
      << '|' << h->segment << '|' << h->codec_id << '|' << h->version << '|' << h->reserved;
    return o.str();
}

std::string units_text(const std::vector<std::uint8_t>& u) {
    return util::to_hex(util::sha256(std::span<const std::uint8_t>(u))).substr(0, 16);
}

// --- the tail statistics ---------------------------------------------------------------------

void test_gauss_z() {
    check::current_step = "gauss_z";
    for (const em_pins::GaussZ& r : em_pins::GAUSS_Z) near(em::gauss_z(r.log_p), r.z, "_gauss_z(" + hex(r.log_p) + ")");
    // 0 at and above log(0.5); the first value below it is positive
    check::equal(em::gauss_z(std::log(0.5)), 0.0, "_gauss_z(log 0.5)");
    check::is_true(em::gauss_z(std::nextafter(std::log(0.5), -1.0)) > 0.0, "_gauss_z just below log 0.5");
}

void test_template_z() {
    check::current_step = "template_z";
    for (const em_pins::TemplateZ& r : em_pins::TEMPLATE_Z) {
        const std::string what = "template_z(" + hex(r.lam) + ", " + std::to_string(r.k) + ", " + hex(r.n_cells) + ")";
        const std::string want_error = r.error;
        try {
            const double z = em::template_z(r.lam, r.k, r.n_cells);
            check::is_true(want_error.empty(), what + " returned " + hex(z) + ", want ValueError " + want_error);
            if (want_error.empty()) near(z, r.z, what);
        } catch (const std::invalid_argument& e) {
            check::equal(std::string(e.what()), want_error, what + " refusal");
        }
    }
}

// --- the references ----------------------------------------------------------------------------

// gen_em_pins.prior_case
std::pair<std::vector<double>, std::vector<double>> prior_case(int k, int n) {
    auto s = units(600 + static_cast<std::uint64_t>(k), static_cast<std::size_t>(n));
    auto w = units(700 + static_cast<std::uint64_t>(k), static_cast<std::size_t>(n));
    for (double& v : s) v = (v - 0.5) * 8.0;
    for (double& v : w) v = v * 5.0 - 1.0;
    if (k % 2 == 1) s[3] = std::numeric_limits<double>::quiet_NaN();
    if (k % 3 == 0) w[5] = std::numeric_limits<double>::infinity();
    if (k == 4) w[2] = std::numeric_limits<double>::quiet_NaN();
    return {s, w};
}

void test_prior() {
    check::current_step = "lmmse_latents / prior_from";
    for (const em_pins::Prior& r : em_pins::PRIOR) {
        const auto [s, w] = prior_case(r.k, r.n);
        const std::string what = "n " + std::to_string(r.n) + " q " + std::to_string(r.q);
        const em::Lmmse l = em::lmmse_latents(s, w);
        check::equal(sha_canon(l.a, l.r), std::string(r.lmmse), "lmmse_latents " + what);
        const types::EmPrior p = em::prior_from_u64(s, w, r.q);
        check::equal(sha_canon(p.x_hat, p.v), std::string(r.prior), "prior_from_u64 " + what);
        if (r.q <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            const types::EmPrior p2 = em::prior_from(s, w, static_cast<std::int64_t>(r.q));
            check::is_true(p2.x_hat.size() == p.x_hat.size() && sha_canon(p2.x_hat, p2.v) == r.prior,
                           "prior_from " + what);
        }
    }
    for (const em_pins::PriorRefusal& r : em_pins::PRIOR_REFUSALS) {
        const std::vector<double> s(static_cast<std::size_t>(r.s_n), 1.0), w(static_cast<std::size_t>(r.w_n), 1.0);
        std::string got;
        try {
            const types::EmPrior p = em::prior_from(s, w, r.q);
            got = sha_canon(p.x_hat, p.v);
        } catch (const std::invalid_argument& e) {
            got = std::string("error: ") + e.what();
        }
        check::equal(got, std::string(r.result),
                     "prior_from(" + std::to_string(r.s_n) + " values, " + std::to_string(r.w_n) + " weights, " +
                         std::to_string(r.q) + ")");
    }
}

// --- the store-backed pieces --------------------------------------------------------------------

void test_scenarios() {
    int ran = 0;
    for (const em_pins::Scenario& sc : em_pins::SCENARIOS) {
        const std::string name = sc.name;
        check::current_step = sc.name;
        const fs::path dir = fresh_dir(name);
        Cleanup cleanup{dir};
        const std::string root = util::path_text(dir / "qrss");
        qs::Store::Seams seams;
        seams.clock = [] { return em_pins::T0; };
        qs::Store st(root, seams);
        const qs::Key key = key_of(sc.key);
        std::map<std::string, types::PassResult> built;
        std::vector<std::pair<std::string, std::int64_t>> members;
        for (const std::string& m : split(sc.members, ' ')) {
            const auto colon = m.rfind(':');
            const std::string pname = m.substr(0, colon);
            members.emplace_back(pname, std::stoll(m.substr(colon + 1)));
            built[pname] = make_pass(pass_spec_named(pname));
            st.add_pass(built[pname]);
        }
        for (const auto& [pname, g] : members) st.attach(key, built[pname].uid, g, "test", sc.mode, 0xD1D8);
        const qs::Accumulator acc = st.accumulator(key);
        check::equal(acc.members.size(), members.size(), name + ": members");
        for (const em_pins::Result& r : em_pins::RESULTS) {
            if (name != r.scenario) continue;
            ++ran;
            const std::string query = r.query, want = r.result;
            const auto f = split(query, '|');
            std::string got;
            if (f[0] == "em_prior") {
                const std::string& uid = built.at(f[1]).uid;
                try {
                    const types::EmPrior p =
                        em::em_prior(acc, uid, st, std::stoll(f[3]), qrss::frame::get_any(f[2]));
                    got = sha_canon(p.x_hat, p.v);
                } catch (const std::invalid_argument& e) {
                    got = "error: " + replace_all(e.what(), uid, "<uid>");
                }
            } else if (f[0] == "mean_w") {
                const double g = em::acc_mean_w_db(acc), w = std::strtod(want.c_str(), nullptr);
                // numpy's SIMD log10 against the C library's: the last place at most
                check::is_true(g == w || std::abs(g - w) <= std::abs(std::nextafter(w, 0.0) - w),
                               name + ": acc_mean_w_db " + hex(g) + ", want " + want);
                continue;
            } else if (f[0] == "known_header") {
                got = header_text(em::known_header(st, acc, std::stoll(f[1])));
            } else if (f[0] == "known_keying") {
                const std::optional<std::string> call = f[1] == "<none>" ? std::nullopt : std::optional(f[1]);
                const auto k = em::known_keying(st, acc, call, std::stoll(f[2]));
                got = k ? units_text(k->units) + "|" + (k->ook ? "1" : "0") : "None";
            } else {
                throw std::runtime_error("unknown query " + query);
            }
            check::equal(got, want, name + ": " + query);
        }
    }
    check::equal(ran, static_cast<int>(std::size(em_pins::RESULTS)), "every pinned result was checked");
}

void test_mean_w_empty() {
    check::current_step = "acc_mean_w_db(empty)";
    const qs::Accumulator acc = qs::Accumulator::empty({"NONE", 1}, 0, 1, 0.0);
    check::equal(em::acc_mean_w_db(acc), -std::numeric_limits<double>::infinity(), "acc_mean_w_db of nothing heard");
}

// --- a stored pass --------------------------------------------------------------------------------

void test_pass_spec() {
    check::current_step = "pass_spec";
    for (const em_pins::SpecCase& c : em_pins::SPEC_CASES) {
        types::PassResult p = make_pass(pass_spec_named("t0"));
        p.frame = c.frame;
        if (c.version >= 0) {
            auto h = hdr_of(0, 0);
            h.version = c.version;
            p.header = h;
        }
        std::string got;
        try {
            const qrss::frame::FrameSpec spec = em::pass_spec(p);
            got = spec.name() + "|" + std::to_string(spec.n_data()) + "|" + (em::is_l(p) ? "1" : "0");
        } catch (const std::invalid_argument& e) {
            got = std::string("error: ValueError: ") + e.what();
        }
        check::equal(got, std::string(c.result), std::string("pass_spec(") + c.frame + ", " + std::to_string(c.version) + ")");
    }
    em::ce_only(qrss::frame::get_any("short"), "em.pass_path");
    try {
        em::ce_only(qrss::frame::get_any("l-short"), "em.pass_path");
        check::is_true(false, "ce_only refuses an L frame");
    } catch (const std::invalid_argument& e) {
        check::equal(std::string(e.what()), std::string("em.pass_path is CE-only (design 8.3): frame 'l-short' is waveform L"),
                     "ce_only's refusal");
    }
}

void test_flat() {
    check::current_step = "flat";
    const types::FreqPath f = em::flat(1234.5);
    check::is_true(f.t_s == std::vector<double>{-20.0, 0.0, 1800.0}, "_flat t_s");
    check::is_true(f.f_hz == std::vector<double>{1234.5, 1234.5, 1234.5}, "_flat f_hz");
    check::is_true(f.weight == std::vector<double>{1.0, 1.0, 1.0}, "_flat weight");
}

// --- the hooks -------------------------------------------------------------------------------------

void test_hooks() {
    check::current_step = "hooks";
    em::Hooks none;
    em::check_cancel(none);   // no flag: never cancelled
    em::log_line(none, "nothing");
    em::report_progress(none, 1, 2, "nothing");

    std::atomic<bool> flag{false};
    std::vector<std::string> lines;
    std::vector<std::string> progress;
    em::Hooks h;
    h.log = [&](const std::string& s) { lines.push_back(s); };
    h.progress = [&](std::int64_t done, std::int64_t total, const std::string& what) {
        progress.push_back(std::to_string(done) + "/" + std::to_string(total) + " " + what);
    };
    h.cancel = &flag;
    em::check_cancel(h);
    em::log_line(h, "em: round 1");
    em::report_progress(h, 3, 7, "passes");
    check::is_true(lines == std::vector<std::string>{"em: round 1"}, "log_line reaches the log");
    check::is_true(progress == std::vector<std::string>{"3/7 passes"}, "report_progress reaches progress");

    // set from another thread, seen at the next poll
    std::thread([&] { flag.store(true, std::memory_order_relaxed); }).join();
    bool threw = false;
    try {
        em::check_cancel(h);
    } catch (const em::Cancelled& e) {
        threw = true;
        check::equal(std::string(e.what()), std::string("cancelled"), "Cancelled's text");
    }
    check::is_true(threw, "check_cancel throws Cancelled once the flag is set");
    // Cancelled is a runtime_error, so a caller catching those sees it
    try {
        em::check_cancel(h);
    } catch (const std::runtime_error&) {
        threw = false;
    }
    check::is_true(!threw, "Cancelled is a std::runtime_error");
    flag.store(false);
    em::check_cancel(h);
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(600.0, "qrss_em");
    try {
        test_gauss_z();
        test_template_z();
        test_prior();
        test_scenarios();
        test_mean_w_empty();
        test_pass_spec();
        test_flat();
        test_hooks();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL in %s: %s\n", check::current_step.load(), e.what());
        return 1;
    }
    std::printf("worst relative error against the pinned statistics: %.3g\n", worst_rel);
    return check::report("qrss_em");
}
