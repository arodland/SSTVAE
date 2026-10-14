// native/core/qrss/listener (the live listener) without Python.
//
// The scenarios of golden_qrss/listener_pins.json (tools/
// gen_qrss_listener_pins.py) were run through the reference's
// LiveListener with its receivers stubbed: this replays each through the
// C++ one with the same stubs (the receivers' answers, call by call, from
// the file) and requires what the reference did -- every receiver call
// with its arguments, each step's result, the log lines, the tiles,
// state.json's and listener.log's text byte for byte, and the pictures of
// a fake codec. tests/test_qrss_native_listener.py compares the two side
// by side where Python is available (more scenarios, the real receiver
// and the real codec); this keeps the C++ honest without it, and covers
// what only the C++ has: its threads.
//
//   * two threads feeding and stepping (and a third reading the tiles and
//     writing state.json) while the background workers receive;
//   * feed(), step() and the readers while a receive is in flight, which
//     must never wait on it (the receive is held until they are done: a
//     lock held across it would deadlock the test, and the watchdog names
//     the step);
//   * close() with a receive in flight: it waits for the receive, which is
//     applied, and runs nothing queued; a second close() returns at once;
//   * an exception in a live receive: logged once, the slot received again
//     at its next refresh (the reference retired it);
//   * run() replaying a source to its end.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "images/images.hpp"
#include "images/types.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/listener.hpp"
#include "qrss/live.hpp"
#include "qrss/picture.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/render.hpp"
#include "qrss/source.hpp"

using namespace sstvae;
namespace lv = sstvae::qrss::live;
namespace fr = sstvae::qrss::frame;
namespace pj = sstvae::qrss::pyjson;
namespace types = sstvae::qrss::types;
namespace rcv = sstvae::qrss::receiver;
namespace aq = sstvae::qrss::acquire;
namespace hdr = sstvae::qrss::header;
namespace qcfg = sstvae::qrss::config;
namespace fs = std::filesystem;

namespace {

std::string g_golden;   // argv[1]: native/tests/golden_qrss
fs::path g_tmp;         // this run's scratch directory

struct Step {
    explicit Step(const char* name) { check::current_step = name; }
};

std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void write_text(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary);
    f << s;
    if (!f) throw std::runtime_error("cannot write " + p.string());
}

fs::path fresh_dir(const std::string& name) {
    const fs::path d = g_tmp / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

// The generator's audio: sample k of a chunk of seed s is
// 1e-3 ((h(k + 7919 s) / 2^32) - 0.5), h(i) = i 2654435761 mod 2^32.
std::vector<double> audio(std::int64_t n, std::int64_t seed) {
    std::vector<double> x(static_cast<std::size_t>(n));
    for (std::int64_t k = 0; k < n; ++k) {
        const std::uint64_t i = static_cast<std::uint64_t>(k) + 7919ULL * static_cast<std::uint64_t>(seed);
        const std::uint64_t h = (i * 2654435761ULL) % 4294967296ULL;
        x[static_cast<std::size_t>(k)] = 1e-3 * (static_cast<double>(h) / 4294967296.0 - 0.5);
    }
    return x;
}

// --- the stubs' records and answers ----------------------------------------------------------

pj::Value num(double v) { return pj::Value::number(v); }
pj::Value integer(std::int64_t v) { return pj::Value::integer(v); }
pj::Value str(std::string s) { return pj::Value::string(std::move(s)); }

pj::Value floats(const std::vector<double>& v) {
    pj::Value a = pj::Value::array();
    for (const double x : v) a.push_back(num(x));
    return a;
}

pj::Value spans(const std::vector<std::pair<double, double>>& e) {
    pj::Value a = pj::Value::array();
    for (const auto& [lo, hi] : e) a.push_back(floats({lo, hi}));
    return a;
}

pj::Value f_of(std::span<const types::Detection> ds) {
    std::vector<double> f;
    for (const auto& d : ds) f.push_back(d.f_hz);
    return floats(f);
}

// a hint as the generator records it: [f, waveform] for (f, waveform), else f
pj::Value hints_of(std::span<const aq::Hint> hs) {
    pj::Value a = pj::Value::array();
    for (const aq::Hint& h : hs) {
        if (h.waveform) {
            pj::Value t = pj::Value::array();
            t.push_back(num(h.f_hz));
            t.push_back(integer(*h.waveform));
            a.push_back(std::move(t));
        } else {
            a.push_back(num(h.f_hz));
        }
    }
    return a;
}

types::Detection det_of(const pj::Value& d) {
    types::Detection o;
    o.f_hz = d.at("f_hz").as_double();
    if (const pj::Value* w = d.find("waveform")) o.waveform = w->as_int();
    if (const pj::Value* z = d.find("z_ref")) o.z_ref = z->as_double();
    if (const pj::Value* m = d.find("method")) o.method = m->as_string();
    return o;
}

std::vector<types::Detection> dets_of(const pj::Value& a) {
    std::vector<types::Detection> out;
    for (const pj::Value& d : a.items()) out.push_back(det_of(d));
    return out;
}

hdr::HeaderFields header_of(const pj::Value& h) {
    hdr::HeaderFields o;
    o.callsign = h.at("callsign").as_string();
    if (const pj::Value* g = h.find("grid"); g && !g->is_null()) o.grid = g->as_string();
    o.picture_id = h.at("picture_id").as_int();
    o.mode = h.at("mode").as_int();
    o.segment = h.at("segment").as_int();
    o.codec_id = h.at("codec_id").as_int();
    return o;
}

// The generator's _Pass: w ones, z (k % 7 - 3) / 4 in float32, n -1 for picture.SENT.
types::PassResult pass_of(const pj::Value& d) {
    types::PassResult p;
    const std::int64_t n0 = d.at("n").as_int();
    const std::size_t n = static_cast<std::size_t>(n0 == -1 ? qrss::picture::SENT : n0);
    p.f_hz = d.at("f_hz").as_double();
    p.uid = d.at("uid").as_string();
    p.report.snr2500_db = d.at("snr").as_double();
    p.report.z_ref = d.at("z_ref").as_double();
    p.report.suspect = false;
    if (const pj::Value* h = d.find("header")) p.header = header_of(*h);
    if (const pj::Value* c = d.find("cw")) {
        types::CwIdResult cw;
        cw.text = c->as_string();
        p.cw = cw;
    }
    if (const pj::Value* w = d.find("waveform")) {
        p.waveform = w->as_int();
        p.frame = p.waveform == qcfg::WAVEFORM_L ? "l-tiny" : "tiny";
    }
    p.w.assign(n, 1.0f);
    p.z.resize(n);
    for (std::size_t k = 0; k < n; ++k) p.z[k] = (static_cast<float>(k % 7) - 3.0f) / 4.0f;
    return p;
}

std::vector<types::PassResult> passes_of(const pj::Value& a) {
    std::vector<types::PassResult> out;
    for (const pj::Value& p : a.items()) out.push_back(pass_of(p));
    return out;
}

// One scenario's stub receivers: each answers from its queue in order, the
// last answer repeated, and records its call as the generator does.
struct Script {
    std::mutex mu;
    std::map<std::string, std::deque<pj::Value>> answers;
    std::vector<std::string> calls;   // json.dumps of each call

    explicit Script(const pj::Value& sc) {
        for (const char* k : {"candidates", "detect", "receive", "finish"})
            for (const pj::Value& a : sc.at(k).items()) answers[k].push_back(a);
    }

    pj::Value take(const std::string& k) {
        std::deque<pj::Value>& q = answers.at(k);
        if (q.size() > 1) {
            pj::Value a = q.front();
            q.pop_front();
            return a;
        }
        return q.front();
    }

    pj::Value call(const pj::Value& record, const std::string& k) {
        std::lock_guard<std::mutex> lk(mu);
        calls.push_back(pj::dumps(record));
        return take(k);
    }

    lv::Seams seams() {
        lv::Seams s;
        s.candidates = [this](const rcv::Prepared& prep, auto, std::span<const aq::Hint> hints) {
            pj::Value r = pj::Value::array();
            r.push_back(str("candidates"));
            r.push_back(integer(prep.q));
            r.push_back(integer(static_cast<std::int64_t>(prep.fe.size())));
            r.push_back(hints_of(hints));
            return dets_of(call(r, "candidates"));
        };
        s.detect = [this](const rcv::Prepared& prep, auto, const rcv::DetectOptions& opt) {
            pj::Value r = pj::Value::array();
            r.push_back(str("detect"));
            r.push_back(integer(prep.q));
            r.push_back(integer(static_cast<std::int64_t>(prep.fe.size())));
            r.push_back(pj::Value::boolean(opt.live_only));
            r.push_back(spans(opt.erase_s));
            r.push_back(opt.cands ? f_of(*opt.cands) : pj::Value::null());
            r.push_back(floats(opt.skip_hz));
            return dets_of(call(r, "detect"));
        };
        s.receive_slot = [this](const rcv::Prepared& prep, auto, const rcv::SlotOptions& opt) {
            pj::Value r = pj::Value::array();
            r.push_back(str("receive"));
            r.push_back(integer(prep.q));
            r.push_back(integer(static_cast<std::int64_t>(prep.fe.size())));
            r.push_back(str(opt.estimator));
            r.push_back(pj::Value::boolean(opt.round_b));
            r.push_back(spans(opt.erase_s));
            r.push_back(opt.dets ? f_of(*opt.dets) : pj::Value::array());
            return passes_of(call(r, "receive"));
        };
        s.finish = [this](std::span<const std::complex<float>> x, std::span<const std::uint8_t> ok, std::int64_t origin,
                          std::int64_t q, const qrss::api::ReceiveConfig& config,
                          std::span<const types::Detection> dets, std::span<const aq::Hint> hints) {
            pj::Value r = pj::Value::array();
            r.push_back(str("finish"));
            r.push_back(integer(q));
            r.push_back(integer(static_cast<std::int64_t>(x.size())));
            r.push_back(integer(static_cast<std::int64_t>(std::count_if(ok.begin(), ok.end(),
                                                                         [](std::uint8_t b) { return b != 0; }))));
            r.push_back(integer(origin));
            r.push_back(str(config.estimator));
            r.push_back(f_of(dets));
            r.push_back(hints_of(hints));
            const pj::Value a = call(r, "finish");
            lv::FinishResult out;
            out.passes = passes_of(a.items().at(0));
            out.heard = a.items().at(1).as_double();
            return out;
        };
        return s;
    }
};

// The generator's fake codec: picture k (counting from 0) is 8 x 6 of
// value (37 k + int(sum of the weight plane) % 200) % 256.
class FakeRenderer : public lv::Renderer {
public:
    std::optional<std::string> load(std::optional<std::int64_t>, const Warn&) override { return std::nullopt; }
    std::optional<std::string> check(std::optional<std::int64_t>) override { return std::nullopt; }
    images::Picture render(std::span<const double> S, std::span<const double> W, std::int64_t mode,
                           std::optional<std::int64_t>) override {
        const auto planes = qrss::render::decoder_planes(S, W, mode);
        double sum = 0.0;
        for (const float w : planes.second) sum += static_cast<double>(w);
        std::lock_guard<std::mutex> lk(mu_);
        const int v = static_cast<int>((37 * k_ + static_cast<std::int64_t>(sum) % 200) % 256);
        ++k_;
        images::Picture p(8, 6);
        std::fill(p.rgb.begin(), p.rgb.end(), static_cast<std::uint8_t>(v));
        return p;
    }

private:
    std::mutex mu_;
    std::int64_t k_ = 0;
};

constexpr double CLOCK_DEFAULT = 1'760'000'000.0;

lv::LiveConfig config_of(const pj::Value& c) {
    lv::LiveConfig cfg;
    cfg.render = false;   // (the generator's default)
    for (std::size_t i = 0; i < c.keys().size(); ++i) {
        const std::string& k = c.keys()[i];
        const pj::Value& v = c.values()[i];
        if (k == "spec") cfg.spec = fr::get(v.as_string());
        else if (k == "waveforms") {
            std::vector<std::int64_t> w;
            for (const pj::Value& x : v.items()) w.push_back(x.as_int());
            cfg.waveforms = w;
        } else if (k == "render") cfg.render = v.as_bool();
        else if (k == "refresh_s") cfg.refresh_s = v.as_double();
        else if (k == "first_s") cfg.first_s = v.as_double();
        else if (k == "detect_until_s") cfg.detect_until_s = v.as_double();
        else if (k == "recheck_until_s") cfg.recheck_until_s = v.as_double();
        else if (k == "empty_refresh_s") cfg.empty_refresh_s = v.as_double();
        else if (k == "keep_done_s") cfg.keep_done_s = v.as_double();
        else throw std::runtime_error("listener_pins: unknown config key " + k);
    }
    return cfg;
}

std::string tiles_json(const std::vector<lv::Tile>& tiles) {
    pj::Value a = pj::Value::array();
    for (const lv::Tile& t : tiles) a.push_back(t.to_json());
    return pj::dumps(a);
}

void equal_strings(const std::vector<std::string>& got, const pj::Value& want, const std::string& what) {
    std::vector<std::string> w;
    for (const pj::Value& x : want.items()) w.push_back(x.as_string());
    ++check::checks;
    if (got == w) return;
    std::string detail = std::to_string(got.size()) + " against " + std::to_string(w.size());
    for (std::size_t k = 0; k < std::max(got.size(), w.size()); ++k) {
        const std::string g = k < got.size() ? got[k] : "(none)";
        const std::string e = k < w.size() ? w[k] : "(none)";
        if (g != e) {
            detail += "; first difference at " + std::to_string(k) + ":\n  got  " + g + "\n  want " + e;
            break;
        }
    }
    check::fail(what, detail);
}

// Where two texts part, for a failure that says where.
void equal_text(const std::string& got, const std::string& want, const std::string& what) {
    ++check::checks;
    if (got == want) return;
    std::size_t k = 0;
    while (k < got.size() && k < want.size() && got[k] == want[k]) ++k;
    const std::size_t a = k > 60 ? k - 60 : 0;
    check::fail(what, "differ at byte " + std::to_string(k) + ":\n  got  ..." + got.substr(a, 160) + "\n  want ..." +
                          want.substr(a, 160));
}

// --- the pinned scenarios -------------------------------------------------------------------

void replay(const pj::Value& sc, double clock) {
    const std::string name = sc.at("name").as_string();
    const pj::Value& expect = sc.at("expect");
    const fs::path d = fresh_dir("pins_" + name);
    if (const pj::Value* st = sc.find("state")) write_text(d / "state.json", pj::dumps(*st));
    if (const pj::Value* df = sc.find("dial_file")) write_text(d / lv::DIAL_FILE, df->as_string());
    const lv::LiveConfig cfg = config_of(sc.at("cfg"));

    Script script(sc);
    std::mutex sink_mu;
    std::vector<std::string> sink;
    lv::Seams seams = script.seams();
    seams.clock = [clock] { return clock; };
    seams.log = [&](const std::string& line) {
        std::lock_guard<std::mutex> lk(sink_mu);
        sink.push_back(line);
    };
    if (const pj::Value* g = sc.find("dial_guard")) seams.dial_guard = {g->items().at(0).as_double(), g->items().at(1).as_double()};
    if (cfg.render) seams.renderer = std::make_shared<FakeRenderer>();

    auto L = std::make_unique<lv::LiveListener>(d.string(), cfg, nullptr, nullptr, seams);
    for (const pj::Value& f : sc.at("feeds").items()) {
        const auto& v = f.items();
        L->feed(audio(v.at(0).as_int(), v.at(1).as_int()), v.at(2).as_double());
    }
    std::vector<std::string> steps;
    std::vector<std::string> want_steps;
    for (const pj::Value& t : sc.at("steps").items()) steps.push_back(L->step(t.as_double()) ? "true" : "false");
    for (const pj::Value& b : expect.at("steps").items()) want_steps.push_back(b.as_bool() ? "true" : "false");
    ++check::checks;
    if (steps != want_steps) check::fail(name + ": steps", "differ");
    L->write_state();

    const std::string what = "pins " + name + ": ";
    {
        std::lock_guard<std::mutex> lk(sink_mu);
        equal_strings(sink, expect.at("sink"), what + "log sink");
    }
    equal_strings(L->lines(), expect.at("lines"), what + "lines");
    equal_text(tiles_json(L->tiles()), expect.at("tiles").as_string(), what + "tiles");
    equal_text(read_text(d / "state.json"), expect.at("state").as_string(), what + "state.json");
    std::vector<std::int64_t> want_finished;
    for (const pj::Value& q : expect.at("finished").items()) want_finished.push_back(q.as_int());
    check::is_true(L->finished() == want_finished, what + "finished");
    const pj::Value& log = expect.at("log");
    const fs::path log_path = d / lv::LOG_FILE;
    if (log.is_null()) {
        check::is_true(!fs::exists(log_path), what + "no listener.log");
    } else {
        equal_text(fs::exists(log_path) ? read_text(log_path) : std::string(), log.as_string(), what + "listener.log");
    }

    // the pictures: the same files, each its size and the one value its pixels take
    std::set<std::string> pngs;
    for (const auto& e : fs::directory_iterator(d / "tiles"))
        if (e.path().extension() == ".png") pngs.insert(e.path().filename().string());
    const pj::Value& pics = expect.at("pictures");
    check::equal(pngs.size(), pics.keys().size(), what + "pictures");
    for (std::size_t i = 0; i < pics.keys().size(); ++i) {
        const std::string& f = pics.keys()[i];
        const auto& want = pics.values()[i].items();
        if (!pngs.count(f)) {
            check::fail(what + f, "missing");
            continue;
        }
        const images::Picture pic = images::load((d / "tiles" / f).string());
        check::equal(pic.width, static_cast<int>(want.at(0).as_int()), what + f + " width");
        check::equal(pic.height, static_cast<int>(want.at(1).as_int()), what + f + " height");
        std::set<std::int64_t> values(pic.rgb.begin(), pic.rgb.end());
        std::set<std::int64_t> want_values;
        for (const pj::Value& v : want.at(2).items()) want_values.insert(v.as_int());
        check::is_true(values == want_values, what + f + " pixels");
    }

    if (const pj::Value* rt = expect.find("restart_tiles")) {
        lv::Seams s2 = script.seams();
        s2.clock = [clock] { return clock; };
        s2.log = [](const std::string&) {};
        s2.dial_guard = seams.dial_guard;
        lv::LiveListener L2(d.string(), cfg, nullptr, nullptr, s2);
        const std::vector<lv::Tile> t2 = L2.tiles();
        equal_text(tiles_json(t2), rt->as_string(), what + "tiles read back");
        if (!t2.empty()) L2.step(t2.front().updated + cfg.keep_done_s + 1);
        equal_text(tiles_json(L2.tiles()), expect.at("expired_tiles").as_string(), what + "tiles expired");
    }
    // every receiver call, the restarted listener's included (it makes none)
    std::lock_guard<std::mutex> lk(script.mu);
    std::vector<std::string> want_calls;
    for (const pj::Value& c : expect.at("calls").items()) want_calls.push_back(pj::dumps(c));
    ++check::checks;
    if (script.calls != want_calls) {
        std::string detail = std::to_string(script.calls.size()) + " calls against " + std::to_string(want_calls.size());
        for (std::size_t k = 0; k < std::max(script.calls.size(), want_calls.size()); ++k) {
            const std::string g = k < script.calls.size() ? script.calls[k] : "(none)";
            const std::string e = k < want_calls.size() ? want_calls[k] : "(none)";
            if (g != e) {
                detail += "; first difference at " + std::to_string(k) + ":\n  got  " + g + "\n  want " + e;
                break;
            }
        }
        check::fail(what + "receiver calls", detail);
    }
}

void test_pinned_scenarios() {
    const pj::Value pins = pj::loads(read_text(fs::path(g_golden) / "listener_pins.json"));
    const double clock = pins.at("clock").as_double();
    check::equal(pins.at("q").as_int(), std::int64_t{1954752}, "pins: Q_TEST");
    check::is_true(pins.at("scenarios").items().size() >= 7, "pins: the scenarios");
    static std::deque<std::string> names;   // (the watchdog may read a step's name at any time)
    for (const pj::Value& sc : pins.at("scenarios").items()) {
        names.push_back("pinned scenario " + sc.at("name").as_string());
        check::current_step = names.back().c_str();
        replay(sc, clock);
    }
}

// --- threads ------------------------------------------------------------------------------------

constexpr std::int64_t Q = 1954752;

double t0() { return static_cast<double>(lv::slot_t0(Q)); }
double tiny_end() { return t0() + lv::frame_seconds(fr::get("tiny")) + lv::PB_TAIL_S + 0.5; }

types::Detection det(double f) {
    types::Detection d;
    d.f_hz = f;
    return d;
}

types::PassResult pass(double f, std::int64_t n = 1024) {
    types::PassResult p;
    p.f_hz = f;
    p.uid = "u" + std::to_string(static_cast<long long>(f));
    p.report.snr2500_db = -12.0;
    p.report.z_ref = 20.0;
    p.w.assign(static_cast<std::size_t>(n), 1.0f);
    p.z.assign(static_cast<std::size_t>(n), 0.25f);
    return p;
}

// Stub receivers that each find the signal at 1500 Hz, after `delay`.
lv::Seams quick_stubs(std::chrono::milliseconds delay, std::atomic<int>* receives = nullptr) {
    lv::Seams s;
    s.clock = [] { return CLOCK_DEFAULT; };
    s.log = [](const std::string&) {};
    s.candidates = [](const rcv::Prepared&, auto, std::span<const aq::Hint>) {
        return std::vector<types::Detection>{det(1500.0)};
    };
    s.detect = [](const rcv::Prepared&, auto, const rcv::DetectOptions&) {
        return std::vector<types::Detection>{det(1500.0)};
    };
    s.receive_slot = [delay, receives](const rcv::Prepared&, auto, const rcv::SlotOptions&) {
        std::this_thread::sleep_for(delay);
        if (receives) ++*receives;
        return std::vector<types::PassResult>{pass(1500.0)};
    };
    s.finish = [delay](std::span<const std::complex<float>>, std::span<const std::uint8_t>, std::int64_t,
                       std::int64_t, const qrss::api::ReceiveConfig&, std::span<const types::Detection>,
                       std::span<const aq::Hint>) {
        std::this_thread::sleep_for(delay);
        lv::FinishResult r;
        r.passes = {pass(1500.0)};
        r.heard = 1.0;
        return r;
    };
    return s;
}

lv::LiveConfig tiny_config(bool background) {
    lv::LiveConfig cfg;
    cfg.spec = fr::get("tiny");
    cfg.refresh_s = 5.0;
    cfg.render = false;
    cfg.background = background;
    return cfg;
}

std::optional<lv::Tile> tile_of(const lv::LiveListener& L, const std::string& id) {
    for (const lv::Tile& t : L.tiles())
        if (t.id == id) return t;
    return std::nullopt;
}

// Two threads feed and step while the workers receive, a third reads.
void test_threads_feed_and_step() {
    Step step("threads: feeding and stepping while the workers run");
    const fs::path d = fresh_dir("threads");
    std::atomic<int> receives{0};
    lv::LiveListener L(d.string(), tiny_config(true), nullptr, nullptr,
                       quick_stubs(std::chrono::milliseconds(15), &receives));
    const double start = t0() - 16.0;
    const double end = tiny_end() + 2.0;
    const std::int64_t chunk = qcfg::FS / 4;   // 0.25 s
    const std::int64_t n_chunks = static_cast<std::int64_t>((end - start) * 4.0);
    std::atomic<std::int64_t> fed{0};
    std::atomic<bool> done{false};
    std::atomic<int> errors{0};

    std::thread feeder([&] {
        try {
            for (std::int64_t k = 0; k < n_chunks; ++k) {
                L.feed(audio(chunk, 100 + k), start + 0.25 * static_cast<double>(k + 1));
                fed = k + 1;
                if (k % 16 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "feeder: %s\n", e.what());
            ++errors;
        }
        done = true;
    });
    std::thread stepper([&] {
        try {
            while (!done) {
                const std::optional<double> now = L.audio_end();
                if (now) L.step(*now);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "stepper: %s\n", e.what());
            ++errors;
        }
    });
    std::thread reader([&] {
        try {
            while (!done) {
                (void)L.tiles();
                (void)L.busy();
                (void)L.lines();
                (void)L.idle();
                L.write_state();
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "reader: %s\n", e.what());
            ++errors;
        }
    });
    feeder.join();
    stepper.join();
    reader.join();
    check::equal(errors.load(), 0, "threads: no exception");
    check::equal(fed.load(), n_chunks, "threads: every chunk fed");

    // the replay's end: the slot finishes once the workers are idle
    check::is_true(L.wait_idle(60.0), "threads: the workers go idle");
    L.step(end);
    check::is_true(L.wait_idle(60.0), "threads: the end-of-slot receive returns");
    L.step(end + 1.0);
    const std::vector<std::int64_t> fin = L.finished();
    check::is_true(std::find(fin.begin(), fin.end(), Q) != fin.end(), "threads: the slot finished");
    const std::optional<lv::Tile> t = tile_of(L, "1954752-1500");
    check::is_true(t && t->status == "complete", "threads: its tile complete");
    check::is_true(receives.load() >= 1, "threads: live receives ran");
    const pj::Value st = pj::loads(read_text(d / "state.json"));
    check::is_true(st.at("tiles").kind() == pj::Value::Kind::Array, "threads: state.json is JSON");
    L.close();
}

// feed(), step() and the readers while a receive is held in flight; then
// close() waits for it, and it is applied.
void test_close_with_a_receive_in_flight() {
    Step step("close: a receive in flight");
    const fs::path d = fresh_dir("close");
    std::mutex mu;
    std::condition_variable cv;
    bool entered = false, release = false, timed_out = false;
    std::atomic<int> receives{0};

    lv::Seams s = quick_stubs(std::chrono::milliseconds(0));
    s.receive_slot = [&](const rcv::Prepared&, auto, const rcv::SlotOptions&) {
        std::unique_lock<std::mutex> lk(mu);
        entered = true;
        cv.notify_all();
        if (!cv.wait_for(lk, std::chrono::seconds(60), [&] { return release; })) timed_out = true;
        ++receives;
        return std::vector<types::PassResult>{pass(1500.0)};
    };
    auto L = std::make_unique<lv::LiveListener>(d.string(), tiny_config(true), nullptr, nullptr, s);
    L->feed(audio(static_cast<std::int64_t>(46.0 * qcfg::FS), 1), t0() + 30.0);
    L->step(t0() + 30.0);
    {
        std::unique_lock<std::mutex> lk(mu);
        check::is_true(cv.wait_for(lk, std::chrono::seconds(30), [&] { return entered; }), "close: the receive started");
    }

    // everything but close() goes on while the receive is held
    check::current_step = "close: feeding while the receive is held";
    std::vector<double> lat;
    for (int k = 0; k < 120; ++k) {
        const auto a = std::chrono::steady_clock::now();
        L->feed(audio(qcfg::FS / 4, 50 + k), t0() + 30.0 + 0.25 * (k + 1));
        lat.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
    }
    check::current_step = "close: stepping and reading while the receive is held";
    L->step(t0() + 60.0);
    const std::optional<std::string> busy = L->busy();
    check::is_true(busy && busy->find("refresh 2025-10-01T00:00:00Z") != std::string::npos, "close: busy says so");
    check::is_true(!L->idle(), "close: not idle");
    (void)L->tiles();
    L->write_state();
    check::is_true(!L->wait_idle(0.05), "close: wait_idle times out");
    std::sort(lat.begin(), lat.end());
    std::printf("feed() during a receive: p50 %.1f us, p95 %.1f us, max %.1f us (%zu chunks of 0.25 s)\n",
                lat[lat.size() / 2] * 1e6, lat[lat.size() * 95 / 100] * 1e6, lat.back() * 1e6, lat.size());

    check::current_step = "close: close() waits for the receive";
    std::atomic<bool> closed{false};
    std::thread closer([&] {
        L->close();
        closed = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    check::is_true(!closed.load(), "close: waits for the receive in flight");
    {
        std::lock_guard<std::mutex> lk(mu);
        release = true;
    }
    cv.notify_all();
    closer.join();
    check::is_true(closed.load(), "close: returns once it has");
    check::is_true(!timed_out, "close: the receive was released, not timed out");
    check::equal(receives.load(), 1, "close: one receive, nothing queued run");
    const std::optional<lv::Tile> t = tile_of(*L, "1954752-1500");
    check::is_true(t.has_value(), "close: the receive in flight was applied");
    check::current_step = "close: a second close";
    const auto a = std::chrono::steady_clock::now();
    L->close();
    check::is_true(std::chrono::steady_clock::now() - a < std::chrono::seconds(5), "close: idempotent");
    check::is_true(L->idle() && L->wait_idle(1.0), "close: idle after");
    check::current_step = "close: the destructor";
    L.reset();
}

// The destructor of a background listener with a receive in flight closes it.
void test_destructor_with_a_receive_in_flight() {
    Step step("close: the destructor with a receive in flight");
    const fs::path d = fresh_dir("dtor");
    std::atomic<bool> entered{false};
    lv::Seams s = quick_stubs(std::chrono::milliseconds(0));
    s.receive_slot = [&](const rcv::Prepared&, auto, const rcv::SlotOptions&) {
        entered = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        return std::vector<types::PassResult>{pass(1500.0)};
    };
    {
        lv::LiveListener L(d.string(), tiny_config(true), nullptr, nullptr, s);
        L.feed(audio(static_cast<std::int64_t>(46.0 * qcfg::FS), 2), t0() + 30.0);
        L.step(t0() + 30.0);
        for (int k = 0; k < 3000 && !entered; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check::is_true(entered.load(), "dtor: the receive started");
    }
    check::is_true(true, "dtor: returned");
}

// An exception in a live receive: logged once (per slot and message), and
// the slot received again at its next refresh, never raised from step().
void test_a_refresh_exception_is_retried() {
    Step step("a refresh exception: logged once and retried");
    const fs::path d = fresh_dir("refresh_error");
    std::atomic<int> calls{0};
    std::vector<std::string> sink;
    lv::Seams s = quick_stubs(std::chrono::milliseconds(0));
    s.log = [&](const std::string& line) { sink.push_back(line); };
    s.receive_slot = [&](const rcv::Prepared&, auto, const rcv::SlotOptions&) {
        const int k = ++calls;
        if (k <= 2) throw std::runtime_error("boom");
        return std::vector<types::PassResult>{pass(1500.0)};
    };
    lv::LiveConfig cfg = tiny_config(false);
    cfg.refresh_s = 10.0;
    lv::LiveListener L(d.string(), cfg, nullptr, nullptr, s);
    L.feed(audio(76 * qcfg::FS, 8), t0() + 60.0);
    bool threw = false;
    try {
        for (const double t : {t0() + 26.0, t0() + 36.0, t0() + 46.0}) L.step(t);
    } catch (const std::exception& e) {
        threw = true;
        std::fprintf(stderr, "step raised: %s\n", e.what());
    }
    check::is_true(!threw, "refresh error: never raised from step()");
    check::equal(calls.load(), 3, "refresh error: the slot received again at each refresh");
    int said = 0;
    for (const std::string& line : sink)
        if (line.find("slot 2025-10-01T00:00:00Z: RuntimeError: boom") != std::string::npos) ++said;
    check::equal(said, 1, "refresh error: said once");
    const std::optional<lv::Tile> t = tile_of(L, "1954752-1500");
    check::is_true(t && t->status == "receiving", "refresh error: the third receive's tile");
    L.step(tiny_end());
    const std::vector<std::int64_t> fin = L.finished();
    check::is_true(std::find(fin.begin(), fin.end(), Q) != fin.end(), "refresh error: the slot still finishes");
    const std::optional<lv::Tile> t2 = tile_of(L, "1954752-1500");
    check::is_true(t2 && t2->status == "complete", "refresh error: its tile complete");
}

// run() over a replayed source: "listening" first, "audio ended" last, the
// slot finished, state.json written with listening false.
class ChunkSource : public lv::Source {
public:
    ChunkSource(double start, double end, double chunk_s) : t_(start), end_(end), chunk_s_(chunk_s) {}
    std::optional<lv::Chunk> next(double) override {
        if (t_ >= end_) {
            eof_ = true;
            return std::nullopt;
        }
        lv::Chunk c;
        c.x = audio(static_cast<std::int64_t>(chunk_s_ * qcfg::FS), ++seed_);
        t_ += chunk_s_;
        c.t_end = t_;
        return c;
    }
    bool eof() const override { return eof_; }
    std::string name() const override { return "replay"; }

private:
    double t_, end_, chunk_s_;
    std::int64_t seed_ = 0;
    bool eof_ = false;
};

void test_run_replays_to_the_end() {
    for (const bool background : {false, true}) {
        Step step(background ? "run: a replay, background" : "run: a replay");
        const fs::path d = fresh_dir(background ? "run_bg" : "run");
        std::vector<std::string> sink;
        std::mutex mu;
        lv::Seams s = quick_stubs(std::chrono::milliseconds(background ? 5 : 0));
        s.log = [&](const std::string& line) {
            std::lock_guard<std::mutex> lk(mu);
            sink.push_back(line);
        };
        lv::LiveListener L(d.string(), tiny_config(background), nullptr, nullptr, s);
        ChunkSource src(t0() - 16.0, t0() + 40.0, 1.0);
        lv::run(L, src, false);
        const std::string tag = background ? "run (background): " : "run: ";
        std::lock_guard<std::mutex> lk(mu);
        check::is_true(!sink.empty() && sink.front().find(" listening: replay, frame tiny, waveform CE (tiny)") !=
                                            std::string::npos,
                       tag + "listening first");
        check::is_true(!sink.empty() && sink.back().size() >= 11 &&
                           sink.back().compare(sink.back().size() - 11, 11, "audio ended") == 0,
                       tag + "audio ended last");
        const std::vector<std::int64_t> fin = L.finished();
        check::is_true(std::find(fin.begin(), fin.end(), Q) != fin.end(), tag + "the slot finished");
        const pj::Value st = pj::loads(read_text(d / "state.json"));
        check::is_true(!st.at("listening").as_bool(), tag + "listening false at the end");
        check::equal(st.at("source").as_string(), std::string("replay"), tag + "the source's name");
        L.close();
    }
    {
        Step step("run: stopped");
        const fs::path d = fresh_dir("run_stop");
        lv::LiveListener L(d.string(), tiny_config(false), nullptr, nullptr, quick_stubs(std::chrono::milliseconds(0)));
        ChunkSource src(t0() - 16.0, t0() + 40.0, 1.0);
        const std::atomic<bool> stop{true};
        lv::run(L, src, false, &stop);
        check::is_true(!L.audio_end().has_value(), "run: a set stop feeds nothing");
        const std::vector<std::string> lines = L.lines();
        check::is_true(!lines.empty() && lines.back().find("audio ended") != std::string::npos, "run: stopped, ended");
    }
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_qrss_core_listener <golden_qrss dir>\n");
        return 2;
    }
    g_golden = argv[1];
    // ~30x the measured runtime: expiring means wedged, and names the step
    check::Watchdog watchdog(110.0, "qrss_listener");
    try {
        g_tmp = fs::temp_directory_path() / ("qrss_listener_test_" + std::to_string(std::random_device{}()));
        fs::create_directories(g_tmp);
        test_pinned_scenarios();
        test_a_refresh_exception_is_retried();
        test_threads_feed_and_step();
        test_close_with_a_receive_in_flight();
        test_destructor_with_a_receive_in_flight();
        test_run_replays_to_the_end();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL (%s): %s\n", check::current_step.load(), e.what());
        return 1;
    }
    std::error_code ec;
    fs::remove_all(g_tmp, ec);
    return check::report("qrss_listener");
}
