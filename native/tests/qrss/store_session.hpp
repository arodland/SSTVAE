// The scripted store session of gen_store_pins.py, replayed through the C++
// port (shared by test_store.cpp and test_associate.cpp): the formula
// passes, each op's result as json.dumps text, and a store's file tree as
// (path, SHA-256) pairs. See gen_store_pins.py for the formulas.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/associate.hpp"
#include "qrss/header.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"
#include "store_pins.hpp"
#include "util/sha256.hpp"

namespace store_session {

namespace qrss = sstvae::qrss;
namespace qs = sstvae::qrss::store;
namespace qa = sstvae::qrss::associate;
using qrss::pyjson::Value;

// lowbias32 (gen_store_pins.h32)
inline std::uint32_t h32(std::uint64_t in) {
    std::uint32_t x = static_cast<std::uint32_t>(in);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

inline qrss::header::HeaderFields hdr_of(int pic, int segment) {
    const store_pins::Pic& p = store_pins::PICS[pic];
    qrss::header::HeaderFields h;
    h.callsign = p.callsign;
    if (p.grid) h.grid = std::string(p.grid);
    h.picture_id = p.picture_id;
    h.mode = p.mode;
    h.segment = segment;
    h.codec_id = p.codec_id;
    return h;
}

inline qrss::types::PassResult make_pass(const store_pins::Pass& s) {
    namespace types = qrss::types;
    const int n = store_pins::N;
    const auto canon = qs::canonical_index(s.segment, n);
    types::PassResult p;
    p.z.resize(static_cast<std::size_t>(n));
    p.w.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const std::uint64_t c = canon[static_cast<std::size_t>(i)];
        const double a =
            (static_cast<double>(h32(c + 65536u * static_cast<std::uint64_t>(s.segment) +
                                     1000003u * static_cast<std::uint64_t>(s.pic)) >> 27) - 15.5) / 8.0;
        const double noise =
            (static_cast<double>(h32(static_cast<std::uint64_t>(i) + 1000003u * static_cast<std::uint64_t>(s.seed + 7)) >>
                                 28) - 7.5) / 8.0;
        const double w = (s.erase_from >= 0 && i >= s.erase_from) ? 0.0 : s.w;
        p.z[static_cast<std::size_t>(i)] = w > 0 ? static_cast<float>(a + noise) : 0.0f;
        p.w[static_cast<std::size_t>(i)] = static_cast<float>(w);
    }
    const auto h = hdr_of(s.hdr_pic, s.hdr_seg);
    const auto bits = qrss::header::encode(h);
    p.hdr_llr.resize(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i)
        p.hdr_llr[i] = s.llr ? static_cast<float>(4.0 * (1.0 - 2.0 * bits[i])) : 0.0f;
    if (s.decoded) p.header = h;
    p.uid = types::pass_uid(s.q, s.f_hz, p.z);
    p.q = s.q;
    p.frame = s.waveform ? "l-short" : "short";
    p.waveform = s.waveform;
    p.f_hz = s.f_hz;
    p.timing = {0.0, 0.0, 0.0, 10.0, sstvae::math::Matrix(2, 2, {1.0, 0.0, 0.0, 1.0})};
    p.report = {s.f_hz, 0.0, 0.0, 0.0, 0.0, -20.0, 10.0, 1.0, true, false};
    p.ch_fs = 250;
    p.ch_t0_index = 0.0;
    p.f_mix_hz = qrss::Rational(1500);
    p.psi.assign(4, 1.0f);
    p.estimator = "joint";
    p.em_round = s.em_round;
    return p;
}

struct Built {
    const store_pins::Pass* spec = nullptr;
    qrss::types::PassResult pass;
};

inline std::map<std::string, Built> make_passes() {
    std::map<std::string, Built> out;
    for (const store_pins::Pass& s : store_pins::PASSES) out[s.name] = Built{&s, make_pass(s)};
    return out;
}

inline std::vector<double> f64(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

inline Value key_json(const std::optional<qs::Key>& key) {
    if (!key) return Value::null();
    return Value::array({Value::string(key->callsign), Value::integer(key->picture_id)});
}

inline Value strings(const std::vector<std::string>& v) {
    Value out = Value::array();
    for (const std::string& s : v) out.push_back(Value::string(s));
    return out;
}

inline std::vector<std::uint8_t> read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), {});
}

// [[path, sha256], ...] of every file under root, sorted (gen_store_pins.tree).
inline std::vector<std::pair<std::string, std::string>> tree(const std::string& root) {
    std::vector<std::pair<std::string, std::string>> out;
    for (const auto& e : std::filesystem::recursive_directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        const auto bytes = read_file(e.path());
        out.emplace_back(std::filesystem::relative(e.path(), root).generic_string(),
                         sstvae::util::to_hex(sstvae::util::sha256(std::span<const std::uint8_t>(bytes))));
    }
    std::sort(out.begin(), out.end());
    return out;
}

inline Value tree_json(const std::vector<std::pair<std::string, std::string>>& t) {
    Value out = Value::array();
    for (const auto& [path, sha] : t) out.push_back(Value::array({Value::string(path), Value::string(sha)}));
    return out;
}

// gen_store_pins.foreign: pass p laid on the canonical grid as its own (S, W).
inline std::pair<std::vector<float>, std::vector<float>> foreign(const qrss::types::PassResult& p, int segment,
                                                                 const std::string& variant) {
    std::vector<float> S(qs::PLANE, 0.0f), W(qs::PLANE, 0.0f);
    const auto canon = qs::canonical_index(segment, static_cast<std::int64_t>(p.z.size()));
    const std::size_t base = static_cast<std::size_t>(segment) * qs::GROUP_LATENTS;
    for (std::size_t i = 0; i < p.z.size(); ++i) {
        S[base + canon[i]] = static_cast<float>(static_cast<double>(p.w[i]) * static_cast<double>(p.z[i]));
        W[base + canon[i]] = p.w[i];
    }
    if (variant == "nan") S[0] = std::numeric_limits<float>::quiet_NaN();
    if (variant == "negative") W[2 * static_cast<std::size_t>(qs::GROUP_LATENTS) + 7] = -1.0f;
    return {S, W};
}

inline qs::Key key_a() { return {store_pins::PICS[0].callsign, store_pins::PICS[0].picture_id}; }

// One op of the session against `st` (rooted at `root`): its result as
// json.dumps text. `soft` replaces rule 2's search (empty: the real one).
inline std::string run_op(const store_pins::Op& op, qs::Store& st, std::map<std::string, Built>& passes,
                          const std::string& root, const qa::SoftHeaderFn& soft = {}) {
    const std::string kind = op.kind, arg = op.arg, extra = op.extra;
    if (kind == "associate") {
        const qa::Result r = qa::associate(passes.at(arg).pass, st, soft);
        return dumps(Value::array({key_json(r.key), Value::string(r.rule)}));
    }
    if (kind == "corr") {
        const Built& b = passes.at(arg);
        const auto& p = b.pass;
        const auto key = st.find_member(p.uid);
        const qs::Accumulator acc = st.accumulator(key.value());
        const auto [m, W] = qa::acc_reference(acc, b.spec->segment, static_cast<std::int64_t>(p.z.size()));
        const auto z = f64(p.z), w = f64(p.w);
        auto [Sl, Wl] = acc.loo(p.uid, st);
        std::vector<double> ml(Wl.size(), 0.0);
        for (std::size_t i = 0; i < Wl.size(); ++i)
            if (Wl[i] > 0) ml[i] = Sl[i] / Wl[i];
        const qa::Corr c = qa::corr_stat(z, w, m, W);
        const qa::Corr cl = qa::corr_stat(z, w, ml, Wl);
        const double te = qa::expected_t(w, W);
        return dumps(Value::array({Value::number(c.t), Value::integer(c.M), Value::number(te),
                                   Value::boolean(qa::corr_accept(c.t, te)), Value::number(cl.t),
                                   Value::integer(cl.M), Value::number(qa::expected_t(w, Wl))}));
    }
    if (kind == "agrees") {
        const store_pins::Pass& o = *passes.at(extra).spec;
        const auto llr = f64(passes.at(arg).pass.hdr_llr);
        return dumps(Value::boolean(qa::header_agrees(hdr_of(o.hdr_pic, o.hdr_seg), std::span<const double>(llr))));
    }
    if (kind == "tree") return dumps(tree_json(tree(root)));
    if (kind == "update") {
        Value out = Value::array();
        for (const qs::Key& k : st.update_pass(passes.at(arg).pass)) out.push_back(key_json(k));
        return dumps(out);
    }
    if (kind == "merge") {
        const Built& b = passes.at(arg);
        const auto [S, W] = foreign(b.pass, b.spec->segment, extra);
        const std::string source = (extra == "nan" || extra == "negative") ? "rx2" : extra;
        return dumps(Value::boolean(st.merge(key_a(), S, W, source)));
    }
    if (kind == "detach") {
        const std::string& uid = passes.at(arg).pass.uid;
        const auto key = st.find_member(uid);
        const qs::Accumulator acc = st.detach(key.value(), uid);
        return dumps(Value::array({key_json(key), strings(acc.uids())}));
    }
    if (kind == "add") {
        st.add_pass(passes.at(arg).pass);
        return dumps(strings(st.pass_uids()));
    }
    if (kind == "prov_add") {
        std::optional<std::string> cluster;
        if (!extra.empty()) cluster = "c_" + passes.at(extra).pass.uid;
        return dumps(Value::string(st.prov_add(passes.at(arg).pass.uid, cluster)));
    }
    if (kind == "prov_remove") {
        st.prov_remove(passes.at(arg).pass.uid);
        return dumps(Value::array(st.prov_clusters()));
    }
    if (kind == "log") {
        Value ev = Value::object();
        ev.set("event", Value::string("note"));
        ev.set("v", Value::number(0.1));
        ev.set("n", Value::array({Value::integer(1), Value::number(2.5), Value::null()}));
        ev.set("s", Value::string("caf\xc3\xa9"));
        st.log(ev);
        return dumps(Value::null());
    }
    if (kind == "due") {
        Value out = Value::array();
        for (const qs::Due& d : st.due(std::stoll(arg))) {
            Value o = Value::object();
            o.set("key", key_json(d.key));
            o.set("segment", Value::integer(d.segment));
            o.set("f_hz", Value::number(d.f_hz));
            o.set("q", Value::integer(d.q));
            o.set("waveform", Value::integer(d.waveform));
            o.set("frame", d.frame);
            out.push_back(std::move(o));
        }
        return dumps(out);
    }
    if (kind == "resolve") {
        Value out = Value::array();
        for (const auto& [cluster, key] : qa::resolve_provisional(st))
            out.push_back(Value::array({cluster, key_json(key)}));
        return dumps(out);
    }
    if (kind == "expire") {
        const qs::Expired r = st.expire(std::stod(arg));
        Value acc = Value::array();
        for (const qs::Key& k : r.acc) acc.push_back(key_json(k));
        Value o = Value::object();
        o.set("acc", std::move(acc));
        o.set("prov", Value::array(r.prov));
        o.set("passes", strings(r.passes));
        return dumps(o);
    }
    throw std::invalid_argument("unknown op " + kind);
}

}  // namespace store_session
