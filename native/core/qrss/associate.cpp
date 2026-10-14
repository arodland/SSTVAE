// Port of sstvae/qrss/associate.py; see associate.hpp.

#include "qrss/associate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "qrss/config.hpp"
#include "qrss/npsum.hpp"

namespace sstvae::qrss::associate {

using pyjson::Value;
using store::Key;
using store::Store;

namespace {

std::string shape1(std::size_t n) { return "(" + std::to_string(n) + ",)"; }

[[noreturn]] void broadcast_error(std::size_t a, std::size_t b) {
    throw std::invalid_argument("operands could not be broadcast together with shapes " + shape1(a) + " " +
                                shape1(b) + " ");
}

std::vector<double> as_double(std::span<const float> x) { return std::vector<double>(x.begin(), x.end()); }

bool in(const std::vector<std::string>& v, std::string_view s) { return std::find(v.begin(), v.end(), s) != v.end(); }

std::vector<std::string> member_uids(const Value& c) {
    std::vector<std::string> out;
    for (const Value& u : c.at("members").items()) out.push_back(u.as_string("a cluster member"));
    return out;
}

}  // namespace

// --- the statistic ------------------------------------------------------------------------

Corr corr_stat(std::span<const double> z, std::span<const double> w, std::span<const double> m,
               std::span<const double> W) {
    const std::size_t n = std::min(z.size(), m.size());
    // z[:n], w[:n], m[:n], W[:n] must broadcast together
    const std::size_t nw = std::min(w.size(), n), nW = std::min(W.size(), n);
    if (nw != nW) broadcast_error(nw, nW);
    if (nw != n) broadcast_error(nw, n);
    std::vector<double> p;   // v * z * m over the kept latents
    p.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double wi = w[i], Wi = W[i], zi = z[i], mi = m[i];
        if (!(wi > 0 && Wi > 0 && std::isfinite(zi) && std::isfinite(mi) && std::isfinite(wi) && std::isfinite(Wi)))
            continue;
        const double v = wi * Wi / (1 + wi + Wi);
        p.push_back(v * zi * mi);
    }
    if (p.empty()) return {};
    std::vector<double> p2(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) p2[i] = p[i] * p[i];
    const double num = npsum::sum(std::span<const double>(p));
    const double den = npsum::sum(std::span<const double>(p2));
    Corr c;
    c.M = static_cast<std::int64_t>(p.size());
    if (den > 0) {
        c.t = num / std::sqrt(den);
        c.numpy = true;
    }
    return c;
}

double expected_t(std::span<const double> w, std::span<const double> W, double kurtosis) {
    const std::size_t n = std::min(w.size(), W.size());
    std::vector<double> v, q;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double wi = w[i], Wi = W[i];
        if (!(wi > 0 && Wi > 0 && std::isfinite(wi) && std::isfinite(Wi))) continue;
        v.push_back(wi * Wi / (1 + wi + Wi));
    }
    if (v.empty()) return 0.0;
    q.resize(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) q[i] = v[i] + kurtosis * v[i] * v[i];
    return npsum::sum(std::span<const double>(v)) / std::sqrt(npsum::sum(std::span<const double>(q)));
}

bool corr_accept(double t, double t_exp) { return t > T_ACCEPT && t >= RHO_MIN * t_exp; }

// --- references --------------------------------------------------------------------------------

std::pair<std::vector<double>, std::vector<double>> acc_reference(const store::Accumulator& acc, std::int64_t g,
                                                                  std::int64_t n) {
    const auto idx = store::canonical_index(g, n);
    const std::size_t base = static_cast<std::size_t>(g) * store::GROUP_LATENTS;
    std::pair<std::vector<double>, std::vector<double>> out;
    out.first.assign(idx.size(), 0.0);
    out.second.resize(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        const double S = acc.S[base + idx[i]];
        const double W = acc.W[base + idx[i]];
        out.second[i] = W;
        if (W > 0) out.first[i] = S / W;
    }
    return out;
}

std::pair<std::vector<double>, std::vector<double>> cluster_sums(Store& store, const std::vector<std::string>& uids) {
    std::vector<store::ZWPtr> zws;
    for (const std::string& u : uids) zws.push_back(store.pass_zw(u));
    if (zws.empty()) throw std::invalid_argument("max() iterable argument is empty");
    std::size_t n = 0;
    for (const auto& zw : zws) n = std::max(n, zw->z.size());
    std::vector<double> S(n, 0.0), W(n, 0.0);
    for (const auto& zw : zws) {
        for (std::size_t i = 0; i < zw->z.size(); ++i) S[i] += zw->w[i] > 0 ? zw->w[i] * zw->z[i] : 0.0;
        for (std::size_t i = 0; i < zw->w.size(); ++i) W[i] += zw->w[i];
    }
    std::vector<double> m(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        if (W[i] > 0) m[i] = S[i] / W[i];
    return {std::move(m), std::move(W)};
}

std::optional<Best> best_accumulator(Store& store, std::span<const double> z, std::span<const double> w,
                                     const std::vector<Key>& exclude) {
    const auto lk = store.hold();
    std::optional<Best> best;
    for (const Key& key : store.open_keys()) {
        if (std::find(exclude.begin(), exclude.end(), key) != exclude.end()) continue;
        const store::Accumulator acc = store.accumulator(key);
        for (std::int64_t g = 0; g < acc.mode + 1; ++g) {
            if (!acc.has_data(g)) continue;
            const auto [m, W] =
                acc_reference(acc, g, std::min<std::int64_t>(static_cast<std::int64_t>(z.size()), store::SENT));
            const double t = corr_stat(z, w, m, W).t;
            if (corr_accept(t, expected_t(w, W)) && (!best || t > best->t)) best = Best{t, key, g};
        }
    }
    return best;
}

// --- rule 1 validation ---------------------------------------------------------------------------

Validation validate_header_match(Store& store, const Key& key, std::int64_t mode, std::int64_t segment,
                                 std::span<const double> z, std::span<const double> w) {
    const auto lk = store.hold();
    Validation out;
    if (!store.has_accumulator(key)) return out;
    const store::Accumulator acc = store.accumulator(key);
    if (mode != acc.mode) {
        out.ok = false;
        out.details.set("reason", Value::string("mode"));
        out.details.set("acc_mode", Value::integer(acc.mode));
        out.details.set("mode", Value::integer(mode));
        return out;
    }
    if (!acc.has_data(segment) || acc.median_w(segment) <= VALIDATE_MEDIAN_W) return out;
    const auto [m, W] = acc_reference(acc, segment, static_cast<std::int64_t>(z.size()));
    const Corr c = corr_stat(z, w, m, W);
    const double t_exp = expected_t(w, W);
    out.ok = !(c.t < HDR_REJECT_T && t_exp > HDR_REJECT_EXPECT);
    out.details.set("t", Value::number(c.t));
    out.t_numpy = c.numpy;
    out.details.set("M", Value::integer(c.M));
    out.details.set("t_expected", Value::number(t_exp));
    return out;
}

std::vector<std::string> attach_by_header(Store& store, const header::HeaderFields& h,
                                          const std::vector<std::string>& uids, const std::string& method) {
    const auto lk = store.hold();
    const Key key{h.callsign, h.picture_id};
    std::vector<std::string> done;
    for (const std::string& u : uids) {
        const store::ZWPtr zw = store.pass_zw(u);
        const Validation v = validate_header_match(store, key, h.mode, h.segment, zw->z, zw->w);
        if (v.ok) {
            store.attach(key, u, h.segment, method, h.mode, h.codec_id);
            done.push_back(u);
        } else {
            Value ev = Value::object();
            ev.set("event", Value::string("collision"));
            ev.set("uid", Value::string(u));
            ev.set("key", Value::array({Value::string(key.callsign), Value::integer(key.picture_id)}));
            ev.set("rule", Value::string(method));
            for (std::size_t k = 0; k < v.details.keys().size(); ++k) ev.set(v.details.keys()[k], v.details.values()[k]);
            store.log(ev);
            if (!store.prov_cluster_of(u)) store.prov_add(u);
        }
    }
    return done;
}

// --- rule 2 ----------------------------------------------------------------------------------

bool header_agrees(const header::HeaderFields& h, std::optional<std::span<const double>> llr) {
    if (!llr) return true;
    const std::vector<std::uint8_t> c = header::encode(h);
    if (llr->size() != c.size()) return true;
    std::vector<double> x, s;   // the heard LLRs, and x * (1 - 2c)
    for (std::size_t i = 0; i < c.size(); ++i) {
        if ((*llr)[i] == 0) continue;
        x.push_back((*llr)[i]);
        s.push_back((*llr)[i] * (1.0 - 2.0 * static_cast<double>(c[i])));
    }
    const std::size_t n = x.size();
    if (n == 0) return true;
    std::vector<double> xx(n);
    for (std::size_t i = 0; i < n; ++i) xx[i] = x[i] * x[i];
    const double a = (std::sqrt(1.0 + npsum::mean(std::span<const double>(xx))) - 1.0) / 2.0;
    // np.dot (BLAS) in numpy: a pairwise sum here, equal to the last bits
    const double t = npsum::sum(std::span<const double>(s));
    const double nd = static_cast<double>(n);
    const double own = 2.0 * nd * a;
    const double other = (1.0 - 2.0 * HDR_OTHER_FLIP) * 2.0 * nd * a;
    const double mid = (own + other) / 2.0;
    const double floor_ = other + HDR_AGREE_Z * 2.0 * std::sqrt(nd * a);
    return t > (floor_ > mid ? floor_ : mid);
}

std::optional<SoftHeader> soft_header(Store& store, const std::string& uid, std::span<const double> llr) {
    const auto lk = store.hold();
    const store::PassInfo me = store.pass_info(uid);
    if (llr.size() != static_cast<std::size_t>(config::N_HDR_BITS)) return std::nullopt;
    const auto near = [&](const store::PassInfo& o) {
        return o.hdr_llr.size() == static_cast<std::size_t>(config::N_HDR_BITS) &&
               static_cast<double>(std::abs(o.q - me.q)) * static_cast<double>(store::SLOT_S) <=
                   PAIR_DAYS * store::DAY_S;
    };
    std::vector<Value> clusters;
    for (Value& c : store.prov_clusters())
        if (!store::contains(c.at("members"), uid)) clusters.push_back(std::move(c));
    for (const Value& c : clusters) {
        const std::vector<std::string> members = member_uids(c);
        std::vector<store::PassInfo> infos;
        for (const std::string& u : members) infos.push_back(store.pass_info(u));
        if (std::none_of(infos.begin(), infos.end(), near)) continue;
        std::vector<double> acc(config::N_HDR_BITS, 0.0);   // sum(...) starts at 0
        for (const store::PassInfo& o : infos) {
            if (o.hdr_llr.size() != static_cast<std::size_t>(config::N_HDR_BITS)) continue;
            for (std::size_t i = 0; i < acc.size(); ++i) acc[i] += o.hdr_llr[i];
        }
        std::vector<double> total(llr.size());
        for (std::size_t i = 0; i < total.size(); ++i) total[i] = llr[i] + acc[i];
        if (auto h = header::decode(total)) {
            SoftHeader out{*h, {uid}};
            out.uids.insert(out.uids.end(), members.begin(), members.end());
            return out;
        }
    }
    for (const Value& c : clusters) {
        const std::vector<std::string> members = member_uids(c);
        if (members.size() < 2) continue;   // already tried as a cluster sum
        for (const std::string& u : members) {
            const store::PassInfo o = store.pass_info(u);
            if (!near(o) || std::abs(o.f_hz - me.f_hz) > PAIR_HZ) continue;
            std::vector<double> total(llr.size());
            for (std::size_t i = 0; i < total.size(); ++i) total[i] = llr[i] + o.hdr_llr[i];
            if (auto h = header::decode(total)) return SoftHeader{*h, {uid, u}};
        }
    }
    return std::nullopt;
}

// --- the rules ----------------------------------------------------------------------------------

std::string rule(Store& store, const Key& key, std::string_view uid, const std::string& r) {
    return in(store.accumulator(key).uids(), uid) ? r : std::string();
}

Result associate(const types::PassResult& p, Store& store, const SoftHeaderFn& soft) {
    const auto lk = store.hold();
    if (!store.has_pass(p.uid)) store.add_pass(p);
    if (std::optional<Key> held = store.find_member(p.uid)) return {std::move(held), ""};
    const std::vector<double> z = as_double(p.z);
    const std::vector<double> w = as_double(p.w);

    // 1. Header decoded on this pass.
    if (p.header) {
        const header::HeaderFields& h = *p.header;
        if (!attach_by_header(store, h, {p.uid}, "header").empty()) {
            resolve_provisional(store);
            const Key key{h.callsign, h.picture_id};
            return {key, rule(store, key, p.uid, "header")};
        }
        return {std::nullopt, "prov"};
    }

    // 2. Soft-combined header.
    const std::vector<double> llr = as_double(p.hdr_llr);
    const std::optional<SoftHeader> found = soft ? soft(store, p.uid, llr) : soft_header(store, p.uid, llr);
    if (found) {
        const header::HeaderFields& h = found->h;
        // only the passes that carry this header themselves (header_agrees)
        std::vector<std::string> uids;
        for (const std::string& u : found->uids) {
            const store::PassInfo info = store.pass_info(u);
            if (header_agrees(h, std::span<const double>(info.hdr_llr))) uids.push_back(u);
        }
        const std::vector<std::string> done =
            uids.empty() ? std::vector<std::string>{} : attach_by_header(store, h, uids, "soft-header");
        if (!done.empty()) resolve_provisional(store);
        if (in(done, p.uid)) {
            const Key key{h.callsign, h.picture_id};
            return {key, rule(store, key, p.uid, "soft-header")};
        }
        if (in(uids, p.uid)) return {std::nullopt, "prov"};   // rule 1's check refused it (a collision)
        // this pass carries another header: on to rules 3 and 4
    }

    // 3. Latent correlation against the open accumulators.
    if (const std::optional<Best> best = best_accumulator(store, z, w)) {
        store.attach(best->key, p.uid, best->segment, "corr");
        return {best->key, rule(store, best->key, p.uid, "corr")};
    }

    // 4. Provisional: join the best-correlated cluster (scored without the
    // pass itself), or start one.
    std::optional<std::pair<double, Value>> best;
    for (const Value& c : store.prov_clusters()) {
        std::vector<std::string> others;
        for (const std::string& u : member_uids(c))
            if (u != p.uid) others.push_back(u);
        if (others.empty()) continue;
        const auto [m, W] = cluster_sums(store, others);
        const double t = corr_stat(z, w, m, W).t;
        if (corr_accept(t, expected_t(w, W)) && (!best || t > best->first)) best.emplace(t, c.at("cluster"));
    }
    const std::optional<Value> current = store.prov_cluster_of(p.uid);
    if (!best) {
        if (!current) store.prov_add(p.uid);
    } else if (!current || !(current->at("cluster") == best->second)) {
        store.prov_add(p.uid, best->second.as_string("a cluster's name"));
    }
    return {std::nullopt, "prov"};
}

std::vector<std::pair<Value, Key>> resolve_provisional(Store& store) {
    const auto lk = store.hold();
    std::vector<std::pair<Value, Key>> out;
    for (const Value& c : store.prov_clusters()) {
        const std::vector<std::string> members = member_uids(c);
        const auto [m, W] = cluster_sums(store, members);
        const std::optional<Best> best = best_accumulator(store, m, W);
        if (!best) continue;
        for (const std::string& u : members) store.attach(best->key, u, best->segment, "corr");
        out.emplace_back(c.at("cluster"), best->key);
    }
    return out;
}

}  // namespace sstvae::qrss::associate
