#include "qrss/types.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

#include "qrss/pyjson.hpp"
#include "util/sha256.hpp"

namespace sstvae::qrss::types {

namespace {

using pyjson::Value;

// int(round(x)) for a float, as Python: half to even, ValueError for NaN,
// OverflowError for infinity (and here for anything outside int64).
std::int64_t py_round_int(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    const double r = std::nearbyint(x);   // default rounding mode: to nearest, ties to even
    if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0))
        throw std::overflow_error("int(round(" + std::to_string(x) + ")) does not fit in int64");
    return static_cast<std::int64_t>(r);
}

Value opt_string(const std::optional<std::string>& s) { return s ? Value::string(*s) : Value::null(); }

// Python's bytes.decode() (strict UTF-8, surrogates refused), as load reads meta.
void check_utf8(std::string_view s) {
    std::size_t i = 0;
    auto bad = [&] { throw std::invalid_argument("PassResult meta is not valid UTF-8 (byte " + std::to_string(i) + ")"); };
    while (i < s.size()) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        std::size_t n = 0;
        std::uint32_t u = 0;
        if (b0 < 0x80) {
            ++i;
            continue;
        } else if ((b0 & 0xE0) == 0xC0) {
            n = 1;
            u = b0 & 0x1Fu;
        } else if ((b0 & 0xF0) == 0xE0) {
            n = 2;
            u = b0 & 0x0Fu;
        } else if ((b0 & 0xF8) == 0xF0) {
            n = 3;
            u = b0 & 0x07u;
        } else {
            bad();
        }
        for (std::size_t k = 1; k <= n; ++k) {
            if (i + k >= s.size()) bad();
            const auto b = static_cast<unsigned char>(s[i + k]);
            if ((b & 0xC0) != 0x80) bad();
            u = (u << 6) | (b & 0x3Fu);
        }
        const std::uint32_t min_of[4] = {0, 0x80, 0x800, 0x10000};
        if (u < min_of[n] || u > 0x10FFFF || (u >= 0xD800 && u <= 0xDFFF)) bad();
        i += n + 1;
    }
}

template <typename T>
std::vector<T> vector_member(const npz::Npz& z, const char* name, npz::DType want, const char* dtype_name) {
    const npz::Array& a = z.at(name);
    if (a.dtype != want)
        throw std::invalid_argument(std::string("PassResult member ") + name + " is " +
                                    std::string(npz::descr(a.dtype)) + ", not " + dtype_name);
    if (a.ndim() != 1)
        throw std::invalid_argument(std::string("PassResult member ") + name + " has " + std::to_string(a.ndim()) +
                                    " dimensions, not 1");
    return a.values<T>();
}

double num(const Value& d, const char* key) { return d.at(key).as_double(key); }

std::int64_t integer(const Value& d, const char* key) {
    const Value& v = d.at(key);
    if (v.kind() == Value::Kind::Bool) return v.as_bool() ? 1 : 0;   // Python's bool is an int
    return v.as_int(key);
}

bool flag(const Value& d, const char* key) {
    const Value& v = d.at(key);
    if (v.kind() == Value::Kind::Int) return v.as_int() != 0;
    return v.as_bool(key);
}

std::optional<std::string> opt_str(const Value& d, const char* key) {
    const Value& v = d.at(key);
    if (v.is_null()) return std::nullopt;
    return v.as_string(key);
}

}  // namespace

std::string pass_uid(std::int64_t q, double f_hz, std::span<const float> z) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(z.data());
    const std::string hex = util::to_hex(util::sha256(std::span<const std::uint8_t>(p, z.size() * sizeof(float))));
    return std::to_string(q) + "_" + std::to_string(py_round_int(f_hz * 1000)) + "_" + hex.substr(0, 8);
}

std::string PassResult::meta_json() const {
    Value meta = Value::object();
    meta.set("uid", Value::string(uid));
    meta.set("q", Value::integer(q));
    meta.set("frame", Value::string(frame));
    meta.set("waveform", Value::integer(waveform));
    meta.set("f_hz", Value::number(f_hz));
    Value tm = Value::object();
    tm.set("tau0", Value::number(timing.tau0));
    tm.set("ppm", Value::number(timing.ppm));
    tm.set("gamma", Value::number(timing.gamma));
    tm.set("z", Value::number(timing.z));
    meta.set("timing", std::move(tm));
    Value rep = Value::object();
    rep.set("offset_hz", Value::number(report.offset_hz));
    rep.set("drift_hz_per_min", Value::number(report.drift_hz_per_min));
    rep.set("wander_hz_rms", Value::number(report.wander_hz_rms));
    rep.set("doppler_hz", Value::number(report.doppler_hz));
    rep.set("ppm", Value::number(report.ppm));
    rep.set("snr2500_db", Value::number(report.snr2500_db));
    rep.set("z_ref", Value::number(report.z_ref));
    rep.set("kappa", Value::number(report.kappa));
    rep.set("slip_free", Value::boolean(report.slip_free));
    rep.set("suspect", Value::boolean(report.suspect));
    meta.set("report", std::move(rep));
    if (header) {
        Value h = Value::object();
        h.set("callsign", Value::string(header->callsign));
        h.set("grid", opt_string(header->grid));
        h.set("picture_id", Value::integer(header->picture_id));
        h.set("mode", Value::integer(header->mode));
        h.set("segment", Value::integer(header->segment));
        h.set("codec_id", Value::integer(header->codec_id));
        h.set("version", Value::integer(header->version));
        h.set("reserved", Value::integer(header->reserved));
        meta.set("header", std::move(h));
    } else {
        meta.set("header", Value::null());
    }
    if (cw) {
        Value c = Value::object();
        c.set("text", opt_string(cw->text));
        c.set("z_match", Value::number(cw->z_match));
        c.set("keying", Value::string(cw->keying));
        c.set("agrees", cw->agrees ? Value::boolean(*cw->agrees) : Value::null());
        meta.set("cw", std::move(c));
    } else {
        meta.set("cw", Value::null());
    }
    meta.set("ch_fs", Value::integer(ch_fs));
    meta.set("ch_t0_index", Value::number(ch_t0_index));
    meta.set("f_mix_hz", Value::string(f_mix_hz.str()));
    meta.set("estimator", Value::string(estimator));
    meta.set("em_round", Value::integer(em_round));
    return pyjson::dumps(meta);
}

npz::Npz PassResult::to_npz() const {
    npz::Npz out;
    const std::string meta = meta_json();
    std::vector<std::uint8_t> mb(meta.begin(), meta.end());
    out.add("meta", npz::Array::of(mb));
    out.add("z", npz::Array::of(z));
    out.add("w", npz::Array::of(w));
    out.add("hdr_llr", npz::Array::of(hdr_llr));
    out.add("ch", npz::Array::of(ch));
    out.add("psi", npz::Array::of(psi));
    out.add("timing_cov", npz::Array::of(timing.cov.data, std::vector<std::size_t>{timing.cov.rows, timing.cov.cols}));
    return out;
}

std::vector<std::uint8_t> PassResult::bytes() const { return npz::npz_bytes(to_npz()); }

void PassResult::save(const std::string& path) const { npz::save_npz(path, to_npz()); }

PassResult PassResult::load(const std::string& path) { return from_npz(npz::load_npz(path)); }

PassResult PassResult::parse(std::span<const std::uint8_t> file) { return from_npz(npz::parse_npz(file)); }

PassResult PassResult::from_npz(const npz::Npz& z) {
    const npz::Array& m = z.at("meta");
    if (m.dtype != npz::DType::u1) throw std::invalid_argument("PassResult member meta is not uint8");
    const std::string text(m.bytes.begin(), m.bytes.end());
    check_utf8(text);
    const Value meta = pyjson::loads(text);
    PassResult p;
    // the arrays, as Python's load reads them before any meta field
    p.z = vector_member<float>(z, "z", npz::DType::f4, "float32");
    p.w = vector_member<float>(z, "w", npz::DType::f4, "float32");
    p.hdr_llr = vector_member<float>(z, "hdr_llr", npz::DType::f4, "float32");
    p.ch = vector_member<std::complex<float>>(z, "ch", npz::DType::c8, "complex64");
    p.psi = vector_member<float>(z, "psi", npz::DType::f4, "float32");
    const npz::Array& cov = z.at("timing_cov");
    if (cov.dtype != npz::DType::f8)
        throw std::invalid_argument("PassResult member timing_cov is " + std::string(npz::descr(cov.dtype)) +
                                    ", not float64");
    if (cov.size() == 0) {
        p.timing.cov = math::Matrix();
    } else if (cov.ndim() == 2) {
        p.timing.cov = math::Matrix(cov.shape[0], cov.shape[1], cov.values<double>());
    } else {
        throw std::invalid_argument("PassResult member timing_cov has " + std::to_string(cov.ndim()) +
                                    " dimensions, not 2");
    }
    // meta["header"] first, then the rest in the constructor call's order, as
    // Python's load reads them (so a file missing several keys names the same one)
    const Value& h = meta.at("header");
    if (!h.is_null()) {
        header::HeaderFields f;
        f.callsign = h.at("callsign").as_string("callsign");
        f.grid = opt_str(h, "grid");
        f.picture_id = integer(h, "picture_id");
        f.mode = integer(h, "mode");
        f.segment = integer(h, "segment");
        f.codec_id = integer(h, "codec_id");
        f.version = integer(h, "version");
        f.reserved = integer(h, "reserved");
        p.header = std::move(f);
    }
    p.uid = meta.at("uid").as_string("uid");
    p.q = integer(meta, "q");
    p.frame = meta.at("frame").as_string("frame");
    p.waveform = integer(meta, "waveform");
    p.f_hz = num(meta, "f_hz");
    const Value& tm = meta.at("timing");
    p.timing.tau0 = num(tm, "tau0");
    p.timing.ppm = num(tm, "ppm");
    p.timing.gamma = num(tm, "gamma");
    p.timing.z = num(tm, "z");
    const Value& rep = meta.at("report");
    p.report.offset_hz = num(rep, "offset_hz");
    p.report.drift_hz_per_min = num(rep, "drift_hz_per_min");
    p.report.wander_hz_rms = num(rep, "wander_hz_rms");
    p.report.doppler_hz = num(rep, "doppler_hz");
    p.report.ppm = num(rep, "ppm");
    p.report.snr2500_db = num(rep, "snr2500_db");
    p.report.z_ref = num(rep, "z_ref");
    p.report.kappa = num(rep, "kappa");
    p.report.slip_free = flag(rep, "slip_free");
    p.report.suspect = flag(rep, "suspect");
    const Value& c = meta.at("cw");
    if (!c.is_null()) {
        CwIdResult r;
        r.text = opt_str(c, "text");
        r.z_match = num(c, "z_match");
        r.keying = c.at("keying").as_string("keying");
        const Value& ag = c.at("agrees");
        if (!ag.is_null()) r.agrees = ag.kind() == Value::Kind::Int ? ag.as_int() != 0 : ag.as_bool("agrees");
        p.cw = std::move(r);
    }
    p.ch_fs = integer(meta, "ch_fs");
    p.ch_t0_index = num(meta, "ch_t0_index");
    p.f_mix_hz = Rational::parse(meta.at("f_mix_hz").as_string("f_mix_hz"));
    p.estimator = meta.at("estimator").as_string("estimator");
    p.em_round = integer(meta, "em_round");

    return p;
}

}  // namespace sstvae::qrss::types
