// qrss/decode.hpp -- qrss_decode.py's helpers -- against the script itself,
// through the pins gen_decode_pins.py wrote (decode_pins.hpp): str.upper()
// of callsigns, f"{v:08x}" of ints of any size, parse_key and its refusals,
// acc_lines of accumulators built in memory (every JSON type a member's
// field can hold, and the exceptions the script ends with), and the --pass
// line. Also planes_from_pass's placement and refusals.
//
// The whole subcommand, on stores written by either implementation, is
// tests/test_qrss_native_decode.py's (the script and sstvae-qrss decode
// side by side).

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "check.hpp"
#include "decode_pins.hpp"
#include "qrss/decode.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"

namespace check = sstvae::check;
namespace dq = sstvae::qrss::decode;
namespace store = sstvae::qrss::store;
namespace pyjson = sstvae::qrss::pyjson;
namespace types = sstvae::qrss::types;
namespace pins = decode_pins;

namespace {

std::vector<std::string> split(std::string_view s, char sep) {
    std::vector<std::string> out;
    if (s.empty()) return out;
    std::size_t i = 0;
    for (;;) {
        const std::size_t j = s.find(sep, i);
        out.emplace_back(s.substr(i, j == std::string_view::npos ? std::string_view::npos : j - i));
        if (j == std::string_view::npos) return out;
        i = j + 1;
    }
}

std::string join(const std::vector<std::string>& lines) {
    std::string out;
    for (std::size_t k = 0; k < lines.size(); ++k) out += (k ? "\n" : "") + lines[k];
    return out;
}

void test_upper() {
    for (const auto& u : pins::UPPERS)
        check::equal(dq::py_upper(u.in), std::string(u.out), std::string("upper of ") + u.in);
}

void test_hex() {
    for (const auto& h : pins::HEXES) {
        check::equal(dq::py_hex(h.decimal, h.width), std::string(h.out), std::string("hex of ") + h.decimal);
        errno = 0;
        char* end = nullptr;
        const long long v = std::strtoll(h.decimal, &end, 10);
        if (errno == 0 && *end == '\0')
            check::equal(dq::py_hex(static_cast<std::int64_t>(v), h.width), std::string(h.out),
                         std::string("hex of int64 ") + h.decimal);
    }
}

void test_parse_key() {
    for (const auto& k : pins::KEYS) {
        const std::string label = std::string("parse_key ") + k.in;
        try {
            const dq::PictureKey key = dq::parse_key(k.in);
            check::is_true(k.ok, label + " parsed");
            check::equal(key.callsign, std::string(k.callsign), label + " callsign");
            check::equal(key.picture_id.decimal, std::string(k.decimal), label + " picture ID");
        } catch (const dq::Refusal& e) {
            check::is_true(!k.ok, label + " refused");
            check::equal(std::string(e.what()), std::string(k.refusal), label + " refusal");
        }
    }
}

store::Accumulator accumulator(const pins::Acc& a) {
    store::Accumulator acc = store::Accumulator::empty(store::Key{a.callsign, a.picture_id}, 0, a.codec_id, 0.0);
    acc.mode = a.mode;
    acc.members = pyjson::loads(a.members).items();
    acc.foreign = pyjson::loads(a.foreign).items();
    for (const std::string& run : split(a.runs, ';')) {
        const std::vector<std::string> f = split(run, ',');
        const std::size_t base = std::stoul(f[0]) * store::GROUP_LATENTS + std::stoul(f[1]);
        const auto value = static_cast<float>(std::strtod(f[3].c_str(), nullptr));
        for (std::size_t k = 0; k < std::stoul(f[2]); ++k) acc.W[base + k] = value;
    }
    return acc;
}

void test_acc_lines() {
    int n = 0;
    for (const auto& a : pins::ACCS) {
        const std::string label = "acc_lines case " + std::to_string(n++);
        const store::Accumulator acc = accumulator(a);
        std::string got;
        bool error = true;
        try {
            got = join(dq::acc_lines(acc));
            error = false;
        } catch (const dq::PyException& e) {
            got = e.cls() + ": " + e.what();
        } catch (const store::KeyError& e) {
            got = e.what();   // "KeyError: 'uid'"
        } catch (const std::exception& e) {
            got = std::string("unexpected: ") + e.what();
        }
        check::equal(error, a.error, label + " raised");
        check::equal(got, std::string(a.out), label);
    }
}

types::PassResult pass_of(const pins::Pass& p) {
    types::PassResult r;
    r.uid = p.uid;
    r.waveform = p.waveform;
    r.frame = p.frame;
    for (const std::string& run : split(p.runs, ';')) {
        const std::vector<std::string> f = split(run, ',');
        const auto value = static_cast<float>(std::strtod(f[1].c_str(), nullptr));
        r.w.insert(r.w.end(), std::stoul(f[0]), value);
    }
    r.z.assign(r.w.size(), 0.0f);
    return r;
}

void test_pass_line() {
    for (const auto& p : pins::PASSES)
        check::equal(dq::pass_line(pass_of(p)), std::string(p.line), std::string("pass line ") + p.runs);
}

void test_planes_from_pass() {
    types::PassResult p;
    p.uid = "u";
    const std::size_t n = 1000;
    for (std::size_t k = 0; k < n; ++k) {
        p.z.push_back(static_cast<float>(k) / 7.0f - 50.0f);
        p.w.push_back(0.25f + static_cast<float>(k % 5));
    }
    bool refused = false;
    try {
        dq::planes_from_pass(p);
    } catch (const dq::Refusal& e) {
        refused = std::string(e.what()) ==
                  "the pass has no decoded header: its latents cannot be placed (store it and decode the "
                  "accumulator instead)";
    }
    check::is_true(refused, "no header: refused");

    p.header.emplace();
    p.header->segment = 1;
    p.header->mode = 2;
    const dq::Planes planes = dq::planes_from_pass(p);
    check::equal(planes.mode, std::int64_t{2}, "mode from the header");
    check::equal(planes.S.size(), store::PLANE, "S size");
    check::equal(planes.W.size(), store::PLANE, "W size");
    const std::span<const std::uint16_t> idx = store::canonical_index(1, static_cast<std::int64_t>(n));
    std::vector<char> placed(store::PLANE, 0);
    bool same = true;
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t at = store::GROUP_LATENTS + static_cast<std::size_t>(idx[k]);
        placed[at] = 1;
        same = same && planes.W[at] == static_cast<double>(p.w[k]) &&
               planes.S[at] == static_cast<double>(p.w[k]) * static_cast<double>(p.z[k]);
    }
    check::is_true(same, "S = w z and W = w at the segment's canonical offsets");
    bool rest = true;
    for (std::size_t k = 0; k < store::PLANE; ++k)
        if (!placed[k]) rest = rest && planes.S[k] == 0.0 && planes.W[k] == 0.0;
    check::is_true(rest, "zero elsewhere");

    p.w.pop_back();
    bool mismatch = false;
    try {
        dq::planes_from_pass(p);
    } catch (const std::invalid_argument&) {
        mismatch = true;
    }
    check::is_true(mismatch, "z and w of different lengths: numpy's broadcast refusal");
}

void test_acc_mean_w_db() {
    store::Accumulator acc = store::Accumulator::empty(store::Key{"K1", 1}, 2, 0, 0.0);
    check::is_true(std::isinf(dq::acc_mean_w_db(acc)) && dq::acc_mean_w_db(acc) < 0, "nothing heard: -inf");
    acc.W[0] = 1.0f;
    acc.W[store::GROUP_LATENTS + 7] = 9.0f;
    acc.W[5] = -3.0f;   // not heard (W > 0 only)
    check::is_true(std::abs(dq::acc_mean_w_db(acc) - 10.0 * std::log10(5.0)) < 1e-12, "mean of the heard W");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    try {
        test_upper();
        test_hex();
        test_parse_key();
        test_acc_lines();
        test_pass_line();
        test_planes_from_pass();
        test_acc_mean_w_db();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_decode");
}
