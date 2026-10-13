#include "qrss/simfile.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "qrss/crc32.hpp"
#include "qrss/npz.hpp"

namespace sstvae::qrss::simfile {

namespace {

[[noreturn]] void bad_zip(const std::string& why) { throw Error("zipfile.BadZipFile", "simfile: " + why); }

bool sig_at(std::span<const std::uint8_t> b, std::size_t at, const char* sig) {
    return at <= b.size() && b.size() - at >= 4 && std::memcmp(b.data() + at, sig, 4) == 0;
}

std::uint64_t le(std::span<const std::uint8_t> b, std::size_t at, int n) {
    if (at > b.size() || b.size() - at < static_cast<std::size_t>(n)) bad_zip("truncated archive");
    std::uint64_t v = 0;
    for (int i = n - 1; i >= 0; --i) v = (v << 8) | b[at + static_cast<std::size_t>(i)];
    return v;
}

constexpr std::size_t SIZE_END = 22, SIZE_CENTRAL = 46, SIZE_LOCAL = 30, MAX_COMMENT = 0xffff;

// The members of an np.savez archive, by name as the central directory
// spells it ("fe.npy"): each one's bytes, a view into the file.
std::unordered_map<std::string, std::span<const std::uint8_t>> members(std::span<const std::uint8_t> f) {
    // The end record, last unless a comment follows it.
    std::size_t end_at = std::string::npos;
    const std::size_t lo = f.size() > MAX_COMMENT + SIZE_END ? f.size() - MAX_COMMENT - SIZE_END : 0;
    for (std::size_t p = f.size() >= SIZE_END ? f.size() - SIZE_END + 1 : 0; p-- > lo;)
        if (sig_at(f, p, "PK\x05\x06")) {
            end_at = p;
            break;
        }
    if (end_at == std::string::npos) bad_zip("File is not a zip file");
    if (end_at >= 20 && sig_at(f, end_at - 20, "PK\x06\x07"))
        bad_zip("a zip64 archive (over 2 GiB or 65535 members) is not read here");
    const std::uint64_t n_entries = le(f, end_at + 10, 2);
    const std::uint64_t cd_size = le(f, end_at + 12, 4);
    const std::uint64_t cd_offset = le(f, end_at + 16, 4);
    if (cd_size > end_at) bad_zip("Bad offset for central directory");
    // zipfile's "concat": bytes before the archive shift every offset.
    const std::uint64_t start_dir = end_at - cd_size;
    if (cd_offset > start_dir) bad_zip("Bad offset for central directory");
    const std::uint64_t concat = start_dir - cd_offset;

    std::unordered_map<std::string, std::span<const std::uint8_t>> out;
    std::size_t p = static_cast<std::size_t>(start_dir);
    std::uint64_t seen = 0;
    while (p < end_at) {
        if (end_at - p < SIZE_CENTRAL || !sig_at(f, p, "PK\x01\x02"))
            bad_zip("Bad magic number for central directory");
        const auto flags = le(f, p + 8, 2), method = le(f, p + 10, 2);
        const auto crc = static_cast<std::uint32_t>(le(f, p + 16, 4));
        std::uint64_t csize = le(f, p + 20, 4), usize = le(f, p + 24, 4);
        const std::size_t nlen = le(f, p + 28, 2), elen = le(f, p + 30, 2), clen = le(f, p + 32, 2);
        std::uint64_t offset = le(f, p + 42, 4);
        const std::size_t name_at = p + SIZE_CENTRAL;
        if (name_at + nlen + elen + clen > end_at) bad_zip("truncated central directory");
        const std::string name(reinterpret_cast<const char*>(f.data() + name_at), nlen);
        // The zip64 extra (np.savez's force_zip64): whichever of the sizes
        // and the offset are 0xFFFFFFFF in the fixed fields, in that order.
        for (std::size_t x = name_at + nlen; x + 4 <= name_at + nlen + elen;) {
            const std::size_t tp = le(f, x, 2), ln = le(f, x + 2, 2), q0 = x + 4, qend = q0 + ln;
            if (qend > name_at + nlen + elen) bad_zip("Corrupt extra field");
            if (tp == 1) {
                std::size_t q = q0;
                auto next = [&] {
                    if (qend - q < 8) bad_zip("Corrupt zip64 extra field");
                    const std::uint64_t v = le(f, q, 8);
                    q += 8;
                    return v;
                };
                if (usize == 0xffffffffu) usize = next();
                if (csize == 0xffffffffu) csize = next();
                if (offset == 0xffffffffu) offset = next();
            }
            x = qend;
        }
        p = name_at + nlen + elen + clen;
        ++seen;
        if (flags & 0x1u) bad_zip("member " + name + " is encrypted");
        if (method != 0) bad_zip("member " + name + " is compressed (np.savez_compressed), which is not read here");
        // The local header: its name must be the directory's, and the data
        // follows its own name and extra field.
        if (offset > f.size() - std::min<std::uint64_t>(f.size(), concat)) bad_zip("Bad local header offset");
        const std::size_t h = static_cast<std::size_t>(offset + concat);
        if (!sig_at(f, h, "PK\x03\x04")) bad_zip("Bad magic number for file header");
        const std::size_t hn = le(f, h + 26, 2), he = le(f, h + 28, 2);
        if (h + SIZE_LOCAL + hn > f.size() ||
            std::memcmp(f.data() + h + SIZE_LOCAL, name.data(), std::min(hn, name.size())) != 0 || hn != nlen)
            bad_zip("File name in directory and header differ");
        const std::uint64_t data_at = h + SIZE_LOCAL + hn + he;
        const std::uint64_t size = std::min(csize, usize);
        if (data_at > f.size() || f.size() - data_at < size) bad_zip("member " + name + " is truncated");
        const std::span<const std::uint8_t> data = f.subspan(static_cast<std::size_t>(data_at),
                                                             static_cast<std::size_t>(size));
        if (crc32(data) != crc) bad_zip("Bad CRC-32 for file " + name);
        out[name] = data;   // a repeated name: zipfile keeps the last
    }
    if (seen != n_entries) bad_zip("the end record's member count disagrees with the central directory");
    return out;
}

// d[key] as NpzFile.__getitem__ finds it ("<key>.npy"), or the KeyError.
std::span<const std::uint8_t> member(const std::unordered_map<std::string, std::span<const std::uint8_t>>& z,
                                     const std::string& key) {
    const auto it = z.find(key + ".npy");
    if (it == z.end()) throw Error("KeyError", "'" + key + " is not a file in the archive'");
    return it->second;
}

npz::Array array(std::span<const std::uint8_t> m) {
    try {
        return npz::parse_npy(m);
    } catch (const std::invalid_argument& e) {
        throw Error("ValueError", e.what());
    }
}

// int(a) / float(a) of a 0-d array, with Python's classes for a refusal.
std::int64_t to_int(const npz::Array& a) {
    try {
        return a.to_int64();
    } catch (const std::overflow_error& e) {
        throw Error("OverflowError", e.what());
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        if (what.rfind("npz: only 0-dimensional arrays", 0) == 0)
            throw Error("TypeError", "only 0-dimensional arrays can be converted to Python scalars");
        if (what.rfind("npz: a complex array", 0) == 0)
            throw Error("TypeError",
                        "int() argument must be a string, a bytes-like object or a real number, not 'complex'");
        throw Error("ValueError", what);
    }
}

double to_float(const npz::Array& a) {
    if (a.ndim() != 0) throw Error("TypeError", "only 0-dimensional arrays can be converted to Python scalars");
    if (a.dtype == npz::DType::c8 || a.dtype == npz::DType::c16)
        throw Error("TypeError", "float() argument must be a string or a real number, not 'complex'");
    return a.to_double()[0];
}

// str(d["frame"]) for a 0-d '<U<n>' array: UTF-32LE code points, trailing
// NULs dropped (numpy's str of a fixed-width string), as UTF-8. The header
// is read by npz::parse_npy with the descr rewritten to '<u4', which checks
// the dict exactly as for any other member; the data are then read here.
[[noreturn]] void refuse_npy(std::span<const std::uint8_t> m) {
    array(m);   // parse_npy's own refusal of a malformed .npy
    throw Error("ValueError", "simfile: the frame member is not a str array");
}

std::string str_member(std::span<const std::uint8_t> m) {
    if (m.size() < 8 || std::memcmp(m.data(), "\x93NUMPY", 6) != 0) refuse_npy(m);
    const unsigned major = m[6];
    if ((major != 1 && major != 2 && major != 3) || m[7] != 0) refuse_npy(m);
    const std::size_t len_bytes = major == 1 ? 2 : 4;
    if (m.size() < 8 + len_bytes) refuse_npy(m);
    std::size_t hlen = 0;
    for (std::size_t i = len_bytes; i-- > 0;) hlen = (hlen << 8) | m[8 + i];
    const std::size_t data_at = 8 + len_bytes + hlen;
    if (data_at > m.size()) refuse_npy(m);
    std::string header(reinterpret_cast<const char*>(m.data() + 8 + len_bytes), hlen);
    const std::size_t d = header.find("'<U");
    std::size_t e = d == std::string::npos ? d : d + 3;
    while (e != std::string::npos && e < header.size() && header[e] >= '0' && header[e] <= '9') ++e;
    if (d == std::string::npos || e == d + 3 || e >= header.size() || header[e] != '\'' ||
        header.find("'<U", e) != std::string::npos)
        throw Error("ValueError", "simfile: the frame member is not a str array");
    const std::size_t chars = std::stoul(header.substr(d + 3, e - d - 3));
    header.replace(d, e + 1 - d, "'<u4'");
    // The rewritten .npy (version 1.0 or 2.0 framing for the new length),
    // then parse_npy: the header's keys, shape and fortran_order checked.
    std::vector<std::uint8_t> npy = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (header.size() <= 0xffff) {
        npy.insert(npy.end(), {1, 0, static_cast<std::uint8_t>(header.size()),
                               static_cast<std::uint8_t>(header.size() >> 8)});
    } else {
        npy.insert(npy.end(), {2, 0});
        for (int i = 0; i < 4; ++i) npy.push_back(static_cast<std::uint8_t>(header.size() >> (8 * i)));
    }
    npy.insert(npy.end(), header.begin(), header.end());
    npy.insert(npy.end(), m.begin() + static_cast<std::ptrdiff_t>(data_at), m.end());
    const npz::Array shape = array(npy);
    if (shape.ndim() != 0) throw Error("ValueError", "simfile: the frame member is not a 0-d str array");
    if (m.size() - data_at < 4 * chars)
        throw Error("ValueError", "npy: EOF reading array data, expected " + std::to_string(4 * chars) +
                                      " bytes got " + std::to_string(m.size() - data_at));
    std::vector<std::uint32_t> cps(chars);
    for (std::size_t i = 0; i < chars; ++i) cps[i] = static_cast<std::uint32_t>(le(m, data_at + 4 * i, 4));
    while (!cps.empty() && cps.back() == 0) cps.pop_back();
    std::string out;
    for (const std::uint32_t c : cps) {
        if (c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff))
            throw Error("ValueError", "simfile: the frame member holds a code point that is not a character");
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            out.push_back(static_cast<char>(0xc0 | (c >> 6)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3f)));
        } else if (c < 0x10000) {
            out.push_back(static_cast<char>(0xe0 | (c >> 12)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (c >> 18)));
            out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3f)));
        }
    }
    return out;
}

}  // namespace

Sim parse(std::span<const std::uint8_t> file) {
    // np.load's first look: six bytes decide zip, .npy or pickle.
    if (file.empty()) throw Error("EOFError", "No data left in file");
    if (!sig_at(file, 0, "PK\x03\x04") && !sig_at(file, 0, "PK\x05\x06")) {
        if (file.size() >= 6 && std::memcmp(file.data(), "\x93NUMPY", 6) == 0)
            throw Error("TypeError", "'numpy.ndarray' object does not support the context manager protocol");
        throw Error("ValueError",
                    "This file contains pickled (object) data. If you trust the file you can load it unsafely "
                    "using the `allow_pickle=` keyword argument or `pickle.load()`.");
    }
    const auto z = members(file);
    const auto has = [&](const std::string& key) { return z.count(key + ".npy") != 0; };

    // load_npz's reads, in its order: the truth (read and checked, then
    // dropped), then what the receiver takes.
    for (const char* k : {"truth_u", "truth_f_hz", "truth_t_pos_s", "truth_impulse_mask"}) array(member(z, k));
    to_float(array(member(z, "truth_noise_var_fe")));
    if (has("truth_agc_gain")) array(member(z, "truth_agc_gain"));
    to_float(array(member(z, "truth_signal_power")));
    to_float(array(member(z, "truth_time_scale")));
    if (has("truth_u_data")) array(member(z, "truth_u_data"));

    Sim sim;
    const npz::Array fe = array(member(z, "fe"));
    if (fe.dtype != npz::DType::c8 || fe.ndim() != 1)
        throw Error("ValueError", "simfile: fe is " + std::string(npz::descr(fe.dtype)) + " of " +
                                      std::to_string(fe.ndim()) + " dimensions, not complex64 (n,) as " +
                                      "qrss_simulate.py writes it");
    sim.fe = fe.values<std::complex<float>>();
    sim.fs = to_int(array(member(z, "fs")));
    sim.t0_index = to_float(array(member(z, "t0_index")));
    sim.q = to_int(array(member(z, "q")));
    sim.frame = str_member(member(z, "frame"));
    return sim;
}

Sim load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    const std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) throw std::runtime_error("cannot read " + path);
    return parse(data);
}

}  // namespace sstvae::qrss::simfile
