// Minimal reader for the .npy files in native/tests/golden.
//
// Deliberately supports only the subset tools/gen_golden_vectors.py
// emits: version 1.0 or 2.0, C-contiguous, little-endian, and one of
// '<f8' / '<c16' / '<i8'. Anything else is a hard error rather than a
// best-effort parse, because a golden vector silently read as the wrong
// dtype would produce a confidently wrong "parity confirmed".
//
// This lives under core/testing/ rather than in tests/ because the
// pybind11 module also loads vectors when the Python suite asks the C++
// side to prove itself against them.

#pragma once

#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sstvae::testing {

struct NpyFile {
    std::string dtype;            // as written, e.g. "<f8"
    std::vector<std::size_t> shape;
    std::vector<char> raw;        // element bytes, C order

    std::size_t size() const {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        return n;
    }

    // Number of rows when the array is treated as 2-D. A 1-D array is
    // one row, which is what every caller here wants.
    std::size_t rows() const { return shape.empty() ? 1 : shape[0]; }
    std::size_t cols() const {
        std::size_t n = 1;
        for (std::size_t i = 1; i < shape.size(); ++i) n *= shape[i];
        return n;
    }
};

namespace detail {

inline std::string header_value(const std::string& header, const std::string& key) {
    const std::size_t k = header.find("'" + key + "'");
    if (k == std::string::npos)
        throw std::runtime_error("npy header has no '" + key + "'");
    std::size_t colon = header.find(':', k);
    if (colon == std::string::npos)
        throw std::runtime_error("npy header is malformed near '" + key + "'");
    ++colon;
    while (colon < header.size() && std::isspace(static_cast<unsigned char>(header[colon])))
        ++colon;
    // Values are either quoted, a bare token, or a parenthesised tuple.
    if (header[colon] == '\'') {
        const std::size_t end = header.find('\'', colon + 1);
        return header.substr(colon + 1, end - colon - 1);
    }
    if (header[colon] == '(') {
        const std::size_t end = header.find(')', colon);
        return header.substr(colon, end - colon + 1);
    }
    std::size_t end = colon;
    while (end < header.size() && header[end] != ',' && header[end] != '}') ++end;
    std::string value = header.substr(colon, end - colon);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.pop_back();
    return value;
}

inline std::size_t itemsize(const std::string& dtype) {
    if (dtype == "<f8" || dtype == "<i8") return 8;
    if (dtype == "<c16") return 16;
    // uint16 is not emitted by the golden generator, but it is the dtype
    // of sstvae/modem/interleaver_perms.npy -- the frozen interleaver,
    // which the framing test reads directly rather than through a copy.
    if (dtype == "<u2") return 2;
    throw std::runtime_error("unsupported npy dtype '" + dtype +
                             "'; supported: <f8, <c16, <i8, <u2");
}

}  // namespace detail

inline NpyFile read_npy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    char magic[6];
    in.read(magic, 6);
    if (in.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error(path + " is not a .npy file");

    std::uint8_t major = 0, minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);

    std::size_t header_len = 0;
    if (major == 1) {
        std::uint16_t len16 = 0;
        in.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (major == 2) {
        std::uint32_t len32 = 0;
        in.read(reinterpret_cast<char*>(&len32), 4);
        header_len = len32;
    } else {
        throw std::runtime_error(path + ": unsupported .npy version");
    }

    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));

    NpyFile out;
    out.dtype = detail::header_value(header, "descr");
    if (detail::header_value(header, "fortran_order") != "False")
        throw std::runtime_error(path + ": Fortran-order arrays are not supported");

    const std::string shape = detail::header_value(header, "shape");
    for (std::size_t i = 0; i < shape.size();) {
        if (std::isdigit(static_cast<unsigned char>(shape[i]))) {
            std::size_t j = i;
            while (j < shape.size() && std::isdigit(static_cast<unsigned char>(shape[j])))
                ++j;
            out.shape.push_back(std::stoull(shape.substr(i, j - i)));
            i = j;
        } else {
            ++i;
        }
    }

    const std::size_t bytes = out.size() * detail::itemsize(out.dtype);
    out.raw.resize(bytes);
    in.read(out.raw.data(), static_cast<std::streamsize>(bytes));
    if (static_cast<std::size_t>(in.gcount()) != bytes)
        throw std::runtime_error(path + ": truncated (expected " +
                                 std::to_string(bytes) + " bytes of data)");
    return out;
}

// Typed accessors. Each checks the dtype rather than reinterpreting
// whatever is there, so a mismatched vector fails loudly at load.

inline std::vector<double> load_f8(const std::string& path) {
    NpyFile f = read_npy(path);
    if (f.dtype != "<f8") throw std::runtime_error(path + ": expected <f8, got " + f.dtype);
    std::vector<double> out(f.size());
    std::memcpy(out.data(), f.raw.data(), f.raw.size());
    return out;
}

inline std::vector<std::int64_t> load_i8(const std::string& path) {
    NpyFile f = read_npy(path);
    if (f.dtype != "<i8") throw std::runtime_error(path + ": expected <i8, got " + f.dtype);
    std::vector<std::int64_t> out(f.size());
    std::memcpy(out.data(), f.raw.data(), f.raw.size());
    return out;
}

inline std::vector<std::uint16_t> load_u2(const std::string& path) {
    NpyFile f = read_npy(path);
    if (f.dtype != "<u2") throw std::runtime_error(path + ": expected <u2, got " + f.dtype);
    std::vector<std::uint16_t> out(f.size());
    std::memcpy(out.data(), f.raw.data(), f.raw.size());
    return out;
}

inline std::vector<std::complex<double>> load_c16(const std::string& path) {
    NpyFile f = read_npy(path);
    if (f.dtype != "<c16")
        throw std::runtime_error(path + ": expected <c16, got " + f.dtype);
    std::vector<std::complex<double>> out(f.size());
    std::memcpy(out.data(), f.raw.data(), f.raw.size());
    return out;
}

}  // namespace sstvae::testing
