// numpy's .npy and .npz files: np.save / np.savez / np.load, for the
// arrays sstvae/qrss stores (.qrsp pictures, PassResult passes, the store's
// accumulators) and nothing else.
//
// Supported: little-endian <f2 <f4 <f8 <c8 <c16 |i1 |u1 <i2 <u2 <i4 <u4 <i8
// <u8 and |b1, any shape (0-d included), C order when written; .npy format
// versions 1.0, 2.0 and 3.0 read (Fortran order is reordered to C on read),
// 1.0 written unless the header needs 2.0, exactly as numpy chooses.
//
// npz is a ZIP_STORED archive of "<name>.npy" members. What numpy writes,
// and what this writes, byte for byte (tests/test_qrss_native_infra.py
// compares the two files whole):
//
// * every member is opened with force_zip64, so its *local* header carries
//   0xFFFFFFFF sizes and a zip64 extra field with the real ones; readers
//   must take sizes from the central directory, which has them plainly
//   (and in a zip64 extra only past 2 GiB). That is why `parse_npz` reads
//   the central directory and never trusts a local header's sizes;
// * flags 0, method 0 (stored), versions 45, made by Unix (3, as Python on
//   Linux and macOS), external attributes 0o600 << 16, no comments;
// * CRC-32 of each member (zlib.crc32), which np.load checks and so does
//   `parse_npz`;
// * the timestamp 1980-01-01 00:00:00 (DOS time 0, date 0x21): np.savez
//   opens each member by name, and zipfile.ZipInfo(name) defaults to that,
//   so numpy's .npz files are reproducible and so are these. DosTime
//   exists to match an archive some other writer stamped.
//
// The one difference: Python on Windows records "made by" 0 (MS-DOS); this
// always writes 3. Both load the same everywhere.
//
// Not supported, and refused with std::invalid_argument rather than read
// wrongly: np.savez_compressed (deflate), big-endian dtypes, structured or
// object dtypes, dtype spellings other than a kind-and-size code (np.dtype
// also reads "d", "float64", a subarray "2f8"), encrypted or multi-disk
// archives, and members that are not .npy arrays (np.load hands those
// back as bytes). As np.load, a file that does not begin with a zip local
// header or end record is refused, though zipfile alone would read past
// prepended data. parse_npz reads every member np.load would, at once, so
// one that np.load would refuse only when it is read refuses the archive.
// Malformed input is std::invalid_argument; I/O failure is
// std::runtime_error.

#ifndef SSTVAE_QRSS_NPZ_HPP
#define SSTVAE_QRSS_NPZ_HPP

#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qrss/half.hpp"

namespace sstvae::qrss::npz {

static_assert(std::endian::native == std::endian::little,
              "npz stores little-endian data and copies it as it lies in memory");

enum class DType : std::uint8_t { f2, f4, f8, c8, c16, i1, u1, i2, u2, i4, u4, i8, u8, b1 };

// numpy's dtype.str: "<f8", "|u1", ...
std::string_view descr(DType t);
std::size_t itemsize(DType t);
// The dtype a descr names, or nullopt when unsupported: numpy's kind and
// size codes with any byte-order prefix np.dtype reads as little-endian.
std::optional<DType> from_descr(std::string_view d);

template <typename T>
struct DTypeOf;
template <> struct DTypeOf<half::Half> { static constexpr DType value = DType::f2; };
template <> struct DTypeOf<float> { static constexpr DType value = DType::f4; };
template <> struct DTypeOf<double> { static constexpr DType value = DType::f8; };
template <> struct DTypeOf<std::complex<float>> { static constexpr DType value = DType::c8; };
template <> struct DTypeOf<std::complex<double>> { static constexpr DType value = DType::c16; };
template <> struct DTypeOf<std::int8_t> { static constexpr DType value = DType::i1; };
template <> struct DTypeOf<std::uint8_t> { static constexpr DType value = DType::u1; };
template <> struct DTypeOf<std::int16_t> { static constexpr DType value = DType::i2; };
template <> struct DTypeOf<std::uint16_t> { static constexpr DType value = DType::u2; };
template <> struct DTypeOf<std::int32_t> { static constexpr DType value = DType::i4; };
template <> struct DTypeOf<std::uint32_t> { static constexpr DType value = DType::u4; };
template <> struct DTypeOf<std::int64_t> { static constexpr DType value = DType::i8; };
template <> struct DTypeOf<std::uint64_t> { static constexpr DType value = DType::u8; };
// |b1 has no C++ element type of its own (std::vector<bool> is a bitset):
// use Array::of_bools / Array::bools.

template <typename T>
inline constexpr DType dtype_of = DTypeOf<T>::value;

// One array: dtype, shape (empty for 0-d) and the elements' bytes in C order.
struct Array {
    DType dtype = DType::f8;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> bytes;

    std::size_t ndim() const { return shape.size(); }
    // Number of elements: the product of the shape, 1 for a 0-d array.
    std::size_t size() const;

    // From typed values. `shape` defaults to 1-D; otherwise its product
    // must equal v.size() (std::invalid_argument).
    template <typename T>
    static Array of(std::span<const T> v, std::optional<std::vector<std::size_t>> shape = std::nullopt) {
        Array a;
        a.dtype = dtype_of<T>;
        a.set_shape(v.size(), std::move(shape));
        a.bytes.resize(v.size() * sizeof(T));
        if (!v.empty()) std::memcpy(a.bytes.data(), v.data(), a.bytes.size());
        return a;
    }
    template <typename T>
    static Array of(const std::vector<T>& v, std::optional<std::vector<std::size_t>> shape = std::nullopt) {
        return of(std::span<const T>(v), std::move(shape));
    }
    // A 0-d array, as np.int64(v) becomes in np.savez.
    template <typename T>
    static Array scalar(T v) {
        Array a = of(std::span<const T>(&v, 1));
        a.shape.clear();
        return a;
    }
    // |b1 from 0/1 bytes (anything nonzero is stored as 1, as numpy does).
    static Array of_bools(std::span<const std::uint8_t> v, std::optional<std::vector<std::size_t>> shape = std::nullopt);

    // The elements as T; std::invalid_argument unless dtype is dtype_of<T>.
    template <typename T>
    std::vector<T> values() const {
        check_dtype(dtype_of<T>);
        std::vector<T> out(size());
        if (!out.empty()) std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
        return out;
    }
    // |b1 elements as 0/1 bytes.
    std::vector<std::uint8_t> bools() const;
    // The elements as float64, as a.astype(np.float64) for every real
    // dtype (float16 exactly, integers by C++'s conversion, which rounds
    // to nearest like numpy's). Complex dtypes are refused.
    std::vector<double> to_double() const;
    // int(a) for a 0-d array, as Python reads the .qrsp scalars: integers
    // and bools as they are, floats truncated toward zero. Like numpy 2.5,
    // only a 0-d array converts -- int(np.array([5])) is a TypeError there
    // -- so any other shape is std::invalid_argument, as is a complex
    // dtype and a NaN ("cannot convert float NaN to integer"). Infinity
    // ("cannot convert float infinity to integer") and a value outside
    // int64 (a uint64 above 2^63 - 1, a float past +-2^63, both of which
    // Python answers with a wide int) are std::overflow_error.
    std::int64_t to_int64() const;

    void check_dtype(DType want) const;

private:
    void set_shape(std::size_t n, std::optional<std::vector<std::size_t>> shape);
};

// The members of an .npz, in archive order, by name without ".npy".
class Npz {
public:
    // Appends; std::invalid_argument if the name is taken, holds a NUL or
    // is not UTF-8. An empty name is allowed, as np.savez allows it.
    void add(std::string name, Array a);
    bool contains(std::string_view name) const;
    // std::out_of_range "<name> is not a file in the archive" (numpy's
    // KeyError text) when absent.
    const Array& at(std::string_view name) const;
    std::vector<std::string> files() const;
    std::size_t size() const { return members_.size(); }
    const std::vector<std::pair<std::string, Array>>& members() const { return members_; }

private:
    std::vector<std::pair<std::string, Array>> members_;
    std::unordered_map<std::string, std::size_t> index_;   // name -> position in members_
};

// A DOS timestamp as zip stores it.
struct DosTime {
    std::uint16_t time = 0;      // hour << 11 | minute << 5 | second / 2
    std::uint16_t date = 0x21;   // (year - 1980) << 9 | month << 5 | day
};

// np.save to a buffer / np.load of one: a whole .npy file.
std::vector<std::uint8_t> npy_bytes(const Array& a);
Array parse_npy(std::span<const std::uint8_t> file);

// np.savez to a buffer / np.load of one. parse_npz returns what reading
// every key of np.load's NpzFile returns: the member names as zipfile
// decodes them (UTF-8 when flagged, else cp437; cut at a NUL), without
// ".npy", in first-appearance order; a key whose exact name is also a
// member reads that member, and a repeated name reads its last entry.
// Members no key reaches are not read, as np.load never opens them.
std::vector<std::uint8_t> npz_bytes(const Npz& z, DosTime stamp = {});
Npz parse_npz(std::span<const std::uint8_t> file);

// The same, on files. The path is used as given (np.savez would append
// ".npz" to a name without it; picture.save_qrsp avoids that by passing a
// file handle, which this matches).
void save_npy(const std::string& path, const Array& a);
Array load_npy(const std::string& path);
void save_npz(const std::string& path, const Npz& z, DosTime stamp = {});
Npz load_npz(const std::string& path);

}  // namespace sstvae::qrss::npz

#endif  // SSTVAE_QRSS_NPZ_HPP
