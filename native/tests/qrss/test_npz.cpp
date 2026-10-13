// native/core/qrss/npz against numpy's own files. NUMPY_NPZ below is
// np.savez's output, byte for byte (numpy 2.5, CPython 3.13 on Linux;
// zipfile stamps a member opened for writing 1980-01-01 00:00, so the
// file is reproducible):
//
//   np.savez(buf,
//       f2=np.array([1.0, -2.5, 65504.0, 6e-8], np.float16),
//       f4=np.array([[1.5, -0.0], [3.25, 1e30]], np.float32),
//       f8=np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 8,
//       c8=np.array([1 + 2j, -3.5j], np.complex64),
//       c16=np.array([0.1 + 0.2j], np.complex128),
//       i1=np.array([-128, 0, 127], np.int8), u1=np.array([0, 255], np.uint8),
//       i2=np.array([-32768, 1234], np.int16), u2=np.array([0xD1D8, 65535], np.uint16),
//       i4=np.array([-2**31, 2**31 - 1], np.int32), u4=np.array([2**32 - 1], np.uint32),
//       i8=np.int64(-(2**62)), u8=np.array([2**64 - 1, 0], np.uint64),
//       b1=np.array([True, False, True]), empty=np.zeros((0,), np.float32),
//       x1500=np.float64(1500.3))
//
// FORTRAN_NPY is np.save(np.asfortranarray(np.arange(6, dtype=np.int32).reshape(2, 3)))
// and COMPRESSED_NPZ is np.savez_compressed(a=np.arange(3, dtype=np.int16)).
// tests/test_qrss_native_infra.py checks both directions against numpy itself.

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/npz.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
namespace npz = sstvae::qrss::npz;
using npz::Array;
using npz::DType;

static const char* const NUMPY_NPZ[] = {
    "504b03042d000000000000002100e867fdefffffffffffffffff0600140066322e6e7079010010008800000000000000",
    "8800000000000000934e554d5059010076007b276465736372273a20273c6632272c2027666f727472616e5f6f726465",
    "72273a2046616c73652c20277368617065273a2028342c292c207d202020202020202020202020202020202020202020",
    "2020202020202020202020202020202020202020202020202020202020202020202020202020200a003c00c1ff7b0100",
    "504b03042d0000000000000021008dc8de6bffffffffffffffff0600140066342e6e7079010010009000000000000000",
    "9000000000000000934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f726465",
    "72273a2046616c73652c20277368617065273a2028322c2032292c207d20202020202020202020202020202020202020",
    "2020202020202020202020202020202020202020202020202020202020202020202020202020200a0000c03f00000080",
    "00005040caf24971504b03042d00000000000000210093bb8aceffffffffffffffff0600140066382e6e707901001000",
    "40010000000000004001000000000000934e554d5059010076007b276465736372273a20273c6638272c2027666f7274",
    "72616e5f6f72646572273a2046616c73652c20277368617065273a2028322c20332c2034292c207d2020202020202020",
    "20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a",
    "0000000000000000000000000000c03f000000000000d03f000000000000d83f000000000000e03f000000000000e43f",
    "000000000000e83f000000000000ec3f000000000000f03f000000000000f23f000000000000f43f000000000000f63f",
    "000000000000f83f000000000000fa3f000000000000fc3f000000000000fe3f00000000000000400000000000000140",
    "000000000000024000000000000003400000000000000440000000000000054000000000000006400000000000000740",
    "504b03042d0000000000000021005107abb7ffffffffffffffff0600140063382e6e7079010010009000000000000000",
    "9000000000000000934e554d5059010076007b276465736372273a20273c6338272c2027666f727472616e5f6f726465",
    "72273a2046616c73652c20277368617065273a2028322c292c207d202020202020202020202020202020202020202020",
    "2020202020202020202020202020202020202020202020202020202020202020202020202020200a0000803f00000040",
    "00000080000060c0504b03042d000000000000002100976407f1ffffffffffffffff070014006331362e6e7079010010",
    "0090000000000000009000000000000000934e554d5059010076007b276465736372273a20273c633136272c2027666f",
    "727472616e5f6f72646572273a2046616c73652c20277368617065273a2028312c292c207d2020202020202020202020",
    "202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020",
    "0a9a9999999999b93f9a9999999999c93f504b03042d000000000000002100ecbf65e5ffffffffffffffff0600140069",
    "312e6e70790100100083000000000000008300000000000000934e554d5059010076007b276465736372273a20277c69",
    "31272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028332c292c207d20202020",
    "202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020",
    "20202020202020200a80007f504b03042d00000000000000210082bd7285ffffffffffffffff0600140075312e6e7079",
    "0100100082000000000000008200000000000000934e554d5059010076007b276465736372273a20277c7531272c2027",
    "666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c292c207d202020202020202020",
    "202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020",
    "2020200a00ff504b03042d0000000000000021007e4f59e5ffffffffffffffff0600140069322e6e7079010010008400",
    "0000000000008400000000000000934e554d5059010076007b276465736372273a20273c6932272c2027666f72747261",
    "6e5f6f72646572273a2046616c73652c20277368617065273a2028322c292c207d202020202020202020202020202020",
    "2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0080",
    "d204504b03042d000000000000002100e7543ac6ffffffffffffffff0600140075322e6e707901001000840000000000",
    "00008400000000000000934e554d5059010076007b276465736372273a20273c7532272c2027666f727472616e5f6f72",
    "646572273a2046616c73652c20277368617065273a2028322c292c207d20202020202020202020202020202020202020",
    "20202020202020202020202020202020202020202020202020202020202020202020202020202020200ad8d1ffff504b",
    "03042d0000000000000021004f22ecc9ffffffffffffffff0600140069342e6e70790100100088000000000000008800",
    "000000000000934e554d5059010076007b276465736372273a20273c6934272c2027666f727472616e5f6f7264657227",
    "3a2046616c73652c20277368617065273a2028322c292c207d2020202020202020202020202020202020202020202020",
    "202020202020202020202020202020202020202020202020202020202020202020202020200a00000080ffffff7f504b",
    "03042d00000000000000210034e0ad8effffffffffffffff0600140075342e6e70790100100084000000000000008400",
    "000000000000934e554d5059010076007b276465736372273a20273c7534272c2027666f727472616e5f6f7264657227",
    "3a2046616c73652c20277368617065273a2028312c292c207d2020202020202020202020202020202020202020202020",
    "202020202020202020202020202020202020202020202020202020202020202020202020200affffffff504b03042d00",
    "00000000000021008a7351fdffffffffffffffff0600140069382e6e7079010010008800000000000000880000000000",
    "0000934e554d5059010076007b276465736372273a20273c6938272c2027666f727472616e5f6f72646572273a204661",
    "6c73652c20277368617065273a2028292c207d2020202020202020202020202020202020202020202020202020202020",
    "2020202020202020202020202020202020202020202020202020202020202020200a00000000000000c0504b03042d00",
    "00000000000021007b762e38ffffffffffffffff0600140075382e6e7079010010009000000000000000900000000000",
    "0000934e554d5059010076007b276465736372273a20273c7538272c2027666f727472616e5f6f72646572273a204661",
    "6c73652c20277368617065273a2028322c292c207d202020202020202020202020202020202020202020202020202020",
    "2020202020202020202020202020202020202020202020202020202020202020200affffffffffffffff000000000000",
    "0000504b03042d00000000000000210016938cbbffffffffffffffff0600140062312e6e707901001000830000000000",
    "00008300000000000000934e554d5059010076007b276465736372273a20277c6231272c2027666f727472616e5f6f72",
    "646572273a2046616c73652c20277368617065273a2028332c292c207d20202020202020202020202020202020202020",
    "20202020202020202020202020202020202020202020202020202020202020202020202020202020200a010001504b03",
    "042d000000000000002100cddb69fcffffffffffffffff09001400656d7074792e6e7079010010008000000000000000",
    "8000000000000000934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f726465",
    "72273a2046616c73652c20277368617065273a2028302c292c207d202020202020202020202020202020202020202020",
    "2020202020202020202020202020202020202020202020202020202020202020202020202020200a504b03042d000000",
    "000000002100c0645d88ffffffffffffffff0900140078313530302e6e70790100100088000000000000008800000000",
    "000000934e554d5059010076007b276465736372273a20273c6638272c2027666f727472616e5f6f72646572273a2046",
    "616c73652c20277368617065273a2028292c207d20202020202020202020202020202020202020202020202020202020",
    "202020202020202020202020202020202020202020202020202020202020202020200a3333333333719740504b01022d",
    "032d000000000000002100e867fdef880000008800000006000000000000000000000080010000000066322e6e707950",
    "4b01022d032d0000000000000021008dc8de6b90000000900000000600000000000000000000008001c000000066342e",
    "6e7079504b01022d032d00000000000000210093bb8ace40010000400100000600000000000000000000008001880100",
    "0066382e6e7079504b01022d032d0000000000000021005107abb7900000009000000006000000000000000000000080",
    "010003000063382e6e7079504b01022d032d000000000000002100976407f19000000090000000070000000000000000",
    "0000008001c80300006331362e6e7079504b01022d032d000000000000002100ecbf65e5830000008300000006000000",
    "000000000000000080019104000069312e6e7079504b01022d032d00000000000000210082bd72858200000082000000",
    "06000000000000000000000080014c05000075312e6e7079504b01022d032d0000000000000021007e4f59e584000000",
    "8400000006000000000000000000000080010606000069322e6e7079504b01022d032d000000000000002100e7543ac6",
    "84000000840000000600000000000000000000008001c206000075322e6e7079504b01022d032d000000000000002100",
    "4f22ecc9880000008800000006000000000000000000000080017e07000069342e6e7079504b01022d032d0000000000",
    "0000210034e0ad8e840000008400000006000000000000000000000080013e08000075342e6e7079504b01022d032d00",
    "00000000000021008a7351fd88000000880000000600000000000000000000008001fa08000069382e6e7079504b0102",
    "2d032d0000000000000021007b762e3890000000900000000600000000000000000000008001ba09000075382e6e7079",
    "504b01022d032d00000000000000210016938cbb83000000830000000600000000000000000000008001820a00006231",
    "2e6e7079504b01022d032d000000000000002100cddb69fc800000008000000009000000000000000000000080013d0b",
    "0000656d7074792e6e7079504b01022d032d000000000000002100c0645d888800000088000000090000000000000000",
    "0000008001f80b000078313530302e6e7079504b0506000000001000100047030000bb0c00000000",
};
static const char* const FORTRAN_NPY[] = {
    "934e554d5059010076007b276465736372273a20273c6934272c2027666f727472616e5f6f72646572273a2054727565",
    "2c20277368617065273a2028322c2033292c207d20202020202020202020202020202020202020202020202020202020",
    "202020202020202020202020202020202020202020202020202020202020200a00000000030000000100000004000000",
    "0200000005000000",
};
static const char* const COMPRESSED_NPZ[] = {
    "504b03042d0000000800000021006acff251ffffffffffffffff05001400612e6e70790100100086000000000000004a",
    "000000000000009bec17ea1b10c9c850c650ad9e925a9c5ca46ea5a06e9369a4aea3a09e965f54529498179f5f94920a",
    "12774bcc294e058a17672416a402f91ac63a9a3a0ab50a14002e06064606260600504b01022d032d0000000800000021",
    "006acff2514a00000086000000050000000000000000000000800100000000612e6e7079504b05060000000001000100",
    "33000000810000000000",
};

static std::vector<std::uint8_t> unhex(const char* const* lines, std::size_t n) {
    std::vector<std::uint8_t> out;
    const auto nib = [](char c) -> unsigned {
        return c <= '9' ? static_cast<unsigned>(c - '0') : static_cast<unsigned>(c - 'a' + 10);
    };
    for (std::size_t i = 0; i < n; ++i)
        for (const char* p = lines[i]; p[0] != '\0' && p[1] != '\0'; p += 2)
            out.push_back(static_cast<std::uint8_t>(nib(p[0]) << 4 | nib(p[1])));
    return out;
}
#define UNHEX(a) unhex(a, std::size(a))

static std::string sha(std::span<const std::uint8_t> b) { return util::to_hex(util::sha256(b)); }

template <typename E, typename F>
static void throws(F f, const std::string& prefix, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        const std::string got = e.what();
        if (got.compare(0, prefix.size(), prefix) != 0) check::fail(what, "threw '" + got + "'");
        return;
    }
    check::fail(what, "did not throw");
}

template <typename T>
static void values_are(const Array& a, DType dt, const std::vector<std::size_t>& shape, const std::vector<T>& want,
                       const std::string& what) {
    check::is_true(a.dtype == dt, what + " dtype");
    check::is_true(a.shape == shape, what + " shape");
    const std::vector<T> got = a.values<T>();
    check::equal(got.size(), want.size(), what + " size");
    bool same = got.size() == want.size();
    for (std::size_t i = 0; same && i < got.size(); ++i)
        same = std::memcmp(&got[i], &want[i], sizeof(T)) == 0;   // bitwise, so -0.0 counts
    check::is_true(same, what + " values");
}

static void test_reads_numpy() {
    const auto f = UNHEX(NUMPY_NPZ);
    check::equal(f.size(), std::size_t(4120), "embedded size");
    check::equal(sha(f), std::string("8db1a1b10593e50b1d51fc38e830ea5e3389a11228ebc42df2bb7f3ee33d0691"),
                 "embedded file");
    const npz::Npz z = npz::parse_npz(f);
    const std::vector<std::string> names = {"f2", "f4", "f8", "c8", "c16", "i1", "u1", "i2",
                                            "u2", "i4", "u4", "i8", "u8", "b1", "empty", "x1500"};
    check::is_true(z.files() == names, "member names, in archive order, without .npy");

    check::is_true(z.at("f2").dtype == DType::f2 && z.at("f2").shape == std::vector<std::size_t>{4}, "f2");
    check::is_true(z.at("f2").to_double() == std::vector<double>{1.0, -2.5, 65504.0, 0x1p-24}, "f2 values");
    values_are<float>(z.at("f4"), DType::f4, {2, 2}, {1.5f, -0.0f, 3.25f, 1e30f}, "f4");
    std::vector<double> f8(24);
    for (std::size_t i = 0; i < f8.size(); ++i) f8[i] = static_cast<double>(i) / 8;
    values_are<double>(z.at("f8"), DType::f8, {2, 3, 4}, f8, "f8, 3-D");
    values_are<std::complex<float>>(z.at("c8"), DType::c8, {2}, {{1.0f, 2.0f}, {-0.0f, -3.5f}}, "c8");   // -3.5j is (-0.0 - 3.5j)
    values_are<std::complex<double>>(z.at("c16"), DType::c16, {1}, {{0.1, 0.2}}, "c16");
    values_are<std::int8_t>(z.at("i1"), DType::i1, {3}, {-128, 0, 127}, "i1");
    values_are<std::uint8_t>(z.at("u1"), DType::u1, {2}, {0, 255}, "u1");
    values_are<std::int16_t>(z.at("i2"), DType::i2, {2}, {-32768, 1234}, "i2");
    values_are<std::uint16_t>(z.at("u2"), DType::u2, {2}, {0xD1D8, 65535}, "u2");
    values_are<std::int32_t>(z.at("i4"), DType::i4, {2}, {std::numeric_limits<std::int32_t>::min(), 2147483647},
                             "i4");
    values_are<std::uint32_t>(z.at("u4"), DType::u4, {1}, {4294967295u}, "u4");
    values_are<std::int64_t>(z.at("i8"), DType::i8, {}, {-(std::int64_t(1) << 62)}, "i8, 0-d");
    check::equal(z.at("i8").to_int64(), -(std::int64_t(1) << 62), "int() of a 0-d int64");
    values_are<std::uint64_t>(z.at("u8"), DType::u8, {2}, {std::numeric_limits<std::uint64_t>::max(), 0}, "u8");
    check::is_true(z.at("b1").dtype == DType::b1 && z.at("b1").bools() == std::vector<std::uint8_t>{1, 0, 1}, "b1");
    check::is_true(z.at("empty").dtype == DType::f4 && z.at("empty").shape == std::vector<std::size_t>{0} &&
                       z.at("empty").size() == 0,
                   "an empty array");
    check::is_true(z.at("x1500").ndim() == 0 && z.at("x1500").to_double() == std::vector<double>{1500.3},
                   "a 0-d float64");

    // Written back, it is numpy's file again, byte for byte.
    const auto again = npz::npz_bytes(z);
    check::is_true(again == f, "npz_bytes(parse_npz(f)) == f");
}

static void test_writes_numpy() {
    // np.save's bytes:
    //   np.save(b, np.float64(1500.3))                                   -> 136 bytes
    //   np.save(b, np.arange(24, dtype=np.int16).reshape(2, 3, 4))       -> 176 bytes
    //   np.save(b, np.array([0xD1D8, 1, 2], np.uint16))                  -> 134 bytes
    const auto s = npz::npy_bytes(Array::scalar(1500.3));
    check::equal(s.size(), std::size_t(136), "0-d npy size");
    check::equal(sha(s), std::string("c96a43cb0ec9bc9a10347cecc98c20093d1f11a1e08ed52fa527de7df90bac5b"), "0-d npy");
    std::vector<std::int16_t> cube(24);
    for (std::size_t i = 0; i < cube.size(); ++i) cube[i] = static_cast<std::int16_t>(i);
    const auto c = npz::npy_bytes(Array::of(cube, std::vector<std::size_t>{2, 3, 4}));
    check::equal(c.size(), std::size_t(176), "3-D npy size");
    check::equal(sha(c), std::string("d29a37c68fa19ddf1d0571b1c47ec7059b8257b9c4330c3174dcaf8520405784"), "3-D npy");
    const auto u = npz::npy_bytes(Array::of(std::vector<std::uint16_t>{0xD1D8, 1, 2}));
    check::equal(sha(u), std::string("c5afb0482b453f09736592d6b7383c7d3a65cfc3989fc28d83235ed3e616ac06"), "u2 npy");
}

static void test_fortran_order() {
    const Array a = npz::parse_npy(UNHEX(FORTRAN_NPY));
    values_are<std::int32_t>(a, DType::i4, {2, 3}, {0, 1, 2, 3, 4, 5}, "a Fortran-ordered file reads in C order");
}

static void test_round_trips() {
    npz::Npz z;
    z.add("h", Array::of(std::vector<qrss::half::Half>{{0x3c00}, {0x7e00}, {0x8001}}));
    z.add("f", Array::of(std::vector<float>{1.0f, std::numeric_limits<float>::infinity()}, std::vector<std::size_t>{2, 1}));
    z.add("d", Array::scalar(-0.0));
    z.add("c", Array::of(std::vector<std::complex<double>>{{1, -1}}));
    z.add("z", Array::of(std::vector<std::complex<float>>{{2, 3}}));
    z.add("b", Array::of_bools(std::vector<std::uint8_t>{0, 7, 1}));
    z.add("i8", Array::scalar<std::int64_t>(1234567890123LL));
    z.add("u2", Array::of(std::vector<std::uint16_t>{1, 2, 3, 4, 5, 6}, std::vector<std::size_t>{1, 2, 3}));
    z.add("i1", Array::of(std::vector<std::int8_t>{-1}));
    z.add("u4", Array::of(std::vector<std::uint32_t>{}));
    z.add("u8", Array::scalar<std::uint64_t>(42));
    z.add("i2", Array::of(std::vector<std::int16_t>{-2}));
    z.add("i4", Array::of(std::vector<std::int32_t>{-3}));
    z.add("u1", Array::of(std::vector<std::uint8_t>{9}));
    const auto bytes = npz::npz_bytes(z);
    const npz::Npz back = npz::parse_npz(bytes);
    check::is_true(back.files() == z.files(), "round trip: names");
    bool same = true;
    for (const auto& [name, a] : z.members()) {
        const Array& b = back.at(name);
        same = same && a.dtype == b.dtype && a.shape == b.shape &&
               (a.dtype == DType::b1 ? b.bools() == std::vector<std::uint8_t>{0, 1, 1} : a.bytes == b.bytes);
    }
    check::is_true(same, "round trip: every member, nonzero bools stored as 1");
    check::is_true(std::signbit(back.at("d").to_double()[0]), "-0.0 survives");

    // Through a file, too.
    const auto dir = std::filesystem::temp_directory_path();
    const std::string path = (dir / ("qrss_npz_test_" + std::to_string(std::bit_cast<std::uintptr_t>(&z)) + ".npz")).string();
    npz::save_npz(path, z);
    const npz::Npz from_file = npz::load_npz(path);
    check::is_true(npz::npz_bytes(from_file) == bytes, "save_npz / load_npz");
    std::filesystem::remove(path);
}

static void test_refusals() {
    const auto f = UNHEX(NUMPY_NPZ);
    // The first member's data starts after its local header, name and extra.
    const std::size_t data_at = 30u + (f[26] | f[27] << 8) + (f[28] | f[29] << 8);
    auto bad_crc = f;
    bad_crc[data_at + 130] ^= 0x01;   // inside f2.npy's values, past its 128-byte header
    throws<std::invalid_argument>([&] { npz::parse_npz(bad_crc); }, "npz: bad CRC-32 for file f2.npy", "bad CRC");
    const std::vector<std::uint8_t> cut(f.begin(), f.end() - 1);
    throws<std::invalid_argument>([&] { npz::parse_npz(cut); }, "npz: file is not a zip file", "truncated");
    throws<std::invalid_argument>([] { npz::parse_npz(UNHEX(COMPRESSED_NPZ)); },
                                  "npz: member a.npy is compressed (method 8)", "np.savez_compressed");
    const npz::Npz z = npz::parse_npz(f);
    throws<std::out_of_range>([&] { (void)z.at("nope"); }, "nope is not a file in the archive", "missing member");
    throws<std::invalid_argument>([&] { (void)z.at("f4").values<double>(); }, "npz: array is <f4, not <f8",
                                  "wrong element type");
    auto npy = npz::npy_bytes(Array::scalar(1500.3));
    auto magic = npy;
    magic[1] = 'X';
    throws<std::invalid_argument>([&] { npz::parse_npy(magic); }, "npy: the magic string is not correct", "magic");
    npy.pop_back();
    throws<std::invalid_argument>([&] { npz::parse_npy(npy); }, "npy: EOF reading array data, expected 8 bytes got 7",
                                  "short data");
    npz::Npz dup;
    dup.add("a", Array::scalar(1.0));
    throws<std::invalid_argument>([&] { dup.add("a", Array::scalar(2.0)); }, "npz: duplicate member a", "duplicate");
    throws<std::invalid_argument>([&] { dup.add("\xff", Array::scalar(2.0)); }, "npz: member name is not valid UTF-8",
                                  "a name a Python str could not hold");

    // np.load reads only a file that starts like a zip archive; data
    // prepended to one (which zipfile alone would skip) is refused.
    std::vector<std::uint8_t> junk = {'j', 'u', 'n', 'k'};
    junk.insert(junk.end(), f.begin(), f.end());
    throws<std::invalid_argument>([&] { npz::parse_npz(junk); }, "npz: file does not start like a zip archive",
                                  "prepended data");
    throws<std::invalid_argument>([] { npz::parse_npz(std::vector<std::uint8_t>{}); }, "npz: file is empty",
                                  "empty file");

    // A zip64 end record claiming a central directory of 2^64 - 1 bytes is
    // corrupt (zipfile: "Corrupt zip64 end of central directory record"),
    // never an empty archive:
    //   body + pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 1, 1, 2**64 - 1, 0)
    //        + pack("<4sLQL", b"PK\x06\x07", 0, len(body), 1) + end
    npz::Npz one;
    one.add("a", Array::of(std::vector<std::int64_t>{0, 1, 2, 3}));
    const auto good = npz::npz_bytes(one);
    std::vector<std::uint8_t> z64(good.begin(), good.end() - 22);
    const std::uint64_t body = z64.size();
    auto le = [&](std::uint64_t v, int n) {
        for (int i = 0; i < n; ++i) z64.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    };
    for (const char c : std::string_view("PK\x06\x06")) z64.push_back(static_cast<std::uint8_t>(c));
    for (const auto& [v, n] : {std::pair<std::uint64_t, int>{44, 8}, {45, 2}, {45, 2}, {0, 4}, {0, 4}, {1, 8}, {1, 8},
                               {~std::uint64_t(0), 8}, {0, 8}})
        le(v, n);
    for (const char c : std::string_view("PK\x06\x07")) z64.push_back(static_cast<std::uint8_t>(c));
    le(0, 4);
    le(body, 8);
    le(1, 4);
    z64.insert(z64.end(), good.end() - 22, good.end());
    throws<std::invalid_argument>([&] { npz::parse_npz(z64); }, "npz: corrupt zip64 end of central directory record",
                                  "zip64 record with a 2^64 - 1 byte directory");
}

static void test_names_and_int() {
    // np.savez(f, **{"": a}) writes ".npy", and np.load reads it back as "".
    npz::Npz z;
    z.add("", Array::of(std::vector<std::int64_t>{0, 1}));
    const npz::Npz back = npz::parse_npz(npz::npz_bytes(z));
    check::is_true(back.files() == std::vector<std::string>{""}, "an empty member name");

    // int(a), as numpy 2.5: only a 0-d array, floats truncated.
    check::equal(Array::scalar(-2.7).to_int64(), std::int64_t(-2), "int(np.float64(-2.7))");
    check::equal(Array::scalar(2.9f).to_int64(), std::int64_t(2), "int(np.float32(2.9))");
    check::equal(Array::scalar(qrss::half::Half{0x3e00}).to_int64(), std::int64_t(1), "int(np.float16(1.5))");
    check::equal(Array::of_bools(std::vector<std::uint8_t>{1}, std::vector<std::size_t>{}).to_int64(), std::int64_t(1),
                 "int(np.bool_(True))");
    throws<std::invalid_argument>([] { (void)Array::of(std::vector<std::int64_t>{5}).to_int64(); },
                                  "npz: only 0-dimensional arrays can be converted to an integer, not shape (1,)",
                                  "int(np.array([5])) is a TypeError");
    throws<std::invalid_argument>([] { (void)Array::scalar(std::nan("")).to_int64(); },
                                  "cannot convert float NaN to integer", "int(nan)");
    throws<std::overflow_error>([] { (void)Array::scalar(-HUGE_VAL).to_int64(); },
                                "cannot convert float infinity to integer", "int(-inf)");
    throws<std::overflow_error>([] { (void)Array::scalar(0x1p63).to_int64(); }, "npz: value does not fit int64",
                                "int(2.0**63) is past int64");
    check::equal(Array::scalar(-0x1p63).to_int64(), std::numeric_limits<std::int64_t>::min(), "int(-2.0**63)");
    throws<std::overflow_error>([] { (void)Array::scalar<std::uint64_t>(std::uint64_t(1) << 63).to_int64(); },
                                "npz: value does not fit int64", "a uint64 past int64");
    throws<std::invalid_argument>([] { (void)Array::scalar(std::complex<double>(1, 0)).to_int64(); },
                                  "npz: a complex array has no integer value", "int() of a complex");
}

int main() {
    try {
        test_reads_numpy();
        test_writes_numpy();
        test_fortran_order();
        test_round_trips();
        test_refusals();
        test_names_and_int();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_npz");
}
