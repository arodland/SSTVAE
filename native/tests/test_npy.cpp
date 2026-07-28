// Tests for the golden-vector loader itself.
//
// Everything else in this directory trusts npy.hpp to say what a file
// contains. A reader that silently mis-parsed a shape, or reinterpreted
// int64 as double, would turn every parity check into a confident
// tautology -- so the loader is checked against facts stated
// independently in the manifest rather than against itself.

#include <string>

#include "check.hpp"
#include "testing/npy.hpp"

using namespace sstvae;
using sstvae::testing::NpyFile;
using sstvae::testing::read_npy;

namespace {

std::string golden_dir;

std::string g(const std::string& name) { return golden_dir + "/" + name + ".npy"; }

void test_shapes_and_dtypes() {
    const NpyFile codewords = read_npy(g("golay/all_codewords"));
    check::equal(codewords.dtype, std::string("<i8"), "all_codewords dtype");
    check::equal(codewords.shape.size(), std::size_t{1}, "all_codewords rank");
    check::equal(codewords.shape[0], std::size_t{4096}, "all_codewords length");

    // 2-D, and the trailing dimension matters: a reader that flattened
    // shape would still pass every 1-D check above.
    const NpyFile soft = read_npy(g("golay/soft_inputs"));
    check::equal(soft.dtype, std::string("<f8"), "soft_inputs dtype");
    check::equal(soft.shape.size(), std::size_t{2}, "soft_inputs rank");
    check::equal(soft.cols(), std::size_t{24}, "soft_inputs columns");
    check::equal(soft.rows() * soft.cols(), soft.size(), "soft_inputs size is rows*cols");

    const NpyFile mod = read_npy(g("ofdm/mod_matrix"));
    check::equal(mod.dtype, std::string("<c16"), "mod_matrix dtype");
    check::equal(mod.rows(), std::size_t{192}, "mod_matrix rows (NSYM)");
    check::equal(mod.cols(), std::size_t{24}, "mod_matrix cols (NC)");
}

void test_values_land_in_the_right_places() {
    // Row-major ordering, checked against a value whose position is
    // known from the format rather than from our own writer: the first
    // row of MOD_MATRIX is sample n = -NCP, and every entry is a unit
    // phasor, so magnitudes are 1 everywhere.
    const std::vector<std::complex<double>> mod =
        sstvae::testing::load_c16(g("ofdm/mod_matrix"));
    bool unit = true;
    for (const auto& v : mod)
        if (std::abs(std::abs(v) - 1.0) > 1e-12) unit = false;
    check::is_true(unit, "every mod_matrix entry is a unit phasor");

    // Golay codewords are 24-bit and the all-zero message encodes to 0.
    const std::vector<std::int64_t> cw =
        sstvae::testing::load_i8(g("golay/all_codewords"));
    check::equal(cw[0], std::int64_t{0}, "codeword for message 0");
    bool in_range = true;
    for (std::int64_t v : cw)
        if (v < 0 || v >= (1 << 24)) in_range = false;
    check::is_true(in_range, "codewords fit in 24 bits");
}

void test_dtype_mismatch_throws() {
    // The important failure mode: reading int64 as double would produce
    // denormal garbage instead of an error.
    bool threw = false;
    try {
        sstvae::testing::load_f8(g("golay/all_codewords"));
    } catch (const std::exception&) {
        threw = true;
    }
    check::is_true(threw, "loading <i8 as <f8 throws");

    threw = false;
    try {
        read_npy(golden_dir + "/does_not_exist.npy");
    } catch (const std::exception&) {
        threw = true;
    }
    check::is_true(threw, "missing file throws");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <golden-dir>\n", argv[0]);
        return 2;
    }
    golden_dir = argv[1];
    try {
        test_shapes_and_dtypes();
        test_values_land_in_the_right_places();
        test_dtype_mismatch_throws();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("npy loader");
}
