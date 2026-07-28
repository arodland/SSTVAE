// pybind11 module: `pytest` drives the C++ core.
//
// This is the inverted binding described in docs/native-app.md -- not a
// way to call C++ from an application, but a way to make the *existing*
// Python test suite the C++ modem's acceptance suite. `test_modem_e2e`,
// `test_beacon`, `test_blind_acquisition` and the slow listener tests
// all apply unchanged once the modules they exercise are ported, at the
// cost of this shim.
//
// Every function here mirrors its Python counterpart's signature
// exactly, including argument names and defaults, because
// tests/conftest.py substitutes them into the reference modules by
// attribute assignment. A shim that "improved" an interface would break
// that substitution, which is the whole point of the file.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <vector>

#include "config.hpp"
#include "dsp/dsp.hpp"
#include "golay/golay.hpp"
#include "ofdm/ofdm.hpp"

namespace py = pybind11;
using sstvae::ofdm::cdouble;

// to_numpy overloads are shared with the dsp bindings below.

namespace {

// forcecast so a test passing an int array, or a non-contiguous slice,
// behaves the way numpy would rather than raising something obscure.
using CArray = py::array_t<cdouble, py::array::c_style | py::array::forcecast>;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<double> to_numpy(const std::vector<double>& v) {
    py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

py::array_t<cdouble> to_numpy(const std::vector<cdouble>& v) {
    py::array_t<cdouble> out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

template <std::size_t N>
py::array_t<cdouble> to_numpy(const std::array<cdouble, N>& v) {
    py::array_t<cdouble> out(static_cast<py::ssize_t>(N));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

template <std::size_t N>
py::array_t<double> to_numpy(const std::array<double, N>& v) {
    py::array_t<double> out(static_cast<py::ssize_t>(N));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

py::array_t<cdouble> matrix_to_numpy(std::span<const cdouble> data,
                                     py::ssize_t rows, py::ssize_t cols) {
    py::array_t<cdouble> out({rows, cols});
    std::copy(data.begin(), data.end(), out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(sstvae_native, m) {
    m.doc() =
        "C++ core of the native SSTVAE app, exposed so the Python test "
        "suite can check it against the reference implementation. Not "
        "shipped in the application: see docs/native-app.md.";

    // Lets a test report which side it is exercising, and lets conftest
    // fail loudly rather than silently skipping if a stale module is
    // found on sys.path.
    m.attr("__sstvae_abi__") = 1;

    py::module_ golay = m.def_submodule("golay");
    golay.def("encode", &sstvae::golay::encode, py::arg("data12"),
              "12 info bits -> 24-bit codeword (data in high bits, parity last).");
    golay.def(
        "codeword_bits",
        [](int data12) {
            const auto bits = sstvae::golay::codeword_bits(data12);
            // int64, matching what the reference's
            // `(cw >> np.arange(23, -1, -1)) & 1` produces -- tests do
            // `bits.copy()` and `bits[flip] ^= 1` on the result.
            py::array_t<std::int64_t> out(sstvae::golay::N_BITS);
            for (int i = 0; i < sstvae::golay::N_BITS; ++i)
                out.mutable_data()[i] = bits[static_cast<std::size_t>(i)];
            return out;
        },
        py::arg("data12"), "24-bit codeword as an array of 0/1, MSB first.");
    golay.def(
        "decode_soft",
        [](DArray soft) {
            auto buf = soft.unchecked<1>();
            std::vector<double> v(static_cast<std::size_t>(buf.shape(0)));
            for (py::ssize_t i = 0; i < buf.shape(0); ++i)
                v[static_cast<std::size_t>(i)] = buf(i);
            return sstvae::golay::decode_soft(v);
        },
        py::arg("soft"),
        "ML-decode 24 soft values (positive => bit 0) to the 12 info bits.");
    golay.def("min_distance", &sstvae::golay::min_distance,
              "Minimum distance of the code (8).");

    py::module_ dsp = m.def_submodule("dsp");
    dsp.def("to_baseband",
            [](DArray x) {
                std::span<const double> in(x.data(),
                                           static_cast<std::size_t>(x.size()));
                return to_numpy(sstvae::dsp::to_baseband(in));
            },
            py::arg("x"), "Real passband -> complex baseband.");
    dsp.def("freq_correct",
            [](CArray z, double f_hz) {
                std::span<const cdouble> in(z.data(),
                                            static_cast<std::size_t>(z.size()));
                return to_numpy(sstvae::dsp::freq_correct(in, f_hz));
            },
            py::arg("z"), py::arg("f_hz"));
    dsp.def("sync_lowpass",
            [](CArray z) {
                std::span<const cdouble> in(z.data(),
                                            static_cast<std::size_t>(z.size()));
                return to_numpy(sstvae::dsp::sync_lowpass(in));
            },
            py::arg("z"));
    dsp.def("tx_condition",
            [](DArray x, double clip_headroom_db, int iterations) {
                std::span<const double> in(x.data(),
                                           static_cast<std::size_t>(x.size()));
                return to_numpy(
                    sstvae::dsp::tx_condition(in, clip_headroom_db, iterations));
            },
            py::arg("x"), py::arg("clip_headroom_db"), py::arg("iterations") = 2);
    dsp.def("papr_db",
            [](DArray x) {
                std::span<const double> in(x.data(),
                                           static_cast<std::size_t>(x.size()));
                return sstvae::dsp::papr_db(in);
            },
            py::arg("x"));
    dsp.def("to_int16",
            [](DArray x, double peak) {
                std::span<const double> in(x.data(),
                                           static_cast<std::size_t>(x.size()));
                const auto v = sstvae::dsp::to_int16(in, peak);
                py::array_t<std::int16_t> out(static_cast<py::ssize_t>(v.size()));
                std::copy(v.begin(), v.end(), out.mutable_data());
                return out;
            },
            py::arg("x"), py::arg("peak") = 0.95);
    // Not substituted into the reference (Python uses scipy for these),
    // but exposed so the parity tests can check them directly -- they
    // are what sync_lowpass and tx_condition are built from, and a bug
    // in one of them would otherwise only surface as a vague
    // whole-filter mismatch.
    dsp.def("hilbert",
            [](DArray x) {
                std::span<const double> in(x.data(),
                                           static_cast<std::size_t>(x.size()));
                return to_numpy(sstvae::dsp::hilbert(in));
            },
            py::arg("x"));
    dsp.def("firwin_lowpass", [](int numtaps, double cutoff_hz) {
        return to_numpy(sstvae::dsp::firwin_lowpass(numtaps, cutoff_hz));
    });
    dsp.def("firwin_bandpass", [](int numtaps, double lo_hz, double hi_hz) {
        return to_numpy(sstvae::dsp::firwin_bandpass(numtaps, lo_hz, hi_hz));
    });
    dsp.def("wrap_cycles", &sstvae::dsp::wrap_cycles, py::arg("cycles"));

    py::module_ ofdm = m.def_submodule("ofdm");
    ofdm.def("modulate_symbols",
             [](CArray symbols) {
                 if (symbols.ndim() != 2 || symbols.shape(1) != sstvae::config::NC)
                     throw std::invalid_argument(
                         "modulate_symbols expects an (n_sym, NC) array");
                 const std::size_t n_sym = static_cast<std::size_t>(symbols.shape(0));
                 std::span<const cdouble> in(symbols.data(),
                                             static_cast<std::size_t>(symbols.size()));
                 return to_numpy(sstvae::ofdm::modulate_symbols(in, n_sym));
             },
             py::arg("symbols"),
             "(n_sym, NC) complex -> real waveform (n_sym * NSYM,).");
    ofdm.def("demod_window",
             [](CArray z, std::int64_t start, std::int64_t backoff) {
                 std::span<const cdouble> in(z.data(),
                                             static_cast<std::size_t>(z.size()));
                 return to_numpy(sstvae::ofdm::demod_window(in, start, backoff));
             },
             py::arg("z"), py::arg("start"), py::arg("backoff") = 0,
             "Demodulate one useful window of baseband signal.");
    ofdm.def("pilot_sequence",
             []() { return to_numpy(sstvae::ofdm::pilot_sequence()); },
             "Fixed unit-magnitude QPSK sequence for preamble and frame pilots.");
    ofdm.def("preamble_waveform",
             []() { return to_numpy(sstvae::ofdm::preamble_waveform()); });
    ofdm.def("preamble_template",
             []() { return to_numpy(sstvae::ofdm::preamble_template()); });
    ofdm.def("pilot_template",
             []() { return to_numpy(sstvae::ofdm::pilot_template()); });

    // The module-level arrays. Exposed as functions *and* as attributes:
    // the reference has them as attributes, so conftest substitutes the
    // attributes, while the functions let a parity test fetch them
    // without depending on that substitution having happened.
    ofdm.def("carrier_freqs", []() { return to_numpy(sstvae::ofdm::carrier_freqs()); });
    ofdm.def("baseband_freqs",
             []() { return to_numpy(sstvae::ofdm::baseband_freqs()); });
    ofdm.def("mod_matrix", []() {
        return matrix_to_numpy(sstvae::ofdm::mod_matrix(), sstvae::config::NSYM,
                               sstvae::config::NC);
    });
    ofdm.def("demod_matrix", []() {
        return matrix_to_numpy(sstvae::ofdm::demod_matrix(), sstvae::config::NC,
                               sstvae::config::M);
    });
    ofdm.attr("CARRIER_FREQS") = to_numpy(sstvae::ofdm::carrier_freqs());
    ofdm.attr("BASEBAND_FREQS") = to_numpy(sstvae::ofdm::baseband_freqs());
    ofdm.attr("MOD_MATRIX") = matrix_to_numpy(
        sstvae::ofdm::mod_matrix(), sstvae::config::NSYM, sstvae::config::NC);
    ofdm.attr("DEMOD_MATRIX") = matrix_to_numpy(
        sstvae::ofdm::demod_matrix(), sstvae::config::NC, sstvae::config::M);

    // The generated constants, so a test can prove the C++ build and the
    // Python package were built from the same config.py rather than
    // assuming it.
    py::module_ config = m.def_submodule("config");
    config.attr("FS") = sstvae::config::FS;
    config.attr("RS") = sstvae::config::RS;
    config.attr("NC") = sstvae::config::NC;
    config.attr("M") = sstvae::config::M;
    config.attr("NCP") = sstvae::config::NCP;
    config.attr("NSYM") = sstvae::config::NSYM;
    config.attr("CARRIER0") = sstvae::config::CARRIER0;
    config.attr("FCENTER") = sstvae::config::FCENTER;
    config.attr("FRAME_SAMPLES") = sstvae::config::FRAME_SAMPLES;
    config.attr("LATENTS_PER_FRAME") = sstvae::config::LATENTS_PER_FRAME;
    config.attr("BEACON_CARRIER") = sstvae::config::BEACON_CARRIER;
    config.attr("FRAMES_PER_GROUP") = sstvae::config::FRAMES_PER_GROUP;
    config.attr("GROUP_LATENTS") = sstvae::config::GROUP_LATENTS;
    config.attr("PREAMBLE_SAMPLES") = sstvae::config::PREAMBLE_SAMPLES;
    config.attr("PROTOCOL_VERSION") = sstvae::config::PROTOCOL_VERSION;
}
