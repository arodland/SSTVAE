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


#include <algorithm>
#include <complex>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>

#include "config.hpp"
#include "beacon/beacon.hpp"
#ifdef SSTVAE_HAVE_CODEC
#include "codec/codec.hpp"
#endif
#include "dsp/dsp.hpp"
#include "framing/framing.hpp"
#include "golay/golay.hpp"
#include "images/images.hpp"
#include "overlay/model.hpp"
#include "settings/settings.hpp"
#include "modem/modem.hpp"
#include "ofdm/ofdm.hpp"
#include "sync/sync.hpp"

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

    // framing's interleave/deinterleave take a ModeSpec, which on the
    // Python side is a frozen dataclass. Rather than bind that type, the
    // shims accept the mode *index* and look it up -- the reference's
    // signature is preserved by the conftest wrapper, which unpacks
    // `mode.index` before calling in. Keeping the C++ boundary free of
    // Python object layout is what lets the same functions serve the app.
    auto mode_by_index = [](int index) -> const sstvae::config::ModeSpec& {
        for (const auto& mode : sstvae::config::MODES)
            if (mode.index == index) return mode;
        throw std::invalid_argument("no such mode index");
    };

    py::module_ framing = m.def_submodule("framing");
    framing.def("interleave",
                [mode_by_index](DArray latents, int mode_index) {
                    std::span<const double> in(
                        latents.data(), static_cast<std::size_t>(latents.size()));
                    return to_numpy(
                        sstvae::framing::interleave(in, mode_by_index(mode_index)));
                },
                py::arg("latents"), py::arg("mode_index"));
    framing.def("deinterleave",
                [mode_by_index](DArray slots, int mode_index) {
                    std::span<const double> in(
                        slots.data(), static_cast<std::size_t>(slots.size()));
                    const auto r =
                        sstvae::framing::deinterleave(in, mode_by_index(mode_index));
                    return py::make_tuple(to_numpy(r.latents), to_numpy(r.weight));
                },
                py::arg("slots"), py::arg("mode_index"));
    framing.def("slot_range_for_frame",
                [](int abs_frame) {
                    const auto r = sstvae::framing::slot_range_for_frame(abs_frame);
                    py::array_t<std::int64_t> idx(
                        static_cast<py::ssize_t>(r.indices.size()));
                    std::copy(r.indices.begin(), r.indices.end(), idx.mutable_data());
                    return py::make_tuple(r.group, idx);
                },
                py::arg("abs_frame"));
    framing.def("slots_to_symbols",
                [](DArray frame_slots) {
                    std::span<const double> in(
                        frame_slots.data(),
                        static_cast<std::size_t>(frame_slots.size()));
                    const auto v = sstvae::framing::slots_to_symbols(in);
                    // (DATA_SYMS_PER_FRAME, NC_LATENT), matching the
                    // reference's reshape -- callers index it 2-D.
                    py::array_t<cdouble> out({
                        static_cast<py::ssize_t>(sstvae::config::DATA_SYMS_PER_FRAME),
                        static_cast<py::ssize_t>(sstvae::config::NC_LATENT)});
                    std::copy(v.begin(), v.end(), out.mutable_data());
                    return out;
                },
                py::arg("frame_slots"));
    framing.def("symbols_to_slots",
                [](CArray symbols) {
                    std::span<const cdouble> in(
                        symbols.data(), static_cast<std::size_t>(symbols.size()));
                    return to_numpy(sstvae::framing::symbols_to_slots(in));
                },
                py::arg("symbols"));
    framing.def("header_bits",
                [mode_by_index](int mode_index) {
                    const auto bits =
                        sstvae::framing::header_bits(mode_by_index(mode_index));
                    py::array_t<std::int64_t> out(
                        static_cast<py::ssize_t>(bits.size()));
                    std::copy(bits.begin(), bits.end(), out.mutable_data());
                    return out;
                },
                py::arg("mode_index"));
    framing.def("header_symbol",
                [mode_by_index](int mode_index) {
                    return to_numpy(
                        sstvae::framing::header_symbol(mode_by_index(mode_index)));
                },
                py::arg("mode_index"));
    framing.def("decode_header",
                [](DArray soft) -> py::object {
                    std::span<const double> in(
                        soft.data(), static_cast<std::size_t>(soft.size()));
                    const auto mode = sstvae::framing::decode_header(in);
                    // None for a rejected header, matching the reference;
                    // the caller maps the index back to its ModeSpec.
                    if (!mode) return py::none();
                    return py::int_(mode->index);
                },
                py::arg("soft"));

    py::module_ beacon = m.def_submodule("beacon");
    beacon.def("callsign_to_codes",
               [](const std::string& callsign) {
                   const auto codes = sstvae::beacon::callsign_to_codes(callsign);
                   py::array_t<std::int64_t> out(
                       static_cast<py::ssize_t>(codes.size()));
                   std::copy(codes.begin(), codes.end(), out.mutable_data());
                   return out;
               },
               py::arg("callsign"));
    beacon.def("codes_to_callsign",
               [](py::array_t<std::int64_t, py::array::c_style |
                                                py::array::forcecast> codes) {
                   std::vector<int> v(static_cast<std::size_t>(codes.size()));
                   for (py::ssize_t i = 0; i < codes.size(); ++i)
                       v[static_cast<std::size_t>(i)] =
                           static_cast<int>(codes.data()[i]);
                   return sstvae::beacon::codes_to_callsign(v);
               },
               py::arg("codes"));
    beacon.def("crc16",
               [](py::array_t<std::int64_t, py::array::c_style |
                                                py::array::forcecast> bits) {
                   std::vector<int> v(static_cast<std::size_t>(bits.size()));
                   for (py::ssize_t i = 0; i < bits.size(); ++i)
                       v[static_cast<std::size_t>(i)] =
                           static_cast<int>(bits.data()[i]);
                   const auto crc = sstvae::beacon::crc16(v);
                   py::array_t<std::int64_t> out(
                       static_cast<py::ssize_t>(crc.size()));
                   std::copy(crc.begin(), crc.end(), out.mutable_data());
                   return out;
               },
               py::arg("bits"));
    beacon.def("encode_chips",
               [](int frame_index, const std::string& callsign) {
                   return to_numpy(
                       sstvae::beacon::encode_chips(frame_index, callsign));
               },
               py::arg("frame_index"), py::arg("callsign"));
    beacon.def("chip_stream",
               [](int start_frame, int n_frames, const std::string& callsign) {
                   return to_numpy(sstvae::beacon::chip_stream(
                       start_frame, n_frames, callsign));
               },
               py::arg("start_frame"), py::arg("n_frames"), py::arg("callsign"));
    beacon.def("find_sync",
               [](DArray chips, double threshold, int max_candidates) {
                   std::span<const double> in(
                       chips.data(), static_cast<std::size_t>(chips.size()));
                   const auto offs =
                       sstvae::beacon::find_sync(in, threshold, max_candidates);
                   // A plain list, matching the reference's list[int].
                   py::list out;
                   for (std::int64_t v : offs) out.append(py::int_(v));
                   return out;
               },
               py::arg("chips"), py::arg("threshold") = 0.6,
               py::arg("max_candidates") = 8);
    beacon.def("decode",
               [](DArray chips, double threshold) -> py::object {
                   std::span<const double> in(
                       chips.data(), static_cast<std::size_t>(chips.size()));
                   const auto r = sstvae::beacon::decode(in, threshold);
                   // A tuple, unpacked by the conftest adapter into the
                   // reference's BeaconResult dataclass -- binding that
                   // dataclass would put Python object layout into the
                   // core the application links.
                   if (!r) return py::none();
                   return py::make_tuple(r->chip_offset, r->frame_index,
                                         r->callsign);
               },
               py::arg("chips"), py::arg("threshold") = 0.6);

    py::module_ modem = m.def_submodule("modem");
    modem.def("modulate",
              [mode_by_index](DArray latents, int mode_index, bool normalize,
                              const std::string& callsign,
                              double clip_headroom_db) {
                  std::span<const double> in(
                      latents.data(), static_cast<std::size_t>(latents.size()));
                  const sstvae::modem::Modem md;
                  return to_numpy(md.modulate(in, mode_by_index(mode_index),
                                              normalize, callsign,
                                              clip_headroom_db));
              },
              py::arg("latents"), py::arg("mode_index"),
              py::arg("normalize") = true, py::arg("callsign") = "",
              py::arg("clip_headroom_db") = sstvae::config::CLIP_HEADROOM_DB);
    // Results come back as dicts rather than bound structs: the conftest
    // adapter rebuilds the reference's dataclasses from them, so the
    // core stays free of any knowledge of Python object layout.
    modem.def("demodulate",
              [](DArray x, std::optional<std::pair<double, double>> search_s) {
                  std::span<const double> in(
                      x.data(), static_cast<std::size_t>(x.size()));
                  const sstvae::modem::Modem md;
                  const auto r = md.demodulate(in, search_s);
                  py::dict out;
                  out["latents"] = to_numpy(r.latents);
                  out["weights"] = to_numpy(r.weights);
                  out["mode_index"] = r.mode.index;
                  out["freq_offset"] = r.freq_offset;
                  out["sync_metric"] = r.sync_metric;
                  out["frames_received"] = r.frames_received;
                  out["callsign"] = r.callsign;
                  out["preamble_start"] = r.preamble_start;
                  out["snr_db"] = r.snr_db;
                  if (r.beacon)
                      out["beacon"] = py::make_tuple(r.beacon->chip_offset,
                                                     r.beacon->frame_index,
                                                     r.beacon->callsign);
                  else
                      out["beacon"] = py::none();
                  return out;
              },
              py::arg("x"), py::arg("search_s") = py::none());
    modem.def("demodulate_blind",
              [](DArray x, std::optional<std::pair<double, double>> search_s) {
                  std::span<const double> in(
                      x.data(), static_cast<std::size_t>(x.size()));
                  const sstvae::modem::Modem md;
                  const auto r = md.demodulate_blind(in, search_s);
                  py::dict out;
                  out["latents"] = to_numpy(r.latents);
                  out["weights"] = to_numpy(r.weights);
                  out["freq_offset"] = r.freq_offset;
                  out["callsign"] = r.callsign;
                  out["n_frames"] = r.n_frames;
                  out["snr_db"] = r.snr_db;
                  out["frame_offset"] =
                      r.frame_offset ? py::cast(*r.frame_offset) : py::none();
                  out["frame0_start"] =
                      r.frame0_start ? py::cast(*r.frame0_start) : py::none();
                  if (r.beacon)
                      out["beacon"] = py::make_tuple(r.beacon->chip_offset,
                                                     r.beacon->frame_index,
                                                     r.beacon->callsign);
                  else
                      out["beacon"] = py::none();
                  return out;
              },
              py::arg("x"), py::arg("search_s") = py::none());

    py::module_ sync = m.def_submodule("sync");
    // SyncError is raised through to Python as the reference's own
    // exception type, registered by the conftest adapter -- a bare
    // RuntimeError would make `pytest.raises(SyncError)` in the existing
    // suite fail for the wrong reason.
    static py::exception<sstvae::sync::SyncError> sync_error(sync, "SyncError");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const sstvae::sync::SyncError& e) {
            py::set_error(sync_error, e.what());
        }
    });
    sync.def("acquire",
             [](CArray z, double threshold, int max_bins,
                std::optional<std::pair<std::int64_t, std::int64_t>> search) {
                 std::span<const cdouble> in(
                     z.data(), static_cast<std::size_t>(z.size()));
                 std::optional<sstvae::sync::SearchWindow> win;
                 if (search) win = sstvae::sync::SearchWindow{search->first,
                                                             search->second};
                 const auto a = sstvae::sync::acquire(in, threshold, max_bins, win);
                 return py::make_tuple(a.preamble_start, a.freq_offset, a.metric);
             },
             py::arg("z"), py::arg("threshold") = 0.5, py::arg("max_bins") = 2,
             py::arg("search") = py::none());
    sync.def("acquire_blind",
             [](CArray z, double max_offset_hz, double bin_step_hz,
                int min_periods, double threshold,
                std::optional<std::pair<std::int64_t, std::int64_t>> search) {
                 std::span<const cdouble> in(
                     z.data(), static_cast<std::size_t>(z.size()));
                 std::optional<sstvae::sync::SearchWindow> win;
                 if (search) win = sstvae::sync::SearchWindow{search->first,
                                                             search->second};
                 const auto a = sstvae::sync::acquire_blind(
                     in, max_offset_hz, bin_step_hz, min_periods, threshold, win);
                 return py::make_tuple(a.frame_start, a.freq_offset, a.metric);
             },
             py::arg("z"), py::arg("max_offset_hz") = 55.0,
             py::arg("bin_step_hz") = 1.7, py::arg("min_periods") = 8,
             py::arg("threshold") = 4.0, py::arg("search") = py::none());

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
    dsp.def("bessel_i0", &sstvae::dsp::bessel_i0, py::arg("x"));
    dsp.def("kaiser", [](int m, double beta) {
        return to_numpy(sstvae::dsp::kaiser(m, beta));
    }, py::arg("m"), py::arg("beta"));
    dsp.def("resample_poly", [](DArray x, int up, int down) {
        std::vector<double> v(x.data(), x.data() + x.size());
        return to_numpy(sstvae::dsp::resample_poly(v, up, down));
    }, py::arg("x"), py::arg("up"), py::arg("down"));

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

    // --- settings ------------------------------------------------------
    //
    // Exposed as JSON text in and out rather than as a bound struct.
    // What matters for the tests is that a config file written by
    // either implementation is understood by the other, and that is a
    // statement about *the file*, not about field accessors -- binding
    // 30 fields would be more code testing less.
    py::module_ settings = m.def_submodule("settings");
    settings.def("round_trip",
                 [](const std::string& text) {
                     std::vector<sstvae::settings::Note> notes;
                     const auto cfg = sstvae::settings::from_json(text, &notes);
                     py::list out;
                     for (const auto& n : notes) out.append(py::make_tuple(n.key, n.problem));
                     return py::make_tuple(sstvae::settings::to_json(cfg), out);
                 },
                 py::arg("text"),
                 "Parse config JSON and re-serialize it; returns (json, notes).");
    settings.def("defaults_json",
                 [] { return sstvae::settings::to_json(sstvae::settings::Config{}); });
    settings.def("format_filename",
                 [](const std::string& tmpl, const std::string& callsign,
                    std::optional<double> freq_hz, const std::string& mode,
                    std::optional<std::int64_t> when) {
                     sstvae::settings::FilenameFields f;
                     f.callsign = callsign;
                     f.freq_hz = freq_hz;
                     f.mode = mode;
                     f.when = when;
                     return sstvae::settings::format_filename(tmpl, f);
                 },
                 py::arg("template"), py::arg("callsign") = std::string(),
                 py::arg("freq_hz") = std::nullopt, py::arg("mode") = std::string(),
                 py::arg("when") = std::nullopt);
    settings.def("config_path", [] { return sstvae::settings::config_path().string(); });
    settings.def("save_and_load",
                 [](const std::string& text, const std::string& path) {
                     std::vector<sstvae::settings::Note> notes;
                     const auto cfg = sstvae::settings::from_json(text, &notes);
                     sstvae::settings::save(cfg, path);
                     return sstvae::settings::load(path).config.callsign;
                 },
                 py::arg("text"), py::arg("path"));

    // --- overlay -------------------------------------------------------
    //
    // The document only; rendering lands in Phase 3 with the editor, so
    // that `item_bbox` has one implementation shared between the drawn
    // picture and the editor's selection handles rather than two.
    py::module_ overlay = m.def_submodule("overlay");
    overlay.def("round_trip",
                [](const std::string& text) {
                    std::vector<sstvae::overlay::Note> notes;
                    const auto doc = sstvae::overlay::from_json(text, &notes);
                    py::list out;
                    for (const auto& n : notes) out.append(py::make_tuple(n.where, n.problem));
                    return py::make_tuple(sstvae::overlay::to_json(doc), out);
                },
                py::arg("text"));
    overlay.attr("CANVAS_W") = sstvae::overlay::CANVAS_W;
    overlay.attr("CANVAS_H") = sstvae::overlay::CANVAS_H;
    overlay.attr("DOC_VERSION") = sstvae::overlay::DOC_VERSION;

    // --- images --------------------------------------------------------
    //
    // Pictures cross as (H, W, 3) uint8, which is what `np.asarray` of a
    // PIL image gives, so a test can hand one straight over.
    py::module_ images = m.def_submodule("images");
    images.def("to_array",
               [](py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>
                      pic) {
                   if (pic.ndim() != 3 || pic.shape(2) != 3) {
                       throw std::runtime_error("to_array wants an (H, W, 3) array");
                   }
                   sstvae::images::Picture p(static_cast<int>(pic.shape(1)),
                                             static_cast<int>(pic.shape(0)));
                   std::copy(pic.data(), pic.data() + pic.size(), p.rgb.begin());
                   const sstvae::images::ImageArray a = sstvae::images::to_array(p);
                   py::array_t<float> out({py::ssize_t{3},
                                           static_cast<py::ssize_t>(a.height),
                                           static_cast<py::ssize_t>(a.width)});
                   std::copy(a.chw.begin(), a.chw.end(), out.mutable_data());
                   return out;
               },
               py::arg("img"));
    images.def("load",
               [](const std::string& path) {
                   const sstvae::images::Picture p = sstvae::images::load(path);
                   py::array_t<std::uint8_t> out({static_cast<py::ssize_t>(p.height),
                                                  static_cast<py::ssize_t>(p.width),
                                                  py::ssize_t{3}});
                   std::copy(p.rgb.begin(), p.rgb.end(), out.mutable_data());
                   return out;
               },
               py::arg("path"));
    images.attr("IMG_W") = sstvae::images::IMG_W;
    images.attr("IMG_H") = sstvae::images::IMG_H;
    images.attr("MIN_W") = sstvae::images::MIN_W;
    images.attr("MIN_H") = sstvae::images::MIN_H;

#ifdef SSTVAE_HAVE_CODEC
    // --- codec ---------------------------------------------------------
    //
    // Unlike every other binding here, this one does *not* mirror its
    // Python counterpart's return type: `codec.OnnxCodec.decode` hands
    // back a PIL image, and a C++ class cannot. It returns the (H, W, 3)
    // uint8 array PIL would have been built from, and conftest wraps it.
    // The wrapper is two lines and it is honest about the seam; a
    // binding that linked Pillow to avoid it would not be.
    py::module_ codec = m.def_submodule("codec");
    py::class_<sstvae::codec::OnnxCodec>(codec, "OnnxCodec")
        .def(py::init([](py::object resolver) {
                 return std::make_unique<sstvae::codec::OnnxCodec>(
                     [resolver](const std::string& part) {
                         py::gil_scoped_acquire gil;
                         return resolver(part).cast<std::string>();
                     });
             }),
             py::arg("resolver"))
        .def("encode",
             [](sstvae::codec::OnnxCodec& self, DArray image) {
                 // (3, H, W) in [0, 1], as images.image_to_array gives.
                 if (image.ndim() != 3 || image.shape(0) != 3) {
                     throw std::runtime_error("encode wants a (3, H, W) array");
                 }
                 sstvae::codec::ImageArray arr;
                 arr.height = static_cast<int>(image.shape(1));
                 arr.width = static_cast<int>(image.shape(2));
                 arr.chw.assign(image.data(), image.data() + image.size());
                 std::vector<double> out;
                 {
                     py::gil_scoped_release unlock;
                     out = self.encode(arr);
                 }
                 return to_numpy(out);
             },
             py::arg("image"))
        .def("decode",
             [](sstvae::codec::OnnxCodec& self, DArray latents, DArray weights) {
                 std::vector<double> z(latents.data(), latents.data() + latents.size());
                 std::vector<double> w(weights.data(), weights.data() + weights.size());
                 sstvae::codec::Picture p;
                 {
                     py::gil_scoped_release unlock;
                     p = self.decode(z, w);
                 }
                 py::array_t<std::uint8_t> out(
                     {static_cast<py::ssize_t>(p.height),
                      static_cast<py::ssize_t>(p.width), py::ssize_t{3}});
                 std::copy(p.rgb.begin(), p.rgb.end(), out.mutable_data());
                 return out;
             },
             py::arg("latents"), py::arg("weights"));
    codec.def("pad_to_full",
              [](DArray vec, double fill) {
                  std::vector<double> v(vec.data(), vec.data() + vec.size());
                  return to_numpy(sstvae::codec::pad_to_full(v, fill));
              },
              py::arg("vec"), py::arg("fill") = 0.0);
    codec.attr("IMG_W") = sstvae::codec::IMG_W;
    codec.attr("IMG_H") = sstvae::codec::IMG_H;
    codec.attr("N_LATENTS") = sstvae::codec::N_LATENTS;
#endif

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
