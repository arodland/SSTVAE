// sstvae_native.qrss.flac: the passband store's FLAC encoder and reader
// (native/core/qrss/flac) with every encoder option, for the tests and
// measurements of tests/test_qrss_native_flac.py. Nothing in the
// reference has these names (soundfile packs the Python listener's
// hours), so nothing substitutes them; sstvae_native.qrss.passband keeps
// its own _flac_encode/_flac_read/_flac_info for the passband tests.
//
//   DEFAULTS: dict of EncodeOptions' defaults (the keyword defaults below)
//   encode(data, fs, **options) -> (bytes, stats)
//   encode_file(path, data, fs, **options) -> stats
//   decode(bytes) -> (int16[frames, channels], info)
//   read(path, start=0, stop=None) -> int16[frames, channels]
//
// options: blocksize, max_fixed_order, max_lpc_order, qlp_precision,
// lpc_orders_tried, partial_tukey, max_partition_order, stereo, seek_every.
// stats: bytes, frames, stereo (L/R, L/S, R/S, M/S frames), subframes
// (CONSTANT, VERBATIM, FIXED 0..4, LPC), lpc_orders (by order, 0..32), md5.
// info: channels, sample_rate, bits_per_sample, frames, md5.
// Errors: ValueError for bad options or data, OSError naming the file,
// RuntimeError for a stream that does not decode. The encode and decode
// run with the GIL released.

#include "qrss/registry.hpp"
#include "qrss/fspath.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "qrss/flac.hpp"
#include "qrss/rawfile.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace flac = sstvae::qrss::flac;
namespace rawfile = sstvae::qrss::rawfile;

int errno_of(const std::error_code& ec) {
    const std::error_condition c = ec.default_error_condition();
    if (c.category() == std::generic_category()) return c.value();
    return EIO;
}

template <typename F>
auto os_guard(F&& fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const rawfile::FileError& e) {
        fspath::raise_os_error(errno_of(e.code()), fspath::fs_text(e.path()));
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
}

py::array_t<std::int16_t, py::array::c_style | py::array::forcecast> frames_of(const py::object& data) {
    auto a = py::array_t<std::int16_t, py::array::c_style | py::array::forcecast>::ensure(data);
    if (!a) throw py::error_already_set();
    if (a.ndim() != 2) throw py::value_error("expected (frames, channels) int16");
    return a;
}

flac::EncodeOptions options(int blocksize, int max_fixed_order, int max_lpc_order, int qlp_precision,
                            int lpc_orders_tried, bool partial_tukey, int max_partition_order, bool stereo,
                            int seek_every) {
    flac::EncodeOptions o;
    o.blocksize = blocksize;
    o.max_fixed_order = max_fixed_order;
    o.max_lpc_order = max_lpc_order;
    o.qlp_precision = qlp_precision;
    o.lpc_orders_tried = lpc_orders_tried;
    o.partial_tukey = partial_tukey;
    o.max_partition_order = max_partition_order;
    o.stereo_decorrelation = stereo;
    o.seek_every_frames = seek_every;
    return o;
}

py::dict stats_dict(const flac::EncodeStats& st) {
    py::dict out;
    out["bytes"] = st.bytes;
    out["frames"] = st.frames;
    out["stereo"] = py::make_tuple(st.stereo[0], st.stereo[1], st.stereo[2], st.stereo[3]);
    py::list sub, orders;
    for (const auto v : st.subframes) sub.append(v);
    for (const auto v : st.lpc_orders) orders.append(v);
    out["subframes"] = sub;
    out["lpc_orders"] = orders;
    out["md5"] = py::bytes(reinterpret_cast<const char*>(st.md5.data()), st.md5.size());
    return out;
}

py::dict info_dict(const flac::Info& info) {
    py::dict out;
    out["channels"] = info.channels;
    out["sample_rate"] = info.sample_rate;
    out["bits_per_sample"] = info.bits_per_sample;
    out["frames"] = info.frames;
    out["md5"] = py::bytes(reinterpret_cast<const char*>(info.md5.data()), info.md5.size());
    return out;
}

py::array_t<std::int16_t> to_frames(const std::vector<std::int16_t>& y, int ch) {
    const std::size_t c = ch > 0 ? static_cast<std::size_t>(ch) : 1;
    py::array_t<std::int16_t> out({static_cast<py::ssize_t>(y.size() / c), static_cast<py::ssize_t>(ch)});
    std::copy(y.begin(), y.end(), out.mutable_data());
    return out;
}

void bind(py::module_& qrss) {
    py::module_ m = qrss.def_submodule("flac", "the passband store's FLAC encoder and reader (native/core/qrss/flac)");
    const flac::EncodeOptions D{};
    py::dict defaults;
    defaults["blocksize"] = D.blocksize;
    defaults["max_fixed_order"] = D.max_fixed_order;
    defaults["max_lpc_order"] = D.max_lpc_order;
    defaults["qlp_precision"] = D.qlp_precision;
    defaults["lpc_orders_tried"] = D.lpc_orders_tried;
    defaults["partial_tukey"] = D.partial_tukey;
    defaults["max_partition_order"] = D.max_partition_order;
    defaults["stereo"] = D.stereo_decorrelation;
    defaults["seek_every"] = D.seek_every_frames;
    m.attr("DEFAULTS") = defaults;

    m.def("encode", [](const py::object& data, int fs, int blocksize, int max_fixed_order, int max_lpc_order,
                       int qlp_precision, int lpc_orders_tried, bool partial_tukey, int max_partition_order,
                       bool stereo, int seek_every) {
        const auto a = frames_of(data);
        const int ch = static_cast<int>(a.shape(1));
        const flac::EncodeOptions o = options(blocksize, max_fixed_order, max_lpc_order, qlp_precision,
                                              lpc_orders_tried, partial_tukey, max_partition_order, stereo,
                                              seek_every);
        const std::span<const std::int16_t> x(a.data(), static_cast<std::size_t>(a.size()));
        flac::EncodeStats st;
        std::string bytes = os_guard([&] {
            py::gil_scoped_release nogil;
            return flac::encode(x, ch, fs, o, &st);
        });
        return py::make_tuple(py::bytes(bytes), stats_dict(st));
    }, py::arg("data"), py::arg("fs"), py::arg("blocksize") = D.blocksize,
          py::arg("max_fixed_order") = D.max_fixed_order, py::arg("max_lpc_order") = D.max_lpc_order,
          py::arg("qlp_precision") = D.qlp_precision, py::arg("lpc_orders_tried") = D.lpc_orders_tried,
          py::arg("partial_tukey") = D.partial_tukey, py::arg("max_partition_order") = D.max_partition_order,
          py::arg("stereo") = D.stereo_decorrelation, py::arg("seek_every") = D.seek_every_frames);

    m.def("encode_file", [](const py::object& path, const py::object& data, int fs, int blocksize,
                            int max_fixed_order, int max_lpc_order, int qlp_precision, int lpc_orders_tried,
                            bool partial_tukey, int max_partition_order, bool stereo, int seek_every) {
        const fspath::FsPath p = fspath::fs_path(path);
        const auto a = frames_of(data);
        const int ch = static_cast<int>(a.shape(1));
        const std::uint64_t frames = static_cast<std::uint64_t>(a.shape(0));
        const std::int16_t* d = a.data();
        const flac::EncodeOptions o = options(blocksize, max_fixed_order, max_lpc_order, qlp_precision,
                                              lpc_orders_tried, partial_tukey, max_partition_order, stereo,
                                              seek_every);
        const flac::EncodeStats st = os_guard([&] {
            py::gil_scoped_release nogil;
            return flac::encode_file(p.bytes, fs, ch, frames, [&](std::uint64_t first, std::size_t n, std::int16_t* out) {
                std::memcpy(out, d + first * static_cast<std::uint64_t>(ch), n * static_cast<std::size_t>(ch) * 2);
                return true;
            }, o);
        });
        return stats_dict(st);
    }, py::arg("path"), py::arg("data"), py::arg("fs"), py::arg("blocksize") = D.blocksize,
          py::arg("max_fixed_order") = D.max_fixed_order, py::arg("max_lpc_order") = D.max_lpc_order,
          py::arg("qlp_precision") = D.qlp_precision, py::arg("lpc_orders_tried") = D.lpc_orders_tried,
          py::arg("partial_tukey") = D.partial_tukey, py::arg("max_partition_order") = D.max_partition_order,
          py::arg("stereo") = D.stereo_decorrelation, py::arg("seek_every") = D.seek_every_frames);

    m.def("decode", [](const py::bytes& data) {
        const std::string bytes = data;
        flac::Info info;
        std::vector<std::int16_t> y;
        {
            py::gil_scoped_release nogil;
            y = flac::decode(bytes, &info);
        }
        return py::make_tuple(to_frames(y, info.channels), info_dict(info));
    }, py::arg("data"));

    m.def("read", [](const py::object& path, std::uint64_t start, const py::object& stop) {
        const fspath::FsPath p = fspath::fs_path(path);
        std::uint64_t n = UINT64_MAX;
        if (!stop.is_none()) {
            const auto s = stop.cast<std::uint64_t>();
            n = s > start ? s - start : 0;
        }
        std::vector<std::int16_t> y;
        int ch = 0;
        os_guard([&] {
            py::gil_scoped_release nogil;
            flac::Reader r(p.bytes);
            ch = r.info().channels;
            y = r.read(start, n);
        });
        return to_frames(y, ch);
    }, py::arg("path"), py::arg("start") = 0, py::arg("stop") = py::none());
}

const Registrar reg("flac", 91, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
