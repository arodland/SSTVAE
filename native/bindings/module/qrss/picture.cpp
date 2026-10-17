// sstvae_native.qrss.picture: bindings of native/core/qrss/picture.
// Names, signatures, dtypes and ValueError texts follow
// sstvae/qrss/picture.py. StoredPicture stays the reference's dataclass:
// these take any object with its attributes and return its fields as a
// dict, which tests/qrss_native/picture.py wraps (see picture_chain.hpp).
// Arrays go through numpy as the reference converts them (np.asarray, with
// float64 where it asks for float64), and arguments it quotes in a refusal
// are quoted by Python's own str().

#include "qrss/registry.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/fspath.hpp"
#include "qrss/picture.hpp"
#include "qrss/picture_chain.hpp"

namespace sstvae_bindings::qrss {
namespace {

using namespace picture_chain;

using Shape = std::vector<py::ssize_t>;

Shape shape_with_last(const py::array& a, py::ssize_t last) {
    Shape s(a.shape(), a.shape() + a.ndim());
    s.back() = last;
    return s;
}

template <typename T>
py::array_t<T> shaped(const std::vector<T>& v, const Shape& shape) {
    py::array_t<T> out(shape);
    if (!v.empty()) std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(T));
    return out;
}

// np.asarray(x), refusing a 0-d array as `x.shape[-1]` does.
py::array with_last_axis(const py::object& x) {
    py::array a = numpy().attr("asarray")(x);
    if (a.ndim() == 0) throw py::index_error("tuple index out of range");
    return a;
}

// A 0-d result as numpy gives it from arithmetic on a 0-d array: the scalar.
py::object scalar_if_0d(const py::array& a) {
    if (a.ndim() == 0) return a[py::tuple()];
    return a;
}

std::string dtype_str(const py::array& a) { return a.dtype().attr("str").cast<std::string>(); }

// _check_group(group), quoting the argument as given.
std::int64_t group_of(const py::object& group) { return pc::check_group(int64_of(group), str_of(group)); }

template <typename T, int F>
std::span<const T> flat(const py::array_t<T, F>& a) {
    return std::span<const T>(a.data(), static_cast<std::size_t>(a.size()));
}

std::span<const pc::Half> flat_half(const py::array& h) {
    return std::span<const pc::Half>(static_cast<const pc::Half*>(h.data()), static_cast<std::size_t>(h.size()));
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("picture", "Port of sstvae.qrss.picture");
    m.attr("GROUP_LATENTS") = pc::GROUP_LATENTS;
    m.attr("SENT") = pc::SENT;
    m.attr("N_GROUPS") = pc::N_GROUPS;
    py::tuple names(pc::MODE_NAMES.size());
    for (std::size_t i = 0; i < pc::MODE_NAMES.size(); ++i) names[i] = py::str(std::string(pc::MODE_NAMES[i]));
    m.attr("MODE_NAMES") = names;
    m.attr("QRSP_VERSION") = pc::QRSP_VERSION;

    m.def("_check_group", [](const py::object& group) { return group_of(group); }, py::arg("group"));
    m.def("_check_mode", [](const py::object& mode) { return pc::check_mode(int64_of(mode), str_of(mode)); },
          py::arg("mode"));

    m.def("air_to_canonical", [](const py::object& group) {
        const auto perm = pc::air_to_canonical(group_of(group));
        py::array_t<py::ssize_t> out(static_cast<py::ssize_t>(perm.size()));
        for (std::size_t k = 0; k < perm.size(); ++k) out.mutable_data()[k] = perm[k];
        out.attr("setflags")(py::arg("write") = false);
        return out;
    }, py::arg("group"));

    m.def("never_sent", [](const py::object& group) {
        const auto mask = pc::never_sent(group_of(group));
        py::array_t<bool> out(static_cast<py::ssize_t>(mask.size()));
        for (std::size_t i = 0; i < mask.size(); ++i) out.mutable_data()[i] = mask[i] != 0;
        return out;
    }, py::arg("group"));

    m.def("prepare_segment", [](const py::object& full, const py::object& group) {
        const std::int64_t g = group_of(group);
        auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            numpy().attr("asarray")(full, py::arg("dtype") = "float64"));
        if (!a) throw py::error_already_set();
        const pc::Segment seg = pc::prepare_segment(flat(a), g);
        return half_array(seg, {static_cast<py::ssize_t>(seg.size())});
    }, py::arg("full"), py::arg("group"));

    m.def("air_values", [](const py::object& seg, const py::object& group) -> py::array {
        const py::array a = with_last_axis(seg);
        const auto row_len = static_cast<std::size_t>(a.shape(a.ndim() - 1));
        // The length is refused before the group, as the reference does.
        const std::int64_t g = row_len == static_cast<std::size_t>(pc::GROUP_LATENTS) ? group_of(group) : 0;
        const Shape shape = shape_with_last(a, pc::SENT);
        const std::string d = dtype_str(a);
        if (d == "<f2") {
            const py::array h = numpy().attr("ascontiguousarray")(a);
            return shaped(pc::air_values(flat_half(h), g, row_len), shape);
        }
        if (d == "<f4") {
            auto v = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(a);
            return shaped(pc::air_values(flat(v), g, row_len), shape);
        }
        // .astype(np.float64) of anything else: numpy's conversion, then exact.
        auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            a.attr("astype")("float64"));
        if (!v) throw py::error_already_set();
        return shaped(pc::air_values(flat(v), g, row_len), shape);
    }, py::arg("seg"), py::arg("group"));

    m.def("canonical_from_air", [](const py::object& air, const py::object& group) -> py::array {
        const py::array a = with_last_axis(air);
        const auto row_len = static_cast<std::size_t>(a.shape(a.ndim() - 1));
        const std::int64_t g = row_len == static_cast<std::size_t>(pc::SENT) ? group_of(group) : 0;
        const Shape shape = shape_with_last(a, pc::GROUP_LATENTS);
        const std::string d = dtype_str(a);
        const char kind = a.dtype().kind();
        if (d == "<f2") {
            const py::array h = numpy().attr("ascontiguousarray")(a);
            const auto v = pc::canonical_from_air(flat_half(h), g, row_len);
            return half_array(v, shape);
        }
        if (d == "<f4") {
            auto v = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(a);
            return shaped(pc::canonical_from_air(flat(v), g, row_len), shape);
        }
        if (kind == 'f' && d != "<f8") {
            // another floating dtype (long double, a byte-swapped one) keeps
            // its dtype: the reference's own scatter.
            if (row_len != static_cast<std::size_t>(pc::SENT)) (void)pc::canonical_from_air(std::span<const double>(), 0, row_len);
            py::tuple dims(shape.size());
            for (std::size_t i = 0; i < shape.size(); ++i) dims[i] = py::int_(shape[i]);
            py::array out = numpy().attr("zeros")(dims, a.dtype());
            const auto perm = pc::air_to_canonical(g);
            py::array_t<py::ssize_t> idx(static_cast<py::ssize_t>(perm.size()));
            for (std::size_t k = 0; k < perm.size(); ++k) idx.mutable_data()[k] = perm[k];
            out[py::make_tuple(py::ellipsis(), idx)] = a;
            return out;
        }
        auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            a.attr("astype")("float64"));
        if (!v) throw py::error_already_set();
        return shaped(pc::canonical_from_air(flat(v), g, row_len), shape);
    }, py::arg("air"), py::arg("group"));

    m.def("picture_id", [](const py::object& codec_id, const py::object& mode, const py::object& segs) {
        return picture_id_of(codec_id, mode, segs);
    }, py::arg("codec_id"), py::arg("mode"), py::arg("segs"));

    m.def("to_int8", [](const py::object& air) {
        auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            numpy().attr("asarray")(air, py::arg("dtype") = "float64"));
        if (!a) throw py::error_already_set();
        return scalar_if_0d(shaped(pc::to_int8(flat(a)), Shape(a.shape(), a.shape() + a.ndim())));
    }, py::arg("air"));

    m.def("from_int8", [](const py::object& qv) {
        const py::array a = numpy().attr("asarray")(qv);
        const Shape shape(a.shape(), a.shape() + a.ndim());
        if (dtype_str(a) == "|i1") {
            auto v = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>::ensure(a);
            return scalar_if_0d(shaped(pc::from_int8(flat(v)), shape));
        }
        auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            numpy().attr("asarray")(a, py::arg("dtype") = "float64"));
        if (!v) throw py::error_already_set();
        return scalar_if_0d(shaped(pc::from_int8(flat(v)), shape));
    }, py::arg("q"));

    // StoredPicture.from_latents -> its fields: {mode, codec_id, segs, picture_id}.
    m.def("from_latents", [](const py::object& full, const py::object& codec_id, const py::object& mode) {
        const std::int64_t mm = pc::check_mode(int64_of(mode), str_of(mode));
        auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            numpy().attr("asarray")(full, py::arg("dtype") = "float64"));
        if (!a) throw py::error_already_set();
        pc::StoredPicture sp;
        sp.mode = mm;
        for (std::int64_t g = 0; g <= mm; ++g) sp.segs.push_back(pc::prepare_segment(flat(a), g));
        sp.codec_id = int64_of(codec_id);
        pc::check_codec_id(sp.codec_id, str_of(codec_id));
        sp.picture_id = pc::picture_id(sp.codec_id, mm, sp.segs);
        return picture_dict(sp);
    }, py::arg("full"), py::arg("codec_id"), py::arg("mode"));

    // StoredPicture.check(sp) on any object with its fields.
    m.def("check", [](const py::object& sp) { (void)checked_picture(sp); }, py::arg("self"));

    m.def("id_hex", [](const py::object& picture_id) { return pc::id_hex(int64_of(picture_id)); },
          py::arg("picture_id"));

    // save_qrsp(path, sp): checked, then the file np.savez writes.
    m.def("save_qrsp", [](const py::object& path, const py::object& sp) {
        const pc::StoredPicture s = checked_picture(sp);
        const std::vector<std::uint8_t> data = pc::qrsp_bytes(s);
        py::object f = builtin("open")(path, "wb");
        try {
            f.attr("write")(as_bytes(data));
        } catch (...) {
            f.attr("close")();
            throw;
        }
        f.attr("close")();
    }, py::arg("path"), py::arg("sp"));

    // load_qrsp(path) -> {mode, codec_id, segs, picture_id}.
    // The refusals name the file as f"{path}" does, str(path): carried as
    // os.fsencode's bytes and decoded back the same way, so a name that is
    // not UTF-8 comes back as the str it was.
    m.def("load_qrsp", [](const py::object& path) {
        const std::vector<std::uint8_t> data = read_file(path);
        const std::string shown = fspath::str_path(path).bytes;
        try {
            return picture_dict(pc::parse_qrsp(data, shown));
        } catch (const std::invalid_argument& e) {
            fspath::raise_text(PyExc_ValueError, e.what());
        }
    }, py::arg("path"));

    // The whole .qrsp file in memory, for the parity test.
    m.def("_qrsp_bytes", [](const py::object& sp) { return as_bytes(pc::qrsp_bytes(checked_picture(sp))); },
          py::arg("sp"));
}

const Registrar reg("picture", 60, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
