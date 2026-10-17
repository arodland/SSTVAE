// sstvae_native.qrss.slot: the slot clock as text (native/core/qrss/slot),
// bound with sequences.py's names so tests/qrss_native/slot.py can assign
// them over sstvae.qrss.sequences.parse_utc / parse_slot /
// quarter_hour_count / quarter_hour_start. Times cross as datetime objects,
// with timezone.utc for a zero offset, as fromisoformat returns them.
// `fromisoformat` (the CPython 3.13 parser alone, fields or None) and
// `slot_iso` are extra, for the parity tests.

#include "qrss/registry.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "qrss/picture_chain.hpp"
#include "qrss/slot.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace sl = sstvae::qrss::slot;
using picture_chain::str_from_utf8;
using picture_chain::utf8_of;

py::object tzinfo_of(std::int64_t offset_us) {
    py::module_ dt = py::module_::import("datetime");
    if (offset_us == 0) return dt.attr("timezone").attr("utc");
    return dt.attr("timezone")(dt.attr("timedelta")(py::arg("microseconds") = offset_us));
}

py::object to_datetime(const sl::UtcTime& t) {
    py::module_ dt = py::module_::import("datetime");
    return dt.attr("datetime")(t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond,
                               tzinfo_of(t.offset_us));
}

// The text argument as the reference reads it: a str (TypeError otherwise,
// as str.strip of anything else raises AttributeError -- close enough for
// an argument no caller gets wrong).
std::string text_arg(const py::handle& s) {
    if (!py::isinstance<py::str>(s))
        throw py::type_error("expected a str, not " + py::type::of(s).attr("__name__").cast<std::string>());
    return utf8_of(s);
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("slot", "Port of sstvae.qrss.sequences' slot clock (parse_utc, parse_slot)");

    m.def("parse_utc", [](const py::object& s) { return to_datetime(sl::parse_utc(text_arg(s))); }, py::arg("s"));

    m.def("parse_slot", [](const py::object& s) { return sl::parse_slot(text_arg(s)); }, py::arg("s"));

    m.def("quarter_hour_count", [](const py::object& utc) {
        py::object tz = utc.attr("tzinfo");
        py::object off = tz.is_none() ? py::object(py::none()) : tz.attr("utcoffset")(utc);
        if (off.is_none()) throw py::value_error("quarter_hour_count needs an aware datetime (UTC)");
        sl::UtcTime t;
        t.year = utc.attr("year").cast<int>();
        t.month = utc.attr("month").cast<int>();
        t.day = utc.attr("day").cast<int>();
        t.hour = utc.attr("hour").cast<int>();
        t.minute = utc.attr("minute").cast<int>();
        t.second = utc.attr("second").cast<int>();
        t.microsecond = utc.attr("microsecond").cast<int>();
        t.offset_us = (off.attr("days").cast<std::int64_t>() * 86400 + off.attr("seconds").cast<std::int64_t>()) *
                          1000000 +
                      off.attr("microseconds").cast<std::int64_t>();
        return sl::quarter_hour_count(t);
    }, py::arg("utc"));

    m.def("quarter_hour_start", [](const py::object& qv) {
        PyObject* r = PyNumber_Long(qv.ptr());   // int(q)
        if (!r) throw py::error_already_set();
        const py::int_ qi = py::reinterpret_steal<py::int_>(r);
        std::int64_t qq = 0;
        try {
            qq = qi.cast<std::int64_t>();
        } catch (const py::cast_error&) {
            // |q| >= 2^63: its days, q // 96, are past a C int (timedelta's
            // own refusal, before any date is formed).
            throw std::overflow_error("Python int too large to convert to C int");
        }
        return to_datetime(sl::quarter_hour_start(qq));
    }, py::arg("q"));

    // datetime.fromisoformat's parse alone: (year, month, day, hour, minute,
    // second, microsecond, offset in microseconds or None), or None for
    // text it refuses.
    m.def("fromisoformat", [](const py::object& s) -> py::object {
        const std::optional<sl::IsoTime> t = sl::fromisoformat(text_arg(s));
        if (!t) return py::none();
        return py::make_tuple(t->year, t->month, t->day, t->hour, t->minute, t->second, t->microsecond,
                              t->offset_us ? py::object(py::int_(*t->offset_us)) : py::object(py::none()));
    }, py::arg("s"));

    m.def("slot_iso", [](std::int64_t qq) { return str_from_utf8(sl::slot_iso(qq)); }, py::arg("q"));
}

const Registrar reg("slot", 1, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
