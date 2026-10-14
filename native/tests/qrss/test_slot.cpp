// native/core/qrss/slot against sequences.parse_slot (sstvae/qrss/sequences.py,
// CPython 3.13's datetime.fromisoformat underneath).
//
// The table is the reference's answer for each text, pinned: q, or the
// ValueError text. Regenerate a row with
//     python -c "from sstvae.qrss import sequences; print(sequences.parse_slot(TEXT))"
// tests/test_qrss_native_slot.py compares the two directly (and fuzzes the
// parser) where Python is available.

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "check.hpp"
#include "qrss/slot.hpp"

using namespace sstvae;
namespace sl = sstvae::qrss::slot;

namespace {

struct Row {
    const char* text;
    std::int64_t q;      // when `refusal` is empty
    const char* refusal;
};

// clang-format off
const Row ROWS[] = {
    {"2026-10-09T06:00Z", 1990584, ""},
    {"2026-10-09T06:00z", 1990584, ""},
    {"2026-10-09t06:00Z", 1990584, ""},
    {"2026-10-09 06:00Z", 1990584, ""},
    {" 2026-10-09T06:00Z\x0a""", 1990584, ""},
    {"\xe3""\x80""\x80""2026-10-09T06:00Z\xc2""\x85""", 1990584, ""},
    {"2026-10-09T06:00:00Z", 1990584, ""},
    {"2026-10-09T06:00:00.000Z", 1990584, ""},
    {"2026-10-09T06:00+00:00", 1990584, ""},
    {"2026-10-09T08:00+02:00", 1990584, ""},
    {"2026-10-09T01:30-04:30", 1990584, ""},
    {"20261009T0600Z", 1990584, ""},
    {"2026-10-09T0600Z", 1990584, ""},
    {"2026-10-09T06Z", 1990584, ""},
    {"2026-W41-5T06:00Z", 1990584, ""},
    {"2026W415T0600Z", 1990584, ""},
    {"2026-W41T06:00Z", 1990200, ""},
    {"2026-10-09T06:00+05:45", 1990561, ""},
    {"2026-10-09T06:00+00:99", 0, "2026-10-09T06:00:00+01:39 is not on a quarter hour"},
    {"2026-10-09\xc3""\xa9""06:00Z", 1990584, ""},
    {"2026-10-09T06.5Z", 0, "2026-10-09T06:00:00.500000+00:00 is not on a quarter hour"},
    {"2026-10-09T06:00:00:00Z", 1990584, ""},
    {"2026-10-09T06:00 Z", 1990584, ""},
    {"2025-10-01T00:00Z", 1954752, ""},
    {"1970-01-01T00:00Z", 0, ""},
    {"1969-12-31T23:45Z", 0, "slot 1969-12-31T23:45Z is before 1970-01-01T00:00Z"},
    {"1969-12-31T19:45-04:00", 0, "slot 1969-12-31T23:45Z is before 1970-01-01T00:00Z"},
    {"0001-01-01T00:00-01:00", 0, "slot 0001-01-01T01:00Z is before 1970-01-01T00:00Z"},
    {"0001-01-01T00:00+01:00", 0, "slot 0001-01-01T00:00:00+01:00 is before 1970-01-01T00:00Z"},
    {"9999-12-31T23:45-01:00", 281558115, ""},
    {"2026-10-09T06:07Z", 0, "2026-10-09T06:07:00+00:00 is not on a quarter hour"},
    {"2026-10-09T06:00:30Z", 0, "2026-10-09T06:00:30+00:00 is not on a quarter hour"},
    {"2026-10-09T06:00:00.5Z", 0, "2026-10-09T06:00:00.500000+00:00 is not on a quarter hour"},
    {"2026-10-09T06:00+05:30:15", 0, "2026-10-09T06:00:00+05:30:15 is not on a quarter hour"},
    {"2026-10-09T06:00:00+00:00:00.000001", 0, "2026-10-09T06:00:00+00:00:00.000001 is not on a quarter hour"},
    {"2026-10-09T06:00-00:00:00.000001", 0, "2026-10-09T06:00:00-00:00:00.000001 is not on a quarter hour"},
    {"2026-10-09T06:00:00.1234567Z", 0, "2026-10-09T06:00:00.123456+00:00 is not on a quarter hour"},
    {"2026-10-09T06:00", 0, "give the time zone: end it with Z for UTC, e.g. 2026-10-09T06:00Z"},
    {"2026-10-09", 0, "give the time zone: end it with Z for UTC, e.g. 2026-10-09T06:00Z"},
    {"2026-10-09Z", 0, "give the time zone: end it with Z for UTC, e.g. 2026-10-09T06:00Z"},
    {"2025-10-01 00:00", 0, "give the time zone: end it with Z for UTC, e.g. 2026-10-09T06:00Z"},
    {"", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"   ", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"x", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-10-09T24:00Z", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-13-01T00:00Z", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-02-29T00:00Z", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2024-02-29T00:00Z", 1899072, ""},
    {"2026-10-09T06:00+24:00", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-10-09T06:00ZZ", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-10-09T6:00Z", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-10-09T06:00:00.Z", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-W53-1T00:00Z", 1998240, ""},
    {"2020-W53-1T00:00Z", 1787904, ""},
    {"0000-01-01T00:00Z", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-10-09T06:00+23:59:59.999999", 0, "2026-10-09T06:00:00+23:59:59.999999 is not on a quarter hour"},
    {"2026-10-09T06:00Zjunk", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-10-9T06:00Z", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-1009T06:00Z", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
    {"2026-10-09T06:00-05:00:30", 0, "2026-10-09T06:00:00-05:00:30 is not on a quarter hour"},
    {"1969-12-31T23:59:59.999999Z", 0, "1969-12-31T23:59:59.999999+00:00 is not on a quarter hour"},
    // Invalid UTF-8 is Python's surrogateescape: one at the separator is a 'T'.
    {"2026-10-09\xff""06:00Z", 1990584, ""},
    {"2026-10-09T06:00Z\xff", 0, "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"},
};
// clang-format on

void test_table() {
    for (const Row& r : ROWS) {
        const std::string text = r.text;
        std::string got;
        std::int64_t q = 0;
        try {
            q = sl::parse_slot(text);
        } catch (const std::invalid_argument& e) {
            got = e.what();
        }
        check::equal(got, std::string(r.refusal), "refusal of '" + text + "'");
        if (std::string(r.refusal).empty()) check::equal(q, r.q, "q of '" + text + "'");
    }
}

void test_fields() {
    const sl::UtcTime t = sl::parse_utc("2026-10-09T06:00:00.25-05:30");
    check::equal(t.year, 2026, "year");
    check::equal(t.hour, 6, "local hour kept");
    check::equal(t.microsecond, 250000, "fraction");
    check::equal(t.offset_us, std::int64_t(-(5 * 3600 + 30 * 60)) * 1000000, "offset");
    check::equal(t.isoformat(), std::string("2026-10-09T06:00:00.250000-05:30"), "isoformat");
    // A zero offset written either way is UTC.
    check::equal(sl::parse_utc("2026-10-09T06:00-00:00").offset_us, std::int64_t(0), "-00:00 is UTC");
    // fromisoformat alone: naive, and refused.
    const auto naive = sl::fromisoformat("2026-10-09T06:00");
    check::is_true(naive && !naive->offset_us, "naive time has no offset");
    check::is_true(!sl::fromisoformat("2026-10-09T06:00Z+"), "trailing sign refused");
}

void test_start_and_iso() {
    check::equal(sl::slot_iso(1954752), std::string("2025-10-01T00:00Z"), "slot_iso(Q_TEST)");
    check::equal(sl::slot_iso(-1), std::string("1969-12-31T23:45Z"), "slot_iso(-1)");
    check::equal(sl::slot_iso(1990585), std::string("2026-10-09T06:15Z"), "slot_iso, a quarter past");
    const sl::UtcTime t = sl::quarter_hour_start(1954752);
    check::equal(t.isoformat(), std::string("2025-10-01T00:00:00+00:00"), "quarter_hour_start");
    check::equal(sl::quarter_hour_count(t), std::int64_t(1954752), "round trip");
    // Years 1..9999 only, as datetime.
    check::equal(sl::slot_iso(-69039552), std::string("0001-01-01T00:00Z"), "first quarter hour");
    check::equal(sl::slot_iso(281558111), std::string("9999-12-31T23:45Z"), "last quarter hour");
    // Further out, timedelta refuses first, with its own texts (Python 3.13's
    // sequences.quarter_hour_start at each boundary).
    const std::pair<std::int64_t, const char*> refused[] = {
        {281558112, "date value out of range"},
        {-69039553, "date value out of range"},
        {95999999999, "date value out of range"},
        {96000000095, "days=1000000000; must have magnitude <= 999999999"},
        {-95999999999, "days=-1000000000; must have magnitude <= 999999999"},
        {206158430207, "days=2147483647; must have magnitude <= 999999999"},
        {-206158430208, "days=-2147483648; must have magnitude <= 999999999"},
        {206158430208, "Python int too large to convert to C int"},
        {-206158430209, "Python int too large to convert to C int"},
        {INT64_MAX, "Python int too large to convert to C int"},
        {INT64_MIN, "Python int too large to convert to C int"},
    };
    for (const auto& [q, text] : refused) {
        std::string got = "(no refusal)";
        try {
            sl::quarter_hour_start(q);
        } catch (const std::overflow_error& e) {
            got = e.what();
        }
        check::equal(got, std::string(text), "quarter_hour_start(" + std::to_string(q) + ")");
    }
}

}  // namespace

int main() {
    try {
        test_table();
        test_fields();
        test_start_and_iso();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_slot");
}
