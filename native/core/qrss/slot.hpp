// The slot clock as text: the quarter hour a pass is sent in, given as ISO
// 8601 UTC time ("2026-10-09T06:00Z"), and back. Port of
// sequences.parse_utc / parse_slot / quarter_hour_count /
// quarter_hour_start (sstvae/qrss/sequences.py, the reference), which the
// qrss_*.py CLIs' --slot and --start go through.
//
// The reference is three lines over datetime.fromisoformat:
//
//     dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00").replace("z", "+00:00"))
//
// so what it accepts is CPython 3.13's fromisoformat, and that is what is
// reproduced here: Modules/_datetimemodule.c's parser, step for step
// (_sanitize_isoformat_str, _find_isoformat_datetime_separator,
// parse_isoformat_date, parse_hh_mm_ss_ff, parse_isoformat_time,
// tzinfo_from_isoformat_results and the datetime constructor's range
// checks), not the pure-Python _pydatetime, which differs from it in
// corners ("06.5" is half past six seconds to the C parser). It reads
// more than ISO 8601's common forms and the CLIs accept all of it:
//
// - dates YYYY-MM-DD, YYYYMMDD and ISO weeks YYYY-Www[-D] / YYYYWww[D];
// - any one character as the date/time separator (a 'T', a space, an
//   'é' -- a multi-byte one is skipped by its UTF-8 length);
// - times HH, HH:MM, HHMM, HH:MM:SS, HHMMSS, with a fraction after '.' or
//   ',' (digits past six dropped), and after a fraction mark straight
//   after the hour or minute ("06.5" is 06:00:00.5); a third ':' after the
//   seconds starts the fraction too ("06:00:00:12" is 06:00:00.12);
// - offsets +-HH[[:]MM[[:]SS[.ffffff]]] with no field range of their own
//   ("+00:99" is +01:39), strictly inside +-24 h; 'Z'/'z' anywhere is
//   "+00:00" before parsing, as the reference's replace() makes it;
// - Python's str.strip() whitespace around it (Unicode's, by code point).
//
// The text is taken as UTF-8 and decoded as Python decodes an argv
// (pytext::decode: an invalid byte is U+DC00 + byte), so a surrogate at
// code point 7, 8 or 10 is the separator 'T' and one anywhere else makes
// the text invalid, as in CPython.
//
// Refusals are std::invalid_argument with the reference's texts:
//   "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z"
//   "give the time zone: end it with Z for UTC, e.g. 2026-10-09T06:00Z"
//   "<isoformat()> is not on a quarter hour"

#ifndef SSTVAE_QRSS_SLOT_HPP
#define SSTVAE_QRSS_SLOT_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace sstvae::qrss::slot {

// An aware datetime with a fixed UTC offset, as fromisoformat returns one:
// local fields plus the offset (the local time minus UTC).
struct UtcTime {
    int year = 1970, month = 1, day = 1;
    int hour = 0, minute = 0, second = 0, microsecond = 0;
    std::int64_t offset_us = 0;   // utcoffset(), strictly inside +-24 h

    // datetime.isoformat(): "2026-10-09T06:07:00+00:00",
    // "...T06:00:00.500000+05:30:15" (fraction and offset seconds only
    // when nonzero).
    std::string isoformat() const;
    // (self - 1970-01-01T00:00Z) in microseconds: the instant.
    std::int64_t unix_us() const;
    bool operator==(const UtcTime&) const = default;
};

// datetime.fromisoformat(text) in CPython 3.13: the fields and the offset
// (nullopt for a naive time), or nullopt for any text it refuses
// (ValueError). `text` is not stripped and 'Z' is not replaced here.
struct IsoTime {
    int year = 1, month = 1, day = 1;
    int hour = 0, minute = 0, second = 0, microsecond = 0;
    std::optional<std::int64_t> offset_us;   // nullopt: no tzinfo
};
std::optional<IsoTime> fromisoformat(std::string_view text);

// sequences.parse_utc: the aware time in ISO 8601 text with a zone.
UtcTime parse_utc(std::string_view text);

// sequences.quarter_hour_count: floor(unix(t) / 900) for a time exactly on
// a quarter hour ("<t.isoformat()> is not on a quarter hour" otherwise).
std::int64_t quarter_hour_count(const UtcTime& t);

// sequences.parse_slot: quarter_hour_count(parse_utc(text)).
std::int64_t parse_slot(std::string_view text);

// sequences.quarter_hour_start: the UTC time of quarter hour q. Outside
// years 1..9999 std::overflow_error with Python's text, which depends on
// how far: timedelta's days (q // 96) past a C int, "Python int too large
// to convert to C int"; past 999999999, "days=<d>; must have magnitude <=
// 999999999"; else "date value out of range".
UtcTime quarter_hour_start(std::int64_t q);

// The quarter hour as the app's sender writes it on a qrss_transmit.py
// command line: "2026-10-10T03:00Z" (UTC, to the minute); quarter_hour_start's
// std::overflow_error outside years 1..9999.
std::string slot_iso(std::int64_t q);

}  // namespace sstvae::qrss::slot

#endif  // SSTVAE_QRSS_SLOT_HPP
