#include "qrss/slot.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "qrss/pytext.hpp"

namespace sstvae::qrss::slot {

namespace {

constexpr const char* NOT_ISO = "not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z";
constexpr const char* NO_ZONE = "give the time zone: end it with Z for UTC, e.g. 2026-10-09T06:00Z";

constexpr int MINYEAR = 1, MAXYEAR = 9999;
constexpr std::int64_t US_PER_S = 1000000;
constexpr std::int64_t US_PER_DAY = 86400 * US_PER_S;
constexpr std::int64_t QUARTER_US = 900 * US_PER_S;
// timedelta's limit on its days, and a C int's range (datetime's own
// conversion of a days count), for quarter_hour_start's refusals.
constexpr std::int64_t MAX_DELTA_DAYS = 999999999;
constexpr std::int64_t INT32_MIN_V = -2147483647 - 1, INT32_MAX_V = 2147483647;

// --- the calendar (Modules/_datetimemodule.c) ------------------------------------------

constexpr int DAYS_IN_MONTH[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
constexpr int DAYS_BEFORE_MONTH[13] = {0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

bool is_leap(int year) {
    const unsigned y = static_cast<unsigned>(year);
    return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
}

int days_in_month(int year, int month) {
    return (month == 2 && is_leap(year)) ? 29 : DAYS_IN_MONTH[month];
}

int days_before_month(int year, int month) {
    return DAYS_BEFORE_MONTH[month] + (month > 2 && is_leap(year) ? 1 : 0);
}

int days_before_year(int year) {
    const int y = year - 1;
    return y * 365 + y / 4 - y / 100 + y / 400;
}

int ymd_to_ord(int year, int month, int day) {
    return days_before_year(year) + days_before_month(year, month) + day;
}

constexpr int DI4Y = 1461, DI100Y = 36524, DI400Y = 146097;

// ordinal (1 = 0001-01-01) -> year, month, day.
void ord_to_ymd(int ordinal, int& year, int& month, int& day) {
    --ordinal;
    const int n400 = ordinal / DI400Y;
    int n = ordinal % DI400Y;
    year = n400 * 400 + 1;
    const int n100 = n / DI100Y;
    n = n % DI100Y;
    const int n4 = n / DI4Y;
    n = n % DI4Y;
    const int n1 = n / 365;
    n = n % 365;
    year += n100 * 100 + n4 * 4 + n1;
    if (n1 == 4 || n100 == 4) {
        year -= 1;
        month = 12;
        day = 31;
        return;
    }
    const bool leapyear = n1 == 3 && (n4 != 24 || n100 == 3);
    month = (n + 50) >> 5;
    int preceding = DAYS_BEFORE_MONTH[month] + (month > 2 && leapyear ? 1 : 0);
    if (preceding > n) {
        month -= 1;
        preceding -= days_in_month(year, month);
    }
    n -= preceding;
    day = n + 1;
}

int weekday(int year, int month, int day) { return (ymd_to_ord(year, month, day) + 6) % 7; }

int iso_week1_monday(int year) {
    const int first_day = ymd_to_ord(year, 1, 1);
    const int first_weekday = (first_day + 6) % 7;
    int week1_monday = first_day - first_weekday;
    if (first_weekday > 3) week1_monday += 7;
    return week1_monday;
}

int iso_to_ymd(int iso_year, int iso_week, int iso_day, int& year, int& month, int& day) {
    if (iso_year < MINYEAR || iso_year > MAXYEAR) return -4;
    if (iso_week <= 0 || iso_week >= 53) {
        bool out_of_range = true;
        if (iso_week == 53) {
            const int first_weekday = weekday(iso_year, 1, 1);
            if (first_weekday == 3 || (first_weekday == 2 && is_leap(iso_year))) out_of_range = false;
        }
        if (out_of_range) return -2;
    }
    if (iso_day <= 0 || iso_day >= 8) return -3;
    const int day_1 = iso_week1_monday(iso_year);
    const int day_offset = (iso_week - 1) * 7 + iso_day - 1;
    ord_to_ymd(day_1 + day_offset, year, month, day);
    return 0;
}

constexpr int EPOCH_ORD = 719163;   // ymd_to_ord(1970, 1, 1)

// --- the parser (Modules/_datetimemodule.c, CPython 3.13), over a NUL-terminated buffer --

bool is_digit(char c) { return static_cast<unsigned>(c - '0') < 10; }

const char* parse_digits(const char* ptr, int* var, std::size_t num_digits) {
    for (std::size_t i = 0; i < num_digits; ++i) {
        const unsigned tmp = static_cast<unsigned>(*(ptr++) - '0');
        if (tmp > 9) return nullptr;
        *var *= 10;
        *var += static_cast<int>(tmp);
    }
    return ptr;
}

int parse_isoformat_date(const char* dtstr, std::size_t len, int* year, int* month, int* day) {
    const char* p = dtstr;
    p = parse_digits(p, year, 4);
    if (!p) return -1;
    const bool uses_separator = (*p == '-');
    if (uses_separator) ++p;
    if (*p == 'W') {
        p++;
        int iso_week = 0, iso_day = 0;
        p = parse_digits(p, &iso_week, 2);
        if (!p) return -3;
        if (static_cast<std::size_t>(p - dtstr) < len) {
            if (uses_separator && *(p++) != '-') return -2;
            p = parse_digits(p, &iso_day, 1);
            if (!p) return -4;
        } else {
            iso_day = 1;
        }
        const int rv = iso_to_ymd(*year, iso_week, iso_day, *year, *month, *day);
        return rv ? -3 + rv : 0;
    }
    p = parse_digits(p, month, 2);
    if (!p) return -1;
    if (uses_separator && *(p++) != '-') return -2;
    p = parse_digits(p, day, 2);
    if (!p) return -1;
    return 0;
}

int parse_hh_mm_ss_ff(const char* tstr, const char* tstr_end, int* hour, int* minute, int* second,
                      int* microsecond) {
    *hour = *minute = *second = *microsecond = 0;
    const char* p = tstr;
    const char* p_end = tstr_end;
    int* vals[3] = {hour, minute, second};
    bool has_separator = true;
    for (std::size_t i = 0; i < 3; ++i) {
        p = parse_digits(p, vals[i], 2);
        if (!p) return -3;
        const char c = *(p++);
        if (i == 0) has_separator = (c == ':');
        if (c == '.' || c == ',') {
            if (p >= p_end) return -3;   // a decimal mark with no digit after it
            break;
        } else if (p >= p_end) {
            return c != '\0';
        } else if (has_separator && c == ':') {
            continue;
        } else if (!has_separator) {
            --p;
        } else {
            return -4;
        }
    }
    const std::size_t len_remains = static_cast<std::size_t>(p_end - p);
    const std::size_t to_parse = len_remains >= 6 ? 6 : len_remains;
    p = parse_digits(p, microsecond, to_parse);
    if (!p) return -3;
    static constexpr int correction[] = {100000, 10000, 1000, 100, 10};
    // to_parse is at least 1 here: every path into the fraction leaves a
    // character before p_end (the C reads correction[-1] otherwise).
    if (to_parse < 6) *microsecond *= correction[to_parse - 1];
    while (is_digit(*p)) ++p;
    return *p != '\0';
}

int parse_isoformat_time(const char* dtstr, std::size_t dtlen, int* hour, int* minute, int* second,
                         int* microsecond, int* tzoffset, int* tzmicrosecond) {
    const char* p = dtstr;
    const char* p_end = dtstr + dtlen;
    const char* tzinfo_pos = p;
    do {
        if (*tzinfo_pos == 'Z' || *tzinfo_pos == '+' || *tzinfo_pos == '-') break;
    } while (++tzinfo_pos < p_end);
    int rv = parse_hh_mm_ss_ff(dtstr, tzinfo_pos, hour, minute, second, microsecond);
    if (rv < 0) return rv;
    if (tzinfo_pos == p_end) return rv == 1 ? -5 : 0;
    if (*tzinfo_pos == 'Z') {
        *tzoffset = 0;
        *tzmicrosecond = 0;
        return *(tzinfo_pos + 1) != '\0' ? -5 : 1;
    }
    const int tzsign = (*tzinfo_pos == '-') ? -1 : 1;
    tzinfo_pos++;
    int tzhour = 0, tzminute = 0, tzsecond = 0;
    rv = parse_hh_mm_ss_ff(tzinfo_pos, p_end, &tzhour, &tzminute, &tzsecond, tzmicrosecond);
    *tzoffset = tzsign * ((tzhour * 3600) + (tzminute * 60) + tzsecond);
    *tzmicrosecond *= tzsign;
    return rv ? -5 : 1;
}

std::ptrdiff_t find_isoformat_datetime_separator(const char* dtstr, std::ptrdiff_t len) {
    if (len == 7) return 7;
    if (dtstr[4] == '-') {
        if (dtstr[5] == 'W') {
            if (len < 8) return -1;
            if (len > 8 && dtstr[8] == '-') {
                if (len == 9) return -1;
                if (len > 10 && is_digit(dtstr[10])) return 8;
                return 10;
            }
            return 8;
        }
        return 10;
    }
    if (dtstr[4] == 'W') {
        std::ptrdiff_t idx = 7;
        for (; idx < len; ++idx)
            if (!is_digit(dtstr[idx])) break;
        if (idx < 9) return idx;
        return idx % 2 == 0 ? 7 : 8;
    }
    return 8;
}

bool is_surrogate(char32_t c) { return c >= 0xD800 && c <= 0xDFFF; }

// Python's str.isspace(), which str.strip() strips.
bool is_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}

void put_utf8(std::string& out, char32_t c) {
    if (c < 0x80) {
        out += static_cast<char>(c);
    } else if (c < 0x800) {
        out += static_cast<char>(0xC0 | (c >> 6));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        out += static_cast<char>(0xE0 | (c >> 12));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (c >> 18));
        out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    }
}

// datetime.fromisoformat of a str (code points).
std::optional<IsoTime> fromisoformat_cp(std::u32string s) {
    // _sanitize_isoformat_str: under 7 characters is invalid; a surrogate
    // at 7, 8 or 10 (the first found) is the separator, 'T'.
    const std::size_t n = s.size();
    if (n < 7) return std::nullopt;
    for (std::size_t pos : {std::size_t(7), std::size_t(8), std::size_t(10)}) {
        if (pos > n) break;
        if (pos < n && is_surrogate(s[pos])) {
            s[pos] = U'T';
            break;
        }
    }
    // PyUnicode_AsUTF8AndSize: any other surrogate cannot be encoded.
    std::string utf8;
    for (char32_t c : s) {
        if (is_surrogate(c)) return std::nullopt;
        put_utf8(utf8, c);
    }
    // At least 7 bytes, as 7 code points are (said again so the compiler
    // sees the separator search's reads of [4] and [5] are in bounds).
    if (utf8.size() < 7) return std::nullopt;
    const char* dt_ptr = utf8.c_str();
    std::ptrdiff_t len = static_cast<std::ptrdiff_t>(utf8.size());
    const std::ptrdiff_t separator_location = find_isoformat_datetime_separator(dt_ptr, len);

    IsoTime t;
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0, microsecond = 0;
    int tzoffset = 0, tzusec = 0;
    int rv = parse_isoformat_date(dt_ptr, static_cast<std::size_t>(separator_location), &year, &month, &day);
    if (!rv && len > separator_location) {
        const char* p = dt_ptr + separator_location;
        if ((p[0] & 0x80) == 0) {
            p += 1;
        } else {
            switch (p[0] & 0xf0) {
                case 0xe0: p += 3; break;
                case 0xf0: p += 4; break;
                default: p += 2; break;
            }
        }
        len -= (p - dt_ptr);
        rv = parse_isoformat_time(p, static_cast<std::size_t>(len), &hour, &minute, &second, &microsecond,
                                  &tzoffset, &tzusec);
    }
    if (rv < 0) return std::nullopt;
    if (rv == 1) {
        // new_timezone(timedelta(seconds=tzoffset, microseconds=tzusec)):
        // strictly between -24 h and 24 h.
        const std::int64_t off = static_cast<std::int64_t>(tzoffset) * US_PER_S + tzusec;
        if (!(off > -US_PER_DAY && off < US_PER_DAY)) return std::nullopt;
        t.offset_us = off;
    }
    // The datetime constructor's range checks.
    if (year < MINYEAR || year > MAXYEAR || month < 1 || month > 12) return std::nullopt;
    if (day < 1 || day > days_in_month(year, month)) return std::nullopt;
    if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59 || microsecond < 0 ||
        microsecond > 999999)
        return std::nullopt;
    t.year = year;
    t.month = month;
    t.day = day;
    t.hour = hour;
    t.minute = minute;
    t.second = second;
    t.microsecond = microsecond;
    return t;
}

std::string two(int v) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%02d", v);
    return buf;
}

}  // namespace

std::string UtcTime::isoformat() const {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d", year, month, day, hour, minute, second);
    std::string out = buf;
    if (microsecond) {
        std::snprintf(buf, sizeof buf, ".%06d", microsecond);
        out += buf;
    }
    // format_utcoffset: the sign, then |offset| as HH:MM[:SS[.ffffff]].
    const bool negative = offset_us < 0;
    const std::int64_t a = negative ? -offset_us : offset_us;
    const std::int64_t us = a % US_PER_S, s = a / US_PER_S;
    const int hours = static_cast<int>(s / 3600), minutes = static_cast<int>(s / 60 % 60),
              seconds = static_cast<int>(s % 60);
    out += negative ? '-' : '+';
    out += two(hours) + ":" + two(minutes);
    if (us || seconds) out += ":" + two(seconds);
    if (us) {
        std::snprintf(buf, sizeof buf, ".%06d", static_cast<int>(us));
        out += buf;
    }
    return out;
}

std::int64_t UtcTime::unix_us() const {
    const std::int64_t days = ymd_to_ord(year, month, day) - EPOCH_ORD;
    const std::int64_t secs = (static_cast<std::int64_t>(hour) * 60 + minute) * 60 + second;
    return days * US_PER_DAY + secs * US_PER_S + microsecond - offset_us;
}

std::optional<IsoTime> fromisoformat(std::string_view text) { return fromisoformat_cp(pytext::decode(text)); }

UtcTime parse_utc(std::string_view text) {
    std::u32string s = pytext::decode(text);
    // s.strip()
    std::size_t a = 0, b = s.size();
    while (a < b && is_space(s[a])) ++a;
    while (b > a && is_space(s[b - 1])) --b;
    // .replace("Z", "+00:00").replace("z", "+00:00")
    std::u32string r;
    r.reserve(b - a + 8);
    for (std::size_t i = a; i < b; ++i) {
        if (s[i] == U'Z' || s[i] == U'z')
            r += U"+00:00";
        else
            r += s[i];
    }
    const std::optional<IsoTime> t = fromisoformat_cp(std::move(r));
    if (!t) throw std::invalid_argument(NOT_ISO);
    if (!t->offset_us) throw std::invalid_argument(NO_ZONE);
    UtcTime u;
    u.year = t->year;
    u.month = t->month;
    u.day = t->day;
    u.hour = t->hour;
    u.minute = t->minute;
    u.second = t->second;
    u.microsecond = t->microsecond;
    u.offset_us = *t->offset_us;
    return u;
}

std::int64_t quarter_hour_count(const UtcTime& t) {
    const std::int64_t us = t.unix_us();
    std::int64_t r = us % QUARTER_US;
    if (r < 0) r += QUARTER_US;
    if (r != 0) throw std::invalid_argument(t.isoformat() + " is not on a quarter hour");
    return (us - r) / QUARTER_US;
}

std::int64_t parse_slot(std::string_view text) {
    const UtcTime t = parse_utc(text);
    const std::int64_t q = quarter_hour_count(t);
    if (q < 0) {
        // no slot: refused by its UTC time, or the time as read where that
        // is before year 1 (quarter_hour_start's overflow)
        std::string when;
        try {
            when = slot_iso(q);
        } catch (const std::overflow_error&) {
            when = t.isoformat();
        }
        throw std::invalid_argument("slot " + when + " is before 1970-01-01T00:00Z");
    }
    return q;
}

UtcTime quarter_hour_start(std::int64_t q) {
    // _EPOCH + timedelta(seconds=900 q). The timedelta first: its days,
    // floor(900 q / 86400) = floor(q / 96), must fit a C int
    // (PyLong_AsInt) and then 999999999 (check_delta_day_range); then
    // the sum must land in years 1..9999.
    std::int64_t days_q = q / 96;
    if (q % 96 != 0 && q < 0) --days_q;
    if (days_q < INT32_MIN_V || days_q > INT32_MAX_V)
        throw std::overflow_error("Python int too large to convert to C int");
    if (days_q < -MAX_DELTA_DAYS || days_q > MAX_DELTA_DAYS)
        throw std::overflow_error("days=" + std::to_string(days_q) + "; must have magnitude <= " +
                                  std::to_string(MAX_DELTA_DAYS));
    constexpr std::int64_t MIN_Q = (std::int64_t(1) - EPOCH_ORD) * 86400 / 900;              // 0001-01-01
    constexpr std::int64_t MAX_Q = (std::int64_t(3652059) + 1 - EPOCH_ORD) * 86400 / 900 - 1;  // 9999-12-31T23:45
    if (q < MIN_Q || q > MAX_Q) throw std::overflow_error("date value out of range");
    const std::int64_t secs = 900 * q;
    std::int64_t days = secs / 86400, rem = secs % 86400;
    if (rem < 0) {
        rem += 86400;
        days -= 1;
    }
    UtcTime t;
    ord_to_ymd(static_cast<int>(days + EPOCH_ORD), t.year, t.month, t.day);
    t.hour = static_cast<int>(rem / 3600);
    t.minute = static_cast<int>(rem / 60 % 60);
    t.second = static_cast<int>(rem % 60);
    return t;
}

std::string slot_iso(std::int64_t q) {
    const UtcTime t = quarter_hour_start(q);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02dZ", t.year, t.month, t.day, t.hour, t.minute);
    return buf;
}

}  // namespace sstvae::qrss::slot
