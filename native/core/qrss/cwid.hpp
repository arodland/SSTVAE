// Reading and matching the QRSSTVAE callsign windows (spec 2.7, design
// 6.7): a port of sstvae/qrss/cwid.py, the reference.
//
//   cwid.py                  here
//   _unit_sums               unit_sums
//   window_soft              window_soft
//   _fsk_only                fsk_only
//   _ln_i0                   ln_i0
//   window_llr               window_llr
//   _combine                 combine
//   decode_units             decode_units
//   _patterns (_CHARS/_PATS) patterns (an immutable table)
//   _dp                      dp
//   _trace                   trace
//   read_ml                  read_ml
//   read                     read
//   match                    match
//   keying_type              keying_type
//
// What the reference reads of a track -- z_d, chan.t0_index, chan.fs,
// timing, and chan.noise_psd_at(chan.t_s(n)) -- is a UnitSource, whose
// noise lookup is a seam (`unit_source` builds the default one from a
// TrackResult: its ChanCapture's own t_s and noise_psd_at), so the Python
// binding can hand it the reference's ChanCapture or any stand-in.
//
// Exactness. Bit-exact: the unit sums (numpy's pairwise complex sums, its
// FMA complex product and glibc's cexp, as numpy calls them), window_soft,
// keying_type's statistic, the 15-unit amplitude averages (np.convolve's
// BLAS dot: a running sum from 0), combine, decode_units, the reader's
// gains, dynamic programme and trace, and so read_ml's call, Z and margin.
// window_llr takes ln I0 through libm's log where numpy calls SVML's (an
// ulp of a float64, under the float32 rounding of the result nearly
// always); match's dot product is OpenBLAS's SkylakeX ddot order (exact
// on a machine whose numpy picks that kernel; an ulp elsewhere).
//
// Errors are the reference's: ValueError (std::invalid_argument, and
// morse::CallsignError for an unsendable call), IndexError
// (std::out_of_range), ZeroDivisionError (qrss::ZeroDivision).
//
// Thread-safe: the pattern table is a function-local static built once;
// everything else is pure (the noise seam is called on the caller's thread).

#ifndef SSTVAE_QRSS_CWID_HPP
#define SSTVAE_QRSS_CWID_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/chan.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::cwid {

inline constexpr int CALL_LO = config::CW_LEAD_UNITS;    // CALL_UNITS = slice(8, 184)
inline constexpr int CALL_N = config::CW_ROOM_UNITS;     // 176
inline constexpr int UNITS = config::CW_UNITS;           // 192

// What _unit_sums reads of a track. z must outlive the source.
struct UnitSource {
    std::span<const std::complex<double>> z;     // tr.z_d (or the ch given)
    double t0_index = 0.0;                       // tr.chan.t0_index
    double fs = config::CH_FS;                   // tr.chan.fs
    types::Timing timing;                        // tr.timing
    // chan.noise_psd_at(chan.t_s(n)) for CH indices n, one per unit (or
    // one value for all of them)
    std::function<std::vector<double>(std::span<const double> n)> noise_at;
};

// The default source of a track: its z_d, and its ChanCapture's t_s and
// noise_psd_at (with `fn`, chan.hpp's seam). The track's z_d must outlive
// the source.
UnitSource unit_source(const track::TrackResult& tr, const track::NoisePsdFn& fn = {});

// _unit_sums: (X0, X1, sigma2), n_win x 192 row-major: each unit summed at
// 0 Hz and at -1 cycle/unit, and the noise variance of such a sum.
struct UnitSums {
    std::size_t n_win = 0;
    std::vector<std::complex<double>> x0;
    std::vector<std::complex<double>> x1;
    std::vector<double> s2;
};
UnitSums unit_sums(const UnitSource& src, const frame::FrameSpec& spec);

// float32[rows, 176], row-major: per window, per Morse unit of the call region.
struct Units32 {
    std::size_t rows = 0;
    std::vector<float> data;
};

// _fsk_only: OOK of an L frame is ValueError.
void fsk_only(const frame::FrameSpec& spec, bool ook);

// window_soft: + = key-down; noise-only units of about unit variance.
// No windows: no rows (and nothing checked).
Units32 window_soft(const UnitSource& src, const frame::FrameSpec& spec, bool ook = false);

// _ln_i0(x) = log(i0e(x)) + x.
double ln_i0(double x);

// window_llr: each unit's log-likelihood ratio, key-down : key-up.
Units32 window_llr(const UnitSource& src, const frame::FrameSpec& spec, bool ook = false);

// _combine: per-unit values summed over rows (a 2-D stack, from 0 one row
// at a time) and the row count, or a 1-D row as it is and 1.
struct Combined {
    std::vector<double> s;
    std::size_t nrow = 1;
};
Combined combine(std::span<const double> values, std::size_t rows, std::size_t cols, bool one_d = false);

// decode_units: run-length Morse decode of key-down flags, or nullopt.
std::optional<std::string> decode_units(std::span<const std::uint8_t> hard);

// _patterns: the alphabet sorted ("/", "0".."9", "A".."Z") and each
// character's keying (1 key-down, 0 key-up between elements).
struct Pattern {
    char ch;
    std::vector<std::int64_t> units;
};
const std::vector<Pattern>& patterns();

// _dp: best[k, e] (k characters, the last ending at unit e) and back[k, e]
// = (character, previous end), (max_chars + 1) x (n + 1) row-major.
struct Dp {
    std::size_t kmax = 0;    // max_chars + 1 rows
    std::size_t n1 = 0;      // n + 1 columns
    std::vector<double> best;
    std::vector<std::int64_t> back;   // [kmax][n1][2]
};
// gains: n_chars rows of n + 1; lens: each character's length. forbid
// (character, start) is excluded. max_chars < 0 is numpy's "negative
// dimensions" ValueError, 0 its IndexError; gains of another length than
// n + 1 std::invalid_argument.
Dp dp(const std::vector<std::vector<double>>& gains, std::span<const std::int64_t> lens, std::int64_t n,
      std::int64_t max_chars, std::optional<std::pair<std::int64_t, std::int64_t>> forbid = std::nullopt);

// _trace: the best read in a Dp (text, score, (character, start) items),
// or nullopt, -inf and none when nothing is finite. lens as for dp.
struct Trace {
    std::optional<std::string> text;
    double score = 0.0;
    std::vector<std::pair<std::int64_t, std::int64_t>> items;
};
Trace trace(const Dp& d, std::span<const std::int64_t> lens);

// read_ml: (call, Z[, margin]) of the call whose keying best fits per-unit
// LLRs. margin is computed only when asked (0 otherwise).
struct ReadMl {
    std::optional<std::string> text;
    double z = 0.0;
    double margin = 0.0;
};
ReadMl read_ml(const Combined& llr, std::int64_t max_chars = 8, bool margin = false);

// read: the call when its margin reaches READ_LLR_MIN, else nullopt.
std::optional<std::string> read(const Combined& llr);

// match: Z of callsign's expected +-1 keying against soft values (176 per
// row). Another length is numpy's "not aligned" ValueError.
double match(const Combined& soft, std::string_view callsign);

// keying_type: "fsk" or "ook". L frames and frames without windows are
// "fsk" with nothing read from src.
std::string keying_type(const UnitSource& src, const frame::FrameSpec& spec,
                        std::optional<std::string_view> callsign = std::nullopt);

}  // namespace sstvae::qrss::cwid

#endif  // SSTVAE_QRSS_CWID_HPP
