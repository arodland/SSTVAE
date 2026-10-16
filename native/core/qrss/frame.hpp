// The QRSSTVAE frame, CE and L: stream layout, positions, callsign windows
// and timing. Port of sstvae/qrss/frame.py and sstvae/qrss/lframe.py (the
// reference), as one FrameSpec type with a waveform field.
//
// A format module, and the one place frame lengths live. Two index spaces:
//
// - stream index s counts the frame's symbols without the callsign windows:
//   preamble [0, n_pre), header [n_pre, n_pre + n_hdr), data. References,
//   latent order, scrambler and precoder are defined on it.
// - position p = pos(s) counts symbols with the windows, so stream symbol s
//   is centred at t0 + pos(s) * t_sym.
//
// Reference symbols are the stream symbols with j = s - n_pre >= 0 and
// j % 16 == 0. CE sends one +-1 value per symbol: header non-reference
// symbol i carries coded bit i (the 2475th is a spare, +1), data
// non-reference symbol i carries precoded value x_i, plus sqrt(hdr_rho) h_i
// when the frame has the spread header. L sends two latents per complex
// symbol at half the rate: header symbol j carries bits 2j, 2j+1 as QPSK and
// data symbol k latents 2k (I) and 2k+1 (Q) scaled by 1/sqrt(2); its spread
// copy scales the latents to (x + sqrt(rho) h) / sqrt(1 + rho).
//
// Every refusal is std::invalid_argument carrying the reference's ValueError
// text, except a spec handed to the other waveform's CE- or L-only operation,
// which is WrongWaveform (Python fails an assertion there).

#ifndef SSTVAE_QRSS_FRAME_HPP
#define SSTVAE_QRSS_FRAME_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"

namespace sstvae::qrss::frame {

enum class Waveform : std::uint8_t {
    CE = config::WAVEFORM_CE,
    L = config::WAVEFORM_L,
};

// A CE-only operation given an L frame, or the reverse (Python: AssertionError).
struct WrongWaveform : std::logic_error {
    using std::logic_error::logic_error;
};

// Header symbols with bits in an L frame: 1,237, two coded bits each.
inline constexpr std::int64_t N_HDR_SLOTS_L =
    config::L_N_HDR - (config::L_N_HDR + config::REF_PERIOD - 1) / config::REF_PERIOD;
static_assert(2 * N_HDR_SLOTS_L == config::N_HDR_BITS);

// The largest count a FrameSpec takes (n_data, |cw_after|): 2^53, past which
// the reference's float timings stop being exact. Python has no limit.
inline constexpr std::int64_t MAX_COUNT = std::int64_t(1) << 53;

class FrameSpec {
public:
    // FrameSpec(...) / LFrameSpec(...) with __post_init__'s validation and its
    // default: hdr_rho absent is SPREAD_RHO with a header block, else 0.
    static FrameSpec make(Waveform waveform, std::string name, std::int64_t n_pre,
                          std::int64_t n_hdr, std::int64_t n_data,
                          std::vector<std::int64_t> cw_after,
                          std::optional<double> hdr_rho = std::nullopt);

    // --- the dataclass fields
    Waveform waveform() const { return waveform_; }
    const std::string& name() const { return name_; }
    std::int64_t n_pre() const { return n_pre_; }
    std::int64_t n_hdr() const { return n_hdr_; }
    std::int64_t n_data() const { return n_data_; }
    const std::vector<std::int64_t>& cw_after() const { return cw_after_; }
    double hdr_rho() const { return hdr_rho_; }

    // --- the class attributes
    double t_sym() const;            // T_SYM or L_T_SYM, seconds per symbol
    std::int64_t cw_window() const;  // symbols per callsign window (384 or 192)

    // --- derived
    bool has_header() const { return n_hdr_ != 0 || hdr_rho_ > 0; }
    // Data symbols that carry latents: n_data for CE, n_data / 2 for L.
    std::int64_t n_data_payload() const;
    std::int64_t n_data_sym() const { return n_data_sym_; }  // data section incl. references
    std::int64_t n_sym() const { return n_pre_ + n_hdr_ + n_data_sym_; }
    std::int64_t n_win() const { return static_cast<std::int64_t>(cw_after_.size()); }
    std::int64_t n_pos() const { return n_sym() + cw_window() * n_win(); }
    std::int64_t n_ref() const;
    double duration_s() const;  // n_pos * t_sym
    std::vector<std::int64_t> win_start_pos() const;  // cw_after[w] + window * w
    // Whether the last window ends the frame (keyed_end_pos is then the
    // float n_pos - 1/2; otherwise the integer pos(n_sym - 1) + SPAN).
    bool last_window_ends_frame() const;
    double keyed_end_pos() const;
    double keyed_start_pos() const { return -static_cast<double>(config::SPAN); }
    // pos(s) = s + window * |{w : cw_after[w] <= s}|, int64 arithmetic
    // (wrapping, as numpy's does).
    std::int64_t pos_of(std::int64_t s) const;
    double time_s(double pos) const { return pos * t_sym(); }
    std::vector<std::pair<double, double>> window_spans_s() const;

    // The dataclass's eq: the same class (waveform) and every field equal.
    friend bool operator==(const FrameSpec& a, const FrameSpec& b);

private:
    FrameSpec() = default;
    Waveform waveform_ = Waveform::CE;
    std::string name_;
    std::int64_t n_pre_ = 0, n_hdr_ = 0, n_data_ = 0;
    std::vector<std::int64_t> cw_after_;
    double hdr_rho_ = 0.0;
    std::int64_t n_data_sym_ = 0;
};

// frame._data_symbols: the smallest L with L - ceil(L/16) == n_data.
std::int64_t data_symbols(std::int64_t n_data);
// lframe._refs_below: references among j' in [0, j), i.e. ceil(j/16).
std::int64_t refs_below(std::int64_t j);
// lframe._data_symbols: the smallest D with n_payload non-references in
// j = j0 .. j0 + D - 1.
std::int64_t data_symbols_l(std::int64_t n_payload, std::int64_t j0);

// --- presets (config::FRAME_PRESETS, generated from the reference) ---------

// One table in the reference's order: frame.PRESETS, LEGACY, PROTOTYPES or
// lframe.PRESETS.
const std::vector<FrameSpec>& presets(config::PresetGroup group);

// frame.get: a CE preset, version 1 frame or prototype by name.
const FrameSpec& get(std::string_view name);
// lframe.get: an L preset by name.
const FrameSpec& get_l(std::string_view name);
// lframe.get_any: an "l-..." name is L's (get_l), anything else CE's (get).
const FrameSpec& get_any(std::string_view name);
// The refusal for an unknown name; `shown` is the name as Python's !r shows
// it (py_repr for a string).
std::string unknown_frame(std::string_view shown, Waveform waveform);
// lframe.twin: the frame of the same shape in `waveform` ("short" <-> "l-short").
FrameSpec twin(const FrameSpec& spec, int waveform);
// frame.plain: `spec` without the spread copy (the version 1 frame of its shape).
FrameSpec plain(const FrameSpec& spec);

// Python's repr of a str ('short'), for the refusal texts.
std::string py_repr(std::string_view s);
// Python's repr of a float (0.054, 1e-05, nan), for the refusal texts.
std::string py_repr(double v);
// Python's format(v, spec) for one printf conversion of a float ("%g",
// "%.1f", "%.2f"): printf's digits, with '.' as the decimal point whatever
// LC_NUMERIC says. Python never reads the locale here; a Qt program has
// set it from the environment (setlocale(LC_ALL, "")), where printf alone
// writes "287,9" in de_DE.
std::string py_format(double v, const char* printf_spec);
// A numpy shape as Python prints the tuple: (3,), (), (2, 3).
std::string shape_repr(std::span<const std::int64_t> shape);

// --- layout -------------------------------------------------------------------

// Where every symbol class sits: int64 stream indices unless named pos.
// CE fills hdr_bits/hdr_spare and known_val_ce; L fills hdr_sym and known_val_l.
struct Layout {
    Waveform waveform = Waveform::CE;
    std::vector<std::int64_t> pre;
    std::vector<std::int64_t> hdr_ref;
    std::vector<std::int64_t> hdr_bits;    // CE: (2474,) or (0,)
    std::vector<std::int64_t> hdr_spare;   // CE: (1,) or (0,)
    std::vector<std::int64_t> hdr_sym;     // L: (1237,) or (0,)
    std::vector<std::int64_t> data_ref;
    std::vector<std::int64_t> data;        // CE (n_data,); L (n_data/2,)
    std::vector<std::int64_t> ref;         // hdr_ref ++ data_ref (reference counter m)
    std::vector<std::int64_t> known_idx;   // preamble + refs (+ CE's spare), ascending
    std::vector<std::int8_t> known_val_ce;                // CE: +-1 at known_idx
    std::vector<std::complex<double>> known_val_l;        // L: QPSK at known_idx
    std::vector<std::int64_t> pos;         // (n_sym,) position of each stream symbol
    std::vector<std::array<std::int64_t, 2>> win_pos;    // first and last position of each window
    std::vector<std::uint8_t> is_ref;      // (n_sym,) 0/1
};

// The symbol layout of `spec` in its own waveform. Cached (the last 16
// specs, as the reference's lru_cache) and immutable; safe from any thread.
std::shared_ptr<const Layout> layout(const FrameSpec& spec);

// Refuse `spec` unless it is of `waveform`: `what` names the operation.
void require_waveform(const FrameSpec& spec, Waveform waveform, std::string_view what);

// --- assembling a frame -------------------------------------------------------

// hdr_bits as the reference receives them: an array whose first axis has
// `len` entries and, when it is 1-D (`one_d`), these float64 values. A null
// pointer is Python's None.
struct HdrBits {
    std::vector<double> values;
    std::size_t len = 0;
    bool one_d = true;
    static HdrBits of(std::span<const std::uint8_t> bits);
};

// The header-argument checks of assemble (both waveforms): bits required
// exactly when spec.has_header(), and then 2,474 values of 0/1.
void check_hdr_bits(const FrameSpec& spec, const HdrBits* bits);
// "x has shape S, expected (n_data,)" unless shape == (n_data,).
void check_x_shape(const FrameSpec& spec, std::span<const std::int64_t> shape);
// "y_sym has K symbols, expected n_sym" unless last_dim == n_sym.
void check_y_symbols(const FrameSpec& spec, std::int64_t last_dim);

// frame.spread_signs: float64[n_data] of h_i = w_i (1 - 2 bit[i mod 2474]).
// Any spec (CE and L share it); bits must be 1-D with 2,474 entries.
std::vector<double> spread_signs(const FrameSpec& spec, const HdrBits* bits);
// frame.spread_symbols (CE): sqrt(hdr_rho) h at the data symbols, 0 elsewhere.
std::vector<double> spread_symbols(const FrameSpec& spec, const HdrBits* bits);
// frame.assemble (CE): float64[n_sym], the frame's symbol values in stream order.
std::vector<double> assemble(const FrameSpec& spec, const HdrBits* bits,
                             std::span<const double> x);

// lframe.data_latents: x, or (x + sqrt(rho) h) / sqrt(1 + rho) with the
// spread header.
std::vector<double> data_latents(const FrameSpec& spec, std::span<const double> x,
                                 const HdrBits* bits);
// lframe.pair: values 2k and 2k+1 to I and Q of symbol k, scaled by 1/sqrt(2)
// (numpy's operation order: (v0 + 1j*v1) / sqrt(2) as a complex division).
std::vector<std::complex<double>> pair(std::span<const double> v);
// lframe.unpair: sqrt(2) Re, sqrt(2) Im, interleaved (elementwise, so a
// batch of rows is one flat call).
std::vector<double> unpair(std::span<const std::complex<double>> z);
// lframe.assemble (L): complex128[n_sym].
std::vector<std::complex<double>> assemble_l(const FrameSpec& spec, const HdrBits* bits,
                                             std::span<const double> x);

// Take columns `idx` of each row of `y` (rows of `row_len` values).
template <typename T>
std::vector<T> gather(std::span<const T> y, std::size_t row_len,
                      std::span<const std::int64_t> idx) {
    const std::size_t rows = row_len ? y.size() / row_len : 0;
    std::vector<T> out(rows * idx.size());
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t k = 0; k < idx.size(); ++k)
            out[r * idx.size() + k] = y[r * row_len + static_cast<std::size_t>(idx[k])];
    return out;
}

// frame.disassemble (CE): (header-symbol values (rows, 2474), data values
// (rows, n_data)) from stream-order rows of n_sym values.
template <typename T>
std::pair<std::vector<T>, std::vector<T>> disassemble(std::span<const T> y, const FrameSpec& spec) {
    require_waveform(spec, Waveform::CE, "frame.disassemble");
    const auto lay = layout(spec);
    const auto n = static_cast<std::size_t>(spec.n_sym());
    return {gather(y, n, lay->hdr_bits), gather(y, n, lay->data)};
}

// lframe.disassemble (L): (header bit values (rows, 2474), data latent values
// (rows, n_data)), each through unpair.
std::pair<std::vector<double>, std::vector<double>> disassemble_l(
    std::span<const std::complex<double>> y, const FrameSpec& spec);

}  // namespace sstvae::qrss::frame

#endif  // SSTVAE_QRSS_FRAME_HPP
