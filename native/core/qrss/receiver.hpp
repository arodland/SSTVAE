// The single-pass receiver: the per-pass half of sstvae/qrss/receiver.py,
// the reference (design 6.1, 6.8; WP6).
//
// Not a format module: nothing here goes on the air. Constants are
// rx_config.hpp's (receiver.py's, generated) and config.hpp's.
//
//   receiver.py              here
//   Prepared, prepare        Prepared, prepare
//   frames, is_joint,        frames, is_joint, single
//   _single
//   centre_hz, _ce_hints     centre_hz, ce_hints (_centres is the binding's:
//                            it is about Python's number types)
//   channel_for              channel_for
//   _psi_frame               psi_frame
//   _cw_result               cw_result
//   receive_pass,            receive_pass (one function: it dispatches on
//   _receive_pass_l          the frame's waveform, as the reference does)
//   decode_header,           decode_header, decode_header_l
//   _decode_header_l
//   detection_from_pass      detection_from_pass
//   capture_from_pass        capture_from_pass
//   mean_w_db, confidence_db mean_w_db, confidence_db
//
// The slot half (candidates, detect, receive_slot, the ghost and alias
// rules, the joint receiver's gate) is the next port's.
//
// Seams. receive_pass calls, through the module, names a test may replace:
// receiver.prepare, receiver.channel_for, track.verify, ltrack.verify_l and
// (under every tracker, the callsign reader and the report) the front end's
// noise_psd. Each is a std::function in ReceiverSeams; an empty one is the
// C++ here, so a caller with no Python gets the C++ throughout. The binding
// fills a seam only where the module attribute is not the native row.
//
// Exactness. receive_pass is a composition: everything it computes itself
// (the channel's span, the header decision, the callsign verdict, the
// PassResult's fields) is the reference's arithmetic in the reference's
// order, so with the same primitives underneath it is the reference to the
// bit (the parity suite's composition test). What the primitives agree to
// is theirs (track, ltrack, demod, cwid). decode_header, confidence_db and
// mean_w_db are bitwise: numpy's pairwise mean, np.bincount's running sums,
// np.linspace's i * step, libm's log10, and Python's round(x, 1) as the
// correctly rounded "%.1f" read back.
//
// Errors are the reference's: std::invalid_argument for ValueError (the
// waveform refusal and frames' refusals, texts included), and whatever the
// primitives raise.
//
// Thread-safe: no state of its own; the seams are called on the caller's
// thread.

#ifndef SSTVAE_QRSS_RECEIVER_HPP
#define SSTVAE_QRSS_RECEIVER_HPP

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "qrss/acquire.hpp"
#include "qrss/chan.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/header.hpp"
#include "qrss/rational.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::receiver {

inline constexpr double MERGE_HZ = rxconfig::receiver::MERGE_HZ;
inline constexpr double MARGIN_S = rxconfig::receiver::MARGIN_S;
inline constexpr double L_EXCLUDE_OFFSET_HZ = rxconfig::receiver::L_EXCLUDE_OFFSET_HZ;
inline constexpr int CONF_BINS = rxconfig::receiver::CONF_BINS;
inline constexpr double CONF_FLOOR_DB = rxconfig::receiver::CONF_FLOOR_DB;

// --- the capture --------------------------------------------------------------------------

// A capture after the blanker and inverse AGC (FE'), with the keep indicator;
// or, at fs = CH_FS, a stored pass's CH capture as it is (keep none).
struct Prepared {
    std::vector<std::complex<float>> fe;
    std::optional<std::vector<float>> keep;
    std::int64_t q = 0;
    double t0_index = 0.0;
    std::int64_t fs = config::FE_FS;
};

// Python's `cap`: a raw frontend.Capture or a Prepared one. Not owned.
using CaptureRef = std::variant<const frontend::Capture*, const Prepared*>;

// prepare(cap) of a raw capture: blanked and normalised (the front end's
// defaults at cap.fs), or at CH_FS taken as it is.
Prepared prepare(const frontend::Capture& cap);
// prepare(cap): a Prepared passes through (the result then points at it and
// owns nothing: it must outlive the result).
std::shared_ptr<const Prepared> prepare(CaptureRef cap);

// --- which frames, and where a signal is ----------------------------------------------------

// frames(spec): the frames of a selection, CE's first. A tuple (list) of
// frames goes through acquire's frame_map (its ValueErrors); one frame is
// itself.
std::vector<frame::FrameSpec> frames(std::span<const frame::FrameSpec> specs);
std::vector<frame::FrameSpec> frames(const frame::FrameSpec& spec);
// is_joint: the selection holds an L frame.
bool is_joint(std::span<const frame::FrameSpec> specs);
bool is_joint(const frame::FrameSpec& spec);
// _single: frames(spec)[0] of a tuple (one frame: itself).
frame::FrameSpec single(std::span<const frame::FrameSpec> specs);

// centre_hz: a signal's band centre -- f_hz for CE, the pilot plus
// L_EXCLUDE_OFFSET_HZ for L.
double centre_hz(double f_hz, std::int64_t waveform = config::WAVEFORM_CE);
double centre_hz(const types::Detection& d);
double centre_hz(const types::PassResult& p);

// _ce_hints: the hints the CE receiver takes -- unchanged when none names a
// waveform; otherwise the CE ones as bare frequencies and the others left
// out (bare ones kept as they are).
std::vector<acquire::Hint> ce_hints(std::span<const acquire::Hint> hints);

// --- one pass ---------------------------------------------------------------------------------

// channel_for(prep, spec, det, exclude_hz, f_mix_hz): the detection's CH
// channel over its keyed span plus margins, mixed at the path's frequency
// at the frame's middle; a CH-rate capture as it is, mixed at f_mix_hz
// (default the detection's frequency on the mix grid).
track::ChanCapture channel_for(const Prepared& prep, const frame::FrameSpec& spec, const types::Detection& det,
                               std::span<const double> exclude_hz = {},
                               const std::optional<Rational>& f_mix_hz = std::nullopt);

// _psi_frame: psi at positions 0 .. n_pos - 1, float32.
std::vector<float> psi_frame(const track::TrackResult& tr);

// _cw_result: (CwIdResult, cw_known for round B) from round A's tracker.
// Both empty for a frame without callsign windows.
struct CwKnown {
    std::vector<std::uint8_t> units;   // morse.keying_units(call)
    bool ook = false;
};
struct CwOutcome {
    std::optional<types::CwIdResult> cw;
    std::optional<CwKnown> known;
};
CwOutcome cw_result(const track::TrackResult& tr, const frame::FrameSpec& spec,
                    const std::optional<header::HeaderFields>& hdr, const track::NoisePsdFn& noise_psd = {});

// decode_header(llr, diag, spec) after round A: the header, which LLRs to
// keep (`block`: diag's llr_block, in llr's dtype; else llr itself), and
// whether to go on with frame::plain(spec) (a version 1 header from the
// block). llr_spread / llr_block are diag's (null: None, or absent). An L
// frame is decode_header_l's (version 2 only; llr and spec kept).
struct HeaderDecode {
    std::optional<header::HeaderFields> header;
    bool block = false;
    bool plain = false;
};
HeaderDecode decode_header(std::span<const double> llr, const std::vector<double>* llr_spread,
                           const std::vector<double>* llr_block, const frame::FrameSpec& spec);
std::optional<header::HeaderFields> decode_header_l(std::span<const double> llr, const std::vector<double>* llr_spread,
                                                    const std::vector<double>* llr_block,
                                                    const frame::FrameSpec& spec);

// receive_pass's refusal of a detection of the other waveform:
// "a waveform <name> detection cannot be received as frame '<name>'
// (waveform <name>): pass lframe.twin(spec, det.waveform)", with the
// detection's waveform named as WAVEFORM_NAMES.get(wf, wf) names it.
std::string waveform_refusal(std::string_view det_waveform_name, const frame::FrameSpec& spec);
// WAVEFORM_NAMES.get(wf, wf) for an integer wf.
std::string waveform_name(std::int64_t wf);

// The names receive_pass calls through its module that a caller may
// replace. Empty: the C++.
struct ReceiverSeams {
    std::function<std::shared_ptr<const Prepared>(CaptureRef cap)> prepare;
    std::function<std::shared_ptr<const track::ChanCapture>(
        const Prepared& prep, const frame::FrameSpec& spec, const types::Detection& det,
        std::span<const double> exclude_hz, const std::optional<Rational>& f_mix_hz)>
        channel_for;
    // track.verify(chan, CH_FS, spec, det, erase_s=erase_s)
    std::function<std::optional<types::Detection>(std::shared_ptr<const track::ChanCapture> chan,
                                                  const frame::FrameSpec& spec, const types::Detection& det,
                                                  std::span<const std::pair<double, double>> erase_s)>
        verify;
    // ltrack.verify_l(chan, CH_FS, spec, det, erase_s=erase_s)
    std::function<std::optional<types::Detection>(std::shared_ptr<const track::ChanCapture> chan,
                                                  const frame::FrameSpec& spec, const types::Detection& det,
                                                  std::span<const std::pair<double, double>> erase_s)>
        verify_l;
    // frontend.noise_psd, wherever the trackers, the reader and the report
    // fill a channel's noise table
    track::NoisePsdFn noise_psd;
};

// receive_pass's keyword arguments.
struct PassArgs {
    std::optional<types::EmPrior> prior;
    std::string estimator = "joint";
    std::vector<double> exclude_hz;
    std::optional<Rational> f_mix_hz;
    bool round_b = true;
    bool return_tracks = false;
    std::vector<std::pair<double, double>> erase_s;
};

// What receive_pass made: the pass (nullopt when gate V refused an
// unverified detection), and with return_tracks round A's tracker and,
// after a header, round B's (an LTrackResult each for an L frame). `plain`:
// the pass went on as frame::plain(spec) (a version 1 header); `chan` is
// the channel every track ran on.
struct PassOutcome {
    std::optional<types::PassResult> pass;
    std::vector<std::shared_ptr<track::TrackResult>> tracks;
    std::shared_ptr<const track::ChanCapture> chan;
    bool plain = false;
};

// receive_pass(cap, spec, det, prior, estimator, *, exclude_hz, f_mix_hz,
// round_b, return_tracks, erase_s) (design 6.8). std::invalid_argument
// (waveform_refusal) for a detection of the other waveform, before
// anything else is touched.
PassOutcome receive_pass(CaptureRef cap, const frame::FrameSpec& spec, const types::Detection& det,
                         const PassArgs& args = {}, const ReceiverSeams& seams = {});

// --- a stored pass -----------------------------------------------------------------------------

// detection_from_pass(p, lead_in_s): a detection reproducing p's frequency
// and timing (for EM re-receives).
types::Detection detection_from_pass(const types::PassResult& p, double lead_in_s = 0.0);
// capture_from_pass(p): the CH capture a pass keeps, as receive_pass input.
Prepared capture_from_pass(const types::PassResult& p);

// mean_w_db: 10 log10(max(mean(w), 1e-30)), numpy's mean in float64.
double mean_w_db(std::span<const double> w);
double mean_w_db(const types::PassResult& p);

// confidence_db(p, spec, elapsed_s, n): per-latent SNR in dB over n bins
// from t0 to the frame's keyed end; nullopt where there is nothing to say.
// `spec` is the frame already resolved (the binding's getattr logic); w,
// hdr_llr (null: None) and psi (null: None) are the pass's.
// std::invalid_argument for n < 0 (np.linspace's refusal) and for n = 0
// with anything to place (np.bincount's).
std::vector<std::optional<double>> confidence_db(const frame::FrameSpec& spec, std::span<const double> w,
                                                 const std::vector<double>* hdr_llr, const std::vector<double>* psi,
                                                 std::optional<double> elapsed_s = std::nullopt,
                                                 std::int64_t n = CONF_BINS);
// The PassResult form: spec null, or of another waveform than the pass's,
// is the pass's own frame (lframe.get_any(p.frame)).
std::vector<std::optional<double>> confidence_db(const types::PassResult& p, const frame::FrameSpec* spec = nullptr,
                                                 std::optional<double> elapsed_s = std::nullopt,
                                                 std::int64_t n = CONF_BINS);

}  // namespace sstvae::qrss::receiver

#endif  // SSTVAE_QRSS_RECEIVER_HPP
