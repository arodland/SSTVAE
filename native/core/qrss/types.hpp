// The receiver's shared records: port of sstvae/qrss/types.py (design 6.9).
//
// Not a format module: nothing here goes on the air. PassResult is the one
// record a single-pass receive produces and everything after it
// (association, accumulators, leave-one-out EM) consumes, so its fields and
// its .npz form are the contract between work packages -- and between the
// two implementations: a pass either one saves, the other loads with every
// field equal.
//
// Orders and scales follow design section 7: z and w are in air order,
// unscrambled and unprecoded; z is unbiased and w = 1/var.
//
// The .npz: members meta, z, w, hdr_llr, ch, psi, timing_cov, in that
// order, written by npz.hpp exactly as np.savez writes them; meta is the
// UTF-8 bytes (uint8) of json.dumps of a dict whose keys and nesting are
// PassResult.save's (pyjson.hpp writes json.dumps' text byte for byte).
// So a pass written here is the same file Python writes for the same
// values -- provided Python's values had the types this struct gives them:
// every float field a Python float (a Python int there would be written
// "0", where this writes "0.0"), z/w/hdr_llr/psi float32 and ch complex64
// (what the receiver builds; Python's save would store any dtype), and
// timing.cov float64 (save converts it).
//
// Loading is stricter than Python's in the same direction: z, w, hdr_llr
// and psi must be 1-D float32, ch 1-D complex64 and timing_cov float64
// with two dimensions (or no elements) -- std::invalid_argument naming the
// member otherwise -- where Python's load hands back whatever the file
// holds. Integer JSON fields must fit int64.

#ifndef SSTVAE_QRSS_TYPES_HPP
#define SSTVAE_QRSS_TYPES_HPP

#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "math/linalg.hpp"
#include "qrss/config.hpp"
#include "qrss/header.hpp"
#include "qrss/npz.hpp"
#include "qrss/rational.hpp"

namespace sstvae::qrss::types {

// Affine (+ optional quadratic) timing of one pass (design 6.4): position p
// is received at tau0 + p*T*(1 + ppm*1e-6) + gamma*(p/n_pos)^2 CH samples
// on the receiver's clock. cov is the estimate's covariance: 3x3 (tau0,
// ppm, gamma) from the tracker, 2x2 from A2, 0x0 when there is none.
struct Timing {
    double tau0 = 0.0;
    double ppm = 0.0;
    double gamma = 0.0;
    double z = 0.0;
    math::Matrix cov;
    bool operator==(const Timing&) const = default;
};

// A frequency trajectory: f_hz at times t_s (s after t0), with weights.
struct FreqPath {
    std::vector<double> t_s;
    std::vector<double> f_hz;
    std::vector<double> weight;
    bool operator==(const FreqPath&) const = default;
};

struct CarrierCandidate {
    double f_hz = 0.0;
    double drift_hz_per_min = 0.0;
    double metric = 0.0;
    std::optional<std::int64_t> waveform;   // a hint's waveform; nullopt: an A1 line (any)
    bool operator==(const CarrierCandidate&) const = default;
};

struct Detection {
    double f_hz = 0.0;
    FreqPath path;
    std::optional<Timing> timing;
    double z_ref = 0.0;
    std::string method;                      // "preamble" | "hint" | "tbd" | "template"
    double lead_in_s = 0.0;
    std::int64_t waveform = config::WAVEFORM_CE;   // which waveform's frame this candidate is
    double a3_z = std::numeric_limits<double>::quiet_NaN();   // a whole-slot (A3) candidate's line Z
};

struct TrackReport {
    double offset_hz = 0.0;
    double drift_hz_per_min = 0.0;
    double wander_hz_rms = 0.0;
    double doppler_hz = 0.0;
    double ppm = 0.0;
    double snr2500_db = 0.0;
    double z_ref = 0.0;
    double kappa = 0.0;
    bool slip_free = false;
    bool suspect = false;
    bool operator==(const TrackReport&) const = default;
};

struct CwIdResult {
    std::optional<std::string> text;
    double z_match = 0.0;
    std::string keying;                      // "fsk" | "ook"
    std::optional<bool> agrees;
    bool operator==(const CwIdResult&) const = default;
};

// Soft data symbols for one pass, in that pass's precoded-symbol domain.
struct EmPrior {
    std::vector<double> x_hat;   // float64[n_data]
    std::vector<double> v;       // float64[n_data]
    bool operator==(const EmPrior&) const = default;
};

// pass_uid(q, f_hz, z): f"{q}_{round(f_hz*1000)}_{sha8 of z's float32 bytes}".
// round is Python's (half to even); a NaN or infinite f_hz is
// std::invalid_argument / std::overflow_error as int() of it is in Python
// (ValueError / OverflowError), and so is a millihertz count outside int64.
std::string pass_uid(std::int64_t q, double f_hz, std::span<const float> z);

// Everything one pass of one signal delivered.
struct PassResult {
    std::string uid;
    std::int64_t q = 0;
    std::string frame;
    std::int64_t waveform = config::WAVEFORM_CE;   // 0 = CE
    double f_hz = 0.0;
    Timing timing;
    TrackReport report;
    std::vector<float> z;                    // float32[n_data], air order
    std::vector<float> w;                    // float32[n_data]
    std::vector<float> hdr_llr;              // float32[2474]
    std::optional<header::HeaderFields> header;
    std::optional<CwIdResult> cw;
    std::vector<std::complex<float>> ch;     // complex64 narrow capture, kept for EM
    std::int64_t ch_fs = 0;
    double ch_t0_index = 0.0;
    Rational f_mix_hz;
    std::vector<float> psi;                  // float32[n_pos]
    std::string estimator;
    std::int64_t em_round = 0;

    // json.dumps(meta) as PassResult.save builds it.
    std::string meta_json() const;
    // The members np.savez writes, in its order.
    npz::Npz to_npz() const;
    // The .npz file's bytes (npz_bytes(to_npz())).
    std::vector<std::uint8_t> bytes() const;
    // PassResult.save(path) / PassResult.load(path).
    void save(const std::string& path) const;
    static PassResult load(const std::string& path);
    // load from an archive already read (npz::parse_npz / load_npz).
    static PassResult from_npz(const npz::Npz& z);
    static PassResult parse(std::span<const std::uint8_t> file);

    bool operator==(const PassResult&) const = default;
};

}  // namespace sstvae::qrss::types

#endif  // SSTVAE_QRSS_TYPES_HPP
