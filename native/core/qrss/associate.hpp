// Associating a pass with a picture: port of sstvae/qrss/associate.py,
// association rules 1 to 4 (design 8.2, spec section 9).
//
// A pass is attached to an accumulator by the first rule that succeeds:
// 1 "header" (its own header, validated by correlation once the segment is
// good), 2 "soft-header" (header LLRs summed with provisional clusters,
// then pairwise), 3 "corr" (latent correlation against every open
// accumulator), 4 "prov" (a provisional cluster). associate() returns the
// rule string, or "" when the pass (or another copy of its transmission,
// Store::twin) was a member already. **The strings are a GUI contract**:
// the listener writes "stored (<rule>)" into a tile's note, and
// native/gui/qrss_window.cpp's plain_note matches "stored (header)",
// "stored (soft-header)", "stored (corr)", "stored (already a member)" and
// "stored (provisional)".
//
// The statistic (D23): v = w W / (1 + w + W) over the latents where both
// weights are positive and every value finite, t = sum v z m / sqrt(sum
// (v z m)^2); accepted when t > 6 and t >= RHO_MIN * expected_t.
//
// Arithmetic: numpy's, operation for operation (sums pairwise as np.sum,
// -ffp-contract=off), so corr_stat, expected_t and every decision built on
// them are bit-identical with the reference -- except header_agrees, whose
// dot product numpy hands to BLAS (OpenBLAS's kernel order); here it is a
// pairwise sum, equal to the last few bits, which can change the decision
// only at an exact tie of a continuous statistic with its threshold.
//
// THREADS: associate() and resolve_provisional() hold the store's mutex
// (Store::hold) for the whole association, so an association is one
// transaction against other threads' use of the same Store; that is
// typically milliseconds per open accumulator and per provisional cluster
// once the passes are cached (rebuilds and file writes included), up to a
// second on a large store from cold. Everything else here is a pure
// function of its arguments, or reads the Store through its own calls.
// The soft-header seam is called with the store's mutex held.

#ifndef SSTVAE_QRSS_ASSOCIATE_HPP
#define SSTVAE_QRSS_ASSOCIATE_HPP

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/header.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::associate {

inline constexpr double T_ACCEPT = rxconfig::associate::T_ACCEPT;                  // 6
inline constexpr double RHO_MIN = rxconfig::associate::RHO_MIN;                    // 0.5
inline constexpr double KURTOSIS = rxconfig::associate::KURTOSIS;                  // 3
inline constexpr double VALIDATE_MEDIAN_W = rxconfig::associate::VALIDATE_MEDIAN_W;  // -10 dB
inline constexpr double HDR_REJECT_T = rxconfig::associate::HDR_REJECT_T;          // 3
inline constexpr double HDR_REJECT_EXPECT = rxconfig::associate::HDR_REJECT_EXPECT;  // 8
inline constexpr double PAIR_HZ = rxconfig::associate::PAIR_HZ;                    // 10
inline constexpr double PAIR_DAYS = rxconfig::associate::PAIR_DAYS;                // 7
inline constexpr double HDR_AGREE_Z = rxconfig::associate::HDR_AGREE_Z;            // 3
inline constexpr double HDR_OTHER_FLIP = rxconfig::associate::HDR_OTHER_FLIP;      // 0.3

// corr_stat's (t, M). `numpy` says the reference's t is an np.float64 (it
// is a Python float 0.0 when nothing correlates: no latent kept, or a zero
// denominator), which decides how a merge's log line writes "accepted"
// (np.bool_ through default=float as 1.0 / 0.0, a bool as true / false).
struct Corr {
    double t = 0.0;
    std::int64_t M = 0;
    bool numpy = false;
};

// The self-normalised correlation of a pass (z, w) with a reference (m, W),
// over the common prefix. std::invalid_argument (numpy's broadcast refusal)
// when w is shorter than that prefix of z, or W of m.
Corr corr_stat(std::span<const double> z, std::span<const double> w, std::span<const double> m,
               std::span<const double> W);
// sum v / sqrt(sum (v + k v^2)): the mean of t for the same picture.
double expected_t(std::span<const double> w, std::span<const double> W, double kurtosis = KURTOSIS);
// t > 6 and t >= RHO_MIN * t_exp.
bool corr_accept(double t, double t_exp);

// Accumulator segment g as (m = S/W, W) in air order, first n latents.
std::pair<std::vector<double>, std::vector<double>> acc_reference(const store::Accumulator& acc, std::int64_t g,
                                                                  std::int64_t n);
// A provisional cluster's air-order (m, W): its passes summed as one.
// std::invalid_argument "max() iterable argument is empty" for no uids.
std::pair<std::vector<double>, std::vector<double>> cluster_sums(store::Store& store,
                                                                 const std::vector<std::string>& uids);

// Rule 3's best accepted (t, key, segment) over the open accumulators.
struct Best {
    double t = 0.0;
    store::Key key;
    std::int64_t segment = 0;
};
std::optional<Best> best_accumulator(store::Store& store, std::span<const double> z, std::span<const double> w,
                                     const std::vector<store::Key>& exclude = {});

// Rule 1's correlation check: (ok, details), details a dict ({} when not
// checked; reason/acc_mode/mode for a mode mismatch; t/M/t_expected).
struct Validation {
    bool ok = true;
    pyjson::Value details = pyjson::Value::object();
};
Validation validate_header_match(store::Store& store, const store::Key& key, std::int64_t mode,
                                 std::int64_t segment, std::span<const double> z, std::span<const double> w);

// Attach each pass in `uids` under header h after rule 1's check; the uids
// attached (the others logged as collisions and kept provisional).
std::vector<std::string> attach_by_header(store::Store& store, const header::HeaderFields& h,
                                          const std::vector<std::string>& uids, const std::string& method);

// Whether a pass's own header LLRs carry header h rather than another
// (nullopt LLRs: none kept, which agree).
bool header_agrees(const header::HeaderFields& h, std::optional<std::span<const double>> llr);

// Rule 2: the header decoded from summed LLRs and the uids involved.
struct SoftHeader {
    header::HeaderFields h;
    std::vector<std::string> uids;
};
std::optional<SoftHeader> soft_header(store::Store& store, const std::string& uid, std::span<const double> llr);

// `rule`, or "" when uid is not a member of key's accumulator.
std::string rule(store::Store& store, const store::Key& key, std::string_view uid, const std::string& rule);

struct Result {
    std::optional<store::Key> key;   // nullopt for a provisional pass
    std::string rule;                // "header" | "soft-header" | "corr" | "prov" | ""
};

// Rule 2's search, replaceable (the binding hands a monkeypatched
// associate._soft_header through here); empty: soft_header.
using SoftHeaderFn =
    std::function<std::optional<SoftHeader>(store::Store& store, const std::string& uid, std::span<const double> llr)>;

// Attach pass p by the first rule that succeeds (storing the pass first
// when the store does not have it).
Result associate(const types::PassResult& p, store::Store& store, const SoftHeaderFn& soft = {});

// Attach whole provisional clusters that now correlate with an
// accumulator: [(cluster, key)] (cluster: the cluster's "cluster" value).
std::vector<std::pair<pyjson::Value, store::Key>> resolve_provisional(store::Store& store);

}  // namespace sstvae::qrss::associate

#endif  // SSTVAE_QRSS_ASSOCIATE_HPP
