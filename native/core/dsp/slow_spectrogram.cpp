#include "dsp/slow_spectrogram.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <utility>

#include "dsp/fft.hpp"

namespace sstvae::dsp {

namespace {

constexpr double FS = 8000.0;

const std::vector<double>& window() {
    static const std::vector<double> w = [] {
        std::vector<double> v(SlowSpectrogram::NFFT);
        for (int i = 0; i < SlowSpectrogram::NFFT; ++i) {
            v[i] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i / (SlowSpectrogram::NFFT - 1));
        }
        return v;
    }();
    return w;
}

// Power of a unit-amplitude sine through the window, so a column reads
// in "amplitude squared" whatever NFFT is.
double power_scale() {
    static const double s = [] {
        double sum = 0.0;
        for (double v : window()) sum += v;
        return 4.0 / (sum * sum);
    }();
    return s;
}

// The columns of `d` whose [t, t + len) overlaps [t0, t1), each with
// the fraction of it that lies inside the range.
template <class Col, class Fn>
int each_overlapping(const std::deque<Col>& d, double len, double t0, double t1, Fn fn) {
    auto it = std::lower_bound(d.begin(), d.end(), t0 - len,
                               [](const Col& c, double t) { return c.t < t; });
    int n = 0;
    for (; it != d.end() && it->t < t1; ++it) {
        if (it->t + len <= t0) continue;
        const double inside = std::min(t1, it->t + len) - std::max(t0, it->t);
        fn(*it, std::clamp(inside / len, 0.0, 1.0));
        ++n;
    }
    return n;
}

}  // namespace

void SlowSpectrogram::restart() { pending_.clear(); }

void SlowSpectrogram::clear() {
    pending_.clear();
    fine_.clear();
    coarse_.clear();
    block_sum_.clear();
    block_t_ = -1.0;
    block_n_ = 0;
}

void SlowSpectrogram::push(const std::vector<double>& samples, double end_time) {
    if (samples.empty()) return;
    const double start = end_time - static_cast<double>(samples.size()) / FS;
    // Audio that does not follow on from what is pending: a gap.
    if (!pending_.empty() && std::abs(start - pending_end_) > 0.5) pending_.clear();
    pending_.insert(pending_.end(), samples.begin(), samples.end());
    pending_end_ = end_time;

    const std::vector<double>& w = window();
    const double scale = power_scale();
    while (static_cast<int>(pending_.size()) >= NFFT) {
        const double t = pending_end_ - static_cast<double>(pending_.size()) / FS;
        std::vector<cdouble> buf(NFFT);
        for (int i = 0; i < NFFT; ++i) buf[i] = pending_[static_cast<std::size_t>(i)] * w[i];
        const std::vector<cdouble> spec = fft(buf, true);
        Column c;
        c.t = std::round(t * 1000.0) / 1000.0;
        c.power.resize(BINS);
        for (int k = 0; k < BINS; ++k) c.power[k] = static_cast<float>(std::norm(spec[k]) * scale);
        add_fine(std::move(c));
        pending_.erase(pending_.begin(), pending_.begin() + HOP);
    }
}

void SlowSpectrogram::add_fine(Column column) {
    const double block = std::floor(column.t / COARSE_S) * COARSE_S;
    if (block != block_t_) {
        flush_coarse();
        block_t_ = block;
        block_sum_.assign(BINS, 0.0);
        block_n_ = 0;
    }
    for (int k = 0; k < BINS; ++k) block_sum_[k] += column.power[k];
    ++block_n_;
    if (!fine_.empty() && column.t <= fine_.back().t) fine_.clear();   // the clock went back
    FineColumn f;
    f.t = column.t;
    f.level.resize(column.power.size());
    for (std::size_t k = 0; k < column.power.size(); ++k) f.level[k] = quantize(column.power[k]);
    fine_.push_back(std::move(f));
    const double newest = fine_.back().t;
    while (!fine_.empty() && fine_.front().t < newest - fine_keep_s_) fine_.pop_front();
    while (!coarse_.empty() && coarse_.front().t < newest - COARSE_KEEP_S) coarse_.pop_front();
}

void SlowSpectrogram::flush_coarse() {
    if (block_n_ == 0) return;
    Column c;
    c.t = block_t_;
    c.power.resize(BINS);
    for (int k = 0; k < BINS; ++k) c.power[k] = static_cast<float>(block_sum_[k] / block_n_);
    if (!coarse_.empty() && c.t <= coarse_.back().t) coarse_.clear();
    coarse_.push_back(std::move(c));
    block_n_ = 0;
}

namespace {

const std::array<float, 256>& level_power() {
    static const std::array<float, 256> lut = [] {
        std::array<float, 256> v{};
        for (int i = 0; i < 256; ++i) {
            v[static_cast<std::size_t>(i)] = static_cast<float>(
                std::pow(10.0, (SlowSpectrogram::DB_MIN + i * SlowSpectrogram::DB_STEP) / 10.0));
        }
        return v;
    }();
    return lut;
}

}  // namespace

std::uint8_t SlowSpectrogram::quantize(float power) {
    if (!(power > 0.0f)) return 0;
    const double l = std::round((10.0 * std::log10(static_cast<double>(power)) - DB_MIN) / DB_STEP);
    return static_cast<std::uint8_t>(std::clamp(l, 0.0, 255.0));
}

float SlowSpectrogram::dequantize(std::uint8_t level) { return level_power()[level]; }

std::vector<float> SlowSpectrogram::FineColumn::powers() const {
    std::vector<float> out(level.size());
    for (std::size_t k = 0; k < level.size(); ++k) out[k] = dequantize(level[k]);
    return out;
}

std::vector<float> SlowSpectrogram::mean_between(double t0, double t1, double* seconds) const {
    // Every column under the range is summed in power, weighted by how
    // much of it lies inside, and divided by the total weight: a column
    // straddling two pixels counts half in each rather than whole in
    // both.
    std::vector<double> sum;
    double weight = 0.0;   // sum of the fractions
    double var = 0.0;      // sum of fraction^2 / seconds, for the noise
    const auto add = [&](const Column& c, double f) {
        if (sum.empty()) sum.assign(c.power.size(), 0.0);
        for (std::size_t k = 0; k < c.power.size(); ++k) sum[k] += f * c.power[k];
        weight += f;
        var += f * f / COARSE_S;
    };
    const std::array<float, 256>& lut = level_power();
    const auto add_fine = [&](const FineColumn& c, double f) {
        if (sum.empty()) sum.assign(c.level.size(), 0.0);
        for (std::size_t k = 0; k < c.level.size(); ++k) sum[k] += f * lut[c.level[k]];
        weight += f;
        var += f * f / COLUMN_S;
    };
    int columns = 0;
    const bool fine_reaches = !fine_.empty() && t0 >= fine_.front().t - COLUMN_S;
    if (!fine_reaches || t1 - t0 >= COARSE_S) {
        columns = each_overlapping(coarse_, COARSE_S, t0, t1, add);
    }
    // Recent enough for the fine store, or the block still filling,
    // which is not in the coarse one yet.
    if (columns == 0 && !fine_.empty()) {
        columns = each_overlapping(fine_, COLUMN_S, t0, t1, add_fine);
    }
    if (columns == 0 || weight <= 0.0) {
        if (seconds != nullptr) *seconds = 0.0;
        return {};
    }
    // How many seconds of audio the weighted mean is worth, for its
    // noise: the whole column when the range is inside one, the sum
    // when it holds several whole.
    if (seconds != nullptr) *seconds = weight * weight / var;
    std::vector<float> out(sum.size());
    for (std::size_t k = 0; k < sum.size(); ++k) out[k] = static_cast<float>(sum[k] / weight);
    return out;
}

// --- time scale -------------------------------------------------------------

TimeScale::TimeScale(double lens_spp, double history_spp, int lens_px)
    : lens_spp_(std::clamp(lens_spp, MIN_SPP, MAX_SPP)),
      history_spp_(std::clamp(history_spp, MIN_SPP, MAX_SPP)),
      lens_px_(std::max(1, lens_px)) {
    rebuild();
}

void TimeScale::set_lens_spp(double spp) {
    lens_spp_ = std::clamp(spp, MIN_SPP, MAX_SPP);
    rebuild();
}

void TimeScale::set_history_spp(double spp) {
    history_spp_ = std::clamp(spp, MIN_SPP, MAX_SPP);
    rebuild();
}

double TimeScale::spp_at(double x) const {
    if (x <= 0) return lens_spp_;
    if (x >= lens_px_) return history_spp_;
    const double u = x / lens_px_;
    const double s = u * u * (3.0 - 2.0 * u);
    return std::exp(std::log(lens_spp_) + (std::log(history_spp_) - std::log(lens_spp_)) * s);
}

void TimeScale::rebuild() {
    // The integral of spp, by the midpoint rule at a quarter pixel.
    age_.assign(static_cast<std::size_t>(lens_px_) + 1, 0.0);
    double a = 0.0;
    for (int x = 0; x < lens_px_; ++x) {
        for (int q = 0; q < 4; ++q) a += 0.25 * spp_at(x + (q + 0.5) * 0.25);
        age_[static_cast<std::size_t>(x) + 1] = a;
    }
}

double TimeScale::age_at(double x) const {
    if (x <= 0) return 0.0;
    if (x >= lens_px_) return age_.back() + history_spp_ * (x - lens_px_);
    const int i = static_cast<int>(x);
    const double f = x - i;
    return age_[static_cast<std::size_t>(i)] +
           f * (age_[static_cast<std::size_t>(i) + 1] - age_[static_cast<std::size_t>(i)]);
}

double TimeScale::x_of(double age) const {
    if (age <= 0) return 0.0;
    if (age >= age_.back()) return lens_px_ + (age - age_.back()) / history_spp_;
    const auto it = std::upper_bound(age_.begin(), age_.end(), age);
    const std::size_t i = static_cast<std::size_t>(it - age_.begin()) - 1;
    return static_cast<double>(i) + (age - age_[i]) / (age_[i + 1] - age_[i]);
}

float median_of(std::vector<float> values) {
    if (values.empty()) return 0.0f;
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

}  // namespace sstvae::dsp
