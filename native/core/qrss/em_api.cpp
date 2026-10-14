#include "qrss/em_api.hpp"

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <limits>
#include <system_error>
#include <utility>

#include "qrss/decode.hpp"
#include "qrss/demod.hpp"
#include "qrss/em_refine.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
#include "qrss/receiver.hpp"
#include "util/fileio.hpp"

namespace sstvae::qrss::em_api {

namespace {

// Path.is_dir() as Python 3.13 decides it: false for a name the OS cannot
// spell (ValueError) and for the errors pathlib reads as "not there"
// (ENOENT, ENOTDIR, EBADF, ELOOP; on Windows ERROR_NOT_READY,
// ERROR_INVALID_NAME, ERROR_CANT_RESOLVE_FILENAME), its OSError otherwise.
bool pathlib_is_dir(const std::string& path) {
    const auto p = util::fs_path(path);
    if (!p) return false;
    std::error_code ec;
    const std::filesystem::file_status st = std::filesystem::status(*p, ec);
    if (ec) {
#ifdef _WIN32
        if (ec.category() == std::system_category() &&
            (ec.value() == 21 || ec.value() == 123 || ec.value() == 1921))
            return false;
#endif
        const int e = util::errno_of(ec);
        if (e == ENOENT || e == ENOTDIR || e == EBADF || e == ELOOP) return false;
        throw std::filesystem::filesystem_error("is_dir", *p, util::errno_code_of(ec));
    }
    return st.type() == std::filesystem::file_type::directory;
}

// refine's estimator: none or "" (each pass's own), else demod's refusal
// unless "joint" or "plain".
void check_estimator(const std::optional<std::string>& estimator) {
    if (estimator && !estimator->empty()) demod::check_estimator(*estimator);
}

double mean_w_now(store::Store& store, const store::Key& key, double otherwise) {
    try {
        return em::acc_mean_w_db(store.accumulator(key));
    } catch (const std::exception&) {
        return otherwise;   // the accumulator went (a hand edit, an expiry): what was last known
    }
}

}  // namespace

// --- the checks -------------------------------------------------------------------------------

void check_picture(const store::Store& store, const store::Key& key) {
    if (!store.has_accumulator(key))
        throw NoPicture("no picture " + key.callsign + " " + decode::py_hex(key.picture_id, 8) + " in " +
                        store.root());
}

std::string default_passband_dir(const store::Store& store) { return pypath::child(store.root(), "passband"); }

const frame::FrameSpec& retro_frame(std::string_view name) {
    const std::string shown = pytext::repr(name);
    if (name.substr(0, 2) == "l-") throw NotCeFrame("--frame " + shown + ": --retro searches for CE passes only");
    try {
        return frame::get(name);
    } catch (const std::invalid_argument&) {
        // sorted(frame.PRESETS)
        std::vector<std::string> names;
        for (const frame::FrameSpec& s : frame::presets(config::PresetGroup::Presets)) names.push_back(s.name());
        std::sort(names.begin(), names.end());
        std::string list;
        for (const std::string& n : names) list += (list.empty() ? "" : ", ") + n;
        throw UnknownFrame("--frame " + shown + ": not one of " + list);
    }
}

void check_passband_dir(const std::string& dir) {
    if (!pathlib_is_dir(dir)) throw NoPassbandStore("--retro: no passband store at " + dir);
}

// --- refine ------------------------------------------------------------------------------------

RefineResult refine(store::Store& store, const store::Key& key, const RefineOptions& opt, const em::Hooks& hooks) {
    check_picture(store, key);
    check_estimator(opt.estimator);

    em::EmRefineArgs args;
    args.rounds = std::min(opt.rounds, MAX_ROUNDS);
    args.tol_db = opt.tol_db;
    args.estimator = opt.estimator && !opt.estimator->empty() ? opt.estimator : std::nullopt;
    args.use_header = opt.use_header;

    RefineResult res;
    // The round boundaries a cancel would otherwise lose: the mean W before,
    // and after each round once the next one starts (read when its first
    // pass is re-received: em_refine itself writes nothing in between).
    std::vector<double> seen;
    std::int64_t round = 0;
    std::optional<RefinedPass> pending;
    if (args.rounds > 0) seen.push_back(em::acc_mean_w_db(store.accumulator(key)));

    em::EmSeams seams;
    seams.rereceive = [&](const types::PassResult& p, const types::EmPrior& prior,
                          const std::optional<header::HeaderFields>& hdr,
                          const std::optional<receiver::CwKnown>& cw_known,
                          const std::optional<std::string>& estimator, std::int64_t em_round,
                          const std::optional<types::FreqPath>& path) {
        if (em_round != round) {
            if (round > 0) seen.push_back(em::acc_mean_w_db(store.accumulator(key)));
            round = em_round;
        }
        // em_refine's own call (its default branch), unchanged
        em::RereceiveArgs ra;
        ra.hdr = hdr;
        ra.cw_known = cw_known;
        ra.estimator = estimator;
        ra.em_round = em_round;
        ra.path = path;
        em::Rereceived got = em::rereceive(p, prior, ra, hooks);
        pending.reset();
        if (got.pass)
            pending = RefinedPass{em_round, p.uid, receiver::mean_w_db(p), receiver::mean_w_db(*got.pass)};
        return got;
    };
    em::Hooks h = hooks;
    // progress comes once the pass is stored: that is what makes it one of `passes`
    h.progress = [&](std::int64_t done, std::int64_t total, const std::string& what) {
        if (pending && what.rfind("round ", 0) == 0) {
            res.passes.push_back(std::move(*pending));
            pending.reset();
        }
        if (hooks.progress) hooks.progress(done, total, what);
    };

    try {
        res.history = em::em_refine(store, key, args, h, seams);
        res.rounds_run = static_cast<std::int64_t>(res.history.size()) - 1;
        res.mean_w_db = res.history.back();
    } catch (const em::Cancelled&) {
        res.cancelled = true;
        res.history = std::move(seen);
        res.rounds_run = static_cast<std::int64_t>(res.history.size()) - 1;
        res.mean_w_db = mean_w_now(store, key, res.history.empty() ? -std::numeric_limits<double>::infinity()
                                                                   : res.history.back());
    }
    return res;
}

// --- retro -------------------------------------------------------------------------------------

RetroResult retro(store::Store& store, const store::Key& key, const Passband& pb, std::string_view frame_name,
                  const RetroOptions& opt, const em::Hooks& hooks) {
    // qrss_decode.py's order: the picture, the directory, the frame
    check_picture(store, key);
    std::string dir;
    if (!pb.open) {
        dir = pb.dir && !pb.dir->empty() ? pypath::pathlib_str(*pb.dir) : default_passband_dir(store);
        check_passband_dir(dir);
    }
    const frame::FrameSpec& spec = retro_frame(frame_name);
    // every name is checked here, "" included: retro has no "each pass's
    // own" (refine's ""), and the search would refuse "" at its first find
    demod::check_estimator(opt.estimator);
    std::shared_ptr<const passband::PassbandStore> store_pb = pb.open;
    if (!store_pb) store_pb = std::make_shared<const passband::PassbandStore>(dir);

    RetroResult res;
    em::RetroSteps steps = em::retro_steps(store, key, store_pb.get(), spec, opt, hooks);
    // what the loop does not hand back: the slot in hand and its search's Z
    std::int64_t q_now = 0;
    double z_now = 0.0;
    std::int64_t seg_now = 0;
    if (opt.slots) {
        res.slots_total = static_cast<std::int64_t>(opt.slots->size());
    } else {
        auto slots = steps.slots;
        steps.slots = [&, slots](double dur) {
            std::vector<std::int64_t> qs = slots(dur);
            res.slots_total = static_cast<std::int64_t>(qs.size());
            return qs;
        };
    }
    auto capture = steps.capture;
    steps.capture = [&, capture](std::int64_t q, double dur) {
        q_now = q;
        return capture(q, dur);
    };
    auto search = steps.search;
    steps.search = [&, search](const em::SlotInputPtr& in) {
        ++res.slots_searched;
        em::SlotSearch r = search(in);
        z_now = r.z;
        seg_now = r.segment.value_or(0);
        return r;
    };
    auto assoc = steps.associate;
    steps.associate = [&, assoc](const types::PassResult& p) {
        associate::Result a = assoc(p);
        res.finds.push_back(RetroEntry{q_now, p.f_hz, z_now, seg_now, a.key, a.rule, p.uid, receiver::mean_w_db(p)});
        return a;
    };
    em::Hooks h = hooks;
    h.progress = [&](std::int64_t done, std::int64_t total, const std::string& what) {
        res.slots_done = done;
        res.slots_total = total;
        if (hooks.progress) hooks.progress(done, total, what);
    };

    try {
        em::retro_detect_steps(store, key, spec, steps, opt.slots, h);
    } catch (const em::Cancelled&) {
        res.cancelled = true;
    }
    if (!res.cancelled) res.slots_done = res.slots_total;
    return res;
}

}  // namespace sstvae::qrss::em_api
