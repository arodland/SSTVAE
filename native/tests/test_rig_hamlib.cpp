// The libhamlib backend, against Hamlib's own dummy rig.
//
// Model 1 is Hamlib's dummy: it opens, keys, and reports a frequency
// with no hardware attached. CLAUDE.md already relies on that trick for
// the Python side (`rigctld -m 1`), and it is what makes the real
// backend -- not a fake of it -- runnable in CI.
//
// What this cannot check is a *radio*: baud rates, CAT quirks, and how
// long a real K4 takes to key are the on-air shakedown's job. What it
// does check is that the library is linked correctly, that the
// configuration path works, that errors arrive as `RigError` with
// something readable in them, and that `list_models()` returns the
// structured data that replaced parsing `rigctld -l`.

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "check.hpp"
#include "rig/controller.hpp"
#include "rig/hamlib.hpp"

using namespace sstvae;

namespace {

void test_list_models() {
    const std::vector<rig::RigModel> models = rig::list_models();
    check::is_true(models.size() > 100,
                   "hamlib/list: Hamlib knows a few hundred rigs (" +
                       std::to_string(models.size()) + ")");

    const auto has = [&](int model) {
        return std::any_of(models.begin(), models.end(),
                           [&](const rig::RigModel& m) { return m.model == model; });
    };
    check::is_true(has(rig::MODEL_DUMMY), "hamlib/list: the dummy rig is listed");
    // The whole "share the radio with WSJT-X" story depends on this one
    // appearing in the picker, and with rig_list_foreach it is free --
    // where the reference had to parse it out of `rigctld -l` text.
    check::is_true(has(rig::MODEL_NET_RIGCTL),
                   "hamlib/list: NET rigctl is a model like any other");

    // Every entry usable as a picker row. The reference's column-slicing
    // parser silently dropped rows whose fields contained single spaces;
    // reading a struct cannot, and this asserts the result rather than
    // the method.
    for (const rig::RigModel& m : models) {
        if (m.name.empty() || m.label().empty()) {
            check::fail("hamlib/list: every model has a name and a label",
                        "model " + std::to_string(m.model) + " does not");
            return;
        }
    }
    check::is_true(true, "hamlib/list: every model has a name and a label");

    // Sorted by manufacturer then name, so the picker is navigable.
    check::is_true(std::is_sorted(models.begin(), models.end(),
                                  [](const rig::RigModel& a, const rig::RigModel& b) {
                                      if (a.manufacturer != b.manufacturer)
                                          return a.manufacturer < b.manufacturer;
                                      if (a.name != b.name) return a.name < b.name;
                                      return a.model < b.model;
                                  }),
                   "hamlib/list: sorted for display");

    // A manufacturer containing a space, which is the shape of the name
    // that broke the text parser ("N2ADR James Ahlstrom").
    const bool spaced = std::any_of(
        models.begin(), models.end(), [](const rig::RigModel& m) {
            return m.manufacturer.find(' ') != std::string::npos ||
                   m.name.find(' ') != std::string::npos;
        });
    check::is_true(spaced, "hamlib/list: names with spaces survive intact");
}

void test_version_is_reported() {
    const std::string v = rig::hamlib_version();
    check::is_true(!v.empty(), "hamlib/version: reported for bug reports (" + v + ")");
}

void test_dummy_rig_opens_keys_and_reports() {
    rig::HamlibConfig config;
    config.model = rig::MODEL_DUMMY;
    std::unique_ptr<rig::RigBackend> backend = rig::make_hamlib_backend(config);

    backend->open();
    check::is_true(!backend->description().empty(),
                   "hamlib/dummy: describes itself as " + backend->description());

    const double hz = backend->frequency_hz();
    check::is_true(hz > 0.0, "hamlib/dummy: reports a frequency");

    // The operation the whole subsystem exists to make safe.
    backend->set_ptt(true);
    backend->set_ptt(false);
    check::is_true(true, "hamlib/dummy: keys and unkeys");

    backend->close();
    // close() is noexcept and idempotent: the worker's exit path calls
    // it, and so does the destructor.
    backend->close();
    check::is_true(true, "hamlib/dummy: closing twice is safe");
}

void test_an_unknown_model_is_refused_readably() {
    rig::HamlibConfig config;
    config.model = 999999;  // not a model Hamlib has ever had
    std::unique_ptr<rig::RigBackend> backend = rig::make_hamlib_backend(config);

    bool threw = false;
    std::string message;
    try {
        backend->open();
    } catch (const rig::RigError& e) {
        threw = true;
        message = e.what();
    }
    check::is_true(threw, "hamlib/bad-model: refused");
    check::is_true(message.find("999999") != std::string::npos,
                   "hamlib/bad-model: the message names the model");
}

void test_using_a_closed_rig_reports_rather_than_crashes() {
    rig::HamlibConfig config;
    config.model = rig::MODEL_DUMMY;
    std::unique_ptr<rig::RigBackend> backend = rig::make_hamlib_backend(config);

    bool threw = false;
    try {
        backend->set_ptt(true);
    } catch (const rig::RigError&) {
        threw = true;
    }
    check::is_true(threw, "hamlib/closed: keying an unopened rig is an error");
}

void test_the_controller_drives_a_real_backend() {
    // The two halves together: the threading design over the real
    // library, which is the combination the app actually ships.
    std::vector<std::string> statuses;
    std::mutex m;
    std::condition_variable cv;
    bool got = false;

    rig::RigController controller(
        [&](std::optional<double> hz) {
            {
                std::lock_guard<std::mutex> lock(m);
                if (hz) got = true;
            }
            cv.notify_all();
        },
        [&](const std::string& text) {
            std::lock_guard<std::mutex> lock(m);
            statuses.push_back(text);
        });

    rig::HamlibConfig config;
    config.model = rig::MODEL_DUMMY;
    rig::RigConfig rc;
    rc.poll_interval_s = 1.0;
    controller.start(rig::make_hamlib_backend(config), rc);

    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return got; });
    }
    check::is_true(controller.frequency_hz().has_value(),
                   "hamlib/controller: a real backend polls through the controller");

    controller.set_ptt(true);
    controller.set_ptt(false);
    check::is_true(true, "hamlib/controller: and keys through it");

    controller.stop();
    check::is_true(!controller.running(), "hamlib/controller: stops cleanly");
}

}  // namespace

int main() {
    try {
        test_list_models();
        test_version_is_reported();
        test_dummy_rig_opens_keys_and_reports();
        test_an_unknown_model_is_refused_readably();
        test_using_a_closed_rig_reports_rather_than_crashes();
        test_the_controller_drives_a_real_backend();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("hamlib backend");
}
