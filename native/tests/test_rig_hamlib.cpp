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
//
// It also carries the one claim the Android rig support rests on: that
// a *native* Hamlib backend -- a Kenwood, not model 2 -- will speak its
// own CAT protocol over a socket, so a device that can never be a
// `/dev/ttyUSB0` is still every radio Hamlib knows. See
// "a Kenwood on the end of a SerialTransport" below.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "rig/bridged.hpp"
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
        [&](const std::string& text, bool /*error*/) {
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
    // Do not return from main with a detached worker still inside
    // libhamlib: see wait_for_shutdown's comment on why that is a hang
    // on Windows rather than merely untidy.
    check::is_true(controller.wait_for_shutdown(),
                   "hamlib/controller: the worker finishes closing the rig");
}

}  // namespace

// Announce each step on stderr, unbuffered, and publish it for the
// watchdog.
//
// This test drives a real library that opens ports and starts threads,
// so its plausible failure mode is not "wrong answer" but "never
// --- a Kenwood on the end of a SerialTransport ------------------------------
//
// **This is the evidence the whole Android CAT design rests on**, and
// it is why it is here rather than in `test_rig_bridge.cpp`: everything
// else about the bridge is checked against a stub, but the claim being
// made is about *Hamlib*, so nothing short of the real library
// establishes it.
//
// The claim: Hamlib will speak a native backend's own CAT protocol over
// a socket, so a phone -- which can never hand Hamlib a `/dev/ttyUSB0`
// -- still reaches every radio Hamlib supports. `rig_open()` puts the
// configured pathname through `parse_hoststr()` and, on a `host:port`
// match, sets `RIG_PORT_NETWORK` for **any** model. That is the
// mechanism behind `rigctld -m <native model> -r <ser2net host>:4001`,
// and `docs/android.md` was wrong to conclude that Hamlib's serial
// layer made rig control structurally impossible there.
//
// So: model 2014 is a Kenwood TS-2000, a genuine serial backend with no
// idea it is not on a wire. It gets a `LoopbackBridge` in front of a
// transport that answers Kenwood commands, and the assertions are on
// what the *radio* received -- because "Hamlib returned the frequency"
// could be satisfied by a cache, while `FA;` arriving at the far end
// could not.
namespace {

constexpr int MODEL_TS2000 = 2014;

// The radio's side of the link, kept alive independently of the
// transport so the test can still read it after the backend has taken
// ownership.
struct Kenwood {
    std::mutex m;
    std::condition_variable cv;
    bool closed = false;

    long long freq = 14'074'000;
    std::vector<std::string> commands;   // every command, in order
    std::deque<char> pending;            // bytes waiting to go back
    int dtr = -1;                        // -1 = never set
    int rts = -1;

    void reply(const std::string& text) {
        for (char c : text) pending.push_back(c);
        cv.notify_all();
    }

    // Called with `m` held.
    void handle(const std::string& command) {
        commands.push_back(command);
        if (command == "ID") {
            reply("ID019;");  // the TS-2000's identifier
        } else if (command == "FA") {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "FA%011lld;", freq);
            reply(buf);
        } else if (command == "IF") {
            // **Exactly 37 characters before the terminator.** Kenwood
            // backends default `caps->if_len` to 37 and reject anything
            // else as a protocol error, which is what an approximately
            // right reply cost here: `rig_get_freq` failed inside
            // `kenwood_get_vfo_if` having already read the frequency
            // correctly. The tail is Hamlib's own `simulators/`
            // canonical reply with our frequency in the first field.
            char buf[64];
            std::snprintf(buf, sizeof(buf),
                          "IF%011lld1000+0000000000030000000;", freq);
            reply(buf);
        } else if (command == "PS") {
            reply("PS1;");
        } else if (command == "TX" || command == "RX") {
            // A real TS-2000 answers neither; the state change is the
            // whole reply.
        } else {
            reply("?;");  // what a Kenwood says to a command it lacks
        }
    }

    std::vector<std::string> log() {
        std::lock_guard<std::mutex> lock(m);
        return commands;
    }

    bool saw(const std::string& want) {
        std::lock_guard<std::mutex> lock(m);
        return std::find(commands.begin(), commands.end(), want) != commands.end();
    }

    bool wait_for(const std::string& want, double seconds = 5.0) {
        std::unique_lock<std::mutex> lock(m);
        return cv.wait_for(lock, std::chrono::duration<double>(seconds), [&] {
            return std::find(commands.begin(), commands.end(), want) != commands.end();
        });
    }
};

class KenwoodTransport : public rig::SerialTransport {
public:
    explicit KenwoodTransport(std::shared_ptr<Kenwood> radio) : radio_(std::move(radio)) {}

    void open() override {
        std::lock_guard<std::mutex> lock(radio_->m);
        radio_->closed = false;
    }

    void close() noexcept override {
        {
            std::lock_guard<std::mutex> lock(radio_->m);
            radio_->closed = true;
        }
        radio_->cv.notify_all();
    }

    std::size_t read(std::uint8_t* dst, std::size_t n, int timeout_ms) override {
        std::unique_lock<std::mutex> lock(radio_->m);
        radio_->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return !radio_->pending.empty() || radio_->closed;
        });
        if (radio_->closed) return 0;
        std::size_t i = 0;
        while (i < n && !radio_->pending.empty()) {
            dst[i++] = static_cast<std::uint8_t>(radio_->pending.front());
            radio_->pending.pop_front();
        }
        return i;
    }

    void write(const std::uint8_t* src, std::size_t n) override {
        {
            std::lock_guard<std::mutex> lock(radio_->m);
            for (std::size_t i = 0; i < n; i++) {
                const char c = static_cast<char>(src[i]);
                if (c == ';') {
                    radio_->handle(partial_);
                    partial_.clear();
                } else {
                    partial_.push_back(c);
                }
            }
        }
        radio_->cv.notify_all();
    }

    void set_dtr(bool on) override {
        std::lock_guard<std::mutex> lock(radio_->m);
        radio_->dtr = on ? 1 : 0;
    }
    void set_rts(bool on) override {
        std::lock_guard<std::mutex> lock(radio_->m);
        radio_->rts = on ? 1 : 0;
    }

    std::string description() const override { return "USB CP2102"; }

private:
    std::shared_ptr<Kenwood> radio_;
    // Only touched from the bridge's single pump-in thread, and always
    // under the radio's lock in `write`.
    std::string partial_;
};

std::unique_ptr<rig::RigBackend> bridged(rig::HamlibConfig config,
                                         const std::shared_ptr<Kenwood>& radio) {
    // A real Kenwood answers `TX;` and `RX;` with nothing at all, and
    // this fake is faithful about that -- so Hamlib waits out its read
    // timeout on every keying command. At the stock 1000 ms that is
    // most of these tests' runtime, and a slow test is a test whose
    // watchdog has to be loose enough to be useless. The link here is a
    // loopback socket to an in-process fake, so 200 ms is still two
    // orders of magnitude more than it can need.
    config.timeout_ms = 200;
    return rig::make_bridged_backend(
        config, std::make_shared<KenwoodTransport>(radio),
        [](const rig::HamlibConfig& c) { return rig::make_hamlib_backend(c); });
}



void test_a_native_backend_works_over_a_socket() {
    auto radio = std::make_shared<Kenwood>();
    rig::HamlibConfig config;
    config.model = MODEL_TS2000;
    config.ptt_method = rig::PttMethod::Cat;
    // Deliberately not a path and not a host: on Android the operator
    // picks a device from a list and this is its identifier. What the
    // bridge hands Hamlib is a loopback address instead.
    config.device = "usb:10c4:ea60";

    auto backend = bridged(config, radio);
    backend->open();

    // Hamlib opened a *Kenwood* over a socket. `ID;` is the TS-2000
    // backend's own handshake, so its arrival is the proof: nothing in
    // the bridge knows what a Kenwood is.
    check::is_true(radio->saw("ID"),
                   "hamlib/bridge: a native backend handshakes over the socket");

    check::equal(backend->frequency_hz(), 14'074'000.0,
                 "hamlib/bridge: and reads the frequency the radio reported");
    check::is_true(radio->saw("FA"),
                   "hamlib/bridge: ...by actually asking for it");

    backend->set_ptt(true);
    check::is_true(radio->wait_for("TX"),
                   "hamlib/bridge: CAT keying reaches the radio");
    backend->set_ptt(false);
    check::is_true(radio->wait_for("RX"),
                   "hamlib/bridge: and unkeying does too");

    backend->close();
    std::lock_guard<std::mutex> lock(radio->m);
    check::is_true(radio->closed, "hamlib/bridge: closing releases the device");
}

void test_serial_defaults_come_from_the_backend_caps() {
    // The other half of the bridged path's line settings. `rig_init`
    // takes a serial port's defaults from `struct rig_caps`, and its own
    // comment on the rate says "fastest !" -- so this reads
    // `serial_rate_max`, deliberately, rather than picking something
    // more conservative. A different choice would make a bridged rig run
    // at a different speed than the same rig on a desktop.
    const rig::SerialDefaults ts2000 = rig::serial_defaults(MODEL_TS2000);
    check::is_true(ts2000.baud > 0,
                   "hamlib/defaults: a real backend reports a rate (" +
                       std::to_string(ts2000.baud) + ")");
    check::is_true(ts2000.data_bits == 7 || ts2000.data_bits == 8,
                   "hamlib/defaults: and a plausible word length");
    check::is_true(ts2000.stop_bits == 1 || ts2000.stop_bits == 2,
                   "hamlib/defaults: and a plausible stop-bit count");

    // A model Hamlib does not know is a configuration the operator has
    // to fix anyway, and `rig_init` will refuse it a moment later with a
    // better message. Falling back rather than throwing is what lets a
    // settings screen still render while it is wrong.
    const rig::SerialDefaults unknown = rig::serial_defaults(999999);
    check::equal(unknown.baud, 9600,
                 "hamlib/defaults: an unknown model falls back to 9600");
    check::equal(unknown.data_bits, 8, "hamlib/defaults: ...8 data bits");
    check::equal(unknown.stop_bits, 1, "hamlib/defaults: ...1 stop bit");
}

void test_dtr_keying_never_reaches_hamlib() {
    // The one thing that cannot go over this transport. `ser_set_dtr`
    // is a TIOCMSET ioctl and Hamlib is holding a socket, so a DTR
    // keying request handed to Hamlib fails -- at the moment somebody
    // presses transmit, which is the worst time to discover it. The
    // line is driven on the transport instead, and Hamlib is configured
    // not to key at all.
    auto radio = std::make_shared<Kenwood>();
    rig::HamlibConfig config;
    config.model = MODEL_TS2000;
    config.ptt_method = rig::PttMethod::Dtr;
    config.device = "usb:10c4:ea60";

    auto backend = bridged(config, radio);
    backend->open();
    {
        std::lock_guard<std::mutex> lock(radio->m);
        check::equal(radio->dtr, 0, "hamlib/bridge: DTR is parked unkeyed on open");
    }

    backend->set_ptt(true);
    {
        std::lock_guard<std::mutex> lock(radio->m);
        check::equal(radio->dtr, 1, "hamlib/bridge: DTR keying drives the line");
    }
    check::is_true(!radio->saw("TX"),
                   "hamlib/bridge: and sends no CAT keying command");

    backend->set_ptt(false);
    std::lock_guard<std::mutex> lock(radio->m);
    check::equal(radio->dtr, 0, "hamlib/bridge: unkeying releases it");
}

void test_the_controller_drives_a_bridged_rig() {
    // The same threading design, over the new transport: nothing about
    // RigController changed to make this work, which is the property
    // `docs/android.md` predicted would hold and the reason a CAT
    // backend could be added later without restructuring anything.
    auto radio = std::make_shared<Kenwood>();
    rig::HamlibConfig config;
    config.model = MODEL_TS2000;
    config.device = "usb:10c4:ea60";

    std::mutex m;
    std::condition_variable cv;
    std::optional<double> reported;

    rig::RigController controller(
        [&](std::optional<double> hz) {
            {
                std::lock_guard<std::mutex> lock(m);
                if (hz) reported = hz;
            }
            cv.notify_all();
        },
        {});
    rig::RigConfig rig_config;
    rig_config.poll_interval_s = 0.05;
    controller.start(bridged(config, radio), rig_config);

    std::unique_lock<std::mutex> lock(m);
    const bool got = cv.wait_for(lock, std::chrono::seconds(10),
                                 [&] { return reported.has_value(); });
    check::is_true(got, "hamlib/bridge: the controller polls a bridged rig");
    if (got) {
        check::equal(*reported, 14'074'000.0,
                     "hamlib/bridge: and publishes what it read");
    }
}

}  // namespace

// returns" -- and a hang with no output tells you nothing at all except
// on which platform it happened. Printing alone was not enough: ctest
// holds a test's output until it finishes, so a live log shows nothing
// either way. The watchdog is what turns the hang into a message,
// because it reports from inside the process and then ends it.
#define STEP(f)                                     \
    do {                                            \
        check::current_step = #f;                   \
        std::fprintf(stderr, "-- %s\n", #f);        \
        std::fflush(stderr);                        \
        f();                                        \
    } while (0)

int main() {
    check::report_crashes_instead_of_prompting();

    // ~23x the measured runtime (5.2 s on Linux, up from 0.65 s when
    // the bridged tests were added). Sized so that expiring can only
    // mean wedged, and so it fires well inside the ctest TIMEOUT -- the
    // two are not redundant, they answer different questions. If the
    // watchdog fires, a step is stuck and it says which. If ctest times
    // out instead, with this test's "ok:" line in the captured output,
    // then everything finished and the wedge is in process teardown --
    // which is a different bug in a different place.
    //
    // A smaller multiple than CLAUDE.md's ~90x, deliberately: most of
    // this test's runtime is Hamlib waiting out deliberate read
    // timeouts on commands a Kenwood does not answer, and a wall-clock
    // wait does not get slower on a slower machine the way a
    // compute-bound test does. The part that *does* scale is well under
    // a second.
    const check::Watchdog watchdog(120.0, "hamlib backend");

    try {
        STEP(test_list_models);
        STEP(test_version_is_reported);
        STEP(test_dummy_rig_opens_keys_and_reports);
        STEP(test_an_unknown_model_is_refused_readably);
        STEP(test_using_a_closed_rig_reports_rather_than_crashes);
        STEP(test_the_controller_drives_a_real_backend);
        STEP(test_a_native_backend_works_over_a_socket);
        STEP(test_serial_defaults_come_from_the_backend_caps);
        STEP(test_dtr_keying_never_reaches_hamlib);
        STEP(test_the_controller_drives_a_bridged_rig);
        check::current_step = "reporting";
        std::fprintf(stderr, "-- done\n");
        std::fflush(stderr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL in %s: %s\n", check::current_step.load(),
                     e.what());
        return 1;
    }
    const int rc = check::report("hamlib backend");
    std::fflush(stdout);
    check::current_step = "process teardown";
    return rc;
}
