// Composing a transport, a bridge and a rig backend.
//
// The inner backend is a stub, which is the point: `sstvae_core` only,
// so the decisions this file makes -- when to bridge at all, what
// device string the rig ends up configured with, and which of the two
// PTT paths a keying request takes -- are all covered in a build with
// no libhamlib. `test_rig_hamlib.cpp` is where the real Hamlib meets a
// real socket.

#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "rig/bridge.hpp"
#include "rig/bridged.hpp"

using namespace sstvae;

namespace {

struct Probe {
    std::mutex m;
    int transport_opens = 0;
    int transport_closes = 0;
    std::vector<std::string> lines;   // set_dtr / set_rts, in order
    std::vector<std::string> inner;   // what the stub backend was told
    rig::HamlibConfig seen{};         // the config the factory received
    bool factory_called = false;

    std::vector<std::string> line_log() {
        std::lock_guard<std::mutex> lock(m);
        return lines;
    }
    std::vector<std::string> inner_log() {
        std::lock_guard<std::mutex> lock(m);
        return inner;
    }
};

class StubTransport : public rig::SerialTransport {
public:
    explicit StubTransport(std::shared_ptr<Probe> probe) : probe_(std::move(probe)) {}
    void open() override {
        std::lock_guard<std::mutex> lock(probe_->m);
        probe_->transport_opens++;
    }
    void close() noexcept override {
        std::lock_guard<std::mutex> lock(probe_->m);
        probe_->transport_closes++;
    }
    std::size_t read(std::uint8_t*, std::size_t, int) override { return 0; }
    void write(const std::uint8_t*, std::size_t) override {}
    void set_dtr(bool on) override { note(on ? "dtr=1" : "dtr=0"); }
    void set_rts(bool on) override { note(on ? "rts=1" : "rts=0"); }
    std::string description() const override { return "USB CP2102"; }

private:
    void note(const std::string& what) {
        std::lock_guard<std::mutex> lock(probe_->m);
        probe_->lines.push_back(what);
    }
    std::shared_ptr<Probe> probe_;
};

class StubBackend : public rig::RigBackend {
public:
    explicit StubBackend(std::shared_ptr<Probe> probe) : probe_(std::move(probe)) {}
    void open() override { note("open"); }
    void close() noexcept override { note("close"); }
    void set_ptt(bool on) override { note(on ? "ptt=1" : "ptt=0"); }
    double frequency_hz() override {
        note("freq");
        return 14'230'000.0;
    }
    std::string description() const override { return "Elecraft K4"; }

private:
    void note(const std::string& what) {
        std::lock_guard<std::mutex> lock(probe_->m);
        probe_->inner.push_back(what);
    }
    std::shared_ptr<Probe> probe_;
};

rig::BackendFactory factory_for(const std::shared_ptr<Probe>& probe) {
    return [probe](const rig::HamlibConfig& config) -> std::unique_ptr<rig::RigBackend> {
        std::lock_guard<std::mutex> lock(probe->m);
        probe->seen = config;
        probe->factory_called = true;
        return std::make_unique<StubBackend>(probe);
    };
}

rig::HamlibConfig usb_config() {
    rig::HamlibConfig config;
    config.model = 2043;  // any native backend; the number is not read here
    config.device = "usb:1a86:7523";  // an app-level device id, not a path
    config.baud = 38400;
    return config;
}

// --- tests ------------------------------------------------------------------

void test_a_host_is_dialled_directly_with_no_bridge() {
    // The operator chose a network rig, so there is no transport and
    // Hamlib opens the socket itself. Putting a bridge in that path
    // would mean two of our threads shuttling bytes between two sockets
    // for no reason -- and it is the *same* Hamlib code path either
    // way, which is what makes NET rigctl and a ser2net-style server
    // one feature rather than two.
    auto probe = std::make_shared<Probe>();
    rig::HamlibConfig config;
    config.model = rig::MODEL_NET_RIGCTL;
    config.device = "192.168.1.20:4532";

    auto backend = rig::make_bridged_backend(config, nullptr, factory_for(probe));
    backend->open();

    std::lock_guard<std::mutex> lock(probe->m);
    check::equal(probe->seen.device, std::string("192.168.1.20:4532"),
                 "bridged: a host reaches the rig unchanged");
    check::equal(probe->transport_opens, 0,
                 "bridged: and nothing was opened for it");
}

void test_a_transport_is_what_asks_for_a_bridge_not_the_string() {
    // The regression this pins, which the first draft had. A USB device
    // identifier is not a path and does not start with `com`, so
    // Hamlib's own rules -- and therefore `is_network_device` -- read it
    // as a hostname. Branching on that skipped the bridge for exactly
    // the devices it exists to serve, and the app then tried to resolve
    // `usb:1a86:7523` as a DNS name.
    check::is_true(rig::is_network_device("usb:1a86:7523"),
                   "bridged: a device id does look like a host to Hamlib's rules");

    auto probe = std::make_shared<Probe>();
    auto backend = rig::make_bridged_backend(usb_config(),
                                             std::make_shared<StubTransport>(probe),
                                             factory_for(probe));
    backend->open();
    std::lock_guard<std::mutex> lock(probe->m);
    check::equal(probe->transport_opens, 1,
                 "bridged: ...and it is bridged anyway, because a transport was given");
    check::is_true(probe->seen.device != "usb:1a86:7523",
                   "bridged: the device id never reaches Hamlib");
}

void test_a_device_gets_a_bridge_and_the_rig_gets_its_address() {
    auto probe = std::make_shared<Probe>();
    auto backend = rig::make_bridged_backend(usb_config(),
                                             std::make_shared<StubTransport>(probe),
                                             factory_for(probe));
    backend->open();

    std::string device;
    {
        std::lock_guard<std::mutex> lock(probe->m);
        device = probe->seen.device;
        check::equal(probe->transport_opens, 1, "bridged: the transport was opened");
    }
    check::is_true(device.rfind("127.0.0.1:", 0) == 0,
                   "bridged: the rig is pointed at the bridge (" + device + ")");
    check::is_true(rig::is_network_device(device),
                   "bridged: and Hamlib will read that as a network address");

    check::equal(backend->frequency_hz(), 14'230'000.0,
                 "bridged: frequency goes straight through");
}

void test_the_serial_speed_still_reaches_the_rig_config() {
    // Not because Hamlib will use it -- over a socket it will not, the
    // transport owns the line settings -- but because nothing here may
    // quietly rewrite a field it does not own. Only `device` and the
    // PTT method change.
    auto probe = std::make_shared<Probe>();
    auto backend = rig::make_bridged_backend(usb_config(),
                                             std::make_shared<StubTransport>(probe),
                                             factory_for(probe));
    backend->open();
    std::lock_guard<std::mutex> lock(probe->m);
    check::equal(probe->seen.baud, 38400, "bridged: baud is passed along untouched");
    check::equal(probe->seen.model, 2043, "bridged: so is the model");
}

void test_cat_keying_goes_through_the_rig() {
    auto probe = std::make_shared<Probe>();
    rig::HamlibConfig config = usb_config();
    config.ptt_method = rig::PttMethod::Cat;

    auto backend = rig::make_bridged_backend(config, std::make_shared<StubTransport>(probe),
                                             factory_for(probe));
    backend->open();
    backend->set_ptt(true);
    backend->set_ptt(false);

    const std::vector<std::string> want{"open", "ptt=1", "ptt=0"};
    check::equal(probe->inner_log().size(), want.size(),
                 "bridged: CAT keying is a CAT command");
    check::is_true(probe->inner_log() == want, "bridged: ...in that order");
    check::is_true(probe->line_log().empty(),
                   "bridged: and no control line is touched");
    std::lock_guard<std::mutex> lock(probe->m);
    check::is_true(probe->seen.ptt_method == rig::PttMethod::Cat,
                   "bridged: the rig is told to key by CAT");
}

void test_dtr_keying_drives_the_transport_and_hamlib_is_told_not_to() {
    // The trap this pins. Hamlib's `ser_set_dtr` is a TIOCMSET ioctl,
    // and over this transport the descriptor it holds is a socket -- so
    // asking Hamlib to key DTR fails, and it fails at the moment
    // somebody presses transmit. The line has to be driven here, and
    // Hamlib has to be told to keep its hands off, which is the value
    // `Vox` carries (hamlib.cpp maps it to ptt_type "None").
    auto probe = std::make_shared<Probe>();
    rig::HamlibConfig config = usb_config();
    config.ptt_method = rig::PttMethod::Dtr;

    auto backend = rig::make_bridged_backend(config, std::make_shared<StubTransport>(probe),
                                             factory_for(probe));
    backend->open();
    backend->set_ptt(true);
    backend->set_ptt(false);

    const std::vector<std::string> want{"dtr=0", "dtr=1", "dtr=0"};
    check::is_true(probe->line_log() == want,
                   "bridged: DTR is parked unkeyed, then keyed, then released");
    const std::vector<std::string> inner{"open"};
    check::is_true(probe->inner_log() == inner,
                   "bridged: and the rig is never asked to key");
    std::lock_guard<std::mutex> lock(probe->m);
    check::is_true(probe->seen.ptt_method == rig::PttMethod::Vox,
                   "bridged: Hamlib is configured not to key at all");
}

void test_rts_keying_drives_the_other_line() {
    auto probe = std::make_shared<Probe>();
    rig::HamlibConfig config = usb_config();
    config.ptt_method = rig::PttMethod::Rts;

    auto backend = rig::make_bridged_backend(config, std::make_shared<StubTransport>(probe),
                                             factory_for(probe));
    backend->open();
    backend->set_ptt(true);

    const std::vector<std::string> want{"rts=0", "rts=1"};
    check::is_true(probe->line_log() == want, "bridged: RTS keys the radio");
}

void test_a_parked_line_is_held_but_never_the_keying_one() {
    // An interface that steals its power from a control line needs it
    // held for the session. The line that keys the radio is the one
    // exception: parking *that* high would put the transmitter on from
    // the moment the app connects, which is the worst thing in this
    // file.
    auto probe = std::make_shared<Probe>();
    rig::HamlibConfig config = usb_config();
    config.ptt_method = rig::PttMethod::Rts;
    config.dtr = rig::LineState::High;
    config.rts = rig::LineState::High;

    auto backend = rig::make_bridged_backend(config, std::make_shared<StubTransport>(probe),
                                             factory_for(probe));
    backend->open();

    const std::vector<std::string> want{"dtr=1", "rts=0"};
    check::is_true(probe->line_log() == want,
                   "bridged: DTR is parked high, RTS is left unkeyed");
}

void test_closing_releases_the_rig_before_the_link() {
    // Some backends write a last command in `rig_close` -- restoring a
    // mode they changed on connect. Pulling the bridge first would turn
    // that orderly close into a broken pipe on the way out.
    auto probe = std::make_shared<Probe>();
    {
        auto backend = rig::make_bridged_backend(usb_config(),
                                                 std::make_shared<StubTransport>(probe),
                                                 factory_for(probe));
        backend->open();
        backend->close();
    }
    const std::vector<std::string> want{"open", "close"};
    check::is_true(probe->inner_log() == want, "bridged: the rig is closed");
    std::lock_guard<std::mutex> lock(probe->m);
    check::equal(probe->transport_closes, 1,
                 "bridged: and the device is released exactly once");
}

void test_a_serial_path_with_no_transport_is_refused() {
    // The validation `is_network_device` is left doing: with no
    // transport the string has to be something Hamlib will dial, and a
    // device path is not. Refused here rather than as a failed
    // connection minutes later.
    auto probe = std::make_shared<Probe>();
    rig::HamlibConfig config;
    config.device = "/dev/ttyUSB0";
    bool threw = false;
    try {
        auto backend = rig::make_bridged_backend(config, nullptr, factory_for(probe));
    } catch (const rig::RigError&) {
        threw = true;
    }
    check::is_true(threw,
                   "bridged: a device path with nothing to open it is refused up front");
    std::lock_guard<std::mutex> lock(probe->m);
    check::is_true(!probe->factory_called, "bridged: and no rig is built");
}

void test_a_rig_that_will_not_open_releases_the_device() {
    auto probe = std::make_shared<Probe>();
    rig::BackendFactory failing = [](const rig::HamlibConfig&) -> std::unique_ptr<rig::RigBackend> {
        throw rig::RigError("the rig refused");
    };
    auto backend = rig::make_bridged_backend(usb_config(),
                                             std::make_shared<StubTransport>(probe), failing);
    bool threw = false;
    try {
        backend->open();
    } catch (const rig::RigError&) {
        threw = true;
    }
    check::is_true(threw, "bridged: a rig that will not open reports it");
    std::lock_guard<std::mutex> lock(probe->m);
    check::equal(probe->transport_closes, 1,
                 "bridged: and the USB claim is not left held");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(120.0, "rig bridged");
    try {
        check::current_step.store("host_direct");
        test_a_host_is_dialled_directly_with_no_bridge();
        check::current_step.store("transport_decides");
        test_a_transport_is_what_asks_for_a_bridge_not_the_string();
        check::current_step.store("device_bridged");
        test_a_device_gets_a_bridge_and_the_rig_gets_its_address();
        check::current_step.store("passthrough");
        test_the_serial_speed_still_reaches_the_rig_config();
        check::current_step.store("cat_ptt");
        test_cat_keying_goes_through_the_rig();
        check::current_step.store("dtr_ptt");
        test_dtr_keying_drives_the_transport_and_hamlib_is_told_not_to();
        check::current_step.store("rts_ptt");
        test_rts_keying_drives_the_other_line();
        check::current_step.store("parked_lines");
        test_a_parked_line_is_held_but_never_the_keying_one();
        check::current_step.store("close_order");
        test_closing_releases_the_rig_before_the_link();
        check::current_step.store("no_transport");
        test_a_serial_path_with_no_transport_is_refused();
        check::current_step.store("rig_open_failure");
        test_a_rig_that_will_not_open_releases_the_device();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("rig bridged");
}
