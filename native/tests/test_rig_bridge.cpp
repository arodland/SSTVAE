// The loopback bridge: a serial transport presented to Hamlib as a
// socket.
//
// No libhamlib here on purpose -- this is `sstvae_core` only, so the
// whole mechanism is covered in a `--no-rig` build. What is under test
// is the plumbing that Hamlib will sit on top of: that bytes cross in
// both directions unmodified, that a shutdown does not wait out a read
// timeout, and that our reading of a device string is Hamlib's reading
// of it.
//
// **Nothing here asserts on elapsed time.** A bridge that failed to
// wake a blocked read would hang rather than run slowly, and the ctest
// TIMEOUT reports that with the watchdog naming the step -- which is
// the same trade `test_rig.cpp` makes and for the same reason.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

#include "check.hpp"
#include "rig/bridge.hpp"
#include "rig/trace.hpp"

using namespace sstvae;

namespace {

// --- a radio made of two queues --------------------------------------------

// Everything the test needs to see, kept alive independently of the
// transport so assertions still work after the bridge has taken it.
struct Probe {
    std::mutex m;
    std::condition_variable cv;

    int opens = 0;
    int closes = 0;
    bool closed = false;
    bool fail_open = false;

    std::vector<std::uint8_t> written;   // what the bridge sent to the "rig"
    std::deque<std::uint8_t> to_deliver; // what the "rig" will answer with
    std::vector<std::string> lines;      // set_dtr/set_rts, in order

    void deliver(const std::string& s) {
        {
            std::lock_guard<std::mutex> lock(m);
            for (char c : s) to_deliver.push_back(static_cast<std::uint8_t>(c));
        }
        cv.notify_all();
    }

    std::string written_text() {
        std::lock_guard<std::mutex> lock(m);
        return std::string(written.begin(), written.end());
    }

    bool wait_for_written(const std::string& want, double seconds = 5.0) {
        std::unique_lock<std::mutex> lock(m);
        return cv.wait_for(lock, std::chrono::duration<double>(seconds), [&] {
            return std::string(written.begin(), written.end()).find(want) !=
                   std::string::npos;
        });
    }
};

class FakeTransport : public rig::SerialTransport {
public:
    explicit FakeTransport(std::shared_ptr<Probe> probe) : probe_(std::move(probe)) {}

    void open() override {
        std::lock_guard<std::mutex> lock(probe_->m);
        if (probe_->fail_open) throw rig::RigError("no such device");
        probe_->opens++;
        probe_->closed = false;
    }

    // Closing wakes a blocked read: the contract `SerialTransport`
    // states, and the one the bridge's shutdown depends on.
    void close() noexcept override {
        {
            std::lock_guard<std::mutex> lock(probe_->m);
            if (!probe_->closed) probe_->closes++;
            probe_->closed = true;
        }
        probe_->cv.notify_all();
    }

    std::size_t read(std::uint8_t* dst, std::size_t n, int timeout_ms) override {
        std::unique_lock<std::mutex> lock(probe_->m);
        probe_->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return !probe_->to_deliver.empty() || probe_->closed;
        });
        if (probe_->closed) return 0;
        std::size_t i = 0;
        while (i < n && !probe_->to_deliver.empty()) {
            dst[i++] = probe_->to_deliver.front();
            probe_->to_deliver.pop_front();
        }
        return i;
    }

    void write(const std::uint8_t* src, std::size_t n) override {
        {
            std::lock_guard<std::mutex> lock(probe_->m);
            if (probe_->closed) throw rig::RigError("write after close");
            probe_->written.insert(probe_->written.end(), src, src + n);
        }
        probe_->cv.notify_all();
    }

    void set_dtr(bool on) override { note(on ? "dtr=1" : "dtr=0"); }
    void set_rts(bool on) override { note(on ? "rts=1" : "rts=0"); }

    std::string description() const override { return "fake rig"; }

private:
    void note(const std::string& what) {
        std::lock_guard<std::mutex> lock(probe_->m);
        probe_->lines.push_back(what);
    }
    std::shared_ptr<Probe> probe_;
};

// --- a client, standing in for Hamlib --------------------------------------

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kBad = INVALID_SOCKET;
void close_socket(socket_t s) { ::closesocket(s); }
void start_sockets() {
    static bool done = false;
    if (!done) {
        WSADATA d;
        ::WSAStartup(MAKEWORD(2, 2), &d);
        done = true;
    }
}
#else
using socket_t = int;
constexpr socket_t kBad = -1;
void close_socket(socket_t s) { ::close(s); }
void start_sockets() {}
#endif

class Client {
public:
    explicit Client(int port) {
        start_sockets();
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ == kBad) throw std::runtime_error("client socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<std::uint16_t>(port));
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            close_socket(fd_);
            throw std::runtime_error("client connect");
        }
    }
    ~Client() { disconnect(); }

    void disconnect() {
        if (fd_ != kBad) {
            close_socket(fd_);
            fd_ = kBad;
        }
    }

    void send_text(const std::string& s) {
        ::send(fd_, s.data(), static_cast<int>(s.size()), 0);
    }

    // Read until `n` bytes or a few seconds pass. Blocking with a
    // deadline enforced by the ctest timeout rather than by a timer
    // here: a bridge that never delivers is a hang, and a hang is the
    // signal.
    std::string receive(std::size_t n) {
        std::string out;
        char buf[256];
        while (out.size() < n) {
            const auto got = ::recv(fd_, buf, static_cast<int>(sizeof(buf)), 0);
            if (got <= 0) break;
            out.append(buf, static_cast<std::size_t>(got));
        }
        return out;
    }

private:
    socket_t fd_ = kBad;
};

// --- tests ------------------------------------------------------------------

void test_endpoint_is_loopback_with_an_ephemeral_port() {
    auto probe = std::make_shared<Probe>();
    rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
    bridge.start();

    const std::string endpoint = bridge.endpoint();
    check::is_true(endpoint.rfind("127.0.0.1:", 0) == 0,
                   "bridge: binds loopback, not every interface (" + endpoint + ")");
    check::is_true(bridge.port() > 0, "bridge: the kernel chose a port");
    check::equal(endpoint, "127.0.0.1:" + std::to_string(bridge.port()),
                 "bridge: endpoint and port agree");
    {
        std::lock_guard<std::mutex> lock(probe->m);
        check::equal(probe->opens, 1, "bridge: start() opened the transport");
    }
}

void test_bytes_cross_from_hamlib_to_the_radio() {
    auto probe = std::make_shared<Probe>();
    rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
    bridge.start();

    Client client(bridge.port());
    client.send_text("FA;");
    check::is_true(probe->wait_for_written("FA;"),
                   "bridge: a CAT command reaches the transport");
    check::is_true(bridge.connected(), "bridge: reports the client connected");
}

void test_bytes_cross_from_the_radio_to_hamlib() {
    auto probe = std::make_shared<Probe>();
    rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
    bridge.start();

    Client client(bridge.port());
    // Wait until the pump is actually connected before answering, so
    // this tests delivery rather than a race with accept().
    client.send_text("FA;");
    check::is_true(probe->wait_for_written("FA;"), "bridge: (setup) command arrived");

    probe->deliver("FA00014074000;");
    check::equal(client.receive(14), std::string("FA00014074000;"),
                 "bridge: the rig's answer reaches Hamlib");
}

void test_binary_bytes_are_not_mangled() {
    // Yaesu and Icom CAT are binary, not ASCII: a NUL or a 0xFE in the
    // middle of a command is ordinary traffic. A bridge that treated
    // its buffer as a C string would pass every ASCII test and lose
    // half the radios Hamlib supports.
    auto probe = std::make_shared<Probe>();
    rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
    bridge.start();

    Client client(bridge.port());
    const std::string icom("\xFE\xFE\x94\xE0\x03\x00\xFD", 7);
    client.send_text(icom);
    check::is_true(probe->wait_for_written(icom),
                   "bridge: NUL and high bytes survive the crossing");

    const std::string reply("\xFE\xFE\xE0\x94\x00\x60\x40\x07\x14\x00\xFD", 11);
    probe->deliver(reply);
    check::equal(client.receive(reply.size()), reply,
                 "bridge: a binary answer survives the crossing");
}

// The trace is the diagnostic this bridge exists to make possible, so
// it is tested rather than eyeballed.
//
// **What it is for**: from above, a radio that ignores a command and a
// bridge that never delivered one are the same log -- Hamlib writes to
// a socket, the write succeeds, and nothing comes back. The only line
// that separates them is one emitted *after* the transport accepted
// the bytes. Both directions and the hex spelling are checked, because
// the whole value of the line is that a frame in it can be compared
// byte for byte against the one in Hamlib's own `dump_hex` output.
void test_the_trace_reports_both_directions() {
    std::mutex mu;
    std::vector<std::string> lines;
    rig::set_trace_sink([&](const std::string& line) {
        std::lock_guard<std::mutex> lock(mu);
        lines.push_back(line);
    });

    {
        auto probe = std::make_shared<Probe>();
        rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
        bridge.start();

        Client client(bridge.port());
        const std::string icom("\xFE\xFE\xA2\xE0\x03\xFD", 6);
        client.send_text(icom);
        check::is_true(probe->wait_for_written(icom), "bridge/trace: (setup) command arrived");

        const std::string reply("\xFE\xFE\xE0\xA2\xFB\xFD", 6);
        probe->deliver(reply);
        check::equal(client.receive(reply.size()), reply, "bridge/trace: (setup) answer arrived");
    }

    bool saw_out = false;
    bool saw_in = false;
    {
        std::lock_guard<std::mutex> lock(mu);
        for (const std::string& line : lines) {
            if (line.find("-> rig 6: fe fe a2 e0 03 fd") != std::string::npos) saw_out = true;
            if (line.find("<- rig 6: fe fe e0 a2 fb fd") != std::string::npos) saw_in = true;
        }
    }
    check::is_true(saw_out, "bridge/trace: what reached the radio, in Hamlib's hex spelling");
    check::is_true(saw_in, "bridge/trace: and what came back");

    // Removing the sink must actually stop it: this sits in the byte
    // pump, and a sink left installed after the view that owns it is
    // gone is the shape of every use-after-free in a logging path.
    rig::set_trace_sink({});
    check::is_true(!rig::tracing(), "bridge/trace: removing the sink turns it off");

    {
        std::lock_guard<std::mutex> lock(mu);
        lines.clear();
    }
    {
        auto probe = std::make_shared<Probe>();
        rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
        bridge.start();
        Client client(bridge.port());
        client.send_text("FA;");
        check::is_true(probe->wait_for_written("FA;"), "bridge/trace: (setup) still bridging");
    }
    std::size_t after = 0;
    {
        std::lock_guard<std::mutex> lock(mu);
        after = lines.size();
    }
    check::equal(static_cast<int>(after), 0, "bridge/trace: and nothing is delivered after");
}

// The placement is the claim: **after** the transport accepted the
// bytes, never before it. A line logged on the way in would say
// "delivered" for a write that threw, which is precisely the false
// reassurance this trace was added to remove.
void test_a_failed_write_is_never_reported_as_delivered() {
    std::mutex mu;
    std::vector<std::string> lines;
    rig::set_trace_sink([&](const std::string& line) {
        std::lock_guard<std::mutex> lock(mu);
        lines.push_back(line);
    });

    {
        auto probe = std::make_shared<Probe>();
        rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
        bridge.start();
        Client client(bridge.port());

        // The transport now refuses every write, the way an unplugged
        // device does.
        {
            std::lock_guard<std::mutex> lock(probe->m);
            probe->closed = true;
        }
        client.send_text("FA;");

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (bridge.last_error().empty() &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        check::is_true(!bridge.last_error().empty(),
                       "bridge/trace: (setup) the write failed");
    }

    bool claimed = false;
    {
        std::lock_guard<std::mutex> lock(mu);
        for (const std::string& line : lines) {
            if (line.find("-> rig") != std::string::npos) claimed = true;
        }
    }
    check::is_true(!claimed,
                   "bridge/trace: a write that threw is not logged as delivered");
    rig::set_trace_sink({});
}

void test_hex_bytes_truncates_rather_than_floods() {
    const std::vector<std::uint8_t> big(200, 0xAB);
    const std::string out = rig::hex_bytes(big.data(), big.size());
    check::is_true(out.find("(200 bytes)") != std::string::npos,
                   "bridge/trace: a long block says how long it was");
    check::is_true(out.size() < 200,
                   "bridge/trace: ...without printing all of it");
}

void test_a_client_hanging_up_ends_the_session() {
    auto probe = std::make_shared<Probe>();
    rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
    bridge.start();

    {
        Client client(bridge.port());
        client.send_text("ID;");
        check::is_true(probe->wait_for_written("ID;"), "bridge: (setup) connected");
    }
    // The pump notices without being told, so a controller polling
    // `connected()` can tell a rig that hung up from one that is quiet.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (bridge.connected() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    check::is_true(!bridge.connected(), "bridge: a disconnect is noticed");
}

void test_stopping_closes_the_transport_and_does_not_wait() {
    auto probe = std::make_shared<Probe>();
    {
        rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
        bridge.start();
        Client client(bridge.port());
        client.send_text("ID;");
        check::is_true(probe->wait_for_written("ID;"), "bridge: (setup) connected");
        // The pump-out thread is inside a blocked read right now. If
        // stop() relied on the read timeout expiring rather than on
        // close() waking it, this would still pass -- but if it relied
        // on nothing at all it would hang, and the watchdog names it.
        bridge.stop();
    }
    std::lock_guard<std::mutex> lock(probe->m);
    check::equal(probe->closes, 1, "bridge: stop() closed the transport exactly once");
}

void test_a_session_that_ended_by_itself_still_releases_the_device() {
    // The failure this pins: `stop()` used to return early when
    // `stopping_` was already set, and a pump thread sets it itself
    // when the far end hangs up. So an ordinary disconnect left the
    // listener and the USB claim held for the life of the process --
    // and nothing failed, which is the worst shape available.
    auto probe = std::make_shared<Probe>();
    {
        rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
        bridge.start();
        {
            Client client(bridge.port());
            client.send_text("ID;");
            check::is_true(probe->wait_for_written("ID;"), "bridge: (setup) connected");
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (bridge.connected() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        bridge.stop();
    }
    std::lock_guard<std::mutex> lock(probe->m);
    check::equal(probe->closes, 1,
                 "bridge: a self-ended session still closes the transport");
}

void test_a_device_that_cannot_be_opened_fails_start() {
    auto probe = std::make_shared<Probe>();
    {
        std::lock_guard<std::mutex> lock(probe->m);
        probe->fail_open = true;
    }
    rig::LoopbackBridge bridge(std::make_shared<FakeTransport>(probe));
    bool threw = false;
    try {
        bridge.start();
    } catch (const rig::RigError&) {
        threw = true;
    }
    check::is_true(threw, "bridge: a device that will not open fails start()");
    check::equal(bridge.port(), 0, "bridge: and no port is left listening");
}

void test_device_strings_are_read_the_way_hamlib_reads_them() {
    // These are `parse_hoststr` in Hamlib's src/misc.c, case for case.
    // Disagreeing means the app opens a bridge Hamlib will not connect
    // to, or dials a hostname while a radio sits idle.
    struct Case {
        const char* device;
        bool network;
        const char* why;
    };
    const Case cases[] = {
        {"127.0.0.1:4532", true, "an address with a port"},
        {"localhost:4532", true, "a name with a port"},
        {"192.168.1.5:4001", true, "a ser2net server"},
        {"[::1]:4532", true, "bracketed IPv6"},
        {"shack.example.com", true, "a bare hostname -- Hamlib accepts this"},
        {"/dev/ttyUSB0", false, "a Linux device path"},
        {"/dev/cu.usbserial-1420", false, "a macOS device path"},
        {"COM3", false, "a Windows port"},
        {"com12", false, "a Windows port, lower case"},
        {"\\\\.\\COM17", false, "an escaped Windows port"},
        {"", false, "nothing at all"},
    };
    for (const Case& c : cases) {
        check::equal(rig::is_network_device(c.device), c.network,
                     std::string("is_network_device: ") + c.why + " (" + c.device + ")");
    }
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    // ~90x the measured runtime, per CLAUDE.md: expiring can then only
    // mean wedged, never "slower than I guessed".
    check::Watchdog watchdog(120.0, "rig bridge");
    try {
        check::current_step.store("endpoint");
        test_endpoint_is_loopback_with_an_ephemeral_port();
        check::current_step.store("to_radio");
        test_bytes_cross_from_hamlib_to_the_radio();
        check::current_step.store("from_radio");
        test_bytes_cross_from_the_radio_to_hamlib();
        check::current_step.store("binary");
        test_binary_bytes_are_not_mangled();
        check::current_step.store("hangup");
        test_a_client_hanging_up_ends_the_session();
        check::current_step.store("stop");
        test_stopping_closes_the_transport_and_does_not_wait();
        check::current_step.store("stop_after_self_end");
        test_a_session_that_ended_by_itself_still_releases_the_device();
        check::current_step.store("open_failure");
        test_a_device_that_cannot_be_opened_fails_start();
        check::current_step.store("device_strings");
        test_device_strings_are_read_the_way_hamlib_reads_them();
        check::current_step.store("trace");
        test_the_trace_reports_both_directions();
        check::current_step.store("trace_failed_write");
        test_a_failed_write_is_never_reported_as_delivered();
        check::current_step.store("trace_hex");
        test_hex_bytes_truncates_rather_than_floods();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("rig bridge");
}
