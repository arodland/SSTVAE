#include "rig/android/androidrig.hpp"

#include "rig/trace.hpp"

#include <jni.h>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sstvae::rig::android {
namespace {

constexpr const char* kBridge = "org/cleverdomain/sstvae/SerialBridge";

JavaVM* g_vm = nullptr;
jclass g_bridge = nullptr;

std::mutex g_callback_mu;
PermissionResult g_permission_callback;

// Attach once per thread, detach when the thread ends.
//
// `core/audio/android/` attaches per *call*, which is right there:
// control calls are rare and the data path runs the other way, with
// Java calling an already-attached thread. Here the data path is ours
// -- `LoopbackBridge`'s two pump threads call in continuously -- so a
// per-call `AttachCurrentThread`/`DetachCurrentThread` pair would run
// on every read timeout for the life of a session. A thread_local with
// a destructor is what makes "attached for as long as this thread
// exists" expressible.
class ThreadEnv {
public:
    ThreadEnv() = default;
    ~ThreadEnv() {
        if (attached_ && g_vm != nullptr) g_vm->DetachCurrentThread();
    }
    ThreadEnv(const ThreadEnv&) = delete;
    ThreadEnv& operator=(const ThreadEnv&) = delete;

    JNIEnv* get() {
        if (env_ != nullptr) return env_;
        if (g_vm == nullptr) throw RigError("android rig: set_java_vm() was never called");
        const jint rc = g_vm->GetEnv(reinterpret_cast<void**>(&env_), JNI_VERSION_1_6);
        if (rc == JNI_EDETACHED) {
            // As a daemon, so a pump thread that outlives an orderly
            // shutdown cannot hold the VM open at exit -- the same
            // reasoning as `RigController::stop()` detaching rather
            // than joining.
            if (g_vm->AttachCurrentThreadAsDaemon(&env_, nullptr) != JNI_OK) {
                throw RigError("android rig: AttachCurrentThread failed");
            }
            attached_ = true;
        } else if (rc != JNI_OK) {
            throw RigError("android rig: GetEnv failed");
        }
        return env_;
    }

private:
    JNIEnv* env_ = nullptr;
    bool attached_ = false;
};

thread_local ThreadEnv t_env;

struct Env {
    JNIEnv* e;
    Env() : e(t_env.get()) {}
    JNIEnv* operator->() const { return e; }

    jclass bridge() const {
        // The cached global reference first: on a thread this library
        // created it is the only thing that resolves, because
        // `FindClass` there falls back to the system class loader.
        if (g_bridge != nullptr) return g_bridge;
        jclass c = e->FindClass(kBridge);
        if (c == nullptr) {
            e->ExceptionClear();
            throw RigError(std::string("android rig: ") + kBridge + " not found");
        }
        return c;
    }

    // A Java exception left pending poisons the *next* unrelated JNI
    // call, which is a long way from where it was raised. Every call
    // below converts it here instead, keeping the Java message: it is
    // the one that says "permission denied" or "device disconnected".
    void rethrow(const char* what) const {
        if (e->ExceptionCheck() == JNI_FALSE) return;
        jthrowable ex = e->ExceptionOccurred();
        e->ExceptionClear();
        std::string detail;
        if (ex != nullptr) {
            jclass cls = e->GetObjectClass(ex);
            jmethodID m = e->GetMethodID(cls, "getMessage", "()Ljava/lang/String;");
            if (m != nullptr) {
                auto s = static_cast<jstring>(e->CallObjectMethod(ex, m));
                if (e->ExceptionCheck() != JNI_FALSE) e->ExceptionClear();
                if (s != nullptr) {
                    const char* c = e->GetStringUTFChars(s, nullptr);
                    if (c != nullptr) {
                        detail = c;
                        e->ReleaseStringUTFChars(s, c);
                    }
                    e->DeleteLocalRef(s);
                }
            }
            e->DeleteLocalRef(cls);
            e->DeleteLocalRef(ex);
        }
        throw RigError(detail.empty() ? std::string(what) : detail);
    }
};

std::string to_std(JNIEnv* env, jstring s) {
    if (s == nullptr) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c == nullptr ? "" : c);
    if (c != nullptr) env->ReleaseStringUTFChars(s, c);
    return out;
}

// Java returns each device as "id\tlabel\t0|1" -- one array rather than
// three, so a device cannot be described by rows from two different
// enumerations. Labels have tabs stripped on the Java side.
std::vector<SerialDevice> device_list(const char* method) {
    Env env;
    jclass cls = env.bridge();
    jmethodID m = env->GetStaticMethodID(cls, method, "()[Ljava/lang/String;");
    if (m == nullptr) {
        env->ExceptionClear();
        throw RigError(std::string("android rig: no ") + method);
    }
    auto arr = static_cast<jobjectArray>(env->CallStaticObjectMethod(cls, m));
    env.rethrow(method);

    std::vector<SerialDevice> out;
    if (arr == nullptr) return out;
    const jsize n = env->GetArrayLength(arr);
    out.reserve(static_cast<std::size_t>(n));
    for (jsize i = 0; i < n; ++i) {
        auto s = static_cast<jstring>(env->GetObjectArrayElement(arr, i));
        const std::string row = to_std(env.e, s);
        env->DeleteLocalRef(s);

        const std::size_t a = row.find('\t');
        if (a == std::string::npos) continue;
        const std::size_t b = row.find('\t', a + 1);
        SerialDevice d;
        d.id = row.substr(0, a);
        d.label = b == std::string::npos ? row.substr(a + 1) : row.substr(a + 1, b - a - 1);
        d.permitted = b != std::string::npos && b + 1 < row.size() && row[b + 1] == '1';
        out.push_back(std::move(d));
    }
    return out;
}

// --- the transport ----------------------------------------------------------

// --- trace helpers ----------------------------------------------------
//
// Deliberately spelled the way a settings screen spells it (`8N1`,
// `flow=none`), because the first thing to do with this line is compare
// it against what the operator set and what the radio's menu says.
const char* parity_letter(int parity) {
    switch (parity) {
        case SerialParams::kOdd: return "O";
        case SerialParams::kEven: return "E";
        default: return "N";
    }
}

const char* flow_name(int flow) {
    switch (flow) {
        case SerialParams::kRtsCts: return "rts/cts";
        case SerialParams::kXonXoff: return "xon/xoff";
        default: return "none";
    }
}

// How many consecutive empty reads between heartbeat lines. The bridge
// reads with a 500 ms timeout, so this is about five seconds -- often
// enough to show a stuck transmit queue promptly, rare enough that a
// working session does not fill its own log.
constexpr unsigned long kIdleReadsPerReport = 10;

// Never throws and never leaves an exception pending: this runs inside
// an open that has already succeeded, and a diagnostic that fails the
// operation it is diagnosing is worse than no diagnostic.
std::string describe_link(const Env& env, jclass cls, const std::string& id) {
    jmethodID m = env->GetStaticMethodID(cls, "describeLink",
                                         "(Ljava/lang/String;)Ljava/lang/String;");
    if (m == nullptr) {
        env->ExceptionClear();
        return id + " (no describeLink)";
    }
    jstring jid = env->NewStringUTF(id.c_str());
    auto out = static_cast<jstring>(env->CallStaticObjectMethod(cls, m, jid));
    if (jid != nullptr) env->DeleteLocalRef(jid);
    if (env->ExceptionCheck() != JNI_FALSE) {
        env->ExceptionClear();
        return id + " (describeLink threw)";
    }
    if (out == nullptr) return id;
    std::string text = to_std(env.e, out);
    env->DeleteLocalRef(out);
    return text;
}

// The same contract as `describe_link`: never throws, never leaves an
// exception pending. Called from the bridge's read thread.
std::string describe_status(const Env& env, jclass cls, int token) {
    jmethodID m = env->GetStaticMethodID(cls, "describeStatus", "(I)Ljava/lang/String;");
    if (m == nullptr) {
        env->ExceptionClear();
        return "no describeStatus";
    }
    auto out = static_cast<jstring>(
        env->CallStaticObjectMethod(cls, m, static_cast<jint>(token)));
    if (env->ExceptionCheck() != JNI_FALSE) {
        env->ExceptionClear();
        return "describeStatus threw";
    }
    if (out == nullptr) return "";
    std::string text = to_std(env.e, out);
    env->DeleteLocalRef(out);
    return text;
}

class AndroidTransport : public SerialTransport {
public:
    explicit AndroidTransport(SerialParams params) : params_(std::move(params)) {}
    ~AndroidTransport() override { close(); }

    void open() override {
        if (token_ >= 0) return;
        Env env;
        jclass cls = env.bridge();
        jmethodID m = env->GetStaticMethodID(cls, "open", "(Ljava/lang/String;IIIIIZZ)I");
        if (m == nullptr) {
            env->ExceptionClear();
            throw RigError("android rig: no SerialBridge.open");
        }
        jstring id = env->NewStringUTF(params_.device_id.c_str());
        const jint token = env->CallStaticIntMethod(
            cls, m, id, params_.baud, params_.data_bits, params_.stop_bits,
            params_.parity, params_.flow,
            static_cast<jboolean>(params_.dtr ? JNI_TRUE : JNI_FALSE),
            static_cast<jboolean>(params_.rts ? JNI_TRUE : JNI_FALSE));
        env->DeleteLocalRef(id);
        env.rethrow("could not open the serial device");
        if (token < 0) throw RigError("could not open " + params_.device_id);
        token_ = token;

        // A buffer per transport, allocated once. `NewByteArray` on
        // every read would be an allocation and a GC root twice a
        // second for the life of a session.
        jbyteArray local = env->NewByteArray(kBufferBytes);
        if (local == nullptr) {
            env->ExceptionClear();
            close();
            throw RigError("android rig: could not allocate a transfer buffer");
        }
        buffer_ = static_cast<jbyteArray>(env->NewGlobalRef(local));
        env->DeleteLocalRef(local);

        // What was opened, and how. This is the half of the picture
        // that neither Hamlib's trace nor the bridge's byte counts can
        // supply -- which driver the prober chose and which interface
        // of how many it claimed. Guarded, because building it is a
        // handful of JNI calls and the answer is only wanted when
        // somebody is looking.
        if (tracing()) {
            trace("transport: opened " + params_.device_id + " at " +
                  std::to_string(params_.baud) + " " +
                  std::to_string(params_.data_bits) + parity_letter(params_.parity) +
                  std::to_string(params_.stop_bits) + ", flow=" +
                  flow_name(params_.flow) + ", dtr=" + (params_.dtr ? "1" : "0") +
                  " rts=" + (params_.rts ? "1" : "0"));
            trace("transport: " + describe_link(env, cls, params_.device_id));
        }
    }

    void close() noexcept override {
        const int token = token_.exchange(-1);
        try {
            Env env;
            if (token >= 0) {
                jclass cls = env.bridge();
                jmethodID m = env->GetStaticMethodID(cls, "close", "(I)V");
                if (m != nullptr) env->CallStaticVoidMethod(cls, m, token);
                env->ExceptionClear();
            }
            if (buffer_ != nullptr) {
                env->DeleteGlobalRef(buffer_);
                buffer_ = nullptr;
            }
        } catch (const std::exception&) {
            // Nobody left to tell, as `SerialTransport::close` says.
        }
    }

    std::size_t read(std::uint8_t* dst, std::size_t n, int timeout_ms) override {
        const int token = token_.load();
        if (token < 0) return 0;
        Env env;
        jclass cls = env.bridge();
        jmethodID m = env->GetStaticMethodID(cls, "read", "(I[BII)I");
        if (m == nullptr) {
            env->ExceptionClear();
            throw RigError("android rig: no SerialBridge.read");
        }
        const std::size_t cap = static_cast<std::size_t>(kBufferBytes);
        const jint want = static_cast<jint>(n < cap ? n : cap);
        const jint got = env->CallStaticIntMethod(cls, m, token, buffer_, want, timeout_ms);
        env.rethrow("the serial device stopped answering");
        // Negative means the link is gone; zero means an idle radio,
        // which is not an error and must not be reported as one.
        if (got < 0) throw RigError("the serial device went away");
        if (got == 0) {
            // **A heartbeat, and the chip's own account with it.** A bulk
            // write returning its length means the bytes reached the
            // chip, not that the UART clocked them out -- and from above,
            // "the radio is ignoring us" and "the chip is holding our
            // bytes" are the same silence. A CP210x will say which:
            // `GET_COMM_STATUS` reports how many bytes are still queued
            // for transmit, how many arrived from the UART, and why
            // transmission is being held. It also proves this read loop
            // is running at all, which nothing else in the log does.
            if (tracing() && ++idle_reads_ % kIdleReadsPerReport == 0) {
                trace("transport: idle, " + std::to_string(idle_reads_) +
                      " empty reads; " + describe_status(env, cls, token));
            }
            return 0;
        }
        idle_reads_ = 0;
        env->GetByteArrayRegion(buffer_, 0, got, reinterpret_cast<jbyte*>(dst));
        return static_cast<std::size_t>(got);
    }

    void write(const std::uint8_t* src, std::size_t n) override {
        const int token = token_.load();
        if (token < 0) throw RigError("the serial device is not open");
        Env env;
        jclass cls = env.bridge();
        jmethodID m = env->GetStaticMethodID(cls, "write", "(I[BI)V");
        if (m == nullptr) {
            env->ExceptionClear();
            throw RigError("android rig: no SerialBridge.write");
        }
        // A fresh array rather than the read buffer: reads and writes
        // run on two different threads by `SerialTransport`'s contract,
        // and sharing one would be a data race whose shape is a
        // corrupted CAT command. CAT commands are a handful of bytes
        // and this runs a few times a poll, so the allocation is not
        // worth a second cached buffer and the lifetime rules to go
        // with it.
        std::size_t sent = 0;
        while (sent < n) {
            const std::size_t cap = static_cast<std::size_t>(kBufferBytes);
            const std::size_t left = n - sent;
            const jint chunk = static_cast<jint>(left < cap ? left : cap);
            jbyteArray out = env->NewByteArray(chunk);
            if (out == nullptr) {
                env->ExceptionClear();
                throw RigError("android rig: out of memory writing to the rig");
            }
            env->SetByteArrayRegion(out, 0, chunk,
                                    reinterpret_cast<const jbyte*>(src + sent));
            env->CallStaticVoidMethod(cls, m, token, out, chunk);
            env->DeleteLocalRef(out);
            env.rethrow("could not write to the serial device");
            sent += static_cast<std::size_t>(chunk);
        }
    }

    void set_dtr(bool on) override { set_line("setDtr", on); }
    void set_rts(bool on) override { set_line("setRts", on); }

    std::string description() const override { return params_.device_id; }

private:
    static constexpr jsize kBufferBytes = 512;

    void set_line(const char* method, bool on) {
        const int token = token_.load();
        if (token < 0) throw RigError("the serial device is not open");
        Env env;
        jclass cls = env.bridge();
        jmethodID m = env->GetStaticMethodID(cls, method, "(IZ)V");
        if (m == nullptr) {
            env->ExceptionClear();
            throw RigError(std::string("android rig: no SerialBridge.") + method);
        }
        env->CallStaticVoidMethod(cls, m, token, static_cast<jboolean>(on));
        env.rethrow("could not set a control line");
    }

    SerialParams params_;
    std::atomic<int> token_{-1};
    jbyteArray buffer_ = nullptr;
    // Consecutive reads that returned nothing. Only read and written by
    // the bridge's single read thread.
    unsigned long idle_reads_ = 0;
};

}  // namespace

void set_java_vm(JavaVM_* vm) {
    g_vm = reinterpret_cast<JavaVM*>(vm);
    if (g_vm == nullptr || g_bridge != nullptr) return;
    try {
        Env env;
        jclass local = env->FindClass(kBridge);
        if (local != nullptr) {
            g_bridge = static_cast<jclass>(env->NewGlobalRef(local));
            env->DeleteLocalRef(local);
        } else {
            env->ExceptionClear();
        }
    } catch (const std::exception&) {
        // Leave it null; `bridge()` falls back to FindClass, which is
        // no worse than before this cache existed.
    }
}

bool ready() { return g_vm != nullptr; }

std::vector<SerialDevice> usb_devices() { return device_list("usbDevices"); }
std::vector<SerialDevice> bluetooth_devices() { return device_list("bluetoothDevices"); }

bool has_permission(const std::string& id) {
    // **A query answers; an action reports.** This is read from a QML
    // property binding, where a thrown exception terminates the
    // process -- so "the layer is not initialised" and "the JNI call
    // failed" both become false, which is the truthful answer to "may
    // the app open this device right now" in either case. The
    // enumerators below are the other half of the rule: they are called
    // from an explicit refresh that catches and shows the message, so
    // there they throw rather than return an empty list that reads as
    // "nothing is plugged in".
    if (!ready()) return false;
    try {
        Env env;
        jclass cls = env.bridge();
        jmethodID m = env->GetStaticMethodID(cls, "hasPermission",
                                             "(Ljava/lang/String;)Z");
        if (m == nullptr) {
            env->ExceptionClear();
            return false;
        }
        jstring s = env->NewStringUTF(id.c_str());
        const jboolean ok = env->CallStaticBooleanMethod(cls, m, s);
        env->DeleteLocalRef(s);
        if (env->ExceptionCheck() != JNI_FALSE) {
            env->ExceptionClear();
            return false;
        }
        return ok != JNI_FALSE;
    } catch (const std::exception&) {
        return false;
    }
}

void request_permission(const std::string& id) {
    Env env;
    jclass cls = env.bridge();
    jmethodID m = env->GetStaticMethodID(cls, "requestPermission", "(Ljava/lang/String;)V");
    if (m == nullptr) {
        env->ExceptionClear();
        throw RigError("android rig: no SerialBridge.requestPermission");
    }
    jstring s = env->NewStringUTF(id.c_str());
    env->CallStaticVoidMethod(cls, m, s);
    env->DeleteLocalRef(s);
    env.rethrow("could not ask for USB permission");
}

void set_permission_callback(PermissionResult callback) {
    std::lock_guard<std::mutex> lock(g_callback_mu);
    g_permission_callback = std::move(callback);
}

std::shared_ptr<SerialTransport> make_transport(const SerialParams& params) {
    return std::make_shared<AndroidTransport>(params);
}

}  // namespace sstvae::rig::android

extern "C" {

JNIEXPORT void JNICALL Java_org_cleverdomain_sstvae_SerialBridge_nativePermissionResult(
    JNIEnv* env, jclass, jstring jid, jboolean granted) {
    std::string id;
    if (jid != nullptr) {
        const char* c = env->GetStringUTFChars(jid, nullptr);
        if (c != nullptr) {
            id = c;
            env->ReleaseStringUTFChars(jid, c);
        }
    }
    sstvae::rig::android::PermissionResult callback;
    {
        std::lock_guard<std::mutex> lock(sstvae::rig::android::g_callback_mu);
        callback = sstvae::rig::android::g_permission_callback;
    }
    // Copied out and invoked outside the lock: the callback goes on to
    // touch the UI, and holding a lock across that is how a settings
    // screen ends up waiting on a broadcast receiver.
    if (callback) callback(id, granted != JNI_FALSE);
}

}  // extern "C"
