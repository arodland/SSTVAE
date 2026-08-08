// The thin half: JNI marshalling and nothing else.
//
// See the header for the design. The Java counterpart is
// `java/org/cleverdomain/sstvae/AudioBridge.java`, which is part of this
// layer rather than of any app -- an app that had to supply its own
// would be free to get the blocking-read loop subtly wrong, and that
// loop is the thing this layer exists to own.

#include "audio/android/androidaudio.hpp"

#include <jni.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>

#include "audio/audio.hpp"
#include "dsp/dsp.hpp"

namespace sstvae::audio::android {
namespace {

constexpr const char* kBridge = "org/cleverdomain/sstvae/AudioBridge";

JavaVM* g_vm = nullptr;

// Attaches the calling thread for the duration of a *control* call.
// Never used on the data path -- Java calls us there, already attached.
class Env {
public:
    Env() {
        if (g_vm == nullptr) {
            throw std::runtime_error(
                "android audio: set_java_vm() was never called");
        }
        const jint rc = g_vm->GetEnv(reinterpret_cast<void**>(&env_), JNI_VERSION_1_6);
        if (rc == JNI_EDETACHED) {
            if (g_vm->AttachCurrentThread(&env_, nullptr) != JNI_OK) {
                throw std::runtime_error("android audio: AttachCurrentThread failed");
            }
            attached_ = true;
        } else if (rc != JNI_OK) {
            throw std::runtime_error("android audio: GetEnv failed");
        }
    }
    ~Env() {
        if (attached_) g_vm->DetachCurrentThread();
    }
    Env(const Env&) = delete;
    Env& operator=(const Env&) = delete;

    JNIEnv* operator->() const { return env_; }
    JNIEnv* get() const { return env_; }

    jclass bridge() const {
        jclass c = env_->FindClass(kBridge);
        if (c == nullptr) {
            env_->ExceptionClear();
            throw std::runtime_error(std::string("android audio: ") + kBridge +
                                     " not found");
        }
        return c;
    }

    // Turns a pending Java exception into a C++ one rather than letting
    // it sit until the next unrelated JNI call fails mysteriously.
    void check(const char* what) const {
        if (env_->ExceptionCheck() == JNI_FALSE) return;
        env_->ExceptionDescribe();
        env_->ExceptionClear();
        throw std::runtime_error(std::string("android audio: ") + what);
    }

private:
    JNIEnv* env_ = nullptr;
    bool attached_ = false;
};

std::string to_std(JNIEnv* env, jstring s) {
    if (s == nullptr) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c == nullptr ? "" : c);
    if (c != nullptr) env->ReleaseStringUTFChars(s, c);
    return out;
}

std::vector<std::string> string_array(const char* method) {
    Env env;
    jclass cls = env.bridge();
    jmethodID m = env->GetStaticMethodID(cls, method, "()[Ljava/lang/String;");
    if (m == nullptr) {
        env->ExceptionClear();
        throw std::runtime_error(std::string("android audio: no ") + method);
    }
    auto arr = static_cast<jobjectArray>(env->CallStaticObjectMethod(cls, m));
    env.check(method);
    std::vector<std::string> out;
    if (arr == nullptr) return out;
    const jsize n = env->GetArrayLength(arr);
    out.reserve(static_cast<std::size_t>(n));
    for (jsize i = 0; i < n; ++i) {
        auto s = static_cast<jstring>(env->GetObjectArrayElement(arr, i));
        out.push_back(to_std(env.get(), s));
        env->DeleteLocalRef(s);
    }
    return out;
}

std::string single_string(const char* method) {
    Env env;
    jclass cls = env.bridge();
    jmethodID m = env->GetStaticMethodID(cls, method, "()Ljava/lang/String;");
    if (m == nullptr) {
        env->ExceptionClear();
        throw std::runtime_error(std::string("android audio: no ") + method);
    }
    auto s = static_cast<jstring>(env->CallStaticObjectMethod(cls, m));
    env.check(method);
    return to_std(env.get(), s);
}

}  // namespace

void set_java_vm(JavaVM_* vm) { g_vm = reinterpret_cast<JavaVM*>(vm); }
bool ready() { return g_vm != nullptr; }

std::vector<std::string> input_device_names() { return string_array("inputDeviceNames"); }
std::vector<std::string> output_device_names() {
    return string_array("outputDeviceNames");
}
std::string default_input_name() { return single_string("defaultInputName"); }
std::string default_output_name() { return single_string("defaultOutputName"); }

// --- capture -----------------------------------------------------------

struct InputStream::Impl {
    rx::RingBuffer& ring;
    int want_rate;
    Report on_opened;
    Report on_error;

    // Guards everything the Java reader thread writes and the UI reads.
    mutable std::mutex m;
    std::unique_ptr<CapturePipeline> pipeline;
    int device_rate = 0;
    std::string routed;
    std::string warning;
    double peak = 0.0;
    double near_zero = 1.0;
    std::atomic<std::uint64_t> captured{0};
    // Wall clock at the first chunk, for the capture-rate check below.
    // Started on the first *chunk* rather than at open, because the
    // gap before audio begins flowing is not lost audio.
    std::chrono::steady_clock::time_point first_chunk{};
    bool have_first = false;
    std::atomic<bool> stopped{false};
    jint token = -1;

    Impl(rx::RingBuffer& r, int rate, Report opened, Report err)
        : ring(r), want_rate(rate), on_opened(std::move(opened)),
          on_error(std::move(err)) {}
};

namespace {
// Live streams, so a callback arriving from Java can find its owner. A
// registry rather than a raw pointer handed to Java: a chunk that
// arrives after `stop()` must find nothing rather than a freed object,
// and on Android a device disappearing mid-read makes that ordinary
// rather than exotic.
std::mutex g_streams_m;
std::vector<std::pair<jint, InputStream::Impl*>> g_streams;

InputStream::Impl* find_stream(jint token) {
    std::lock_guard<std::mutex> lk(g_streams_m);
    for (const auto& [t, p] : g_streams) {
        if (t == token) return p;
    }
    return nullptr;
}
}  // namespace

InputStream::InputStream(const std::string& device_name, rx::RingBuffer& ring,
                         int samplerate, Report on_opened, Report on_error)
    : impl_(std::make_unique<Impl>(ring, samplerate, std::move(on_opened),
                                   std::move(on_error))) {
    Env env;
    jclass cls = env.bridge();
    jmethodID m = env->GetStaticMethodID(cls, "openInput", "(Ljava/lang/String;I)I");
    if (m == nullptr) {
        env->ExceptionClear();
        throw std::runtime_error("android audio: no openInput");
    }
    jstring jname = env->NewStringUTF(device_name.c_str());
    const jint token = env->CallStaticIntMethod(cls, m, jname, samplerate);
    env->DeleteLocalRef(jname);
    env.check("openInput");
    if (token < 0) throw std::runtime_error("android audio: could not open capture");

    impl_->token = token;
    {
        std::lock_guard<std::mutex> lk(g_streams_m);
        g_streams.emplace_back(token, impl_.get());
    }
}

InputStream::~InputStream() {
    stop();
    std::lock_guard<std::mutex> lk(g_streams_m);
    for (auto it = g_streams.begin(); it != g_streams.end(); ++it) {
        if (it->second == impl_.get()) {
            g_streams.erase(it);
            break;
        }
    }
}

void InputStream::stop() {
    if (impl_->stopped.exchange(true)) return;
    try {
        Env env;
        jclass cls = env.bridge();
        jmethodID m = env->GetStaticMethodID(cls, "closeInput", "(I)V");
        if (m != nullptr) {
            env->CallStaticVoidMethod(cls, m, impl_->token);
            env->ExceptionClear();
        }
    } catch (const std::exception&) {
        // Shutting down; there is nothing useful to do with a failure to
        // tell Java we are done, and throwing from here would propagate
        // out of a destructor.
    }
}

int InputStream::device_rate() const {
    std::lock_guard<std::mutex> lk(impl_->m);
    return impl_->device_rate;
}
std::string InputStream::routed_device() const {
    std::lock_guard<std::mutex> lk(impl_->m);
    return impl_->routed;
}
std::string InputStream::routing_warning() const {
    std::lock_guard<std::mutex> lk(impl_->m);
    return impl_->warning;
}
std::uint64_t InputStream::samples_captured() const {
    return impl_->captured.load(std::memory_order_relaxed);
}
double InputStream::peak_level() const {
    std::lock_guard<std::mutex> lk(impl_->m);
    return impl_->peak;
}
double InputStream::near_zero_fraction() const {
    std::lock_guard<std::mutex> lk(impl_->m);
    return impl_->near_zero;
}

// How far the capture rate has drifted from the device's nominal one,
// in parts per million, negative when samples are going missing.
//
// **This is the instrument for the failure mode that looks like
// everything working.** Lost input samples do not arrive as noise or as
// a gap; they arrive as *timing error*, so sync still succeeds, every
// frame is still reported, and the picture is quietly mangled -- which
// is how the PortAudio-on-JACK bug survived several rounds of being
// investigated as a decoder problem. The project's own numbers set the
// scale: a clean path measured +211 ppm, and 3500 ppm of loss cost 5 dB
// and the picture. Anything past about -1000 ppm is audio being
// dropped, not a crystal being imprecise.
//
// Returns 0 before enough audio has arrived to mean anything: over a
// short window the quantisation of the first chunk dominates.
double InputStream::capture_drift_ppm() const {
    std::lock_guard<std::mutex> lk(impl_->m);
    if (!impl_->have_first || impl_->device_rate <= 0) return 0.0;
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - impl_->first_chunk)
                               .count();
    if (elapsed < 5.0) return 0.0;
    const double expected = elapsed * impl_->device_rate;
    const double actual =
        static_cast<double>(impl_->captured.load(std::memory_order_relaxed));
    return (actual / expected - 1.0) * 1e6;
}

// --- playback ----------------------------------------------------------

bool play(const std::string& device_name, std::span<const double> samples,
          int samplerate, const std::function<void(double)>& on_progress,
          const std::function<bool()>& should_stop, const Report& on_error) {
    try {
        Env env;
        jclass cls = env.bridge();
        jmethodID open = env->GetStaticMethodID(cls, "openOutput",
                                                "(Ljava/lang/String;I)I");
        if (open == nullptr) {
            env->ExceptionClear();
            throw std::runtime_error("no openOutput");
        }
        jstring jname = env->NewStringUTF(device_name.c_str());
        const jint rate = env->CallStaticIntMethod(cls, open, jname, samplerate);
        env->DeleteLocalRef(jname);
        env.check("openOutput");
        if (rate <= 0) throw std::runtime_error("could not open playback");

        // Resample the whole waveform once. Playback can, capture cannot,
        // and the per-chunk alternative is the 4.7 dB bug wearing a hat.
        std::vector<double> out(samples.begin(), samples.end());
        if (rate != samplerate) {
            const Ratio r = resample_ratio(samplerate, rate);
            out = dsp::resample_poly(out, r.up, r.down);
        }
        const std::vector<std::byte> raw =
            mono_to_bytes(out, SampleFormat::Int16, 1);

        jmethodID write = env->GetStaticMethodID(cls, "writeOutput", "([BII)I");
        jmethodID close = env->GetStaticMethodID(cls, "closeOutput", "()V");
        if (write == nullptr || close == nullptr) {
            env->ExceptionClear();
            throw std::runtime_error("no writeOutput/closeOutput");
        }

        // A quarter second per call: small enough that a cancel is
        // responsive, large enough that the JNI crossing is noise.
        const std::size_t chunk =
            static_cast<std::size_t>(rate / 4) * 2 /* bytes per sample */;
        jbyteArray buf = env->NewByteArray(static_cast<jsize>(chunk));
        bool ok = true;
        for (std::size_t i = 0; i < raw.size(); i += chunk) {
            if (should_stop && should_stop()) {
                ok = false;
                break;
            }
            const auto n = static_cast<jsize>(std::min(chunk, raw.size() - i));
            env->SetByteArrayRegion(buf, 0, n,
                                    reinterpret_cast<const jbyte*>(raw.data() + i));
            if (env->CallStaticIntMethod(cls, write, buf, 0, n) < 0) {
                ok = false;
                if (on_error) on_error("[audio out] write failed");
                break;
            }
            if (on_progress) {
                on_progress(static_cast<double>(i + static_cast<std::size_t>(n)) /
                            static_cast<double>(raw.size()));
            }
        }
        env->DeleteLocalRef(buf);
        env->CallStaticVoidMethod(cls, close);
        env->ExceptionClear();
        return ok;
    } catch (const std::exception& e) {
        if (on_error) on_error(std::string("[audio out] ") + e.what());
        return false;
    }
}

}  // namespace sstvae::audio::android

// --- the data path, Java -> C++ ----------------------------------------
//
// One call per read. The buffer is direct, so the bytes are not copied
// across the boundary, and everything that happens to them --
// mixdown, format conversion, stateful resampling -- is
// `audio::CapturePipeline` in the Qt-free layer.

extern "C" {

JNIEXPORT void JNICALL Java_org_cleverdomain_sstvae_AudioBridge_nativeOpened(
    JNIEnv* env, jclass, jint token, jint device_rate, jint channels,
    jstring jroute, jstring jwarning) {
    using namespace sstvae::audio;
    auto* s = android::find_stream(token);
    if (s == nullptr) return;

    const char* r = env->GetStringUTFChars(jroute, nullptr);
    const char* w = env->GetStringUTFChars(jwarning, nullptr);
    std::string route(r == nullptr ? "" : r);
    std::string warning(w == nullptr ? "" : w);
    if (r != nullptr) env->ReleaseStringUTFChars(jroute, r);
    if (w != nullptr) env->ReleaseStringUTFChars(jwarning, w);

    android::Report opened;
    {
        std::lock_guard<std::mutex> lk(s->m);
        s->device_rate = device_rate;
        s->routed = route;
        s->warning = warning;
        s->pipeline = std::make_unique<CapturePipeline>(
            SampleFormat::Int16, channels, device_rate, s->want_rate);
        opened = s->on_opened;
    }
    if (opened) {
        std::string msg = "capturing from " + route + " at " +
                          std::to_string(device_rate) + " Hz";
        if (device_rate != s->want_rate) {
            msg += ", resampled to " + std::to_string(s->want_rate) + " Hz";
        }
        opened(msg);
    }
    if (!warning.empty()) {
        android::Report err;
        {
            std::lock_guard<std::mutex> lk(s->m);
            err = s->on_error;
        }
        if (err) err("[audio in] " + warning);
    }
}

JNIEXPORT void JNICALL Java_org_cleverdomain_sstvae_AudioBridge_nativePush(
    JNIEnv* env, jclass, jint token, jobject buf, jint length, jint peak,
    jdouble near_zero) {
    auto* s = sstvae::audio::android::find_stream(token);
    if (s == nullptr || length <= 0) return;
    auto* data = static_cast<std::byte*>(env->GetDirectBufferAddress(buf));
    if (data == nullptr) return;

    std::vector<double> out;
    {
        std::lock_guard<std::mutex> lk(s->m);
        if (!s->pipeline) return;
        out = (*s->pipeline)(std::span<const std::byte>(
            data, static_cast<std::size_t>(length)));
        s->peak = peak / 32768.0;
        s->near_zero = near_zero;
        s->captured.store((*s->pipeline).samples_in(), std::memory_order_relaxed);
        if (!s->have_first) {
            s->first_chunk = std::chrono::steady_clock::now();
            s->have_first = true;
        }
    }
    // Written outside the lock: `RingBuffer::write` takes its own, and
    // nesting the two would put this thread's progress at the mercy of
    // whatever is reading our status.
    if (!out.empty()) s->ring.write(out);
}

JNIEXPORT void JNICALL Java_org_cleverdomain_sstvae_AudioBridge_nativeError(
    JNIEnv* env, jclass, jint token, jstring jmsg) {
    auto* s = sstvae::audio::android::find_stream(token);
    if (s == nullptr) return;
    const char* c = env->GetStringUTFChars(jmsg, nullptr);
    std::string msg(c == nullptr ? "" : c);
    if (c != nullptr) env->ReleaseStringUTFChars(jmsg, c);
    sstvae::audio::android::Report err;
    {
        std::lock_guard<std::mutex> lk(s->m);
        err = s->on_error;
    }
    if (err) err("[audio in] " + msg);
}

}  // extern "C"
