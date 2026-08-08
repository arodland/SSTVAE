// The smoke test's native half: a capture session with a decode loop.
//
// **This is deliberately not the Tier 0 app** (docs/android.md). It
// exists to answer the one question that decides the Android port --
// can a phone select an audio device, capture from it, and decode a
// picture -- and it answers it through the *real* code path:
// `audio::CapturePipeline` into `rx::RingBuffer` into `rx::decode_loop`,
// the same objects the desktop app uses, with nothing stubbed but the
// UI. What is throwaway here is the JSON-ish status string and the
// single global session; what is permanent is everything it calls.
//
// The layering rule this file lives under: Java calls into C++ on the
// data path and never the reverse. The capture thread is a Java thread
// that is already attached, so pushing a chunk is a plain call; a C++
// thread calling *back* into Java would need AttachCurrentThread and
// buys nothing, so the decode loop publishes into `SharedState` and the
// UI polls it.

#include <jni.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "audio/audio.hpp"
#include "audio/wavio.hpp"
#include "config.hpp"
#include "images/images.hpp"
#include "rx/engine.hpp"
#include "rx/ringbuffer.hpp"

#ifdef SSTVAE_SMOKE_HAVE_CODEC
#include "codec/codec.hpp"
#endif

#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "sstvae", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "sstvae", __VA_ARGS__)

namespace {

using namespace sstvae;

std::string to_std(JNIEnv* env, jstring s) {
    if (s == nullptr) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c == nullptr ? "" : c);
    if (c != nullptr) env->ReleaseStringUTFChars(s, c);
    return out;
}

// One capture-and-decode session.
//
// The ring buffer is the desktop's, at the desktop's default depth, so
// the retrospective decoding that makes a mid-stream lock recover
// earlier frames works here exactly as it does there.
class Session {
public:
    Session(audio::SampleFormat fmt, int channels, int device_rate,
            const std::string& model_dir, const std::string& out_dir)
        : ring_(130.0),
          pipeline_(fmt, channels, device_rate, config::FS),
          out_dir_(out_dir) {
        rx::RxConfig cfg;
        cfg.out_dir = out_dir;
        cfg.poll_interval = 5.0;

        rx::Decoder decoder = make_decoder(model_dir);

        // The sink writes the picture and remembers where, so the UI can
        // load it. Saving is the sink's job, not the loop's -- the same
        // split the desktop has, and the reason this file can decide to
        // keep the PNG in memory later without touching the engine.
        rx::Sink sink = [this](const rx::Reception& r) -> std::optional<std::string> {
            const std::string path = rx::timestamped_path(out_dir_);
            try {
                images::save_png(r.image, path);
            } catch (const std::exception& e) {
                LOGE("save failed: %s", e.what());
                return std::nullopt;
            }
            {
                std::lock_guard<std::mutex> lk(m_);
                last_saved_ = path;
                ++completed_;
            }
            LOGI("reception saved: %s", path.c_str());
            return path;
        };

        thread_ = std::thread([this, decoder, cfg, sink] {
            try {
                rx::decode_loop(ring_, decoder, state_, cfg, stop_, sink);
            } catch (const std::exception& e) {
                LOGE("decode loop died: %s", e.what());
                std::lock_guard<std::mutex> lk(m_);
                fatal_ = e.what();
            }
        });
    }

    ~Session() {
        stop_.set();
        if (thread_.joinable()) thread_.join();
    }

    void push(const std::byte* data, std::size_t n) {
        const std::vector<double> out =
            pipeline_(std::span<const std::byte>(data, n));
        if (!out.empty()) ring_.write(out);
    }

    // A tiny JSON object. Deliberately hand-rolled: the alternative is a
    // JSON dependency in a throwaway status channel, and the shape is
    // fixed by the one Java reader on the other side.
    std::string status() {
        const rx::Progress p = state_.get();
        std::string saved;
        std::string fatal;
        int completed = 0;
        {
            std::lock_guard<std::mutex> lk(m_);
            saved = last_saved_;
            fatal = fatal_;
            completed = completed_;
        }
        std::string j = "{";
        j += "\"status\":\"" + std::string(rx::status_name(p.status)) + "\"";
        j += ",\"polls\":" + std::to_string(p.polls);
        j += ",\"captured_s\":" + std::to_string(pipeline_.samples_in() /
                                                 static_cast<double>(device_rate_or(1)));
        j += ",\"ring_s\":" + std::to_string(p.seconds_captured);
        j += ",\"frames\":" + std::to_string(p.frames_received.value_or(0));
        j += ",\"expected\":" + std::to_string(p.n_frames_expected.value_or(0));
        j += ",\"progress\":" + std::to_string(p.progress_frac);
        j += ",\"mode\":\"" + p.mode_name.value_or("") + "\"";
        j += ",\"callsign\":\"" + p.callsign + "\"";
        j += ",\"snr\":" + (std::isnan(p.snr_db) ? std::string("null")
                                                 : std::to_string(p.snr_db));
        j += ",\"completed\":" + std::to_string(completed);
        j += ",\"saved\":\"" + saved + "\"";
        j += ",\"error\":\"" + fatal + "\"";
        j += ",\"resampling\":" + std::string(pipeline_.resampling() ? "true" : "false");
        j += "}";
        return j;
    }

    void set_device_rate(int r) { device_rate_ = r; }

    void dump(const std::string& path) {
        const std::vector<double> samples = ring_.snapshot();
        audio::write_wav_float(path, samples);
        LOGI("dumped %zu samples to %s", samples.size(), path.c_str());
    }

private:
    int device_rate_or(int fallback) const {
        return device_rate_ > 0 ? device_rate_ : fallback;
    }

    // With the codec the loop makes pictures; without it, the state
    // machine still runs and still reports frames, SNR and callsign --
    // which is most of what this smoke test is for, and is why
    // `rx::Decoder` being a seam matters here and not only in the tests.
    static rx::Decoder make_decoder(const std::string& model_dir) {
#ifdef SSTVAE_SMOKE_HAVE_CODEC
        auto codec = std::make_shared<codec::OnnxCodec>(
            [model_dir](const std::string& part) {
                return model_dir + "/v3-" + part + "-fp16.onnx";
            });
        // **Load the decoder now, not on first use.** The parts are
        // deliberately lazy -- a receive-only station never touches the
        // encoder -- but laziness puts a missing artifact's error at the
        // worst possible moment: on real hardware this surfaced as
        // "file doesn't exist" *when a picture finally arrived*, after
        // the operator had waited through a whole transmission, with
        // everything up to that point reporting success.
        //
        // `preload` exists for exactly this, and the error propagates
        // out of `Native.start` so it lands on the Start button.
        codec->preload("decoder");
        return [codec](std::span<const double> latents,
                       std::span<const double> weights) {
            return codec->decode(std::vector<double>(latents.begin(), latents.end()),
                                 std::vector<double>(weights.begin(), weights.end()));
        };
#else
        (void)model_dir;
        return [](std::span<const double>, std::span<const double>) {
            return images::Picture{};
        };
#endif
    }

    rx::RingBuffer ring_;
    audio::CapturePipeline pipeline_;
    rx::SharedState state_;
    rx::StopFlag stop_;
    std::thread thread_;
    std::string out_dir_;
    int device_rate_ = 0;

    std::mutex m_;
    std::string last_saved_;
    std::string fatal_;
    int completed_ = 0;
};

std::mutex g_m;
std::unique_ptr<Session> g_session;

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL
Java_org_cleverdomain_sstvae_smoke_Native_start(JNIEnv* env, jclass, jstring jformat,
                                                jint channels, jint device_rate,
                                                jstring jmodel_dir, jstring jout_dir) {
    const std::string format = to_std(env, jformat);
    const std::optional<audio::SampleFormat> fmt =
        audio::sample_format_from_name(format);
    if (!fmt) {
        return env->NewStringUTF(("unsupported sample format: " + format).c_str());
    }
    try {
        std::lock_guard<std::mutex> lk(g_m);
        g_session = std::make_unique<Session>(*fmt, channels, device_rate,
                                              to_std(env, jmodel_dir),
                                              to_std(env, jout_dir));
        g_session->set_device_rate(device_rate);
    } catch (const std::exception& e) {
        return env->NewStringUTF(e.what());
    }
    LOGI("session started: %s x%d @%d Hz", format.c_str(), channels, device_rate);
    return env->NewStringUTF("");
}

// Called from the Java capture thread for every read. A direct
// ByteBuffer, so the bytes are not copied across the boundary.
JNIEXPORT void JNICALL Java_org_cleverdomain_sstvae_smoke_Native_push(JNIEnv* env, jclass,
                                                                     jobject buf,
                                                                     jint length) {
    auto* data = static_cast<std::byte*>(env->GetDirectBufferAddress(buf));
    if (data == nullptr || length <= 0) return;
    std::lock_guard<std::mutex> lk(g_m);
    if (g_session) g_session->push(data, static_cast<std::size_t>(length));
}

JNIEXPORT jstring JNICALL
Java_org_cleverdomain_sstvae_smoke_Native_status(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lk(g_m);
    if (!g_session) return env->NewStringUTF("{\"status\":\"Idle\"}");
    return env->NewStringUTF(g_session->status().c_str());
}

JNIEXPORT void JNICALL Java_org_cleverdomain_sstvae_smoke_Native_stop(JNIEnv*, jclass) {
    std::unique_ptr<Session> doomed;
    {
        std::lock_guard<std::mutex> lk(g_m);
        doomed = std::move(g_session);
    }
    // Destroyed outside the lock: the destructor joins the decode
    // thread, which can be mid-poll for several seconds, and holding
    // g_m across that would block every status() the UI makes.
    doomed.reset();
    LOGI("session stopped");
}

// Write the ring buffer to a WAV, exactly as captured.
//
// This is the Android equivalent of the desktop's `receive.save_audio`,
// and it is here because it is *the* diagnostic for this class of bug:
// two simultaneous captures of one playback, compared. Windows that
// correlate at 1.000 but at a drifting lag prove sample loss rather
// than added noise, and the interval between lag steps names the
// culprit -- which is how the PortAudio/JACK bug was finally pinned.
//
// `write_wav_float` and not `write_wav`, because a dump that rescales
// and rounds has destroyed the evidence it was taken to preserve.
JNIEXPORT jstring JNICALL
Java_org_cleverdomain_sstvae_smoke_Native_dumpAudio(JNIEnv* env, jclass, jstring jpath) {
    const std::string path = to_std(env, jpath);
    std::lock_guard<std::mutex> lk(g_m);
    if (!g_session) return env->NewStringUTF("no session");
    try {
        g_session->dump(path);
    } catch (const std::exception& e) {
        return env->NewStringUTF(e.what());
    }
    return env->NewStringUTF("");
}

JNIEXPORT jboolean JNICALL
Java_org_cleverdomain_sstvae_smoke_Native_hasCodec(JNIEnv*, jclass) {
#ifdef SSTVAE_SMOKE_HAVE_CODEC
    return JNI_TRUE;
#else
    return JNI_FALSE;
#endif
}

}  // extern "C"
