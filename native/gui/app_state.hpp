// Shared, mutable application state.
//
// A straight port of `AppState` in `sstvae/gui/app.py`, and it exists
// for the same reason: the receive and transmit panels both need the
// configuration, the codec and the rig, and neither should have to
// reach into the other to get them.
//
// Two threading rules carry over unchanged, because both were bought
// with real bugs:
//
// **The codec loads on a worker thread.** Resolving the published
// checkpoint can mean an HTTP download on first run, and a window that
// takes thirty seconds to appear looks broken.
//
// **Nothing here may call the rig.** Every rig operation is blocking,
// and a radio that is powered off costs the timeout twice. All of it
// lives on `RigController`'s own thread; what this class exposes is the
// last *cached* answer and a way to post work.

#ifndef SSTVAE_GUI_APP_STATE_HPP
#define SSTVAE_GUI_APP_STATE_HPP

#include <QObject>
#include <QString>

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "codec/codec.hpp"
#include "rig/controller.hpp"
#include "settings/settings.hpp"

namespace sstvae::gui {

class AppState : public QObject {
    Q_OBJECT

public:
    explicit AppState(QObject* parent = nullptr);
    ~AppState() override;

    settings::Config& config() { return config_; }
    const settings::Config& config() const { return config_; }

    // --- codec ---------------------------------------------------------
    // Start loading in the background; `modelLoaded` follows either way.
    void load_model_async();

    // Null until loaded, and null again if loading failed -- which is
    // why `model_error` is separate rather than encoded as an empty
    // pointer with no explanation.
    codec::OnnxCodec* model();
    QString model_error() const;

    // --- rig -----------------------------------------------------------
    // The last polled dial frequency: a cached value, never a request.
    std::optional<double> current_frequency_hz() const;

    // What the transmit engine keys, or nothing if rig control is off.
    // Returning nothing rather than a no-op is what tells the engine
    // there is nothing to key -- VOX or a manual PTT switch.
    std::function<void(bool)> ptt();

    void connect_rig();
    void disconnect_rig();
    void pause_rig_polling();
    void resume_rig_polling();

    // A read-only config directory must not break the session.
    void save_config();

signals:
    void modelLoaded();
    void rigStatus(const QString& text);

private:
    settings::Config config_;

    mutable std::mutex model_mutex_;
    std::unique_ptr<codec::OnnxCodec> model_;
    QString model_error_;
    std::thread model_thread_;

    rig::RigController rig_;
};

}  // namespace sstvae::gui

#endif
