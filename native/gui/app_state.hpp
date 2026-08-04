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

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "codec/codec.hpp"
#include "log/log.hpp"
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

    // A read-only config directory must not break the session -- but it
    // must not be silent either: a failure is logged.
    void save_config();

    // --- status log ----------------------------------------------------
    // Append an entry. Thread-safe; callable from any thread. Every
    // entry also reaches the file writer and the `logEntry` signal.
    void log_event(const char* source, log::Severity severity,
                   const QString& text);

    // For the log pane: backfill via snapshot(), then follow logEntry.
    const log::StatusLog& status_log() const { return log_; }
    // Empty while the file log is healthy; a description otherwise.
    QString log_file_note() const;
    // Where the file log is being written, or empty if there is none.
    // Asked rather than re-derived: a second copy of this path would
    // keep compiling after the writer's location moved, and it is the
    // path an operator is told to attach to a bug report.
    QString log_file_path() const;

signals:
    void modelLoaded();
    // `error` distinguishes a CAT failure from a routine status; the
    // text alone cannot (failure text is the backend's exception).
    void rigStatus(const QString& text, bool error);
    // Every log entry, queued to the GUI thread for the pane.
    void logEntry(qlonglong ms, const QString& source, int severity,
                  const QString& text);
    // The file log's first failure, whenever it happens. Emitted from
    // inside a StatusLog sink -- i.e. under the log's lock -- so the
    // consumer MUST connect with Qt::QueuedConnection; a direct slot
    // that calls log_event would deadlock on the non-recursive mutex.
    void fileLogFailed(const QString& description);
    // Model artifact download, live. Emitted from the model thread.
    void modelProgress(qlonglong received, qlonglong total);

private:
    settings::Config config_;

    log::StatusLog log_;
    std::unique_ptr<log::FileWriter> file_log_;
    // First-failure guard for fileLogFailed; the failure itself is
    // permanent (FileWriter stops writing), so one report is the truth.
    std::atomic<bool> file_log_reported_{false};

    mutable std::mutex model_mutex_;
    std::unique_ptr<codec::OnnxCodec> model_;
    QString model_error_;
    std::thread model_thread_;

    rig::RigController rig_;
};

}  // namespace sstvae::gui

#endif
