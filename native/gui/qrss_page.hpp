// The main window's QRSSTVAE mode: the QRSS spectrogram
// (gui/qrss_spectrogram.hpp) and, beside it across a divider, the
// received pane of tiles (gui/qrss_window.hpp), with a row above them
// for receiving audio at all and for the transmit schedule.
//
// Receiving is the receive pane's: one sound-card stream feeds both the
// SSTVAE decoder and the QRSS listener, so Start here and Start on the
// SSTVAE mode are one switch, shown the same in both places. The page only
// asks for it (`listenRequested`) and shows what it is told
// (`set_listening`), so it holds no audio of its own and a test can drive
// it without a sound card.

#ifndef SSTVAE_GUI_QRSS_PAGE_HPP
#define SSTVAE_GUI_QRSS_PAGE_HPP

#include <QString>
#include <QWidget>

class QLabel;
class QPushButton;
class QSplitter;

namespace sstvae::gui {

class QrssSpectrogram;
class QrssWindow;

class QrssPage : public QWidget {
    Q_OBJECT

public:
    // Takes `signals_` into its layout.
    explicit QrssPage(QrssWindow* signals_view, QWidget* parent = nullptr);

    QrssWindow* signals_view() const { return signals_; }
    QrssSpectrogram* spectrogram() const { return spectrogram_; }
    QSplitter* splitter() const { return splitter_; }
    // The Received pane's width beside the spectrogram, in pixels; 0
    // leaves the initial split. Applied once the page has its real width.
    void set_received_width(int px);
    int received_width() const;

public slots:
    void set_listening(bool on);
    // While a send is on the air receiving is suspended (half duplex) and
    // comes back by itself afterwards, so the button turns red and halts
    // the send instead (`haltRequested`).
    void set_transmitting(bool on);
    // A send getting ready before its quarter hour: nothing is keyed and
    // receiving carries on, so the button stays as it is and the line
    // under it says what the send is doing. Empty clears it.
    void set_preparing(const QString& message);
    void set_schedule_summary(const QString& text);

signals:
    // Start (true) or stop (false) receiving.
    void listenRequested(bool on);
    void scheduleRequested();
    // Stop the send on the air.
    void haltRequested();
    // The operator dragged the divider: the Received pane's new width.
    void receivedWidthChanged(int px);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    void apply_received_width();
    int pending_received_w_ = 0;
    QrssWindow* signals_ = nullptr;
    QrssSpectrogram* spectrogram_ = nullptr;
    QSplitter* splitter_ = nullptr;
    QPushButton* listen_ = nullptr;
    QLabel* listen_state_ = nullptr;
    QLabel* schedule_summary_ = nullptr;
    QPushButton* details_ = nullptr;
    bool listening_ = false;
    bool transmitting_ = false;
    QString preparing_;
    void show_state();
    QPalette normal_palette_;
};

}  // namespace sstvae::gui

#endif
