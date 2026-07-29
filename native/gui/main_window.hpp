// The application window.
//
// A straight port of `MainWindow` in `sstvae/gui/app.py`: a tab widget
// with the receive and transmit panels, a File menu, and a status bar
// carrying callsign, rig and model state.
//
// The wiring between the panels lives here rather than in either of
// them, because all of it is about the pair: half duplex (our own
// transmission must not be decoded back into a received picture),
// pausing frequency polling while keyed, and handing the most recent
// received picture to the transmitter as an overlay inset.

#ifndef SSTVAE_GUI_MAIN_WINDOW_HPP
#define SSTVAE_GUI_MAIN_WINDOW_HPP

#include <QMainWindow>

class QLabel;
class QTabWidget;

namespace sstvae::gui {

class AppState;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

public slots:
    void open_settings();

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void build_menu();
    void build_status_bar();
    void update_station_label();
    void on_model_loaded();

    AppState* state_ = nullptr;
    QTabWidget* tabs_ = nullptr;
    QLabel* station_label_ = nullptr;
    QLabel* rig_label_ = nullptr;
    QLabel* model_label_ = nullptr;
};

}  // namespace sstvae::gui

#endif
