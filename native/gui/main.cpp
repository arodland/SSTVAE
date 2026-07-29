// The desktop application's entry point.

#include <QApplication>

#include "checkpoint/qt_fetcher.hpp"
#include "main_window.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("SSTVAE"));
    QCoreApplication::setOrganizationName(QStringLiteral("SSTVAE"));

    // Make the published checkpoint fetchable before anything asks for
    // it. Only the *download* needs this; resolving an explicit --model
    // and finding a warm cache are path arithmetic in sstvae_core, so a
    // second run works with the network gone.
    sstvae::checkpoint::install_qt_fetcher();

    sstvae::gui::MainWindow window;
    window.show();
    return app.exec();
}
