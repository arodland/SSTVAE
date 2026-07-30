// The desktop application's entry point.

#include <QApplication>
#include <QIcon>

#include "checkpoint/qt_fetcher.hpp"
#include "main_window.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("SSTVAE"));
    QCoreApplication::setOrganizationName(QStringLiteral("SSTVAE"));
    // Also the identifier the desktop uses to match a window to its
    // .desktop file. Wayland has no other way to do it, so without this
    // the taskbar shows an unnamed, iconless second entry beside the
    // launcher the user clicked.
    QGuiApplication::setDesktopFileName(
        QStringLiteral("org.cleverdomain.sstvae"));

    // From the compiled-in resource, so it is present in the build tree,
    // in an AppImage and inside a .app alike. Two sizes; Qt picks.
    QIcon icon;
    icon.addFile(QStringLiteral(":/sstvae-48.png"));
    icon.addFile(QStringLiteral(":/sstvae-256.png"));
    QApplication::setWindowIcon(icon);

    // Make the published checkpoint fetchable before anything asks for
    // it. Only the *download* needs this; resolving an explicit --model
    // and finding a warm cache are path arithmetic in sstvae_core, so a
    // second run works with the network gone.
    sstvae::checkpoint::install_qt_fetcher();

    sstvae::gui::MainWindow window;
    window.show();
    return app.exec();
}
