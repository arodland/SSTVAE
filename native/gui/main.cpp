// The desktop application's entry point.

#include <QApplication>
#include <QIcon>

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

    // The checkpoint fetcher is installed by AppState (inside the
    // window), which supplies the download-progress hook -- the fetch
    // itself cannot start before then, because only the model load
    // triggers it and AppState is what starts that.
    sstvae::gui::MainWindow window;
    window.show();
    return app.exec();
}
