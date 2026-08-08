import QtQuick
import QtQuick.Controls

ApplicationWindow {
    visible: true
    title: "SSTVAE"

    Column {
        anchors.centerIn: parent
        spacing: 16
        Text { text: "SSTVAE"; font.pixelSize: 32; font.bold: true }
        Text { text: waveformText; font.family: "monospace"; horizontalAlignment: Text.AlignHCenter }
    }
}
