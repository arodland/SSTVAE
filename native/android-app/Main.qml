import QtQuick
import QtQuick.Controls
import SSTVAE

ApplicationWindow {
    visible: true
    title: "SSTVAE"

    Listener { id: listener }

    Column {
        anchors.fill: parent
        anchors.margins: 16
        anchors.topMargin: 48
        spacing: 12

        Text { text: "SSTVAE"; font.pixelSize: 28; font.bold: true }

        ComboBox {
            id: devices
            width: parent.width
            model: listener.inputDevices
            enabled: !listener.listening
        }

        Button {
            width: parent.width
            text: listener.listening ? "Stop" : "Start"
            onClicked: listener.listening ? listener.stop()
                                          : listener.start(devices.currentText)
        }

        Text {
            text: listener.modelStatus
            color: listener.modelReady ? "#080" : "#a60"
            font.family: "monospace"
            font.pixelSize: 12
            width: parent.width
            wrapMode: Text.Wrap
        }
        Text { text: listener.status; font.family: "monospace"; font.pixelSize: 12 }
        Text { text: listener.audioRoute; font.family: "monospace"; font.pixelSize: 12 }
        Text { text: listener.level; font.family: "monospace"; font.pixelSize: 12 }
        Text {
            text: listener.lastError
            color: "#c00"
            font.pixelSize: 12
            width: parent.width
            wrapMode: Text.Wrap
        }
    }
}
