import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// The transmit screen: choose a picture, frame it, send it.
//
// **No overlay, and no automatic callsign caption** (Andrew,
// 2026-08-09). The desktop's transmit panel is an overlay editor because
// a desktop operator composes; here the picture goes out exactly as
// framed. The station is identified by the beacon carrier — which every
// receiver decodes regardless of whether the picture came through well
// enough to read text in — and, if the operator wants it, by a CW ID a
// human can copy by ear. Neither costs a pixel of the picture.
//
// Three things and a button, in the order they are done. Everything with
// a sensible default (mode, level, CW, VOX) lives on Settings, so this
// screen stays the picture and the send.
ColumnLayout {
    id: pane
    spacing: 8

    required property var transmitter

    CropView {
        Layout.fillWidth: true
        Layout.fillHeight: true
        Layout.margins: 8
        transmitter: pane.transmitter
    }

    RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        spacing: 8
        enabled: !pane.transmitter.transmitting

        Button {
            Layout.fillWidth: true
            text: "Choose picture"
            onClicked: pane.transmitter.pickImage()
        }
        Button {
            Layout.fillWidth: true
            text: "Camera"
            onClicked: pane.transmitter.takePhoto()
        }
    }

    // Mode belongs here rather than in Settings, alone among the
    // transmit settings: it is the one an operator changes per
    // transmission and per band condition, and its cost is the airtime
    // printed beside it.
    RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        spacing: 8

        Label { text: "Mode" }
        ComboBox {
            id: modeBox
            Layout.preferredWidth: 90
            enabled: !pane.transmitter.transmitting
            model: pane.transmitter.modes
            currentIndex: Math.max(0, model.indexOf(pane.transmitter.mode))
            onActivated: pane.transmitter.mode = currentValue
        }
        Label {
            text: pane.transmitter.airtime
            color: "#888"
        }
        Item { Layout.fillWidth: true }
    }

    ProgressBar {
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        visible: pane.transmitter.transmitting
        // Indeterminate until the audio is actually playing: encoding
        // and modulating have no meaningful fraction, and a bar sitting
        // at 0% through them reads as a stall.
        indeterminate: pane.transmitter.txProgress <= 0.0
        value: pane.transmitter.txProgress
    }

    Label {
        text: pane.transmitter.txStatus
        visible: text.length > 0
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        Layout.maximumHeight: implicitHeight
        wrapMode: Text.Wrap
    }

    // The encoder is a separate download a receive-only station never
    // makes, so this line is normal on a first visit rather than an
    // error. Hidden once it is ready: nothing to say then.
    Label {
        text: pane.transmitter.encoderStatus
        visible: !pane.transmitter.encoderReady
        color: "#a60"
        font.pixelSize: 12
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        Layout.maximumHeight: implicitHeight
        wrapMode: Text.Wrap
    }

    Label {
        text: pane.transmitter.lastError
        color: "#c00"
        font.pixelSize: 12
        visible: text.length > 0
        Layout.fillWidth: true
        Layout.margins: 12
        Layout.maximumHeight: implicitHeight
        wrapMode: Text.Wrap
    }

    // **A callsign is not required to send** (Andrew, 2026-08-09), and
    // an earlier draft of this gated the button on one. Identifying is
    // required of an amateur station, but this app does not know it is
    // connected to a radio and does not take responsibility for the
    // operator's identification even when it is: the beacon callsign and
    // the CW ID are two ways to do it, and voice is another the app
    // never sees. Refusing to transmit would be the app claiming an
    // authority it does not have.
    Button {
        Layout.fillWidth: true
        Layout.margins: 12
        text: pane.transmitter.transmitting ? "Stop transmitting" : "Send"
        // Cancel is always available while transmitting, whatever else
        // is true: the operator has to be able to get off the air.
        enabled: pane.transmitter.transmitting || pane.transmitter.canSend
        onClicked: pane.transmitter.transmitting ? pane.transmitter.cancel()
                                                 : pane.transmitter.send()
    }
}
