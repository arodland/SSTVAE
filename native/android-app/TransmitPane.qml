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

    // Why Send is disabled, when it is disabled for a reason the
    // operator can fix. A control that is off and silent sends people to
    // look at the wrong thing.
    Label {
        text: pane.transmitter.cwIdProblem
        color: "#c00"
        font.pixelSize: 12
        visible: text.length > 0
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
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
    // authority it does not have. The first-transmit prompt below is
    // the same position stated out loud rather than a departure from it.
    Button {
        Layout.fillWidth: true
        Layout.margins: 12
        text: pane.transmitter.transmitting ? "Stop transmitting" : "Send"
        // Cancel is always available while transmitting, whatever else
        // is true: the operator has to be able to get off the air.
        enabled: pane.transmitter.transmitting || pane.transmitter.canSend
        onClicked: {
            if (pane.transmitter.transmitting) {
                pane.transmitter.cancel()
            } else if (pane.transmitter.needsFirstTransmitPrompt) {
                // The prompt is reached *through* Send rather than
                // shown on first launch: someone who only ever listens
                // should never see it, and the moment it is worth
                // reading is the moment before the first over.
                firstTransmit.open()
            } else {
                pane.transmitter.send()
            }
        }
    }

    // The first-transmit prompt. Shown once per install.
    //
    // **Not a licence check and not a hard gate**, and it is worth being
    // clear about why. The app cannot tell whether it is connected to a
    // radio at all; the service may not be amateur; the operator may be
    // identifying by voice, or on a band where the rules are different,
    // or handling the whole question themselves. Refusing to transmit
    // would be the app claiming an authority it does not have -- the
    // same argument that keeps a callsign optional (see the Send button
    // note above). What this is is a roadblock to casual misuse:
    // somebody who has not thought about any of it has now been asked
    // to, once, in the place where it matters.
    //
    // So the callsign and the CW ID here are *offers*, and the only
    // thing that is required is the acknowledgement.
    Popup {
        id: firstTransmit
        parent: Overlay.overlay
        anchors.centerIn: parent
        width: Math.min(parent.width - 32, 520)
        // **Bounded and scrolled, never merely tall.** This is a long
        // read with two controls and a button row, and on a short
        // screen a Popup that outgrows its parent does not compress --
        // it puts the buttons off the bottom, where there is no way to
        // reach them and nothing to say so. Same failure the settings
        // tabs have a QScrollArea for on the desktop.
        height: Math.min(parent.height - 32, content.implicitHeight + 32)
        modal: true
        // No click-outside-to-dismiss: the acknowledgement should be
        // declined by a button that says so, not by a stray tap.
        closePolicy: Popup.NoAutoClose

        contentItem: Flickable {
            contentHeight: content.implicitHeight
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}

        ColumnLayout {
            id: content
            width: parent.width
            spacing: 8

            Label {
                text: "Before your first transmission"
                font.bold: true
                font.pixelSize: 18
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }
            Label {
                text: "Transmitting on the amateur bands requires a license. "
                      + "It's your job to know and follow all applicable laws "
                      + "about permitted frequencies, modes, and identifying "
                      + "your station. They depend on your license and your "
                      + "country, and the app can't do it for you."
                font.pixelSize: 13
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }

            Label {
                text: "Callsign (optional)"
                font.bold: true
                Layout.topMargin: 8
                Layout.fillWidth: true
            }
            TextField {
                Layout.fillWidth: true
                placeholderText: "Callsign"
                text: pane.transmitter.callsign
                // Per keystroke, for the same reason the Settings field
                // is: the back gesture dismisses the keyboard without
                // ever firing editingFinished.
                onTextEdited: pane.transmitter.callsign = text
                inputMethodHints: Qt.ImhUppercaseOnly | Qt.ImhNoPredictiveText
            }
            Label {
                text: "Sent on the beacon carrier with every transmission, "
                      + "and decoded by SSTVAE receivers. You can set this "
                      + "later in Settings if you don't enter it now."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }

            Switch {
                text: "Send a CW ID"
                Layout.topMargin: 8
                checked: pane.transmitter.cwId
                onToggled: pane.transmitter.cwId = checked
            }
            Label {
                text: "Morse at the end of each transmission, for "
                      + "identification purposes and so that anyone "
                      + "listening can discover what mode it is."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }
            // The one combination that cannot work. Reachable from here
            // because this screen can create it: turn the switch on,
            // leave the callsign empty.
            Label {
                text: pane.transmitter.cwIdProblem
                font.pixelSize: 11
                color: "#c00"
                visible: text.length > 0
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }

            // **A CheckBox beside a Label, not a CheckBox with a
            // wrapping `contentItem`.** Overriding the content item was
            // the obvious way to get a multi-line label and it puts the
            // indicator in the *middle of the text*: the control centres
            // its indicator against the whole content height, so a
            // three-line label leaves the box floating over line two.
            // Two widgets in a row have no such interaction, and the
            // label becomes a tap target as well, which a 24 px box on
            // a phone badly needs.
            RowLayout {
                Layout.topMargin: 8
                Layout.fillWidth: true
                spacing: 8

                CheckBox {
                    id: acknowledge
                    Layout.alignment: Qt.AlignTop
                }
                Label {
                    text: "I hold any license required to transmit on the "
                          + "frequencies I am using, and I am responsible for "
                          + "operating this app legally."
                    font.pixelSize: 13
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    MouseArea {
                        anchors.fill: parent
                        onClicked: acknowledge.toggle()
                    }
                }
            }

            RowLayout {
                Layout.topMargin: 8
                Layout.fillWidth: true
                spacing: 8
                Button {
                    Layout.fillWidth: true
                    text: "Not now"
                    // Closes without recording the acknowledgement, so
                    // the prompt comes back on the next Send. Declining
                    // is a real option and it costs nothing but the
                    // transmission.
                    onClicked: firstTransmit.close()
                }
                Button {
                    Layout.fillWidth: true
                    text: "Transmit"
                    enabled: acknowledge.checked
                             && pane.transmitter.cwIdProblem.length === 0
                    onClicked: {
                        pane.transmitter.acknowledgeFirstTransmit()
                        firstTransmit.close()
                        pane.transmitter.send()
                    }
                }
            }
        }
        }
    }
}
