import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Rig control settings: CAT and PTT over USB, Bluetooth or a network.
//
// A section of the Settings tab rather than a tab of its own. The tab
// bar already carries five and a sixth is a crowded row on a phone --
// and this is a screen an operator visits once per radio, not once per
// session. What *is* worth seeing every session, the dial frequency,
// goes on the Listen screen where the waterfall is.
//
// **Why any of this exists**: `docs/android.md` recorded rig control as
// structurally impossible here, because Hamlib's serial layer opens a
// device path and Android gives an unprivileged app none. Hamlib takes
// a socket for any backend, so the radio list on this screen is the
// desktop's radio list. See `core/rig/transport.hpp`.
ColumnLayout {
    id: pane
    spacing: 8

    // Set by the parent, so this file does not reach out of itself for
    // the object it drives.
    required property var rig

    // Re-enumerate whenever this becomes visible: USB devices come and
    // go with the cable, and a picker showing a radio that was
    // unplugged five minutes ago is worse than an empty one.
    onVisibleChanged: if (visible) rig.refreshDevices()

    Label {
        text: "Rig control"
        font.bold: true
        Layout.margins: 12
        Layout.bottomMargin: 0
    }

    Switch {
        text: "Control the radio"
        Layout.leftMargin: 4
        checked: pane.rig.enabled
        onToggled: pane.rig.enabled = checked
    }

    Label {
        // Says what it is *for*, not what it is. An operator who is
        // acoustically coupling a phone to a handheld has no radio to
        // talk to and should be able to tell that from one sentence.
        text: "Reads the dial frequency and can key the transmitter, over a "
              + "USB serial adapter, a Bluetooth radio, or a station computer "
              + "on the network. Leave it off if the phone is not wired to a "
              + "radio — receiving and VOX transmitting need none of it."
        font.pixelSize: 11
        color: "#666"
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        wrapMode: Text.Wrap
    }

    // Everything below is meaningless with the switch off, and hiding it
    // rather than disabling it keeps the Settings page short for the
    // majority who never turn this on.
    ColumnLayout {
        Layout.fillWidth: true
        spacing: 8
        visible: pane.rig.enabled

        // ---- how it is connected -------------------------------------
        Label {
            text: "Connection"
            Layout.leftMargin: 12
            Layout.rightMargin: 12
        }
        ComboBox {
            id: connectionBox
            Layout.fillWidth: true
            Layout.leftMargin: 12
            Layout.rightMargin: 12
            textRole: "text"
            valueRole: "value"
            model: [
                { text: "USB serial", value: "usb" },
                { text: "Bluetooth", value: "bluetooth" },
                { text: "Network (rigctld or ser2net)", value: "network" }
            ]
            Component.onCompleted: currentIndex = indexOfValue(pane.rig.connection)
            onActivated: {
                pane.rig.connection = currentValue
                pane.rig.refreshDevices()
            }
        }

        // ---- which radio ---------------------------------------------
        Label {
            text: "Radio"
            Layout.leftMargin: 12
            Layout.rightMargin: 12
        }
        Button {
            Layout.fillWidth: true
            Layout.leftMargin: 12
            Layout.rightMargin: 12
            text: pane.rig.modelLabel
            onClicked: modelDialog.open()
        }

        // ---- which device --------------------------------------------
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            visible: pane.rig.connection !== "network"

            Label {
                text: "Device"
                Layout.leftMargin: 12
                Layout.rightMargin: 12
            }
            ComboBox {
                id: deviceCombo
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                model: pane.rig.devices
                textRole: "label"
                valueRole: "id"
                enabled: !pane.rig.running
                // Rebound whenever the list changes, because the index
                // of a given device is not stable across a replug.
                function sync() { currentIndex = indexOfValue(pane.rig.device) }
                Component.onCompleted: sync()
                Connections {
                    target: pane.rig
                    function onDevicesChanged() { deviceCombo.sync() }
                }
                onActivated: pane.rig.device = currentValue
            }
            RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                Button {
                    text: "Refresh"
                    onClicked: pane.rig.refreshDevices()
                }
                Button {
                    // **In front of Connect rather than behind it.**
                    // Android will not open a USB device without
                    // permission and forgets it on every detach, so the
                    // alternative is a connection that fails with a
                    // message the operator can do nothing about from the
                    // screen they are looking at.
                    text: "Grant access"
                    visible: !pane.rig.devicePermitted
                    onClicked: pane.rig.requestPermission()
                }
                Item { Layout.fillWidth: true }
            }
            Label {
                // Three different empty lists, and they are not the same
                // problem: Bluetooth access not granted, nothing paired,
                // and nothing plugged in. Telling an operator to go and
                // pair a radio they already paired is the kind of wrong
                // advice that ends in a bug report.
                text: pane.rig.connection === "bluetooth" && !pane.rig.bluetoothReady
                      ? "Android needs Bluetooth access before paired radios can "
                        + "be listed."
                      : pane.rig.devices.length === 0
                        ? (pane.rig.connection === "bluetooth"
                           ? "No paired radios. Pair the radio in Android's Bluetooth "
                             + "settings first — this app deliberately does not scan, "
                             + "so it never asks for location access."
                           : "Nothing plugged in. Connect the radio or its serial "
                             + "adapter, then Refresh.")
                        : (pane.rig.devicePermitted
                           ? ""
                           : "Android needs permission before this device can be opened.")
                visible: text !== ""
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }
        }

        // ---- or which host -------------------------------------------
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            visible: pane.rig.connection === "network"

            Label {
                text: "Host"
                Layout.leftMargin: 12
                Layout.rightMargin: 12
            }
            TextField {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                placeholderText: "192.168.1.10:4532"
                text: pane.rig.host
                inputMethodHints: Qt.ImhNoAutoUppercase | Qt.ImhNoPredictiveText
                onEditingFinished: pane.rig.host = text
            }
            Label {
                text: "With the radio set to \"Hamlib NET rigctl\" this is a "
                      + "computer running rigctld — the usual way to share one "
                      + "radio with WSJT-X. With a real radio chosen instead, it "
                      + "is a serial-over-network server such as ser2net. The "
                      + "port defaults to 4532."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }
        }

        // ---- speed ----------------------------------------------------
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            visible: pane.rig.connection === "usb"

            Label {
                text: "Speed"
                Layout.leftMargin: 12
                Layout.rightMargin: 12
            }
            ComboBox {
                id: baudBox
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                model: pane.rig.bauds
                enabled: !pane.rig.running
                Component.onCompleted: {
                    currentIndex = pane.rig.baud === 0
                                   ? 0 : Math.max(0, model.indexOf(String(pane.rig.baud)))
                }
                onActivated: pane.rig.baud = currentIndex === 0
                                             ? 0 : parseInt(currentValue)
            }
            Label {
                // "Default" is not a synonym for one of the numbers, and
                // saying so is the difference between an operator
                // leaving it alone and guessing.
                text: "Default uses whatever speed Hamlib would set for the radio "
                      + "chosen above, which is right for almost every rig."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }
        }

        // ---- keying ---------------------------------------------------
        Label {
            text: "Transmit keying"
            Layout.leftMargin: 12
            Layout.rightMargin: 12
        }
        ComboBox {
            id: pttBox
            Layout.fillWidth: true
            Layout.leftMargin: 12
            Layout.rightMargin: 12
            // The list itself narrows on Bluetooth: RFCOMM has no modem
            // control lines, so DTR and RTS are not offered rather than
            // offered and failing when somebody presses Send.
            model: pane.rig.pttMethods
            enabled: !pane.rig.running
            function label(v) {
                switch (v) {
                case "vox": return "VOX (the audio keys the radio)"
                case "cat": return "CAT command"
                case "dtr": return "DTR line"
                case "rts": return "RTS line"
                }
                return v
            }
            displayText: label(currentValue !== undefined ? currentValue : pane.rig.pttMethod)
            delegate: ItemDelegate {
                width: pttBox.width
                text: pttBox.label(modelData)
                highlighted: pttBox.highlightedIndex === index
            }
            function sync() { currentIndex = Math.max(0, model.indexOf(pane.rig.pttMethod)) }
            Component.onCompleted: sync()
            Connections {
                target: pane.rig
                function onChanged() { if (!pttBox.pressed) pttBox.sync() }
            }
            onActivated: pane.rig.pttMethod = currentValue
        }
        Label {
            text: "VOX means this app does not key the radio at all — the "
                  + "transmit audio does, and the leader tone on the Send screen "
                  + "is what trips it. Anything else keys the radio directly, and "
                  + "the leader is skipped."
            font.pixelSize: 11
            color: "#666"
            Layout.fillWidth: true
            Layout.leftMargin: 12
            Layout.rightMargin: 12
            wrapMode: Text.Wrap
        }

        // ---- connect --------------------------------------------------
        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 12
            Layout.rightMargin: 12
            Button {
                text: pane.rig.running ? "Disconnect" : "Connect"
                onClicked: pane.rig.running ? pane.rig.disconnectRig()
                                            : pane.rig.connectRig()
            }
            Label {
                Layout.fillWidth: true
                // The status text, and whether it was a failure, come
                // from two properties rather than one: a failure message
                // is whatever the backend's exception said and has no
                // shape to recognise it by.
                text: pane.rig.status
                color: pane.rig.failed ? "#c62828" : "#666"
                font.pixelSize: 12
                wrapMode: Text.Wrap
            }
        }
        Label {
            text: pane.rig.frequency
            visible: text !== ""
            font.pixelSize: 16
            Layout.leftMargin: 12
            Layout.rightMargin: 12
        }
    }

    // ---- the radio picker ---------------------------------------------
    //
    // A dialog with a search box, because Hamlib knows several hundred
    // rigs. `findModels` caps what it returns, so this list is never the
    // whole set at once.
    Dialog {
        id: modelDialog
        parent: Overlay.overlay
        anchors.centerIn: parent
        width: Math.min(parent ? parent.width - 24 : 320, 480)
        height: Math.min(parent ? parent.height - 80 : 400, 560)
        modal: true
        title: "Radio"
        standardButtons: Dialog.Cancel

        onOpened: {
            search.text = ""
            results.model = pane.rig.findModels("")
            search.forceActiveFocus()
        }

        ColumnLayout {
            anchors.fill: parent
            spacing: 8

            TextField {
                id: search
                Layout.fillWidth: true
                placeholderText: "Search — maker or model"
                inputMethodHints: Qt.ImhNoAutoUppercase | Qt.ImhNoPredictiveText
                onTextChanged: results.model = pane.rig.findModels(text)
            }
            ListView {
                id: results
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                ScrollBar.vertical: ScrollBar {}
                delegate: ItemDelegate {
                    required property var modelData
                    width: results.width
                    text: modelData.label
                    onClicked: {
                        pane.rig.model = modelData.number
                        modelDialog.close()
                    }
                }
            }
        }
    }
}
