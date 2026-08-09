import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import SSTVAE

// Tier 0's three screens.
//
// Deliberately *not* a port of the desktop's layout (docs/android.md,
// Andrew 2026-08-08): that layout history is a record of QtWidgets on a
// desktop, and reaching for it here would inherit answers to questions
// nobody is asking. A phone in portrait, held one-handed, with no rig
// panel to compete for space, wants a different arrangement -- so the
// waterfall gets real vertical extent as the tuning instrument it is,
// and the three areas are a bottom tab bar rather than a splitter.
ApplicationWindow {
    id: window
    visible: true
    title: "SSTVAE"

    Listener { id: listener }
    PictureList { id: pictures }

    // **Edge-to-edge is mandatory from targetSdk 35 up**, so the window
    // extends under the status bar and the navigation bar and it is on
    // us to inset. Andrew hit the consequence on a device with the
    // 3-button nav bar: the tab bar painted *behind* it and could not
    // be tapped at all. Gesture navigation hides this almost
    // completely -- its inset is a few pixels -- which is exactly why
    // it has to be handled by asking the system rather than by looking
    // at an emulator.
    header: ToolBar {
        topPadding: SafeArea.margins.top

        // `contentItem`, not a child with `anchors.fill: parent` --
        // anchoring to the Control bypasses its padding entirely, so
        // the inset above would have been computed correctly and then
        // ignored, leaving the title under the status bar.
        contentItem: RowLayout {
            Label {
                text: "SSTVAE"
                font.bold: true
                font.pixelSize: 18
                Layout.leftMargin: 12
            }
            Item { Layout.fillWidth: true }
            Label {
                text: listener.modelReady ? "●" : "○"
                color: listener.modelReady ? "#3a3" : "#a60"
                Layout.rightMargin: 12
                // The model indicator lives in the header rather than
                // on the Listen screen, because it applies to the whole
                // app and its absence is the thing that quietly costs
                // you a picture.
                ToolTip.visible: false
            }
        }
    }

    footer: TabBar {
        id: tabs
        bottomPadding: SafeArea.margins.bottom
        TabButton { text: "Listen" }
        TabButton { text: "Pictures"; onClicked: pictures.refresh() }
        TabButton { text: "Settings" }
    }

    StackLayout {
        anchors.fill: parent
        currentIndex: tabs.currentIndex

        // ---- Listen -------------------------------------------------
        ColumnLayout {
            spacing: 8

            // The tuning instrument. First, and given room: with no CAT
            // there is no frequency readout at all, so this is the only
            // feedback the operator has about where the radio is
            // pointed.
            Waterfall {
                Layout.fillWidth: true
                // Taller while there is no picture, which is both when
                // the space is free and when the waterfall is the thing
                // being used: tuning happens before a decode, not
                // during one. Once a picture is arriving the picture is
                // what the operator is looking at, and the strip goes
                // back to being a check that the signal is still there.
                Layout.preferredHeight: listener.hasLiveImage ? 180 : 300
                Behavior on Layout.preferredHeight {
                    NumberAnimation { duration: 150 }
                }
            }

            Image {
                Layout.fillWidth: true
                Layout.fillHeight: true
                fillMode: Image.PreserveAspectFit
                visible: listener.hasLiveImage
                cache: false
                // **Asynchronous, because the provider does real work.**
                // A synchronous provider runs on the GUI thread, so
                // every refresh converts a 640x480 picture there while
                // the decode thread already has the CPU busy. This is
                // the cheapest part of the lag to remove.
                asynchronous: true
                source: listener.hasLiveImage
                        ? "image://sstvae/live/" + listener.liveImageId
                        : ""
            }

            // What the operator sees for most of a session: no picture
            // yet. An empty half-screen of white reads as a broken
            // layout, and it is also the moment with the most to say --
            // whether anything is being heard at all, and that a
            // picture builds up over a minute rather than arriving.
            Item {
                Layout.fillWidth: true
                Layout.fillHeight: true
                visible: !listener.hasLiveImage

                ColumnLayout {
                    anchors.centerIn: parent
                    width: parent.width - 48
                    spacing: 6

                    Label {
                        Layout.fillWidth: true
                        horizontalAlignment: Text.AlignHCenter
                        font.pixelSize: 15
                        color: "#888"
                        text: !listener.listening ? "Not listening"
                            : !listener.modelReady ? "Waiting for the model"
                            : "Listening for a transmission"
                    }
                    Label {
                        Layout.fillWidth: true
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.Wrap
                        font.pixelSize: 12
                        color: "#aaa"
                        text: !listener.listening
                              ? "Tune the radio, then start. The waterfall above shows the band whether or not anything is decoding."
                              : !listener.modelReady
                              ? "Reception has already begun; the picture appears as soon as the model finishes downloading."
                              : "A picture appears here as it decodes, and fills in over the length of the transmission."
                    }
                }
            }

            Label {
                text: listener.status
                // Monospace only when it is a table of numbers. Plain
                // prose in a fixed pitch is the house style of a
                // diagnostic, and this line is not one with the switch
                // off.
                font.family: listener.showTechnical ? "monospace" : Qt.application.font.family
                font.pixelSize: listener.showTechnical ? 12 : 14
                visible: text.length > 0
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.maximumHeight: implicitHeight
                wrapMode: Text.Wrap
            }
            LevelMeter {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                visible: listener.listening
                peak: listener.peakLevel
                dropping: listener.droppingAudio
                technical: listener.showTechnical
            }

            // Empty unless the technical switch is on; see
            // Listener::level.
            Label {
                text: listener.level
                visible: text.length > 0
                font.family: "monospace"
                font.pixelSize: 12
                // The drift line turns red on its own threshold; this
                // is the readout that separates "the band is quiet"
                // from "the capture path is eating your audio".
                color: listener.level.indexOf("DROPPING") >= 0 ? "#c00" : "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.maximumHeight: implicitHeight
                wrapMode: Text.Wrap
            }
            Label {
                text: listener.lastError
                color: "#c00"
                font.pixelSize: 12
                visible: text.length > 0
                Layout.fillWidth: true
                Layout.margins: 12
                Layout.maximumHeight: implicitHeight
                wrapMode: Text.Wrap
            }

            Button {
                Layout.fillWidth: true
                Layout.margins: 12
                text: listener.listening ? "Stop" : "Start listening"
                onClicked: listener.listening ? listener.stop()
                                              : listener.start(deviceBox.currentText)
            }
        }

        // ---- Pictures -----------------------------------------------
        ListView {
            clip: true
            model: pictures
            spacing: 1

            Label {
                anchors.centerIn: parent
                visible: pictures.count === 0
                text: "No receptions yet"
                color: "#888"
            }

            delegate: ItemDelegate {
                width: ListView.view.width
                height: 96
                onClicked: viewer.open(model.path, model.summary)

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 12

                    Image {
                        Layout.preferredWidth: 106
                        Layout.preferredHeight: 80
                        fillMode: Image.PreserveAspectFit
                        source: "image://sstvae/file/" + model.path
                        asynchronous: true
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 2
                        Label { text: model.received; font.pixelSize: 13 }
                        // The metadata comes from the sidecar, not from
                        // shared state -- which is why it is still here
                        // days later.
                        Label {
                            text: model.summary
                            font.family: "monospace"
                            font.pixelSize: 11
                            color: "#666"
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                    }
                }
            }
        }

        // ---- Settings -----------------------------------------------
        ColumnLayout {
            spacing: 12

            Label {
                text: "Audio input"
                font.bold: true
                Layout.margins: 12
                Layout.bottomMargin: 0
            }
            ComboBox {
                id: deviceBox
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                model: listener.inputDevices
                enabled: !listener.listening
            }
            Label {
                text: listener.listening ? listener.audioRoute
                                         : "Choose before starting; the device cannot change mid-session."
                font.family: "monospace"
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }
            Button {
                text: "Rescan devices"
                Layout.leftMargin: 12
                enabled: !listener.listening
                onClicked: listener.refreshDevices()
            }

            Label {
                text: "Model"
                font.bold: true
                Layout.margins: 12
                Layout.bottomMargin: 0
            }
            Label {
                text: listener.modelStatus
                font.family: "monospace"
                font.pixelSize: 11
                color: listener.modelReady ? "#3a3" : "#a60"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }
            Button {
                text: "Retry model download"
                Layout.leftMargin: 12
                visible: !listener.modelReady
                onClicked: listener.loadModel()
            }

            Label {
                text: "Advanced"
                font.bold: true
                Layout.margins: 12
                Layout.bottomMargin: 0
            }
            Switch {
                text: "Show technical details"
                Layout.leftMargin: 4
                checked: listener.showTechnical
                onToggled: listener.showTechnical = checked
            }
            Label {
                // Says what it is *for*, not what it shows. Someone
                // reading this is either curious or being walked
                // through a problem by whoever wrote the app, and the
                // second reader is the one who needs to know the
                // switch exists.
                text: "Signal levels, capture timing, decode cost and poll counts, "
                      + "on the Listen screen and in the notification. Useful when "
                      + "something is not decoding and worth reporting with a bug."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }

            Item { Layout.fillHeight: true }
        }
    }

    // Full-screen view of a saved reception. A Popup rather than a page
    // push: it is a look at one picture, and Back should return to the
    // list rather than unwind a navigation stack.
    Popup {
        id: viewer
        anchors.centerIn: parent
        width: parent.width
        height: parent.height
        modal: true
        property alias source: full.source
        property string caption
        property string path

        function open(p, summary) {
            full.source = "image://sstvae/file/" + p
            viewer.path = p
            viewer.caption = summary
            visible = true
        }

        ColumnLayout {
            anchors.fill: parent
            Image {
                id: full
                Layout.fillWidth: true
                Layout.fillHeight: true
                fillMode: Image.PreserveAspectFit
            }
            Label {
                text: viewer.caption
                font.family: "monospace"
                font.pixelSize: 12
                Layout.fillWidth: true
                Layout.maximumHeight: implicitHeight
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.Wrap
            }
            Button {
                text: "Share"
                Layout.fillWidth: true
                Layout.topMargin: 8
                // Receptions live in app-private storage, so this is
                // the only way a picture leaves the app at all.
                onClicked: listener.sharePicture(viewer.path, viewer.caption)
            }
        }
    }
}
