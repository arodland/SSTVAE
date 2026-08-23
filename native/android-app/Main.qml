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
    Transmitter { id: transmitter }
    PictureList { id: pictures }
    // Owns nothing: the rig session lives in `Session`, so keying
    // survives a rotation in the middle of an over.
    // **`rigControl`, not `rig`.** A `RigPane` declares `required
    // property var rig`, so inside its binding block the name `rig`
    // resolves to that property rather than to an outer id -- `rig: rig`
    // is a binding loop, not a hand-off. Naming the id differently is
    // what makes the two unambiguous.
    RigControl { id: rigControl }

    // **Back closes the picture viewer instead of the app.**
    //
    // Android delivers the back gesture (and the 3-button Back) as a
    // close request on the window, and nothing was refusing it — so
    // backing out of a full-screen reception ended the activity, which
    // is not what Back means when something is open on top. `Popup`'s
    // own `CloseOnEscape` does not cover this: that is Qt::Key_Escape,
    // and Android sends a close request, not an Escape.
    //
    // Declining the close is the whole fix for that case.
    //
    // **At the root, Back backgrounds the app rather than ending it,
    // whenever there is a session to protect.** Ending the activity ends
    // the *process*, and the process is what owns the engine — so the
    // single most ordinary gesture on a phone silently killed a
    // reception in progress and left the shade's promise that we were
    // listening untrue. Measured on API 36: the process died on SIGABRT
    // in Android's own HWUI teardown, which is a tombstone and would be
    // counted as a native crash, and the notification's last text froze
    // wherever the poller had left it.
    //
    // Backgrounding is what recorders, navigation and media apps do, and
    // it is what the ongoing notification already implies: the session
    // continues, the poller keeps the notification honest, and the task
    // stays in Recents so the launcher or the notification returns to
    // the screen that was left. Stopping is still one tap away — the
    // notification's own Stop action — which is where an operator
    // already looks for it.
    //
    // With nothing running there is nothing to protect, so Back leaves
    // the app exactly as Android users expect. Hijacking Back
    // unconditionally is the thing this deliberately does not do.
    onClosing: function(close) {
        if (viewer.visible) {
            viewer.close()
            close.accepted = false
        } else if (listener.listening || transmitter.transmitting) {
            listener.moveToBackground()
            close.accepted = false
        }
    }

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
        // The encoder is a second 9 MB artifact, fetched on the first
        // visit to this tab rather than at startup: a station that only
        // ever listens must never pay for it, which is the whole reason
        // `load_codec`'s parts are lazy and independent.
        TabButton { text: "Send"; onClicked: transmitter.loadEncoder() }
        TabButton { text: "Pictures"; onClicked: pictures.refresh() }
        TabButton { text: "Settings" }
        TabButton { text: "About" }
    }

    StackLayout {
        anchors.fill: parent
        currentIndex: tabs.currentIndex

        // ---- Listen -------------------------------------------------
        ColumnLayout {
            spacing: 8

            // The dial frequency, when there is a rig session to read it
            // from.
            //
            // **This is new, and the comment below it used to be the
            // whole story.** "With no CAT there is no frequency readout
            // at all" was true until Hamlib turned out to take a socket
            // (core/rig/transport.hpp), and the waterfall was the only
            // thing telling an operator where the radio was pointed.
            // It still is when this is absent, which is every session
            // without a cable — so the waterfall keeps its height and
            // this is one line above it rather than a panel beside it.
            RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                Layout.topMargin: 4
                visible: rigControl.running
                // The dial, when the radio is actually answering.
                Label {
                    text: rigControl.frequency
                    visible: rigControl.frequency !== ""
                    font.pixelSize: 18
                }
                // ...and what is wrong when it is not. This replaced a
                // bare "—" beside a screen that still said Connected:
                // with the cable pulled out the operator got a dash and
                // no account of it. A healthy rig says nothing here,
                // because a line that is present every second is one
                // the eye learns to skip.
                Label {
                    text: rigControl.connectionState
                    visible: rigControl.frequency === ""
                    color: "#c62828"
                    font.pixelSize: 13
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
                Item { Layout.fillWidth: true; visible: rigControl.frequency !== "" }
            }

            // The tuning instrument. First, and given room: without a
            // rig session there is no frequency readout at all, so this
            // is the only feedback the operator has about where the
            // radio is pointed.
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
                        // **Transmitting has to be said out loud here.**
                        // Half duplex stops capture for the whole over,
                        // so the waterfall freezes and nothing decodes —
                        // which is byte for byte what a wedged capture
                        // looks like. The desktop learned this when the
                        // receive pane stayed on screen through an over.
                        text: transmitter.transmitting ? "Transmitting"
                            : !listener.listening ? "Not listening"
                            : !listener.modelReady ? "Waiting for the model"
                            : "Listening for a transmission"
                    }
                    Label {
                        Layout.fillWidth: true
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.Wrap
                        font.pixelSize: 12
                        color: "#aaa"
                        text: transmitter.transmitting
                              ? "Receiving is paused until the transmission finishes, and resumes on its own."
                              : !listener.listening
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
                // Half duplex owns the audio path for the duration of an
                // over, and the session resumes by itself afterwards --
                // so starting or stopping it by hand here would either
                // do nothing or fight the resume.
                enabled: !transmitter.transmitting
                text: listener.listening ? "Stop" : "Start listening"
                onClicked: listener.listening ? listener.stop()
                                              : listener.start(deviceBox.currentText)
            }
        }

        // ---- Send ---------------------------------------------------
        TransmitPane {
            transmitter: transmitter
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
                onClicked: viewer.showPicture(model.path, model.summary)

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
        //
        // **Scrolled, because a settings page that does not fit does not
        // compress -- it truncates**, and what goes first is the
        // explanatory text at the bottom of the last section. The
        // desktop has the identical construct for the identical reason
        // (`QScrollArea` per settings tab): there is no default size
        // that is right on every panel, and a reader cannot tell
        // clipped text from text that simply ends. It became true here
        // the moment the transmit settings landed -- before them the
        // page fitted, and "Model" was the first thing to disappear
        // behind the tab bar.
        //
        // Horizontal scrolling off: the width is the window's, so a
        // horizontal bar could only ever mean a label refusing to wrap.
        ScrollView {
            id: settingsScroll
            contentWidth: availableWidth
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
            clip: true

        ColumnLayout {
            // **Bound to the ScrollView by id, and `parent` will not do
            // it.** A ScrollView reparents its content into a Flickable's
            // contentItem, so `parent` here is neither the ScrollView nor
            // anything with its width -- and the layout then takes its
            // own implicit width, which is wider than the viewport. The
            // symptom is not a missing scrollbar but text running off the
            // right edge with nothing to scroll it back.
            width: settingsScroll.availableWidth
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
            // **One button, both directions.** It sits under "Audio
            // input" because that is where the first device picker is,
            // but plugging in a USB interface adds a capture *and* a
            // playback device at the same instant, and a rescan that
            // refreshed only the list you happened to be looking at is
            // a button that appears not to work. Two buttons would be
            // worse: two lists that can disagree about when they were
            // last looked at.
            Button {
                text: "Rescan audio devices"
                Layout.leftMargin: 12
                enabled: !listener.listening && !transmitter.transmitting
                onClicked: {
                    listener.refreshDevices()
                    transmitter.refreshDevices()
                }
            }
            Label {
                text: "Refreshes both the input and output lists."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }

            Label {
                text: "Station"
                font.bold: true
                Layout.margins: 12
                Layout.bottomMargin: 0
            }
            TextField {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                placeholderText: "Callsign"
                text: transmitter.callsign
                // Written on every keystroke rather than on editing
                // finished: a phone keyboard is dismissed by the back
                // gesture as often as by a done key, and that path fires
                // no editingFinished at all — so the callsign would be
                // silently lost by the most ordinary way of leaving the
                // field.
                onTextEdited: transmitter.callsign = text
                inputMethodHints: Qt.ImhUppercaseOnly | Qt.ImhNoPredictiveText
            }
            Label {
                text: "Sent on the beacon carrier with every transmission, and "
                      + "used by the CW ID. Every receiver decodes it, so nothing "
                      + "is written into the picture."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }

            Label {
                text: "Transmit"
                font.bold: true
                Layout.margins: 12
                Layout.bottomMargin: 0
            }
            ComboBox {
                id: outputBox
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                model: transmitter.outputDevices
                enabled: !transmitter.transmitting
                currentIndex: Math.max(0, model.indexOf(transmitter.outputDevice))
                onActivated: transmitter.outputDevice = currentValue
            }
            RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                Label { text: "Level" }
                Slider {
                    Layout.fillWidth: true
                    from: 0.1
                    to: 1.0
                    value: transmitter.level
                    enabled: !transmitter.transmitting
                    onMoved: transmitter.level = value
                }
                Label {
                    text: Math.round(transmitter.level * 100) + "%"
                    color: "#666"
                }
            }
            Label {
                text: "Set this so the radio's ALC barely moves. The waveform is "
                      + "already clipped to its designed 4.2 dB peak-to-average; "
                      + "driving it harder splatters rather than getting out further."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }

            Switch {
                text: "VOX leader tone"
                Layout.leftMargin: 4
                checked: transmitter.voxLead > 0
                enabled: !transmitter.transmitting
                onToggled: transmitter.voxLead = checked ? 0.5 : 0.0
            }
            Label {
                text: "Half a second of swept tone before each transmission, to "
                      + "bring a VOX-keyed radio up before the signal starts. "
                      + "Leave it off if the radio is keyed any other way — it is "
                      + "airtime."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }

            Switch {
                text: "CW identification"
                Layout.leftMargin: 4
                checked: transmitter.cwId
                enabled: !transmitter.transmitting
                onToggled: transmitter.cwId = checked
            }
            TextField {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                visible: transmitter.cwId
                text: transmitter.cwMessage
                onTextEdited: transmitter.cwMessage = text
            }
            Label {
                text: "Morse at 18 wpm after the picture, under the same key-up. "
                      + "{callsign} is replaced. The default also advertises the "
                      + "mode, so someone who hears the signal and does not know "
                      + "what it is can find out."
                font.pixelSize: 11
                color: "#666"
                visible: transmitter.cwId
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }
            // Shown here as well as on the transmit screen, because this
            // is the screen the fix is on. Send is disabled while this
            // has anything to say.
            Label {
                text: transmitter.cwIdProblem
                font.pixelSize: 11
                color: "#c00"
                visible: text.length > 0
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }

            Label {
                text: "Receive"
                font.bold: true
                Layout.margins: 12
                Layout.bottomMargin: 0
            }
            // Both take effect on the *next* Start, not mid-session --
            // same as the device picker above, and disabled the same
            // way, so there is nothing here that looks live while
            // listening and silently isn't.
            Switch {
                text: "Wide frequency search"
                Layout.leftMargin: 4
                enabled: !listener.listening
                checked: listener.blindWide
                onToggled: listener.blindWide = checked
            }
            Label {
                text: "Picks up a station whose dial is off by up to 625 Hz, at "
                      + "some extra CPU cost. Only affects picking up a "
                      + "transmission already in progress — finding the *start* "
                      + "of one is always this wide, because there it's free."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }
            RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                Label { text: "Track drift" }
                ComboBox {
                    id: driftTrackBox
                    Layout.fillWidth: true
                    enabled: !listener.listening
                    textRole: "text"
                    valueRole: "value"
                    model: [
                        { text: "Off", value: "off" },
                        { text: "Slow (drifting radio)", value: "slow" },
                        { text: "Fast (VHF, satellite)", value: "fast" }
                    ]
                    Component.onCompleted: currentIndex = indexOfValue(listener.driftTrack)
                    onActivated: listener.driftTrack = currentValue
                }
            }
            Label {
                // Off/slow/fast, not a gain: the two settings are not
                // more and less of one thing, so "fast" is not simply
                // better at everything "slow" does — see the desktop
                // dialog's identical note.
                text: "Follows a carrier that moves during a transmission. Off "
                      + "suits HF with a modern radio, which usually doesn't "
                      + "drift far enough to matter. Fast is for VHF, satellite "
                      + "and EME use, where it costs more if the signal is "
                      + "fading heavily rather than drifting."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }

            RigPane {
                rig: rigControl
                Layout.fillWidth: true
            }

            Label {
                text: "Received pictures"
                font.bold: true
                Layout.margins: 12
                Layout.bottomMargin: 0
            }
            Switch {
                text: "Save to gallery"
                Layout.leftMargin: 4
                checked: listener.saveToGallery
                onToggled: listener.saveToGallery = checked
            }
            Label {
                // Says what it does *and* what follows from it. The
                // consequence is the part an operator cannot guess:
                // "save to gallery" sounds local, and on a phone with
                // photo backup switched on it is not.
                text: "Copies each reception into Pictures/SSTVAE, where the "
                      + "gallery and Google Photos will show it as a device "
                      + "folder. Whatever arrives on the band goes into your "
                      + "camera roll — and into your photo backup, if you have "
                      + "one. Receptions are always kept in the app either way."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
            }
            Label {
                // The export runs in the service, after the operator has
                // stopped looking, so this is the only place a failure
                // can surface at all. Cleared by the next success.
                text: "Last export failed: " + listener.galleryError
                visible: listener.saveToGallery && listener.galleryError !== ""
                font.pixelSize: 11
                color: "#a60"
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                wrapMode: Text.Wrap
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

            // Breathing room under the last control, so it is not
            // pressed against the tab bar at the end of the scroll.
            Item { Layout.preferredHeight: 24 }
        }
        }

        // ---- About --------------------------------------------------
        ColumnLayout {
            spacing: 12

            Item { Layout.fillHeight: true }

            Label {
                text: "SSTVAE"
                font.pixelSize: 32
                font.bold: true
                Layout.alignment: Qt.AlignHCenter
            }
            Label {
                text: "by Andrew Rodland, KC2G"
                Layout.alignment: Qt.AlignHCenter
            }
            Label {
                text: "Image transmission over HF radio."
                Layout.alignment: Qt.AlignHCenter
            }
            Label {
                text: "Version " + Application.version
                Layout.alignment: Qt.AlignHCenter
                opacity: 0.7
            }

            // Buttons rather than rich-text links: a link inside a
            // paragraph is a touch target the width of the words.
            Button {
                text: "GitHub"
                Layout.alignment: Qt.AlignHCenter
                onClicked: Qt.openUrlExternally("https://github.com/arodland/SSTVAE")
            }
            Button {
                text: "Discord"
                Layout.alignment: Qt.AlignHCenter
                onClicked: Qt.openUrlExternally("https://discord.gg/UKUFmMR75")
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

        // **Not called `open`.** `Popup` already has an `open()`, and
        // shadowing it with a different signature leaves the type's own
        // machinery calling something that is no longer its method.
        function showPicture(p, summary) {
            full.source = "image://sstvae/file/" + p
            viewer.path = p
            viewer.caption = summary
            open()
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
