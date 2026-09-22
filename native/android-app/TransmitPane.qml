import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// The transmit screen: choose a picture, frame it, send it.
//
// **There was no overlay here, and that was a decision** (Andrew,
// 2026-08-09) -- reversed 2026-09-14 (see `composition.hpp`). The
// station is still identified by the beacon carrier -- which every
// receiver decodes regardless of whether the picture came through well
// enough to read text in -- and, if the operator wants it, by a CW ID a
// human can copy by ear; neither of those can say *whom* a transmission
// is addressed to, which is what a template's `{theircall}` is for.
// There is still no editor here (docs/overlay-templates.md step 4, not
// built) -- only a row of template chips and a small fields row, and an
// automatic callsign caption is still not offered: `{mycall}` exists
// only inside a template the operator chose, on purpose.
//
// Everything with a sensible default (mode, level, CW, VOX) lives on
// Settings, so this screen stays the picture, the template, and the
// send.
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

    // **Templates (docs/overlay-templates.md).** Chips, not a combo box
    // -- a phone picks between a handful of things by tapping, not by
    // opening a menu first. "None" is always first and is how a
    // template is left; choosing one *replaces* the whole composition,
    // same as the desktop's combo, so switching mid-QSO is a deliberate
    // tap rather than an accident. The preview above is already
    // `Composition::preview()`, so nothing further is needed to show
    // the result -- the same rule every other preview in this project
    // follows.
    RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        spacing: 8

        // Named, like the Mode row below: a bare row of buttons said
        // nothing about what picking one does.
        Label { text: "Template" }

        ListView {
            id: chips
            Layout.fillWidth: true
            // The chips' own height, not a number: at 40 the Basic
            // style's buttons were clipped top and bottom by a few
            // pixels, and a constant would be wrong again on the next
            // style or font.
            Layout.preferredHeight: contentItem.childrenRect.height
            orientation: ListView.Horizontal
            spacing: 6
            clip: true
            model: pane.transmitter.templateNames
            delegate: Button {
                text: modelData
                highlighted: index === pane.transmitter.templateIndex
                onClicked: pane.transmitter.templateIndex = index
                // Hold to delete -- the platform's own gesture for "do
                // something to this one". Built-ins get the same dialog
                // with Delete disabled, so the gesture is never silent.
                onPressAndHold: {
                    deleteTemplate.index = index;
                    deleteTemplate.name = modelData;
                    deleteTemplate.open();
                }
            }

            // Taking a template from another station. It rides at the
            // end of the chips rather than on Settings because this is
            // where templates are, and there is nowhere else on this
            // screen that would not be a menu -- which is the thing the
            // chip row exists instead of. There is no editor here (step
            // 5), so importing is the only way a phone gets a template
            // it did not ship with.
            footer: Button {
                text: "Import…"
                flat: true
                onClicked: importTemplate.open()
            }
        }

        // A clipped row reads as complete; this says there is more.
        // Outside the list rather than a fade over it, so it needs no
        // knowledge of the background colour.
        Label {
            text: "›"
            font.pixelSize: 22
            opacity: 0.6
            visible: chips.contentWidth - chips.contentX > chips.width + 1
        }
    }

    // **The fields the current template uses, and only those.**
    // `theircall` gets its own line because it is the one field that can
    // block Send; every custom `{field}` lives behind a pop-up, because
    // a template may declare several and an empty one is the normal
    // state rather than something worth keeping in view (same reasoning
    // as the desktop's "Reply fields" box -- a pop-up rather than a pane
    // that would have to grow and shrink the strip around it).
    RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        spacing: 8
        visible: pane.transmitter.wantsTheirCall || pane.transmitter.hasCustomFields

        TextField {
            Layout.fillWidth: true
            visible: pane.transmitter.wantsTheirCall
            placeholderText: "Their call"
            text: pane.transmitter.theirCall
            // Per keystroke, for the reason every other field on this
            // screen is: the back gesture dismisses the keyboard
            // without ever firing editingFinished.
            onTextEdited: pane.transmitter.theirCall = text
            inputMethodHints: Qt.ImhUppercaseOnly | Qt.ImhNoPredictiveText
        }
        Button {
            text: pane.transmitter.customFieldsLabel
            visible: pane.transmitter.hasCustomFields
            onClicked: customFields.open()
        }
    }

    // Blocks Send, same tier as `cwIdProblem` -- "  de KC2G" on the air
    // is the whole point of a reply template gone wrong.
    Label {
        text: pane.transmitter.templateFieldProblem
        color: "#c00"
        font.pixelSize: 12
        visible: text.length > 0
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        Layout.maximumHeight: implicitHeight
        wrapMode: Text.Wrap
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
        // Which of the two keying stories this over will use. Only worth
        // a line when there is a choice to have got wrong -- with no rig
        // session VOX is the only thing there is, and saying so every
        // time trains the eye to skip the line.
        text: pane.transmitter.keying
        visible: pane.transmitter.rigKeyed
        font.pixelSize: 11
        color: "#666"
        Layout.fillWidth: true
        Layout.leftMargin: 12
        Layout.rightMargin: 12
        wrapMode: Text.Wrap
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

    // The custom-field pop-up (docs/overlay-templates.md). A sheet of
    // labelled text boxes, one per `{field}` the current template
    // declares, in the order they first appear -- never mandatory, so
    // leaving one blank just drops its line (rule 2 of
    // `sstvae/overlay/template.py`).
    //
    // `fieldLabels` is captured on open rather than bound: `Transmitter`
    // exposes the label list as `Q_INVOKABLE`, not a property (see its
    // header for why -- the desktop's is a pop-up for the identical
    // reason), so nothing re-evaluates it on its own. The button that
    // opens this is visible only while the current template has
    // custom fields, so the list is always current at the moment it
    // matters.
    Popup {
        id: customFields
        parent: Overlay.overlay
        anchors.centerIn: parent
        width: Math.min(parent.width - 32, 480)
        height: Math.min(parent.height - 32, fieldsContent.implicitHeight + 32)
        modal: true
        property var fieldLabels: []
        onAboutToShow: fieldLabels = pane.transmitter.customFieldLabels()

        contentItem: Flickable {
            contentHeight: fieldsContent.implicitHeight
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}

        ColumnLayout {
            id: fieldsContent
            width: parent.width
            spacing: 8

            Label {
                text: "Never mandatory — leave one blank and its line is left out."
                font.pixelSize: 11
                color: "#666"
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }

            Repeater {
                model: customFields.fieldLabels
                delegate: ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 2
                    Label { text: modelData; font.bold: true; font.pixelSize: 12 }
                    TextField {
                        Layout.fillWidth: true
                        text: pane.transmitter.customFieldValue(modelData)
                        onTextEdited: pane.transmitter.setCustomFieldValue(modelData, text)
                    }
                }
            }

            Button {
                Layout.fillWidth: true
                Layout.topMargin: 8
                text: "Done"
                onClicked: customFields.close()
            }
        }
        }
    }

    // Scan or paste, and both land in the same `importTemplate`. The
    // scanner is Google Play services' (see TemplateScanner.java), so
    // the popup closes when it opens -- it takes the whole screen and
    // reports back through `lastError` or by the new chip appearing
    // selected. Paste is the path for a phone with no camera, or one
    // where Play services has not got the scanner module yet.
    Popup {
        id: importTemplate
        parent: Overlay.overlay
        anchors.centerIn: parent
        width: Math.min(parent.width - 32, 480)
        height: Math.min(parent.height - 32, importContent.implicitHeight + 32)
        modal: true
        onAboutToShow: {
            importField.text = "";
            importResult.text = "";
        }

        contentItem: Flickable {
            contentHeight: importContent.implicitHeight
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}

        ColumnLayout {
            id: importContent
            width: parent.width
            spacing: 8

            Label {
                text: "Import a template"
                font.bold: true
                font.pixelSize: 18
                Layout.fillWidth: true
            }
            Label {
                text: "Scan the desktop app's Share window, or paste its text."
                font.pixelSize: 13
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }
            Button {
                text: "Scan QR code"
                Layout.fillWidth: true
                onClicked: {
                    importTemplate.close();
                    pane.transmitter.scanTemplate();
                }
            }
            TextArea {
                id: importField
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                wrapMode: TextEdit.WrapAnywhere
                placeholderText: "{\"version\": 1, …}"
            }
            Label {
                id: importResult
                Layout.fillWidth: true
                wrapMode: Text.Wrap
                font.pixelSize: 13
                visible: text !== ""
            }
            RowLayout {
                Layout.fillWidth: true
                Button {
                    text: "Paste"
                    onClicked: importField.paste()
                }
                Item { Layout.fillWidth: true }
                Button {
                    text: "Cancel"
                    onClicked: importTemplate.close()
                }
                Button {
                    text: "Import"
                    enabled: importField.text.trim() !== ""
                    onClicked: {
                        // The message is shown here rather than in a
                        // toast: "that does not look like a template"
                        // is about the text still on screen, and
                        // closing the popup to say so would throw away
                        // what the operator would need to fix.
                        importResult.text = pane.transmitter.importTemplate(importField.text);
                        if (importResult.text === "") importTemplate.close();
                    }
                }
            }
        }
        }
    }

    // Confirming a delete. One dialog for both cases: a built-in shows
    // the same sheet with Delete disabled and a line saying why, because
    // a long-press that does nothing looks like a gesture that does not
    // exist.
    Popup {
        id: deleteTemplate
        parent: Overlay.overlay
        anchors.centerIn: parent
        width: Math.min(parent.width - 32, 420)
        modal: true
        property int index: -1
        property string name: ""
        property string problem: ""
        readonly property bool deletable: pane.transmitter.templateDeletable(index)
        onAboutToShow: problem = ""

        contentItem: ColumnLayout {
            spacing: 8
            Label {
                text: "Delete template \u201c" + deleteTemplate.name + "\u201d?"
                font.bold: true
                font.pixelSize: 18
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }
            Label {
                text: deleteTemplate.deletable
                      ? "This removes it from the phone. Import it again to get it back."
                      : "Built-in templates cannot be deleted."
                font.pixelSize: 13
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }
            Label {
                text: deleteTemplate.problem
                visible: text !== ""
                font.pixelSize: 13
                Layout.fillWidth: true
                wrapMode: Text.Wrap
            }
            RowLayout {
                Layout.fillWidth: true
                Item { Layout.fillWidth: true }
                Button {
                    text: "Cancel"
                    onClicked: deleteTemplate.close()
                }
                Button {
                    text: "Delete"
                    enabled: deleteTemplate.deletable
                    onClicked: {
                        deleteTemplate.problem = pane.transmitter.deleteTemplate(deleteTemplate.index);
                        if (deleteTemplate.problem === "") deleteTemplate.close();
                    }
                }
            }
        }
    }
}
