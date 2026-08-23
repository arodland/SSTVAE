import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// A collapsible group for the Settings page.
//
// The Settings tab used to be one long scroll of every control the app
// has, with the debug switches and the multi-sentence help text inline
// between the things an operator actually reaches for -- so finding one
// setting meant reading past all the others, and the debug toggle sat in
// the same column as the callsign. Each group is now a section that
// starts collapsed: the page is a short, scannable list of headings, and
// the descriptive text and the rarely-touched controls are one tap away
// rather than always on screen.
//
// **Sub-sections rather than a sub-tab bar.** The bottom tab bar already
// carries five and `RigPane.qml` records why a sixth was refused on a
// phone; a second strip of tabs *inside* the page would be the same
// crowding one level down, and it would not shorten the page the way
// collapsing does. Collapsing is also what keeps the back gesture simple
// -- there is no navigation stack to pop, so `Main.qml`'s deliberately
// careful `onClosing` is untouched.
//
// The header carries an optional one-line `summary` of the section's
// current state, the way Android's own settings show a preference's
// value under its title -- so "is the rig on, is the model ready" is
// answerable without expanding anything.
ColumnLayout {
    id: section

    // Children written between the tags land in the expandable body.
    default property alias content: body.data
    property string title
    property string summary: ""
    property bool expanded: false

    Layout.fillWidth: true
    spacing: 0

    // The whole header is the touch target, not a chevron beside it: a
    // thumb on a phone should not have to find a 24 px glyph.
    ItemDelegate {
        Layout.fillWidth: true
        padding: 12
        onClicked: section.expanded = !section.expanded
        contentItem: RowLayout {
            spacing: 8
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 0
                Label {
                    text: section.title
                    font.bold: true
                    font.pixelSize: 15
                }
                Label {
                    text: section.summary
                    visible: text.length > 0
                    font.pixelSize: 11
                    color: "#888"
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
            }
            // Rotated rather than swapped for a different glyph, so the
            // change reads as one control turning rather than two
            // characters trading places.
            Label {
                text: "▼"
                color: "#888"
                rotation: section.expanded ? 180 : 0
                Behavior on rotation { NumberAnimation { duration: 120 } }
            }
        }
    }

    // A hairline under a collapsed header, so the closed list reads as a
    // list. Dropped while open, where the body's own spacing separates it
    // from the next section.
    Rectangle {
        Layout.fillWidth: true
        Layout.preferredHeight: 1
        color: "#33808080"
        visible: !section.expanded
    }

    // A ColumnLayout with `visible: false` is excluded from its parent
    // layout entirely, so a collapsed section costs only its header --
    // which is the whole point. `Component.onCompleted` on the controls
    // inside still runs (creation is independent of visibility), so a
    // ComboBox that reads its index once is correct on first expand.
    ColumnLayout {
        id: body
        Layout.fillWidth: true
        Layout.bottomMargin: section.expanded ? 8 : 0
        visible: section.expanded
        spacing: 8
    }
}
