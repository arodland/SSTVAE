import QtQuick
import QtQuick.Controls

// Framing the picture, by touching it.
//
// **What is shown is `images::fit`'s output, not a scaled Image with a
// clip on it** — the same rule the desktop's composition preview
// follows, for the same reason. A QML-side imitation of a crop would be
// a second implementation of the framing arithmetic, and the whole point
// of this screen is deciding exactly what goes on the air. Here the
// preview *is* the transmission's first 640x480, re-rendered through the
// same call as the drag moves.
//
// The gestures are the two everyone already has: drag to move, pinch to
// zoom. There are no handles — a desktop crop rectangle with corner
// grabs is a mouse idiom, and `overlay_editor.cpp`'s small drag targets
// are exactly what docs/android.md says not to port.
Item {
    id: root

    required property var transmitter

    // 4:3, always, because that is what the codec sends. Sized to
    // whichever of the two axes runs out first, so the frame never
    // pushes a minimum onto the pane around it — the trap
    // `picture_box.cpp` records, where a fixed height ratcheted the
    // whole window taller and could never lower it again.
    readonly property real frameW: Math.min(width, height * 4 / 3)
    readonly property real frameH: frameW * 3 / 4

    Rectangle {
        id: frame
        width: root.frameW
        height: root.frameH
        anchors.centerIn: parent
        color: "#111"
        border.color: "#444"
        border.width: 1
        clip: true

        // **Two images, swapped on load, so the preview never blanks.**
        //
        // The obvious single `Image` whose `source` follows `previewId`
        // goes empty for the duration of every load, so a drag is a
        // flicker with the picture appearing only once the finger stops
        // — which makes framing something you do by guesswork. Binding
        // one image and revealing it only when it is `Ready` keeps the
        // previous frame up meanwhile, so the preview lags a frame
        // instead of vanishing.
        //
        // The other half is the throttle. `previewId` moves on every
        // touch event and each load is a real `images::fit` — a resize
        // to 640x480 — so requesting one per event would queue work
        // faster than it drains and the preview would fall further
        // behind the longer the drag went on. At most one load is in
        // flight; moves during it collapse into a single `pending`
        // repeat, so the cost is bounded by how fast the renders finish
        // and the last position is always the one that ends up drawn.
        Item {
            id: frames
            anchors.fill: parent

            property Image front: a
            property bool pending: false

            function refresh() {
                const back = front === a ? b : a
                if (back.status === Image.Loading) {
                    pending = true
                    return
                }
                back.source = root.transmitter.hasPicture
                        ? "image://sstvae/compose/" + root.transmitter.previewId
                        : ""
            }

            function settle(who) {
                if (who.status !== Image.Ready)
                    return
                front = who
                if (pending) {
                    pending = false
                    refresh()
                }
            }

            Image {
                id: a
                anchors.fill: parent
                fillMode: Image.PreserveAspectFit
                cache: false
                asynchronous: true
                opacity: frames.front === a ? 1 : 0
                onStatusChanged: frames.settle(a)
            }
            Image {
                id: b
                anchors.fill: parent
                fillMode: Image.PreserveAspectFit
                cache: false
                asynchronous: true
                opacity: frames.front === b ? 1 : 0
                onStatusChanged: frames.settle(b)
            }

            // `previewId` rather than the `changed` signal: that one
            // also fires on the transmitter's twice-a-second status
            // poll, which would re-render the preview for nothing all
            // the way through a transmission.
            readonly property int watched: root.transmitter.previewId
            onWatchedChanged: refresh()
            Component.onCompleted: refresh()
        }

        // Rule-of-thirds guides, only while the picture is being moved.
        // Permanent ones would be drawn over every preview the operator
        // is trying to judge, and this preview is the deliverable.
        Item {
            anchors.fill: parent
            visible: drag.active || pinch.active
            opacity: 0.5
            Repeater {
                model: 2
                Rectangle {
                    required property int index
                    color: "#fff"; width: 1; height: parent.height
                    x: parent.width * (index + 1) / 3
                }
            }
            Repeater {
                model: 2
                Rectangle {
                    required property int index
                    color: "#fff"; height: 1; width: parent.width
                    y: parent.height * (index + 1) / 3
                }
            }
        }

        DragHandler {
            id: drag
            enabled: root.transmitter.hasPicture && !root.transmitter.transmitting
            // Position-based rather than translation-based: a handler's
            // cumulative translation keeps growing while the framing is
            // being clamped at an edge, so releasing and dragging back
            // would do nothing until the accumulated slack unwound.
            property point last
            onActiveChanged: if (active) last = centroid.position
            onCentroidChanged: {
                if (!active)
                    return
                // Negated: dragging right moves the *picture* right,
                // which moves the crop window left.
                root.transmitter.panBy(-(centroid.position.x - last.x) / frame.width,
                                       -(centroid.position.y - last.y) / frame.height)
                last = centroid.position
            }
        }

        PinchHandler {
            id: pinch
            enabled: root.transmitter.hasPicture && !root.transmitter.transmitting
            target: null
            // The handler's own limits are left wide open and the real
            // floor is `Composition`'s: `minZoom` depends on the
            // picture's aspect, so a constant here would either stop
            // short of showing all of a panorama or let a 4:3 source
            // shrink into a frame of black.
            minimumScale: 0.05
            maximumScale: 100.0
            property real base: 1.0
            onActiveChanged: if (active) base = root.transmitter.zoom
            onActiveScaleChanged: root.transmitter.zoom = base * activeScale
        }

        Label {
            anchors.centerIn: parent
            visible: !root.transmitter.hasPicture
            text: "No picture chosen"
            color: "#888"
            z: 1
        }
    }

    // Zoom is discoverable by pinching, but a slider says it exists,
    // shows where the travel ends in both directions, and is the only
    // way back to a round number without fighting the clamp.
    //
    // `from` is the picture's own `minZoom`, so the left end is "all of
    // it, letterboxed" and the right is 8x in. For a 4:3 source the two
    // ends of the pad-out travel coincide and the slider simply starts
    // at 1.
    Slider {
        anchors.top: frame.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 4
        width: frame.width
        visible: root.transmitter.hasPicture
        enabled: !root.transmitter.transmitting
        from: root.transmitter.minZoom
        to: 8.0
        value: root.transmitter.zoom
        onMoved: root.transmitter.zoom = value
    }
}
